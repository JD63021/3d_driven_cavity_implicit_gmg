function driver_ns_3d_optimized()
% 3D Navier-Stokes BDF2 + SUPG + Restart Capability
% Updated for BDF2 Time Stepping with "Lowest Tau" stabilization.
% 1. Uses BDF1 for startup (step 1).
% 2. Uses BDF2 for steps > 1.
% 3. Uses "Effective" parameters to reuse existing kernels.

setenv('OMP_NUM_THREADS', '32');
maxNumCompThreads(32);
clearvars -except setenv; clc;

%% ---------------- User Parameters ----------------
Re_book = 2000;          
L_domain = 1;           
mu = L_domain / Re_book; 
rho = 1.0;

% Time Stepping
T_final = 200;
dt = 0.1;               
time_steps = ceil(T_final/dt);

% Output & Restart
output_stride = 5;         
backup_stride = 20;       
restart_file  = sprintf('latest_restart_Re%d.mat', Re_book);
pvd_filename  = sprintf('ns_bdf2_Re%d.pvd', Re_book);
DO_RESTART    = false;      

% Grid
Nx = 32; Ny = 32; Nz = 32;                 

% Stabilization
CI = 4;                 
gamma_gd = 0*mu;         

% Solver
maxPicard = 15; tolPicard = 1e-5; omega = 0.9;            
restart = 100; maxit = 1000; tol_gmres_base = 1e-3;
tol_rel = 1e-6; % Tighter relative tolerance for solution increment

fprintf('--- 3D NS BDF2 Optimized ---\n');
fprintf('Grid: %dx%dx%d | dt: %.4f | Re: %d\n', Nx, Ny, Nz, dt, Re_book);

%% ---------------- 1. Grid & FEM Setup ----------------
[mesh, fem] = setup_grid_and_fem(Nx, Ny, Nz, L_domain);
Nn = mesh.Nn; Np = mesh.Np; ndof = fem.ndof;

%% ---------------- 2. Initialization / Restart Logic ----------------
ts_start = 1;
time = 0;
frame_count = 0;

% Current Solution
Ux = zeros(mesh.nx, mesh.ny, mesh.nz);
Uy = zeros(mesh.nx, mesh.ny, mesh.nz);
Uz = zeros(mesh.nx, mesh.ny, mesh.nz);
Pp = zeros(Nx+1, Ny+1, Nz+1);

% History Terms (n and n-1)
Ux_n = Ux; Uy_n = Uy; Uz_n = Uz;
Ux_nn = Ux; Uy_nn = Uy; Uz_nn = Uz;

pvd_data = {};

if DO_RESTART && isfile(restart_file)
    fprintf('>> Found restart file: %s. Loading...\n', restart_file);
    loaded = load(restart_file);
    
    Ux = loaded.Ux; Uy = loaded.Uy; Uz = loaded.Uz; Pp = loaded.Pp;
    % For restart, we assume the immediate previous step is valid.
    % We don't always have n-1 in standard restarts. 
    % Strategy: We treat the first step AFTER restart as BDF1 to rebuild history safely.
    Ux_n = Ux; Uy_n = Uy; Uz_n = Uz; 
    
    time = loaded.time;
    ts_start = loaded.ts_next; 
    frame_count = loaded.frame_count;
    pvd_data = loaded.pvd_data;
    
    fprintf('   Resuming from Time Step %d (t=%.4f)\n', ts_start, time);
else
    % BC Initialization for Fresh Start
    [vel_mask, valUx, valUy, valUz, vel_dofs, p_pin] = setup_bcs(mesh);
    Ux(vel_mask) = valUx(vel_mask); 
    Uy(vel_mask) = valUy(vel_mask); 
    Uz(vel_mask) = valUz(vel_mask);
    
    % Initial history is same as IC
    Ux_n = Ux; Uy_n = Uy; Uz_n = Uz;
    Ux_nn = Ux; Uy_nn = Uy; Uz_nn = Uz;
end

% Reload BCs 
[vel_mask, valUx, valUy, valUz, vel_dofs, p_pin] = setup_bcs(mesh);

%% ---------------- 3. PRE-ASSEMBLY (STATIC PHASES) ----------------
fprintf('>> Assembling Global Static Linear Matrices...\n');
[K_static, M_global, Mp, Kp, Rup] = assemble_static_linear_parts(mesh, fem, mu, gamma_gd);

% --- BDF MATRICES ---
M_global_blk = blkdiag(M_global, M_global, M_global, sparse(Np,Np));

% 1. BDF1 Matrix (Used for Step 1 or Restart)
% Coeff: 1/dt
coef_bdf1 = rho/dt;
K_linear_bdf1 = K_static + coef_bdf1 * M_global_blk;

% 2. BDF2 Matrix (Used for Step 2+)
% Coeff: 3/(2*dt)
coef_bdf2 = (1.5 * rho) / dt;
K_linear_bdf2 = K_static + coef_bdf2 * M_global_blk;

% Preconditioner Hierarchies
fprintf('>> Building Preconditioner Static Hierarchies...\n');
% Note: We assume BDF2 timescale (1.5/dt) for the static preconditioner
% because it is used for 99% of the simulation.
[pcd_static, gmg_Mp, gmg_Ap, gmg_vel_static] = setup_pcd_static(mesh, fem, mu, dt, rho);

%% ---------------- 4. Time Loop ----------------
total_gmres_iters = 0;

for ts = ts_start:time_steps
    time = time + dt;
    fprintf('\n=== Time Step %d / %d (t=%.3f) ===\n', ts, time_steps, time);
    
    % --- Step Type Selection ---
    if ts == 1 || (DO_RESTART && ts == ts_start)
        % STARTUP: Use BDF1
        is_bdf2 = false;
        K_linear_total = K_linear_bdf1;
        fprintf('   [Integrator]: BDF1 (Startup)\n');
        
        % BDF1 RHS History: (rho/dt) * u_n
        u_n_vec = [Ux_n(:); Uy_n(:); Uz_n(:)];
        b_history = coef_bdf1 * (M_global_blk * [u_n_vec; zeros(Np,1)]);
        
        % Params for MEX
        dt_eff = dt;
        Ux_eff = Ux_n; Uy_eff = Uy_n; Uz_eff = Uz_n;
        
    else
        % NORMAL: Use BDF2
        is_bdf2 = true;
        K_linear_total = K_linear_bdf2;
        % fprintf('   [Integrator]: BDF2\n'); 
        
        % BDF2 RHS History: (rho/2dt) * (4u_n - u_nn)
        u_n_vec  = [Ux_n(:); Uy_n(:); Uz_n(:)];
        u_nn_vec = [Ux_nn(:); Uy_nn(:); Uz_nn(:)];
        
        % Construct 4*u_n - u_nn
        rhs_comb = 4*u_n_vec - u_nn_vec;
        
        % Multiply by mass matrix and coeff (rho / 2dt)
        b_history = (0.5 * rho / dt) * (M_global_blk * [rhs_comb; zeros(Np,1)]);
        
        % --- EFFECTIVE INPUTS FOR MEX ---
        % We trick the BDF1 MEX to do BDF2:
        % 1. dt_eff = 2/3 * dt. This makes mass coeff 1/dt_eff = 3/2dt.
        % 2. u_eff  = 4/3 u_n - 1/3 u_nn. This maps the residual correctly.
        
        dt_eff = dt * (2.0/3.0);
        
        Ux_eff = (4.0/3.0)*Ux_n - (1.0/3.0)*Ux_nn;
        Uy_eff = (4.0/3.0)*Uy_n - (1.0/3.0)*Uy_nn;
        Uz_eff = (4.0/3.0)*Uz_n - (1.0/3.0)*Uz_nn;
    end
    
    % Picard Loop
    for it = 1:maxPicard
        u_curr_vec = [Ux(:); Uy(:); Uz(:); Pp(:)];
        
        % --- A. Dynamic Assembly (Nonlinear) ---
        % Pass EFFECTIVE dt and EFFECTIVE history to reuse kernel
        [K_nl, b_supg] = assemble_nonlinear_supg(mesh, fem, Ux, Uy, Uz, ...
                                                 Ux_eff, Uy_eff, Uz_eff, ...
                                                 rho, mu, dt_eff, CI, gamma_gd);
        
        % --- B. Combine System ---
        K = K_linear_total + K_nl;
        b = b_history + b_supg;
        
        clear K_nl b_supg; 
        
        % --- C. Apply BCs ---
        [K, b] = apply_boundary_conditions(K, b, vel_dofs, valUx, valUy, valUz, p_pin, mesh, fem);
        
        % --- D. Residual Check (Informational Only) ---
        res_norm = norm(K * u_curr_vec - b);
        fprintf('   Iter %d | Abs Res: %.3e', it, res_norm);
        % REMOVED PREMATURE BREAK HERE to ensure we check du/u
        
        % --- E. Update Preconditioner ---
        % Note: We pass standard dt here, but setup_pcd_static used BDF2 scaling
        Mfun = update_pcd_picard(K, Kp, Mp, Rup, Ux, Uy, Uz, mesh, fem, ...
                                 pcd_static, gmg_Mp, gmg_Ap, gmg_vel_static, ...
                                 rho, dt, CI, mu, p_pin);
        % 
        % % --- F. Solve ---
        x0 = u_curr_vec;
        tol_gmres = max(1e-12, tol_gmres_base * res_norm);
        [U_gmres, ~, relres, iter] = gmres(K, b, restart, tol_gmres, maxit, Mfun, [], x0);

        clear Mfun;

        its = (iter(1)-1)*restart + iter(2);
        total_gmres_iters = total_gmres_iters + its;
        fprintf(' | GMRES: %d (rel %.1e)', its, relres);
        % U_gmres = K\b;
        % 
        U_new = (1-omega)*u_curr_vec + omega*U_gmres;
        
        % --- G. Relative Increment Check (Robust Convergence) ---
        du_norm = norm(U_new - u_curr_vec);
        u_norm  = norm(U_new);
        rel_err = du_norm / (u_norm + 1e-12);
        
        fprintf(' | dU/U: %.2e', rel_err);

        % Unpack Solution
        Ux = reshape(U_new(fem.dof_u), mesh.nx, mesh.ny, mesh.nz);
        Uy = reshape(U_new(fem.dof_v), mesh.nx, mesh.ny, mesh.nz);
        Uz = reshape(U_new(fem.dof_w), mesh.nx, mesh.ny, mesh.nz);
        Pp = reshape(U_new(fem.dof_p), Nx+1, Ny+1, Nz+1);
        
        % Check Convergence (Require both small residual AND small update)
        if rel_err < tol_rel && res_norm < tolPicard
            fprintf(' -> Converged.\n');
            clear u_curr_vec U_gmres U_new;
            break; 
        else
            fprintf('\n');
        end
        
        clear u_curr_vec U_gmres U_new;
    end
    
    % --- End Step: Update History ---
    % Shift n -> nn, current -> n
    Ux_nn = Ux_n; Uy_nn = Uy_n; Uz_nn = Uz_n;
    Ux_n = Ux;    Uy_n = Uy;    Uz_n = Uz;
    
    % --- OUTPUT & SAVING ---
    if mod(ts, output_stride) == 0 || ts == time_steps
        frame_count = frame_count + 1;
        save_vtu_output(Ux, Uy, Uz, Pp, mesh, Re_book, frame_count);
        pvd_data{end+1} = struct('time', time, 'file', sprintf('ns_opt_Re%d_%04d.vtu', Re_book, frame_count));
        write_pvd(pvd_filename, pvd_data);
    end
    
    if mod(ts, output_stride) == 0 || mod(ts, backup_stride) == 0 || ts == time_steps
        ts_next = ts + 1; 
        % Note: We only save Ux, Uy, Uz. On restart, we accept one step of BDF1.
        save(restart_file, 'Ux', 'Uy', 'Uz', 'Pp', 'time', 'ts_next', 'frame_count', 'pvd_data');
    end
    
    if mod(ts, backup_stride) == 0
        backup_name = sprintf('backup_Re%d_step%04d.mat', Re_book, ts);
        copyfile(restart_file, backup_name);
    end
end
disp('Simulation Complete.');
end

%% ========================================================================
%% CORE ASSEMBLY 1: STATIC LINEAR
%% ========================================================================
function [K_static, M_global, Mp, Kp, Rup] = assemble_static_linear_parts(mesh, fem, mu, gamma_gd)
    % Prepare Inputs for MEX
    grid_str.Xc = mesh.Xc; 
    grid_str.Yc = mesh.Yc; 
    grid_str.Zc = mesh.Zc;
    
    mat_str.M2 = fem.M2; mat_str.K2 = fem.K2; 
    mat_str.G2 = fem.G2; mat_str.GT2 = fem.GT2;
    mat_str.M1 = fem.M1; mat_str.K1 = fem.K1;
    mat_str.M21 = fem.M21; mat_str.G21 = fem.G21;
    mat_str.M12 = fem.M12; mat_str.G12 = fem.G12;
    
    fprintf('   >> MEX Full Assembly (Vel+Pres+Coup) started...\n');
    
    % --- CALL MEX ---
    [rk, ck, vk, rm, cm, vm, rmp, cmp, vmp, rkp, ckp, vkp, rrup, crup, vrup] ...
        = assemble_all_static_mex(...
            [mesh.Nx, mesh.Ny, mesh.Nz], ...
            grid_str, mat_str, [mu, gamma_gd]);
            
    ndof = fem.ndof; Nn = mesh.Nn; Np = mesh.Np;
    
    % K_static (Viscosity + Pressure Coupling + GradDiv)
    K_static = sparse(rk, ck, vk, ndof, ndof);
    
    % M_global (Scalar Velocity Mass) - Pure Mass Matrix (No 1/dt scaling yet)
    M_global = sparse(rm, cm, vm, Nn, Nn); 
    
    % Mp (Pressure Mass)
    Mp = sparse(rmp, cmp, vmp, Np, Np);
    
    % Kp (Pressure Laplacian)
    Kp = sparse(rkp, ckp, vkp, Np, Np);
    
    % Rup (Projection)
    Rup = sparse(rrup, crup, vrup, Np, Nn);
end

%% ========================================================================
%% CORE ASSEMBLY 2: DYNAMIC NONLINEAR
%% ========================================================================
function [K_nl, b_supg] = assemble_nonlinear_supg(mesh, fem, Ux, Uy, Uz, Ux_n, Uy_n, Uz_n, rho, mu, dt, CI, gamma_gd)
    % Just wraps the MEX. 
    % Note: The inputs Ux_n and dt here might be "Effective" values for BDF2.
    
    dims = [mesh.Nx, mesh.Ny, mesh.Nz];
    params = [rho, mu, dt, CI];
    
    [rows, cols, vals, b_supg] = assemble_nonlinear_mex(...
        dims, mesh.Xc, mesh.Yc, mesh.Zc, ...
        Ux, Uy, Uz, Ux_n, Uy_n, Uz_n, params);
        
    ndof = fem.ndof;
    K_nl = sparse(rows, cols, vals, ndof, ndof);
end

%% ========================================================================
%% PRECONDITIONER SETUP (Updated for BDF2)
%% ========================================================================
function [pcd_static, gmg_Mp, gmg_Ap, gmg_vel_static] = setup_pcd_static(mesh, fem, mu, dt, rho)
    % 1. Pressure Matrices
    [Mp, Kp] = assemble_pressure_static(mesh, fem);
    
    % 2. GMG Hierarchies
    p_pin_idx = sub2ind([mesh.Nx+1, mesh.Ny+1, mesh.Nz+1], ...
                        round(mesh.Nx/2)+1, round(mesh.Ny/2)+1, round(mesh.Nz/2)+1);
    
    gmg_Mp = setup_gmg_levels(Mp, mesh.Nx, mesh.Ny, mesh.Nz, mesh.Xc, mesh.Yc, mesh.Zc, 'mass', p_pin_idx);
    gmg_Ap = setup_gmg_levels(Kp, mesh.Nx, mesh.Ny, mesh.Nz, mesh.Xc, mesh.Yc, mesh.Zc, 'laplacian', p_pin_idx);
    
    % 3. Pressure Operator Static Part
    % BDF2 Scaling: reaction term is 1.5/dt
    reaction_coeff = 1.5 / dt;
    Fp_static = mu * Kp + reaction_coeff * Mp;
    
    % 4. Velocity GMG
    gmg_vel_static = setup_gmg_velocity_static(mesh, dt, mu, rho);
    
    pcd_static.Fp_static = Fp_static;
    pcd_static.Mp = Mp; 
    pcd_static.Kp = Kp;
end

% ... (Existing update_pcd_picard, apply_pcd_optimized, GMG kernels, and Output Helpers remain largely the same) ...
% ... (Include the rest of the file content here as per original structure) ...

% ---------------- RE-INCLUDED HELPER FUNCTIONS FOR COMPLETENESS ----------------
function Mfun = update_pcd_picard(K, Kp, Mp, Rup, Ux, Uy, Uz, mesh, fem, ...
                                  pcd_static, gmg_Mp, gmg_Ap, gmg_vel_static, ...
                                  rho, dt, CI, mu, p_pin)
    Nn = mesh.Nn; Np = mesh.Np;
    offUx=0; offUy=Nn; offUz=2*Nn; offP=3*Nn;
    iu = [offUx+(1:Nn), offUy+(1:Nn), offUz+(1:Nn)];
    ip = offP+(1:Np);
    
    % 1. Velocity GMG
    Fhat_fine = K(iu,iu);
    % Note: update_gmg_velocity_dynamic internally builds BDF1-like coarse operators. 
    % We accept this approximation for the preconditioner.
    gmg_Vel = update_gmg_velocity_dynamic(gmg_vel_static, Fhat_fine, Ux, Uy, Uz, dt, rho, mu, CI);
    
    % 2. Pressure Advection
    Cp_dynamic = assemble_pressure_advection(mesh, fem, Ux, Uy, Uz, rho, dt, CI, mu);
    
    Fp_total = pcd_static.Fp_static + Cp_dynamic;
    Fp_total(p_pin,:) = 0; Fp_total(:,p_pin) = 0; Fp_total(p_pin,p_pin) = 1;
    
    rhs_x = Rup * Ux(:); rhs_y = Rup * Uy(:); rhs_z = Rup * Uz(:);
    apx = v_cycle(gmg_Mp, 1, rhs_x, zeros(Np,1));
    apy = v_cycle(gmg_Mp, 1, rhs_y, zeros(Np,1));
    apz = v_cycle(gmg_Mp, 1, rhs_z, zeros(Np,1));
    
    data.gmg_Vel = gmg_Vel; data.gmg_Mp = gmg_Mp; data.gmg_Shat= gmg_Ap; data.Ap = Fp_total;
    data.iu = iu; data.ip = ip; data.B = K(ip,iu);       
    Mfun = @(v) apply_pcd_optimized(v, data);
end

function y = apply_pcd_optimized(v, D)
  vu = v(D.iu); vu = reshape(vu, [], 1); 
  vp = v(D.ip); vp = reshape(vp, [], 1);
  yu = v_cycle_vector(D.gmg_Vel, 1, vu, zeros(size(vu)));
  rhs_p = vp - D.B * yu;
  t1 = v_cycle(D.gmg_Shat, 1, rhs_p, zeros(size(rhs_p)));
  t2 = D.Ap * t1;
  yp = v_cycle(D.gmg_Mp, 1, t2, zeros(size(t2)));
  y = zeros(size(v)); y(D.iu) = yu; y(D.ip) = -yp;
end

% --- GMG KERNELS (Shortened for brevity, assumes standard logic) ---
function gmg = setup_gmg_velocity_static(mesh, dt, mu, rho)
    % Uses BDF1 scaling (1/dt) for coarse grids - acceptable approximation for PCD
    levels = {};
    lvl.Nx = mesh.Nx; lvl.Ny = mesh.Ny; lvl.Nz = mesh.Nz; lvl.Nn = mesh.Nn;
    lvl.Xc = mesh.Xc; lvl.Yc = mesh.Yc; lvl.Zc = mesh.Zc; lvl.type = 'fine';
    levels{1} = lvl;
    curr_mesh = mesh;
    level_idx = 1;
    while curr_mesh.Nx > 2
        level_idx = level_idx + 1;
        next_Nx = curr_mesh.Nx / 2; next_Ny = curr_mesh.Ny / 2; next_Nz = curr_mesh.Nz / 2;
        next_Xc = curr_mesh.Xc(1:2:end); next_Yc = curr_mesh.Yc(1:2:end); next_Zc = curr_mesh.Zc(1:2:end);
        P_scalar = build_prolongation_Q2_3d(next_Nx, next_Ny, next_Nz);
        lvl_c.P_scalar = P_scalar; lvl_c.R_scalar = P_scalar'; 
        A_static = assemble_velocity_linear_coarse(next_Nx, next_Ny, next_Nz, next_Xc, next_Yc, next_Zc, dt, mu, rho);
        lvl_c.A_static = A_static; lvl_c.Nx = next_Nx; lvl_c.Ny = next_Ny; lvl_c.Nz = next_Nz;
        lvl_c.Xc = next_Xc; lvl_c.Yc = next_Yc; lvl_c.Zc = next_Zc; lvl_c.Nn = size(A_static,1); 
        if next_Nx == 2, lvl_c.type = 'coarse_coupled'; else, lvl_c.type = 'coarse_scalar'; end
        levels{level_idx} = lvl_c;
        curr_mesh.Nx=next_Nx; curr_mesh.Ny=next_Ny; curr_mesh.Nz=next_Nz; curr_mesh.Xc=next_Xc; curr_mesh.Yc=next_Yc; curr_mesh.Zc=next_Zc;
    end
    gmg.levels = levels; gmg.num_levels = level_idx;
end

% function gmg = update_gmg_velocity_dynamic(gmg_static, K_fine, Ux, Uy, Uz, dt, rho, mu, CI)
%     gmg = gmg_static;
%     Nn_fine = size(K_fine,1)/3;
%     gmg.levels{1}.A_scalar = K_fine(1:Nn_fine, 1:Nn_fine);
%     gmg.levels{1}.lambda_max = norm(gmg.levels{1}.A_scalar, 1); 
%     cur_Ux = Ux; cur_Uy = Uy; cur_Uz = Uz;
%     params = [rho, mu, dt, CI];
%     for i = 2:gmg.num_levels
%         lvl = gmg.levels{i};
%         [cur_Ux, cur_Uy, cur_Uz] = restrict_velocity_components(cur_Ux, cur_Uy, cur_Uz, gmg.levels{i-1}.Nx, gmg.levels{i-1}.Ny, gmg.levels{i-1}.Nz);
%         dims = [lvl.Nx, lvl.Ny, lvl.Nz];
%         [rows, cols, vals, ~] = assemble_nonlinear_mex(dims, lvl.Xc, lvl.Yc, lvl.Zc, cur_Ux, cur_Uy, cur_Uz, cur_Ux, cur_Uy, cur_Uz, params);
%         ndof_c = 3 * lvl.Nn;
%         K_full_coarse = sparse(rows, cols, vals, ndof_c, ndof_c);
%         A_nonlinear_scalar = K_full_coarse(1:lvl.Nn, 1:lvl.Nn);
%         A_total = lvl.A_static + A_nonlinear_scalar;
%         if strcmp(lvl.type, 'coarse_coupled')
%             A_blk = blkdiag(A_total, A_total, A_total); gmg.levels{i}.LU = decomposition(A_blk, 'lu');
%         else
%             gmg.levels{i}.A_scalar = A_total; gmg.levels{i}.lambda_max = norm(A_total, 1);
%         end
%     end
% end
function gmg = update_gmg_velocity_dynamic(gmg_static, K_fine, Ux, Uy, Uz, dt, rho, mu, CI)
    gmg = gmg_static;
    
    % Update Level 1 (Fine)
    Nn_fine = size(K_fine,1)/3;
    gmg.levels{1}.A_scalar = K_fine(1:Nn_fine, 1:Nn_fine);
    gmg.levels{1}.lambda_max = norm(gmg.levels{1}.A_scalar, 1); 
    
    cur_Ux = Ux; cur_Uy = Uy; cur_Uz = Uz;
    params = [rho, mu, dt, CI];
    
    for i = 2:gmg.num_levels
        lvl = gmg.levels{i};
        
        % Restrict Solution to this level
        [cur_Ux, cur_Uy, cur_Uz] = restrict_velocity_components(cur_Ux, cur_Uy, cur_Uz, ...
            gmg.levels{i-1}.Nx, gmg.levels{i-1}.Ny, gmg.levels{i-1}.Nz);
            
        dims = [lvl.Nx, lvl.Ny, lvl.Nz];
        
        % Call MEX
        [rows, cols, vals, ~] = assemble_nonlinear_mex(dims, lvl.Xc, lvl.Yc, lvl.Zc, ...
            cur_Ux, cur_Uy, cur_Uz, cur_Ux, cur_Uy, cur_Uz, params);
        
        % --- FIX STARTS HERE ---
        % 1. Calculate Full Dimensions (Vel + Pressure) to prevent sparse error
        Np_c = (lvl.Nx+1)*(lvl.Ny+1)*(lvl.Nz+1); 
        ndof_vel = 3 * lvl.Nn;
        ndof_full = ndof_vel + Np_c; 
        
        % 2. Build Full Matrix (so indices don't exceed bounds)
        K_full_temp = sparse(rows, cols, vals, ndof_full, ndof_full);
        
        % 3. Extract only the Velocity-Velocity block (discard pressure coupling)
        K_vel_only = K_full_temp(1:ndof_vel, 1:ndof_vel);
        
        % 4. Extract Scalar part (u-u block) for the GMG Operator
        A_nonlinear_scalar = K_vel_only(1:lvl.Nn, 1:lvl.Nn);
        % --- FIX ENDS HERE ---
        
        A_total = lvl.A_static + A_nonlinear_scalar;
        
        if strcmp(lvl.type, 'coarse_coupled')
            A_blk = blkdiag(A_total, A_total, A_total); 
            gmg.levels{i}.LU = decomposition(A_blk, 'lu');
        else
            gmg.levels{i}.A_scalar = A_total; 
            gmg.levels{i}.lambda_max = norm(A_total, 1);
        end
    end
end

% --- Standard Helpers and Utils (Same as provided code) ---
function K = assemble_velocity_linear_coarse(Nx, Ny, Nz, Xc, Yc, Zc, dt, mu, rho)
    nx = 2*Nx+1; ny = 2*Ny+1; nz = 2*Nz+1; Nn = nx*ny*nz; 
    K = spalloc(Nn, Nn, Nn*27);
    phi2=@(xi)[2*xi.^2-3*xi+1; 4*xi-4*xi.^2; 2*xi.^2-xi]; 
    dphi2=@(xi)[4*xi-3; 4-8*xi; 4*xi-1];
    gp=0.5*([-sqrt(3/5),0,sqrt(3/5)]+1); gw=0.5*[5/9,8/9,5/9];
    M2=zeros(3); K2=zeros(3);
    for g=1:3, M2=M2+(phi2(gp(g))*phi2(gp(g))')*gw(g); K2=K2+(dphi2(gp(g))*dphi2(gp(g))')*gw(g); end
    for ez=1:Nz, hz=Zc(ez+1)-Zc(ez); 
    for ey=1:Ny, hy=Yc(ey+1)-Yc(ey); 
    for ex=1:Nx, hx=Xc(ex+1)-Xc(ex); 
        ix=[2*ex-1,2*ex,2*ex+1]; iy=[2*ey-1,2*ey,2*ey+1]; iz=[2*ez-1,2*ez,2*ez+1];
        [II,JJ,KK]=ndgrid(ix,iy,iz); idx=II(:)+(JJ(:)-1)*nx+(KK(:)-1)*nx*ny;
        Kxx=kron(M2,kron(M2,K2))*(hy*hz/hx); Kyy=kron(M2,kron(K2,M2))*(hx*hz/hy); Kzz=kron(K2,kron(M2,M2))*(hx*hy/hz);
        M_loc=(rho/dt)*(hx*hy*hz)*kron(M2,kron(M2,M2));
        K(idx,idx) = K(idx,idx) + M_loc + mu*(Kxx+Kyy+Kzz);
    end,end,end
    [Ix, Iy, Iz] = ndgrid(1:nx, 1:ny, 1:nz); 
    is_bdry = (Ix==1 | Ix==nx | Iy==1 | Iy==ny | Iz==1 | Iz==nz);
    bdry_nodes = find(is_bdry);
    K(bdry_nodes, :) = 0; K = K + sparse(bdry_nodes, bdry_nodes, 1, Nn, Nn);
end

function Fp = assemble_pressure_advection(mesh, fem, Ux, Uy, Uz, rho, dt, CI, mu)
    dims = [mesh.Nx, mesh.Ny, mesh.Nz]; params = [rho, mu, dt, CI];
    [rows, cols, vals] = assemble_pressure_mex(dims, mesh.Xc, mesh.Yc, mesh.Zc, Ux, Uy, Uz, params);
    Np = mesh.Np; Fp = sparse(rows, cols, vals, Np, Np);
end

function [Mp, Kp] = assemble_pressure_static(mesh, fem)
    Np = mesh.Np; Mp = spalloc(Np,Np,Np*27); Kp = spalloc(Np,Np,Np*27);
    [M1, K1] = deal(fem.M1, fem.K1);
    for ez=1:mesh.Nz, hz=mesh.Zc(ez+1)-mesh.Zc(ez);
    for ey=1:mesh.Ny, hy=mesh.Yc(ey+1)-mesh.Yc(ey);
    for ex=1:mesh.Nx, hx=mesh.Xc(ex+1)-mesh.Xc(ex);
        [~, NlocP] = get_local_dofs(ex,ey,ez,mesh);
        Mp_e = (hx*hy*hz) * kron(M1, kron(M1, M1));
        Kp_e = (hy*hz/hx)*kron(M1,kron(M1,K1)) + (hx*hz/hy)*kron(M1,kron(K1,M1)) + (hx*hy/hz)*kron(K1,kron(M1,M1));
        Mp(NlocP,NlocP)=Mp(NlocP,NlocP)+Mp_e; Kp(NlocP,NlocP)=Kp(NlocP,NlocP)+Kp_e;
    end,end,end
end

function [rc_u, rc_v, rc_w] = restrict_velocity_components(r_u, r_v, r_w, Nf_x, Nf_y, Nf_z)
    rc_u = restrict_Q2(r_u, Nf_x, Nf_y, Nf_z); rc_v = restrict_Q2(r_v, Nf_x, Nf_y, Nf_z); rc_w = restrict_Q2(r_w, Nf_x, Nf_y, Nf_z);
end
function Uc = restrict_Q2(Uf, Nf_x, Nf_y, Nf_z)
    Uf_3d = reshape(Uf, 2*Nf_x+1, 2*Nf_y+1, 2*Nf_z+1); Uc = Uf_3d(1:2:end, 1:2:end, 1:2:end); Uc = Uc(:);
end
function P = build_prolongation_Q2_3d(Nc_x, Nc_y, Nc_z)
    Px = build_prolongation_Q2_1d(Nc_x); Py = build_prolongation_Q2_1d(Nc_y); Pz = build_prolongation_Q2_1d(Nc_z); P = kron(Pz, kron(Py, Px));
end
function P1 = build_prolongation_Q2_1d(Nc)
    N_coarse = 2*Nc + 1; N_fine = 4*Nc + 1; P1 = spalloc(N_fine, N_coarse, 3*N_fine);
    for e = 1:Nc, cL = 2*(e-1) + 1; cM = cL+1; cR = cL+2; f1 = 4*(e-1) + 1; P1(f1:f1+4, cL:cR) = [1,0,0; 0.375,0.75,-0.125; 0,1,0; -0.125,0.75,0.375; 0,0,1]; end
end
function x = v_cycle(mg, lvl, b, x)
    L = mg.levels{lvl};
    if lvl == mg.num_levels, x = L.LU \ b; return; end
    x = smooth_chebyshev(L.A, b, x, 2, L.lambda_max);
    r = b - L.A * x; if ~isempty(L.p_pin), r(L.p_pin) = 0; end
    next_L = mg.levels{lvl+1}; rc = next_L.R * r;
    xc = v_cycle(mg, lvl+1, rc, zeros(size(rc)));
    x = x + next_L.P * xc;
    x = smooth_chebyshev(L.A, b, x, 2, L.lambda_max);
end
function x = v_cycle_vector(mg, lvl, b, x)
    L = mg.levels{lvl}; if strcmp(L.type, 'coarse_coupled'), x = L.LU \ b; return; end
    Nn = L.Nn; bx=b(1:Nn); by=b(Nn+1:2*Nn); bz=b(2*Nn+1:end); xx=x(1:Nn); xy=x(Nn+1:2*Nn); xz=x(2*Nn+1:end);
    xx=smooth_chebyshev(L.A_scalar,bx,xx,2,L.lambda_max); xy=smooth_chebyshev(L.A_scalar,by,xy,2,L.lambda_max); xz=smooth_chebyshev(L.A_scalar,bz,xz,2,L.lambda_max);
    rx=bx-L.A_scalar*xx; ry=by-L.A_scalar*xy; rz=bz-L.A_scalar*xz;
    bc = [mg.levels{lvl+1}.R_scalar*rx; mg.levels{lvl+1}.R_scalar*ry; mg.levels{lvl+1}.R_scalar*rz];
    xc = v_cycle_vector(mg, lvl+1, bc, zeros(size(bc)));
    Nnc = mg.levels{lvl+1}.Nn;
    xx=xx+mg.levels{lvl+1}.P_scalar*xc(1:Nnc); xy=xy+mg.levels{lvl+1}.P_scalar*xc(Nnc+1:2*Nnc); xz=xz+mg.levels{lvl+1}.P_scalar*xc(2*Nnc+1:end);
    xx=smooth_chebyshev(L.A_scalar,bx,xx,2,L.lambda_max); xy=smooth_chebyshev(L.A_scalar,by,xy,2,L.lambda_max); xz=smooth_chebyshev(L.A_scalar,bz,xz,2,L.lambda_max);
    x=[xx;xy;xz];
end
function x = smooth_chebyshev(A, b, x, steps, lambda_max)
    r = b - A*x; D_inv = 1./diag(A); D_inv(isinf(D_inv)) = 0; omega = 2/3; 
    for k = 1:steps, x = x + omega * (D_inv .* r); if k < steps, r = b - A*x; end, end
end
function [mesh, fem] = setup_grid_and_fem(Nx, Ny, Nz, L_domain)
    mesh.Nx=Nx; mesh.Ny=Ny; mesh.Nz=Nz;
    mesh.Xc = tanh_grid_range(Nx+1, 0, L_domain, 0); mesh.Yc = tanh_grid_range(Ny+1, 0, L_domain, 0); mesh.Zc = tanh_grid_range(Nz+1, 0, L_domain, 0);
    nx = 2*Nx + 1;  ny = 2*Ny + 1;  nz = 2*Nz + 1; mesh.nx=nx; mesh.ny=ny; mesh.nz=nz;
    xv = zeros(1,nx); yv = zeros(1,ny); zv = zeros(1,nz);
    for ex=1:Nx, xv(2*ex-1)=mesh.Xc(ex); xv(2*ex)=0.5*(mesh.Xc(ex)+mesh.Xc(ex+1)); end, xv(end)=mesh.Xc(end);
    for ey=1:Ny, yv(2*ey-1)=mesh.Yc(ey); yv(2*ey)=0.5*(mesh.Yc(ey)+mesh.Yc(ey+1)); end, yv(end)=mesh.Yc(end);
    for ez=1:Nz, zv(2*ez-1)=mesh.Zc(ez); zv(2*ez)=0.5*(mesh.Zc(ez)+mesh.Zc(ez+1)); end, zv(end)=mesh.Zc(end);
    mesh.xv=xv; mesh.yv=yv; mesh.zv=zv;
    mesh.Nn = nx*ny*nz; mesh.Np = (Nx+1)*(Ny+1)*(Nz+1); fem.ndof = 3*mesh.Nn + mesh.Np;
    fem.dof_u = 1:mesh.Nn; fem.dof_v = mesh.Nn+1:2*mesh.Nn; fem.dof_w = 2*mesh.Nn+1:3*mesh.Nn; fem.dof_p = 3*mesh.Nn+1:fem.ndof;
    phi2  = @(xi) [ 2*xi.^2 - 3*xi + 1 ; 4*xi - 4*xi.^2 ; 2*xi.^2 - xi ];
    dphi2 = @(xi) [ 4*xi - 3 ; 4 - 8*xi ; 4*xi - 1 ];
    ddphi2= @(xi) [ 4 ; -8 ; 4 ] .* ones(1,numel(xi)); 
    phi1  = @(xi) [1 - xi ; xi]; dphi1 = @(xi) [-ones(1,numel(xi)); ones(1,numel(xi))];
    t  = [-sqrt(3/5), 0, sqrt(3/5)]; w  = [5/9, 8/9, 5/9]; gp = 0.5*(t+1); gw = 0.5*w; fem.gw = gw;
    fem.Ph2=phi2(gp); fem.DPh2=dphi2(gp); fem.DDPh2=ddphi2(gp); fem.Ph1=phi1(gp); fem.DPh1=dphi1(gp);
    Ph2=fem.Ph2; DPh2=fem.DPh2; Ph1=fem.Ph1; DPh1=fem.DPh1;
    M2=zeros(3); K2=zeros(3); G2=zeros(3); GT2=zeros(3);
    for g=1:3, 
        M2=M2+(Ph2(:,g)*Ph2(:,g)')*gw(g); 
        K2=K2+(DPh2(:,g)*DPh2(:,g)')*gw(g); 
        G2=G2+(Ph2(:,g)*DPh2(:,g)')*gw(g); % CHANGED FROM DPh1 to DPh2
        GT2=GT2+(DPh2(:,g)*Ph2(:,g)')*gw(g); 
    end
    M1=zeros(2); K1=zeros(2);
    for g=1:3, M1=M1+(Ph1(:,g)*Ph1(:,g)')*gw(g); K1=K1+(DPh1(:,g)*DPh1(:,g)')*gw(g); end
    M21=zeros(3,2); G21=zeros(3,2); M12=zeros(2,3); G12=zeros(2,3);
    for g=1:3, M21=M21+(Ph2(:,g)*Ph1(:,g)')*gw(g); G21=G21+(Ph2(:,g)*DPh1(:,g)')*gw(g); M12=M12+(Ph1(:,g)*Ph2(:,g)')*gw(g); G12=G12+(Ph1(:,g)*DPh2(:,g)')*gw(g); end
    fem.M2=M2; fem.K2=K2; fem.G2=G2; fem.GT2=GT2; fem.M1=M1; fem.K1=K1; fem.M21=M21; fem.G21=G21; fem.M12=M12; fem.G12=G12;
end
function [vel_mask, valUx, valUy, valUz, vel_dofs, p_pin] = setup_bcs(mesh)
    nx=mesh.nx; ny=mesh.ny; nz=mesh.nz;
    [Ix, Iy, Iz] = ndgrid(1:nx, 1:ny, 1:nz); nodes = Ix + (Iy-1)*nx + (Iz-1)*nx*ny;
    left   = nodes(1, :, :); right = nodes(end, :, :); front  = nodes(:, 1, :); back  = nodes(:, end, :); bottom = nodes(:, :, 1); top   = nodes(:, :, end);
    vel_mask = false(mesh.Nn,1); valUx = zeros(mesh.Nn,1); valUy = zeros(mesh.Nn,1); valUz = zeros(mesh.Nn,1);
    noslip = unique([left(:); right(:); front(:); back(:); bottom(:)]);
    vel_mask(noslip) = true; vel_mask(top) = true; valUx(top) = 1.0; 
    vel_dofs = find(vel_mask);
    p_pin = sub2ind([mesh.Nx+1, mesh.Ny+1, mesh.Nz+1], round(mesh.Nx/2)+1, round(mesh.Ny/2)+1, round(mesh.Nz/2)+1);
end
function [NlocV, NlocP] = get_local_dofs(ex, ey, ez, mesh)
    nx = mesh.nx; ny = mesh.ny;
    ix = [2*ex-1, 2*ex, 2*ex+1]; iy = [2*ey-1, 2*ey, 2*ey+1]; iz = [2*ez-1, 2*ez, 2*ez+1];
    [II,JJ,KK] = ndgrid(ix,iy,iz); NlocV = II(:) + (JJ(:)-1)*nx + (KK(:)-1)*nx*ny;
    Jx = [ex, ex+1]; Jy = [ey, ey+1]; Jz = [ez, ez+1];
    [IIp,JJp,KKp] = ndgrid(Jx,Jy,Jz); NlocP = IIp(:) + (JJp(:)-1)*(mesh.Nx+1) + (KKp(:)-1)*(mesh.Nx+1)*(mesh.Ny+1);
end
function [K, b] = apply_boundary_conditions(K, b, vel_dofs, valUx, valUy, valUz, p_pin, mesh, fem)
    ndof = fem.ndof; Nn = mesh.Nn;
    Ux_rows = vel_dofs; Uy_rows = Nn + vel_dofs; Uz_rows = 2*Nn + vel_dofs;
    prow = 3*Nn + p_pin;
    rows_to_clear = [Ux_rows; Uy_rows; Uz_rows; prow];
    K(rows_to_clear, :) = 0; K = K + sparse(rows_to_clear, rows_to_clear, 1, ndof, ndof);
    b(Ux_rows) = valUx(vel_dofs); b(Uy_rows) = valUy(vel_dofs); b(Uz_rows) = valUz(vel_dofs); b(prow) = 0;
end
function x = tanh_grid_range(n, min_val, max_val, beta)
    s = linspace(0,1,n); if beta <= 0, xi = s; else, xi = 0.5*(1 + tanh(beta*(2*s-1))/tanh(beta)); end
    x = min_val + (max_val - min_val) * xi;
end
% function save_vtu_output(Ux, Uy, Uz, Pp, mesh, Re, frame)
%     [X_mesh, Y_mesh, Z_mesh] = ndgrid(mesh.xv, mesh.yv, mesh.zv); points = [X_mesh(:), Y_mesh(:), Z_mesh(:)];
%     xp_vec = mesh.xv(1:2:end); yp_vec = mesh.yv(1:2:end); zp_vec = mesh.zv(1:2:end);
%     P_interp = interpn(xp_vec, yp_vec, zp_vec, Pp, X_mesh, Y_mesh, Z_mesh, 'linear');
%     fields.velocity = [Ux(:), Uy(:), Uz(:)]; fields.pressure = P_interp(:);
%     [I_m, J_m, K_m] = ndgrid(1:mesh.nx-1, 1:mesh.ny-1, 1:mesh.nz-1);
%     idx_m = @(i,j,k) i + (j-1)*mesh.nx + (k-1)*mesh.nx*mesh.ny;
%     n1=idx_m(I_m,J_m,K_m); n2=idx_m(I_m+1,J_m,K_m); n3=idx_m(I_m+1,J_m+1,K_m); n4=idx_m(I_m,J_m+1,K_m);
%     n5=idx_m(I_m,J_m,K_m+1); n6=idx_m(I_m+1,J_m,K_m+1); n7=idx_m(I_m+1,J_m+1,K_m+1); n8=idx_m(I_m,J_m+1,K_m+1);
%     conn = [n1(:), n2(:), n3(:), n4(:), n5(:), n6(:), n7(:), n8(:)] - 1;
%     fname = sprintf('ns_opt_Re%d_%04d.vtu', Re, frame);
%     write_vtu_hex(fname, points, conn, fields);
%     fprintf('   >> Saved Frame %d: %s\n', frame, fname);
% end
% function write_vtu_hex(filename, points, conn, fields)
%     n_pts = size(points,1); n_cells = size(conn,1); fid = fopen(filename, 'w');
%     fprintf(fid, '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n');
%     fprintf(fid, '  <UnstructuredGrid>\n    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n', n_pts, n_cells);
%     fprintf(fid, '      <Points>\n        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n          %.6f %.6f %.6f\n        </DataArray>\n      </Points>\n');
%     fprintf(fid, '      <Cells>\n        <DataArray type="Int32" Name="connectivity" format="ascii">\n          %d %d %d %d %d %d %d %d\n        </DataArray>\n');
%     fprintf(fid, '        <DataArray type="Int32" Name="offsets" format="ascii">\n          %d\n        </DataArray>\n');
%     fprintf(fid, '        <DataArray type="UInt8" Name="types" format="ascii">\n          %d\n        </DataArray>\n      </Cells>\n');
%     fprintf(fid, '      <PointData Scalars="pressure" Vectors="velocity">\n');
%     fprintf(fid, '        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n          %.6f %.6f %.6f\n        </DataArray>\n');
%     fprintf(fid, '        <DataArray type="Float32" Name="pressure" format="ascii">\n          %.6f\n        </DataArray>\n      </PointData>\n');
%     fprintf(fid, '    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n'); fclose(fid);
% end
% function write_pvd(pvd_filename, pvd_data)
%     fid_pvd = fopen(pvd_filename, 'w');
%     fprintf(fid_pvd, '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n  <Collection>\n');
%     for i = 1:length(pvd_data), fprintf(fid_pvd, '    <DataSet timestep="%.4f" group="" part="0" file="%s"/>\n', pvd_data{i}.time, pvd_data{i}.file); end
%     fprintf(fid_pvd, '  </Collection>\n</VTKFile>\n'); fclose(fid_pvd);
% end
%% ========================================================================
%% PART 7: OUTPUT HELPERS (VTK/PVD)
%% ========================================================================

function save_vtu_output(Ux, Uy, Uz, Pp, mesh, Re, frame)
    % Interpolate pressure to velocity nodes for visualization
    [X_mesh, Y_mesh, Z_mesh] = ndgrid(mesh.xv, mesh.yv, mesh.zv);
    points = [X_mesh(:), Y_mesh(:), Z_mesh(:)];
    
    % Pressure is on a coarser grid (Q1), interpolate to Q2 grid
    xp_vec = mesh.xv(1:2:end); yp_vec = mesh.yv(1:2:end); zp_vec = mesh.zv(1:2:end);
    P_interp = interpn(xp_vec, yp_vec, zp_vec, Pp, X_mesh, Y_mesh, Z_mesh, 'linear');
    
    fields.velocity = [Ux(:), Uy(:), Uz(:)];
    fields.pressure = P_interp(:);
    
    % Generate Connectivity for Q2 Hex elements (subdivided into linear hexes for Paraview)
    % Actually, we just output the Q2 grid points as a fine linear mesh
    [I_m, J_m, K_m] = ndgrid(1:mesh.nx-1, 1:mesh.ny-1, 1:mesh.nz-1);
    idx_m = @(i,j,k) i + (j-1)*mesh.nx + (k-1)*mesh.nx*mesh.ny;
    
    n1=idx_m(I_m,J_m,K_m); n2=idx_m(I_m+1,J_m,K_m); 
    n3=idx_m(I_m+1,J_m+1,K_m); n4=idx_m(I_m,J_m+1,K_m);
    n5=idx_m(I_m,J_m,K_m+1); n6=idx_m(I_m+1,J_m,K_m+1); 
    n7=idx_m(I_m+1,J_m+1,K_m+1); n8=idx_m(I_m,J_m+1,K_m+1);
    
    conn = [n1(:), n2(:), n3(:), n4(:), n5(:), n6(:), n7(:), n8(:)] - 1;
    
    fname = sprintf('ns_opt_Re%d_%04d.vtu', Re, frame);
    write_vtu_hex(fname, points, conn, fields);
    fprintf('   >> Saved Frame %d: %s\n', frame, fname);
end

function write_vtu_hex(filename, points, conn, fields)
    n_pts = size(points,1); n_cells = size(conn,1);
    fid = fopen(filename, 'w');
    fprintf(fid, '<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n');
    fprintf(fid, '  <UnstructuredGrid>\n');
    fprintf(fid, '    <Piece NumberOfPoints="%d" NumberOfCells="%d">\n', n_pts, n_cells);
    fprintf(fid, '      <Points>\n        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n');
    fprintf(fid, '          %.6f %.6f %.6f\n', points');
    fprintf(fid, '        </DataArray>\n      </Points>\n');
    fprintf(fid, '      <Cells>\n        <DataArray type="Int32" Name="connectivity" format="ascii">\n');
    fprintf(fid, '          %d %d %d %d %d %d %d %d\n', conn');
    fprintf(fid, '        </DataArray>\n        <DataArray type="Int32" Name="offsets" format="ascii">\n');
    fprintf(fid, '          %d\n', (8:8:8*n_cells));
    fprintf(fid, '        </DataArray>\n        <DataArray type="UInt8" Name="types" format="ascii">\n');
    fprintf(fid, '          %d\n', 12*ones(1,n_cells));
    fprintf(fid, '        </DataArray>\n      </Cells>\n');
    fprintf(fid, '      <PointData Scalars="pressure" Vectors="velocity">\n');
    fprintf(fid, '        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n');
    fprintf(fid, '          %.6f %.6f %.6f\n', fields.velocity');
    fprintf(fid, '        </DataArray>\n');
    fprintf(fid, '        <DataArray type="Float32" Name="pressure" format="ascii">\n');
    fprintf(fid, '          %.6f\n', fields.pressure');
    fprintf(fid, '        </DataArray>\n      </PointData>\n');
    fprintf(fid, '    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n');
    fclose(fid);
end

function write_pvd(pvd_filename, pvd_data)
    fid_pvd = fopen(pvd_filename, 'w');
    fprintf(fid_pvd, '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n');
    fprintf(fid_pvd, '  <Collection>\n');
    for i = 1:length(pvd_data)
        fprintf(fid_pvd, '    <DataSet timestep="%.4f" group="" part="0" file="%s"/>\n', pvd_data{i}.time, pvd_data{i}.file);
    end
    fprintf(fid_pvd, '  </Collection>\n</VTKFile>\n');
    fclose(fid_pvd);
    fprintf('Saved PVD Collection: %s\n', pvd_filename);
end
function P = build_prolongation_3d(Nc_x, Nc_y, Nc_z)
    Px = build_prolongation_1d(Nc_x); Py = build_prolongation_1d(Nc_y); Pz = build_prolongation_1d(Nc_z); P = kron(Pz, kron(Py, Px));
end
function P1 = build_prolongation_1d(Nc)
    Nf = 2*Nc; P1 = spalloc(Nf+1, Nc+1, 3*(Nf+1));
    for i = 1:Nc, P1(2*i-1, i) = 1.0; P1(2*i, i) = 0.5; end
    for i=1:Nc, P1(2*i, i+1) = 0.5; P1(2*i+1, i+1) = 1.0; end
end
function mg_struct = setup_gmg_levels(A_fine, Nx, Ny, Nz, Xc, Yc, Zc, type, p_pin)
    % Generic Scalar GMG Builder (for Pressure Mass/Laplacian)
    levels = {};
    lvl.A = A_fine; lvl.Nx = Nx; lvl.Ny = Ny; lvl.Nz = Nz; lvl.p_pin = p_pin;
    try opts.tol=1e-2; opts.maxit=15; opts.disp=0; lvl.lambda_max = abs(eigs(A_fine, 1, 'lm', opts))*1.1; catch; lvl.lambda_max = norm(A_fine, 1); end
    levels{1} = lvl;
    current_Nx = Nx; current_Ny = Ny; current_Nz = Nz; current_Xc = Xc; current_Yc = Yc; current_Zc = Zc;
    level_idx = 1;
    while current_Nx > 2
        level_idx = level_idx + 1;
        next_Nx = current_Nx / 2; next_Ny = current_Ny / 2; next_Nz = current_Nz / 2;
        next_Xc = current_Xc(1:2:end); next_Yc = current_Yc(1:2:end); next_Zc = current_Zc(1:2:end);
        
        if strcmp(type, 'mass')
            A_coarse = assemble_pressure_static_coarse(next_Nx, next_Ny, next_Nz, next_Xc, next_Yc, next_Zc, 'mass');
        else
            A_coarse = assemble_pressure_static_coarse(next_Nx, next_Ny, next_Nz, next_Xc, next_Yc, next_Zc, 'laplacian');
        end
        
        p_pin_coarse = [];
        if ~isempty(p_pin)
             p_pin_coarse = sub2ind([next_Nx+1, next_Ny+1, next_Nz+1], round(next_Nx/2)+1, round(next_Ny/2)+1, round(next_Nz/2)+1);
            A_coarse(p_pin_coarse,:) = 0; A_coarse(:,p_pin_coarse) = 0; A_coarse(p_pin_coarse,p_pin_coarse) = 1;
        end
        lvl_c.A = A_coarse; lvl_c.p_pin = p_pin_coarse;
        lvl_c.P = build_prolongation_3d(next_Nx, next_Ny, next_Nz); lvl_c.R = lvl_c.P';
        try lvl_c.lambda_max = abs(eigs(A_coarse, 1, 'lm', opts))*1.1; catch; lvl_c.lambda_max = norm(A_coarse, 1); end
        if next_Nx == 2 || next_Nx == 1, lvl_c.LU = decomposition(A_coarse, 'lu'); end
        levels{level_idx} = lvl_c;
        current_Nx = next_Nx; current_Ny = next_Ny; current_Nz = next_Nz; current_Xc = next_Xc; current_Yc = next_Yc; current_Zc = next_Zc;
    end
    mg_struct.levels = levels; mg_struct.num_levels = level_idx;
end

function K = assemble_pressure_static_coarse(Nx, Ny, Nz, Xc, Yc, Zc, type)
    Np = (Nx+1)*(Ny+1)*(Nz+1); K = spalloc(Np, Np, Np*27);
    phi1=@(x)[1-x;x]; dphi1=@(x)[-1;1]; gp=0.5*([-sqrt(3/5),0,sqrt(3/5)]+1); gw=0.5*[5/9,8/9,5/9];
    M1=zeros(2); K1=zeros(2);
    for g=1:3, M1=M1+(phi1(gp(g))*phi1(gp(g))')*gw(g); K1=K1+(dphi1(gp(g))*dphi1(gp(g))')*gw(g); end
    for ez=1:Nz, hz=Zc(ez+1)-Zc(ez); 
    for ey=1:Ny, hy=Yc(ey+1)-Yc(ey); 
    for ex=1:Nx, hx=Xc(ex+1)-Xc(ex); 
        [II,JJ,KK]=ndgrid([ex,ex+1],[ey,ey+1],[ez,ez+1]); idx=II(:)+(JJ(:)-1)*(Nx+1)+(KK(:)-1)*(Nx+1)*(Ny+1);
        if strcmp(type, 'mass'), Ke = (hx*hy*hz) * kron(M1, kron(M1, M1));
        else, Kx = kron(M1, kron(M1, K1))*(hy*hz/hx); Ky = kron(M1, kron(K1, M1))*(hx*hz/hy); Kz = kron(K1, kron(M1, M1))*(hx*hy/hz); Ke = Kx+Ky+Kz; end
        K(idx,idx) = K(idx,idx) + Ke;
    end,end,end
end