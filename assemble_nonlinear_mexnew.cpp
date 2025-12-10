#include "mex.h"
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm> // For std::max

// --- SHAPE FUNCTIONS (Q2) ---
const double GP_loc[] = {0.112701665379258, 0.5, 0.887298334620742};
const double GW_loc[] = {0.277777777777778, 0.444444444444444, 0.277777777777778};

void eval_shape_1d(double xi, double* phi, double* dphi, double* ddphi) {
    phi[0] = 2*xi*xi - 3*xi + 1;   dphi[0] = 4*xi - 3;   ddphi[0] = 4.0;
    phi[1] = 4*xi - 4*xi*xi;       dphi[1] = 4 - 8*xi;   ddphi[1] = -8.0;
    phi[2] = 2*xi*xi - xi;         dphi[2] = 4*xi - 1;   ddphi[2] = 4.0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // --- FORCE PARALLELISM ---
    // This overrides MATLAB's environment settings
    int num_procs = omp_get_num_procs();
    omp_set_dynamic(0);             // Disable dynamic adjustment
    omp_set_num_threads(num_procs); // Force use of all physical cores
    
    // Debug Print (Only prints once per session usually, or you can comment it out later)
    //mexPrintf("Running on %d threads.\n", num_procs); 

    if(nrhs < 7) mexErrMsgIdAndTxt("MEX:Error", "Not enough inputs.");

    const double* dims = mxGetPr(prhs[0]);
    int Nx = (int)dims[0]; int Ny = (int)dims[1]; int Nz = (int)dims[2];
    int nx = 2*Nx + 1; int ny = 2*Ny + 1; int nz = 2*Nz + 1;
    int Nn = nx * ny * nz;
    
    double* Xc = mxGetPr(prhs[1]); double* Yc = mxGetPr(prhs[2]); double* Zc = mxGetPr(prhs[3]);
    double* Ux = mxGetPr(prhs[4]); double* Uy = mxGetPr(prhs[5]); double* Uz = mxGetPr(prhs[6]);
    double* Ux_n = mxGetPr(prhs[7]); double* Uy_n = mxGetPr(prhs[8]); double* Uz_n = mxGetPr(prhs[9]);
    double* params = mxGetPr(prhs[10]);
    
    double rho = params[0]; double mu = params[1]; double dt = params[2]; double CI = params[3];

    // Precompute Basis
    double Ph[3][3], DPh[3][3], DDPh[3][3];
    for(int g=0; g<3; g++) {
        double p[3], dp[3], ddp[3];
        eval_shape_1d(GP_loc[g], p, dp, ddp);
        for(int n=0; n<3; n++) { Ph[n][g]=p[n]; DPh[n][g]=dp[n]; DDPh[n][g]=ddp[n]; }
    }

    size_t est_nz = (size_t)Nx * Ny * Nz * 2000; 
    std::vector<double> rows, cols, vals;
    rows.reserve(est_nz); cols.reserve(est_nz); vals.reserve(est_nz);
    std::vector<double> b_supg(3 * Nn + (Nx+1)*(Ny+1)*(Nz+1), 0.0);

    #pragma omp parallel
    {
        std::vector<double> loc_rows, loc_cols, loc_vals;
        loc_rows.reserve(est_nz / omp_get_num_threads());
        loc_cols.reserve(est_nz / omp_get_num_threads());
        loc_vals.reserve(est_nz / omp_get_num_threads());
        std::vector<double> loc_b_supg(b_supg.size(), 0.0);
        
        #pragma omp for collapse(3) nowait
        for (int ez = 0; ez < Nz; ez++) {
            for (int ey = 0; ey < Ny; ey++) {
                for (int ex = 0; ex < Nx; ex++) {
                    double hx = Xc[ex+1] - Xc[ex];
                    double hy = Yc[ey+1] - Yc[ey];
                    double hz = Zc[ez+1] - Zc[ez];
                    double h_iso = std::pow(hx*hy*hz, 1.0/3.0);
                    
                    int ix[3] = {2*ex, 2*ex+1, 2*ex+2};
                    int iy[3] = {2*ey, 2*ey+1, 2*ey+2};
                    int iz[3] = {2*ez, 2*ez+1, 2*ez+2};
                    
                    int nodes[27];
                    int count = 0;
                    for(int k=0; k<3; k++) for(int j=0; j<3; j++) for(int i=0; i<3; i++)
                        nodes[count++] = ix[i] + iy[j]*nx + iz[k]*nx*ny;
                    
                    double u_loc[27], v_loc[27], w_loc[27], un_loc[27], vn_loc[27], wn_loc[27];
                    double u_avg=0, v_avg=0, w_avg=0;
                    for(int i=0; i<27; i++) {
                        u_loc[i] = Ux[nodes[i]]; v_loc[i] = Uy[nodes[i]]; w_loc[i] = Uz[nodes[i]];
                        un_loc[i] = Ux_n[nodes[i]]; vn_loc[i] = Uy_n[nodes[i]]; wn_loc[i] = Uz_n[nodes[i]];
                        u_avg += u_loc[i]; v_avg += v_loc[i]; w_avg += w_loc[i];
                    }
                    u_avg/=27.0; v_avg/=27.0; w_avg/=27.0;
                    
                    double u_mag = std::sqrt(u_avg*u_avg + v_avg*v_avg + w_avg*w_avg) + 1e-12;
                    double inv2 = 4.0/(dt*dt) + std::pow(rho*2*u_mag/h_iso, 2) + 
                                  CI*CI*mu*mu*( std::pow(2.0/hx,4) + std::pow(2.0/hy,4) + std::pow(2.0/hz,4) );
                    double tau = 1.0 / std::sqrt(inv2 + 1e-16);
                    
                    double Ke_adv[729] = {0}; 

                    for(int gz=0; gz<3; gz++) {
                        for(int gy=0; gy<3; gy++) {
                            for(int gx=0; gx<3; gx++) {
                                double wJ = GW_loc[gx]*GW_loc[gy]*GW_loc[gz] * hx*hy*hz;
                                
                                double N[27], dNx[27], dNy[27], dNz[27], Lap[27];
                                double u_val=0, v_val=0, w_val=0, un_val=0, vn_val=0, wn_val=0;
                                
                                int idx = 0;
                                for(int k=0; k<3; k++) {
                                    double pz = Ph[k][gz], dpz = DPh[k][gz], ddpz = DDPh[k][gz];
                                    for(int j=0; j<3; j++) {
                                        double py = Ph[j][gy], dpy = DPh[j][gy], ddpy = DDPh[j][gy];
                                        for(int i=0; i<3; i++) {
                                            double px = Ph[i][gx], dpx = DPh[i][gx], ddpx = DDPh[i][gx];
                                            
                                            double val = px*py*pz;
                                            N[idx] = val;
                                            dNx[idx] = dpx*py*pz / hx;
                                            dNy[idx] = px*dpy*pz / hy;
                                            dNz[idx] = px*py*dpz / hz;
                                            Lap[idx] = (ddpx*py*pz)/(hx*hx) + (px*ddpy*pz)/(hy*hy) + (px*py*ddpz)/(hz*hz);
                                            
                                            u_val += u_loc[idx]*val; v_val += v_loc[idx]*val; w_val += w_loc[idx]*val;
                                            un_val += un_loc[idx]*val; vn_val += vn_loc[idx]*val; wn_val += wn_loc[idx]*val;
                                            idx++;
                                        }
                                    }
                                }
                                
                                for(int r=0; r<27; r++) {
                                    double u_grad_N = u_val*dNx[r] + v_val*dNy[r] + w_val*dNz[r];
                                    double L_supg = tau * u_grad_N;
                                    double common = wJ * (rho/dt) * L_supg;
                                    
                                    loc_b_supg[nodes[r]]       += common * un_val;
                                    loc_b_supg[nodes[r] + Nn]   += common * vn_val;
                                    loc_b_supg[nodes[r] + 2*Nn] += common * wn_val;

                                    for(int c=0; c<27; c++) {
                                        double u_grad_Nc = u_val*dNx[c] + v_val*dNy[c] + w_val*dNz[c];
                                        double val = rho * N[r] * u_grad_Nc + L_supg * (rho * u_grad_Nc) + L_supg * (rho/dt * N[c]);
                                        val += (-mu) * L_supg * Lap[c];
                                        Ke_adv[r + c*27] += wJ * val;
                                    }
                                }
                            }
                        }
                    }
                    
                    for(int r=0; r<27; r++) {
                        for(int c=0; c<27; c++) {
                            double val = Ke_adv[r + c*27];
                            if(std::abs(val) > 1e-14) {
                                loc_rows.push_back((double)(nodes[r] + 1));
                                loc_cols.push_back((double)(nodes[c] + 1));
                                loc_vals.push_back(val);
                                loc_rows.push_back((double)(nodes[r] + Nn + 1));
                                loc_cols.push_back((double)(nodes[c] + Nn + 1));
                                loc_vals.push_back(val);
                                loc_rows.push_back((double)(nodes[r] + 2*Nn + 1));
                                loc_cols.push_back((double)(nodes[c] + 2*Nn + 1));
                                loc_vals.push_back(val);
                            }
                        }
                    }
                }
            }
        }
        #pragma omp critical
        {
            rows.insert(rows.end(), loc_rows.begin(), loc_rows.end());
            cols.insert(cols.end(), loc_cols.begin(), loc_cols.end());
            vals.insert(vals.end(), loc_vals.begin(), loc_vals.end());
            for(size_t i=0; i<b_supg.size(); i++) b_supg[i] += loc_b_supg[i];
        }
    } 
    
    mxArray* mxRows = mxCreateDoubleMatrix(rows.size(), 1, mxREAL);
    mxArray* mxCols = mxCreateDoubleMatrix(cols.size(), 1, mxREAL);
    mxArray* mxVals = mxCreateDoubleMatrix(vals.size(), 1, mxREAL);
    std::copy(rows.begin(), rows.end(), mxGetPr(mxRows));
    std::copy(cols.begin(), cols.end(), mxGetPr(mxCols));
    std::copy(vals.begin(), vals.end(), mxGetPr(mxVals));
    plhs[0] = mxRows; plhs[1] = mxCols; plhs[2] = mxVals;
    mxArray* mxRHS = mxCreateDoubleMatrix(b_supg.size(), 1, mxREAL);
    std::copy(b_supg.begin(), b_supg.end(), mxGetPr(mxRHS));
    plhs[3] = mxRHS;
}