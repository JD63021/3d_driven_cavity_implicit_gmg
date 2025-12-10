#include "mex.h"
#include <vector>
#include <cmath>

// Helper: Column-major access
// dimR = number of ROWS in the matrix
inline double get(const double* M, int r, int c, int dimR) {
    return M[r + c * dimR];
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs < 4) mexErrMsgTxt("Need inputs: Dims, Grid, Mats, Params");

    double *dims = mxGetPr(prhs[0]);
    int Nx = (int)dims[0]; int Ny = (int)dims[1]; int Nz = (int)dims[2];

    double *Xc = mxGetPr(mxGetField(prhs[1], 0, "Xc"));
    double *Yc = mxGetPr(mxGetField(prhs[1], 0, "Yc"));
    double *Zc = mxGetPr(mxGetField(prhs[1], 0, "Zc"));

    const mxArray *mats = prhs[2];
    double *M2 = mxGetPr(mxGetField(mats, 0, "M2")); 
    double *K2 = mxGetPr(mxGetField(mats, 0, "K2"));
    double *G2 = mxGetPr(mxGetField(mats, 0, "G2")); 
    double *GT2 = mxGetPr(mxGetField(mats, 0, "GT2"));
    
    double *M1 = mxGetPr(mxGetField(mats, 0, "M1")); 
    double *K1 = mxGetPr(mxGetField(mats, 0, "K1"));

    double *M21 = mxGetPr(mxGetField(mats, 0, "M21")); // 3x2 (Stride 3)
    double *G21 = mxGetPr(mxGetField(mats, 0, "G21")); // 3x2 (Stride 3)
    double *M12 = mxGetPr(mxGetField(mats, 0, "M12")); // 2x3 (Stride 2)
    double *G12 = mxGetPr(mxGetField(mats, 0, "G12")); // 2x3 (Stride 2)

    double *params = mxGetPr(prhs[3]);
    double mu = params[0];
    double gamma = params[1];

    int nx_v = 2*Nx + 1; int ny_v = 2*Ny + 1; int nz_v = 2*Nz + 1;
    int Nn = nx_v * ny_v * nz_v;
    int nx_p = Nx + 1; int ny_p = Ny + 1; int nz_p = Nz + 1;
    int Np = nx_p * ny_p * nz_p;

    int off_u = 0; int off_v = Nn; int off_w = 2*Nn; int off_p = 3*Nn;

    std::vector<double> rK, cK, vK;       
    std::vector<double> rM, cM, vM;       
    std::vector<double> rMp, cMp, vMp;    
    std::vector<double> rKp, cKp, vKp;    
    std::vector<double> rRup, cRup, vRup; 

    int n_elem = Nx*Ny*Nz;
    rK.reserve(n_elem * 300); rM.reserve(n_elem * 27); rMp.reserve(n_elem * 8);

    for (int ez = 0; ez < Nz; ez++) {
        double hz = Zc[ez+1] - Zc[ez];
        for (int ey = 0; ey < Ny; ey++) {
            double hy = Yc[ey+1] - Yc[ey];
            for (int ex = 0; ex < Nx; ex++) {
                double hx = Xc[ex+1] - Xc[ex];
                double vol = hx*hy*hz;
                double i_hx = 1.0/hx; double i_hy = 1.0/hy; double i_hz = 1.0/hz;
                
                int idx_v[27]; int idx_p[8];
                int cnt = 0;
                for(int k=0; k<3; k++) for(int j=0; j<3; j++) for(int i=0; i<3; i++) {
                    idx_v[cnt++] = (2*ex+i) + (2*ey+j)*nx_v + (2*ez+k)*nx_v*ny_v;
                }
                cnt = 0;
                for(int k=0; k<2; k++) for(int j=0; j<2; j++) for(int i=0; i<2; i++) {
                    idx_p[cnt++] = (ex+i) + (ey+j)*nx_p + (ez+k)*nx_p*ny_p;
                }

                // 1. VELOCITY (27x27)
                for (int r=0; r<27; r++) {
                    int row_g = idx_v[r];
                    int rk = r/9; int rem = r%9; int rj = rem/3; int ri = rem%3;
                    for (int c=0; c<27; c++) {
                        int col_g = idx_v[c];
                        int ck = c/9; rem = c%9; int cj = rem/3; int ci = rem%3;

                        double mx = get(M2,ri,ci,3); double kx = get(K2,ri,ci,3);
                        double my = get(M2,rj,cj,3); double ky = get(K2,rj,cj,3);
                        double mz = get(M2,rk,ck,3); double kz = get(K2,rk,ck,3);
                        
                        double val_m = vol * mx*my*mz;
                        rM.push_back(row_g+1); cM.push_back(col_g+1); vM.push_back(val_m);

                        double val_diff = mu * ( (hy*hz*i_hx)*kx*my*mz + (hx*hz*i_hy)*mx*ky*mz + (hx*hy*i_hz)*mx*my*kz );
                        rK.push_back(off_u + row_g + 1); cK.push_back(off_u + col_g + 1); vK.push_back(val_diff);
                        rK.push_back(off_v + row_g + 1); cK.push_back(off_v + col_g + 1); vK.push_back(val_diff);
                        rK.push_back(off_w + row_g + 1); cK.push_back(off_w + col_g + 1); vK.push_back(val_diff);
                    }
                }

                // 2. PRESSURE (8x8)
                for(int r=0; r<8; r++){
                    int row_g = idx_p[r];
                    int rk = r/4; int rem = r%4; int rj = rem/2; int ri = rem%2;
                    for(int c=0; c<8; c++){
                        int col_g = idx_p[c];
                        int ck = c/4; rem = c%4; int cj = rem/2; int ci = rem%2;

                        double mx = get(M1,ri,ci,2); double kx = get(K1,ri,ci,2);
                        double my = get(M1,rj,cj,2); double ky = get(K1,rj,cj,2);
                        double mz = get(M1,rk,ck,2); double kz = get(K1,rk,ck,2);

                        rMp.push_back(row_g+1); cMp.push_back(col_g+1); vMp.push_back(vol * mx*my*mz);
                        double val_kp = (hy*hz*i_hx)*kx*my*mz + (hx*hz*i_hy)*mx*ky*mz + (hx*hy*i_hz)*mx*my*kz;
                        rKp.push_back(row_g+1); cKp.push_back(col_g+1); vKp.push_back(val_kp);
                    }
                }

                // 3. COUPLING (27x8)
                for(int i_p=0; i_p<8; i_p++){
                     int idx_P_global = idx_p[i_p];
                     int pk = i_p/4; int prem = i_p%4; int pj = prem/2; int pi = prem%2;

                     for(int i_v=0; i_v<27; i_v++){
                         int idx_V_global = idx_v[i_v];
                         int vk = i_v/9; int vrem = i_v%9; int vj = vrem/3; int vi = vrem%3;

                         // --- FIX STARTS HERE (CORRECT STRIDES) ---
                         
                         // M12 is 2x3. Stride is 2.
                         double m12_x = get(M12, pi, vi, 2); // <--- FIX (was 3)
                         double m12_y = get(M12, pj, vj, 2); // <--- FIX (was 3)
                         double m12_z = get(M12, pk, vk, 2); // <--- FIX (was 3)
                         
                         rRup.push_back(idx_P_global+1); cRup.push_back(idx_V_global+1);
                         vRup.push_back(vol * m12_x * m12_y * m12_z);

                         // M21, G21 are 3x2. Stride is 3.
                         double m21_x = get(M21, vi, pi, 3); // <--- FIX (was 2)
                         double g21_x = get(G21, vi, pi, 3); // <--- FIX (was 2)
                         
                         double m21_y = get(M21, vj, pj, 3); // <--- FIX (was 2)
                         double g21_y = get(G21, vj, pj, 3); // <--- FIX (was 2)
                         
                         double m21_z = get(M21, vk, pk, 3); // <--- FIX (was 2)
                         double g21_z = get(G21, vk, pk, 3); // <--- FIX (was 2)
                         
                         // --- FIX ENDS HERE ---

                         double k_xp =  +(hy*hz) * m21_z * m21_y * g21_x;
                         double k_yp =  +(hx*hz) * m21_z * g21_y * m21_x;
                         double k_zp = + (hx*hy) * g21_z * m21_y * m21_x;

                         rK.push_back(off_u + idx_V_global + 1); cK.push_back(off_p + idx_P_global + 1); vK.push_back(k_xp);
                         rK.push_back(off_v + idx_V_global + 1); cK.push_back(off_p + idx_P_global + 1); vK.push_back(k_yp);
                         rK.push_back(off_w + idx_V_global + 1); cK.push_back(off_p + idx_P_global + 1); vK.push_back(k_zp);

                         // Divergence (M12, G12 are 2x3 -> Stride 2)
                         double m12_x_d = get(M12, pi, vi, 2); double g12_x_d = get(G12, pi, vi, 2); // <--- FIX
                         double m12_y_d = get(M12, pj, vj, 2); double g12_y_d = get(G12, pj, vj, 2); // <--- FIX
                         double m12_z_d = get(M12, pk, vk, 2); double g12_z_d = get(G12, pk, vk, 2); // <--- FIX

                         double k_cx = (hy*hz) * m12_z_d * m12_y_d * g12_x_d;
                         double k_cy = (hx*hz) * m12_z_d * g12_y_d * m12_x_d;
                         double k_cz = (hx*hy) * g12_z_d * m12_y_d * m12_x_d;
                         
                         rK.push_back(off_p + idx_P_global + 1); cK.push_back(off_u + idx_V_global + 1); vK.push_back(k_cx);
                         rK.push_back(off_p + idx_P_global + 1); cK.push_back(off_v + idx_V_global + 1); vK.push_back(k_cy);
                         rK.push_back(off_p + idx_P_global + 1); cK.push_back(off_w + idx_V_global + 1); vK.push_back(k_cz);
                     }
                }
            }
        }
    }

    auto copy_to_mx = [](const std::vector<double>& v) {
        mxArray* m = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
        std::copy(v.begin(), v.end(), mxGetPr(m));
        return m;
    };
    plhs[0] = copy_to_mx(rK); plhs[1] = copy_to_mx(cK); plhs[2] = copy_to_mx(vK);
    plhs[3] = copy_to_mx(rM); plhs[4] = copy_to_mx(cM); plhs[5] = copy_to_mx(vM);
    plhs[6] = copy_to_mx(rMp); plhs[7] = copy_to_mx(cMp); plhs[8] = copy_to_mx(vMp);
    plhs[9] = copy_to_mx(rKp); plhs[10] = copy_to_mx(cKp); plhs[11] = copy_to_mx(vKp);
    plhs[12] = copy_to_mx(rRup); plhs[13] = copy_to_mx(cRup); plhs[14] = copy_to_mx(vRup);
}