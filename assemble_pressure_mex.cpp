#include "mex.h"
#include <vector>
#include <cmath>
#include <omp.h>
#include <cstring>

// --- CONSTANTS ---
const double GP_loc[] = {0.112701665379258, 0.5, 0.887298334620742};
const double GW_loc[] = {0.277777777777778, 0.444444444444444, 0.277777777777778};

// Q1 (Linear) Shape Functions
void eval_shape_Q1(double xi, double* phi, double* dphi) {
    phi[0] = 1.0 - xi;  dphi[0] = -1.0;
    phi[1] = xi;        dphi[1] = 1.0;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // --- FORCE PARALLELISM ---
    int num_procs = omp_get_num_procs();
    omp_set_dynamic(0);             
    omp_set_num_threads(num_procs); 
    // Inputs: [dims, Xc, Yc, Zc, Ux, Uy, Uz, params]
    if(nrhs < 8) mexErrMsgIdAndTxt("MEX:Error", "Not enough inputs.");
    
    const double* dims = mxGetPr(prhs[0]);
    int Nx = (int)dims[0]; int Ny = (int)dims[1]; int Nz = (int)dims[2];
    double* Xc = mxGetPr(prhs[1]); double* Yc = mxGetPr(prhs[2]); double* Zc = mxGetPr(prhs[3]);
    double* Ux = mxGetPr(prhs[4]); double* Uy = mxGetPr(prhs[5]); double* Uz = mxGetPr(prhs[6]);
    double* params = mxGetPr(prhs[7]);
    
    double rho = params[0]; double mu = params[1]; double dt = params[2]; double CI = params[3];
    
    int nx = Nx + 1; int ny = Ny + 1; int nz = Nz + 1; 
    int nx_v = 2*Nx + 1; int ny_v = 2*Ny + 1;

    // Precompute Q1 Basis
    double Ph1[3][2], DPh1[3][2];
    for(int g=0; g<3; g++) {
        double p[2], dp[2];
        eval_shape_Q1(GP_loc[g], p, dp);
        for(int n=0; n<2; n++) { Ph1[n][g] = p[n]; DPh1[n][g] = dp[n]; }
    }

    size_t est_nz = (size_t)Nx * Ny * Nz * 64; 
    std::vector<double> rows, cols, vals;
    rows.reserve(est_nz); cols.reserve(est_nz); vals.reserve(est_nz);

    #pragma omp parallel
    {
        std::vector<double> loc_rows, loc_cols, loc_vals;
        loc_rows.reserve(est_nz / omp_get_num_threads());
        loc_cols.reserve(est_nz / omp_get_num_threads());
        loc_vals.reserve(est_nz / omp_get_num_threads());

        #pragma omp for collapse(3) nowait
        for (int ez = 0; ez < Nz; ez++) {
            for (int ey = 0; ey < Ny; ey++) {
                for (int ex = 0; ex < Nx; ex++) {
                    double hx = Xc[ex+1] - Xc[ex];
                    double hy = Yc[ey+1] - Yc[ey];
                    double hz = Zc[ez+1] - Zc[ez];
                    double h_iso = std::pow(hx*hy*hz, 1.0/3.0);

                    // Q1 Pressure Nodes
                    int nodes[8];
                    int count = 0;
                    for(int k=0; k<2; k++) for(int j=0; j<2; j++) for(int i=0; i<2; i++)
                         nodes[count++] = (ex+i) + (ey+j)*nx + (ez+k)*nx*ny;

                    // Q2 Velocity Nodes (for averaging)
                    int v_nodes[27];
                    int cv = 0;
                    for(int k=0; k<3; k++) for(int j=0; j<3; j++) for(int i=0; i<3; i++)
                         v_nodes[cv++] = (2*ex+i) + (2*ey+j)*nx_v + (2*ez+k)*nx_v*ny_v;
                    
                    double u_avg = 0, v_avg = 0, w_avg = 0;
                    for(int i=0; i<27; i++) {
                        u_avg += Ux[v_nodes[i]]; v_avg += Uy[v_nodes[i]]; w_avg += Uz[v_nodes[i]];
                    }
                    u_avg /= 27.0; v_avg /= 27.0; w_avg /= 27.0;

                    double u_mag = std::sqrt(u_avg*u_avg + v_avg*v_avg + w_avg*w_avg) + 1e-12;
                    double inv2 = 4.0/(dt*dt) + std::pow(rho*2*u_mag/h_iso, 2) + 
                                  CI*CI*mu*mu*( std::pow(2.0/hx,4) + std::pow(2.0/hy,4) + std::pow(2.0/hz,4) );
                    double tau = 1.0 / std::sqrt(inv2 + 1e-16);

                    double Ke[64] = {0}; 

                    for(int gz=0; gz<3; gz++) {
                        for(int gy=0; gy<3; gy++) {
                            for(int gx=0; gx<3; gx++) {
                                double wJ = GW_loc[gx]*GW_loc[gy]*GW_loc[gz] * hx*hy*hz;

                                double N[8], SugN[8];
                                int idx = 0;
                                for(int k=0; k<2; k++) {
                                    double pz = Ph1[k][gz], dpz = DPh1[k][gz];
                                    for(int j=0; j<2; j++) {
                                        double py = Ph1[j][gy], dpy = DPh1[j][gy];
                                        for(int i=0; i<2; i++) {
                                            double px = Ph1[i][gx], dpx = DPh1[i][gx];
                                            N[idx] = px*py*pz;
                                            double dNx = dpx*py*pz / hx;
                                            double dNy = px*dpy*pz / hy;
                                            double dNz = px*py*dpz / hz;
                                            SugN[idx] = u_avg*dNx + v_avg*dNy + w_avg*dNz;
                                            idx++;
                                        }
                                    }
                                }

                                for(int r=0; r<8; r++) {
                                    for(int c=0; c<8; c++) {
                                        double val = N[r]*SugN[c] + 
                                                     tau * SugN[r]*SugN[c] + 
                                                     tau * (rho/dt) * SugN[r]*N[c];
                                        Ke[r + c*8] += wJ * val;
                                    }
                                }
                            }
                        }
                    }

                    for(int r=0; r<8; r++) {
                        for(int c=0; c<8; c++) {
                            double val = Ke[r + c*8];
                            if(std::abs(val) > 1e-15) {
                                loc_rows.push_back((double)(nodes[r] + 1));
                                loc_cols.push_back((double)(nodes[c] + 1));
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
        }
    }

    mxArray* mxRows = mxCreateDoubleMatrix(rows.size(), 1, mxREAL);
    mxArray* mxCols = mxCreateDoubleMatrix(cols.size(), 1, mxREAL);
    mxArray* mxVals = mxCreateDoubleMatrix(vals.size(), 1, mxREAL);
    std::copy(rows.begin(), rows.end(), mxGetPr(mxRows));
    std::copy(cols.begin(), cols.end(), mxGetPr(mxCols));
    std::copy(vals.begin(), vals.end(), mxGetPr(mxVals));
    plhs[0] = mxRows; plhs[1] = mxCols; plhs[2] = mxVals;
}