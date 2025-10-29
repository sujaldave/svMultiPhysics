// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "ArtificialNeuralNetMaterial.h"
#include "ComMod.h"
#include "mat_fun.h"
using namespace mat_fun;

/// @brief 0th layer output of CANN for activation func kf, input x
void ArtificialNeuralNetMaterial::uCANN_h0(const double x, const int kf, double &f, double &df, double &ddf) const {
    if (kf == 1) {
        f = x;
        df = 1;
        ddf = 0;
    } else if (kf == 2) {
        if (x == 0) {
            f = (std::abs(x) + x) / 2;
            df = 0;
            ddf = 0;
        }
        else {
            f = (std::abs(x) + x) / 2;
            df = 0.5 * (std::abs(x) / x + 1);
            ddf = 0;
        }
    } else if (kf == 3) {
        f = std::abs(x);
        df = std::abs(x) / x;
        ddf = 0;
    }
}

/// @brief 1st layer output of CANN for activation func kf, input x, weight W
void ArtificialNeuralNetMaterial::uCANN_h1(const double x, const int kf, const double W, double &f, double &df, double &ddf) const {
    if (kf == 1) {
        f = W * x;
        df = W;
        ddf = 0;
    } else if (kf == 2) {
        f = W * W * x * x;
        df = 2 * W * W * x;
        ddf = 2 * W * W;
    }
}

/// @brief 2nd layer output of CANN for activation func kf, input x, weight W
void ArtificialNeuralNetMaterial::uCANN_h2(const double x, const int kf, const double W, double &f, double &df, double &ddf) const {
    if (kf == 1) {
        f = W * x;
        df = W;
        ddf = 0;
    } else if (kf == 2) {
        f = std::exp(W * x) - 1;
        df = W * std::exp(W * x);
        ddf = W * W * std::exp(W * x);
    } else if (kf == 3) {
        f = -std::log(1 - W * x);
        df = W / (1 - W * x);
        ddf = -W * W / ((1 - W * x) * (1 - W * x));
    }
}

/// @brief Updates psi and its derivatives
void ArtificialNeuralNetMaterial::uCANN(
    const double xInv, const int kInv,
    const int kf0, const int kf1, const int kf2,
    const double W0, const double W1, const double W2,
    double &psi, double (&dpsi)[9], double (&ddpsi)[9]
) const {
    double f0, df0, ddf0;
    uCANN_h0(xInv, kf0, f0, df0, ddf0);
    double f1, df1, ddf1;
    uCANN_h1(f0, kf1, W0, f1, df1, ddf1);
    double f2, df2, ddf2;
    uCANN_h2(f1, kf2, W1, f2, df2, ddf2);

    psi += W2 * f2;
    dpsi[kInv - 1] += W2 * df2 * df1 * df0;
    ddpsi[kInv - 1] += W2 * ((ddf2 * df1 * df1 + df2 * ddf1) * df0 * df0 + df2 * df1 * ddf0);
}

/// @brief function to build psi and dpsidI1 to 9
void ArtificialNeuralNetMaterial::evaluate(const double aInv[9], double &psi, double (&dpsi)[9], double (&ddpsi)[9]) const {
    // Initializing
    psi = 0;
    for (int i = 0; i < 9; ++i) {
        dpsi[i] = 0;
        ddpsi[i] = 0;
    }

    double ref[9] = {3, 3, 1, 1, 1, 0, 0, 1, 1};

    for (int i = 0; i < num_rows; ++i) {
        int kInv = this->invariant_indices(i);
        int kf0 = this->activation_functions(i, 0);
        int kf1 = this->activation_functions(i, 1);
        int kf2 = this->activation_functions(i, 2);
        double W0 = this->weights(i, 0);
        double W1 = this->weights(i, 1);
        double W2 = this->weights(i, 2);

        double xInv = aInv[kInv - 1] - ref[kInv - 1];
        uCANN(xInv, kInv, kf0, kf1, kf2, W0, W1, W2, psi, dpsi, ddpsi);
    }
}

template<size_t nsd>
void ArtificialNeuralNetMaterial::computeInvariantsAndDerivatives(
const Matrix<nsd>& C, const Matrix<nsd>& fl, int nfd, double J2d, double J4d, const Matrix<nsd>& Ci,
const Matrix<nsd>& Idm, const double Tfa, Matrix<nsd>& N1, double& psi, double (&Inv)[9],std::array<Matrix<nsd>,9>& dInv,
std::array<Tensor<nsd>,9>& ddInv) const {

    // Create raw arrays to avoid dynamic initialization
    double C_data[3][3] = {{0}};  // Store C matrix data
    double fl_data[3][2] = {{0}}; // Store fl matrix data
    double Ci_data[3][3] = {{0}}; // Store Ci matrix data
    double Idm_data[3][3] = {{0}}; // Store Idm matrix data
    double N1_data[3][3] = {{0}}; // Store N1 matrix data
    double N2_data[3][3] = {{0}}; // Store N2 matrix data
    double N12_data[3][3] = {{0}}; // Store N12 matrix data
    double dInv_data[9][3][3] = {{{0}}}; // Store first derivatives
    double ddInv_data[9][3][3][3][3] = {{{{0}}}}; // Store second derivatives
    double C2_data[3][3] = {{0}}; // Store C*C data

    // Copy input matrices to raw arrays
    for(int i = 0; i < nsd; i++) {
        for(int j = 0; j < nsd; j++) {
            C_data[i][j] = C(i,j);
            Ci_data[i][j] = Ci(i,j);
            Idm_data[i][j] = Idm(i,j);
            if(j < 2) {
                fl_data[i][j] = fl(i,j);
            }
        }
    }

    // Calculate C2 = C * C using raw arrays
    for(int i = 0; i < nsd; i++) {
        for(int j = 0; j < nsd; j++) {
            C2_data[i][j] = 0;
            for(int k = 0; k < nsd; k++) {
                C2_data[i][j] += C_data[i][k] * C_data[k][j];
            }
        }
    }

    // Calculate traces
    double trC = 0, trC2 = 0;
    for(int i = 0; i < nsd; i++) {
        trC += C_data[i][i];
        trC2 += C2_data[i][i];
    }

    // Calculate invariants using raw arrays
    Inv[0] = J2d * trC;
    Inv[1] = 0.50 * (Inv[0]*Inv[0] - J4d * trC2);
    
    if(nsd == 2) {
        Inv[2] = C_data[0][0]*C_data[1][1] - C_data[0][1]*C_data[1][0];
    } else {
        Inv[2] = C_data[0][0]*C_data[1][1]*C_data[2][2] + 
                 C_data[0][1]*C_data[1][2]*C_data[2][0] + 
                 C_data[0][2]*C_data[1][0]*C_data[2][1] - 
                 C_data[0][2]*C_data[1][1]*C_data[2][0] - 
                 C_data[0][0]*C_data[1][2]*C_data[2][1] - 
                 C_data[0][1]*C_data[1][0]*C_data[2][2];
    }

    // Initialize N1 and other direction tensors
    for(int i = 0; i < nsd; i++) {
        for(int j = 0; j < nsd; j++) {
            N1_data[i][j] = fl_data[i][0] * fl_data[j][0];
            N1(i,j) = N1_data[i][j]; // Also update N1 matrix for output
            
            if(nfd == 2) {
                N2_data[i][j] = fl_data[i][1] * fl_data[j][1];
                N12_data[i][j] = 0.5*(fl_data[i][0]*fl_data[j][1] + fl_data[i][1]*fl_data[j][0]);
            }
        }
    }

    // Calculate additional fiber invariants
    double flC[3] = {0}; // fl_data[*][0] * C_data
    double flC2[3] = {0}; // fl_data[*][0] * C2_data
    
    for(int i = 0; i < nsd; i++) {
        for(int j = 0; j < nsd; j++) {
            flC[i] += fl_data[i][0] * C_data[i][j];
            flC2[i] += fl_data[i][0] * C2_data[i][j];
        }
    }

    double flCfl = 0, flC2fl = 0;
    for(int i = 0; i < nsd; i++) {
        flCfl += flC[i] * fl_data[i][0];
        flC2fl += flC2[i] * fl_data[i][0];
    }

    Inv[3] = J2d * flCfl;
    Inv[4] = J4d * flC2fl;

    // Calculate first derivatives using raw arrays
    for(int i = 0; i < nsd; i++) {
        for(int j = 0; j < nsd; j++) {
            dInv_data[0][i][j] = -Inv[0]/3.0 * Ci_data[i][j] + J2d * Idm_data[i][j];
            dInv_data[1][i][j] = (trC2/3.0)*Ci_data[i][j] + Inv[0]*dInv_data[0][i][j] + J4d*C_data[i][j];
            dInv_data[2][i][j] = Inv[2]*Ci_data[i][j];
            dInv_data[3][i][j] = -Inv[3]/3.0*Ci_data[i][j] + J2d*N1_data[i][j];
            dInv_data[4][i][j] = J4d*(N1_data[i][j]*C_data[i][j] + C_data[i][j]*N1_data[i][j]) - 
                               Inv[4]/3.0*Ci_data[i][j];

            // Zero remaining matrices
            for(int k = 5; k < 9; k++) {
                dInv_data[k][i][j] = 0.0;
            }
        }
    }

    // Initialize second derivatives
    for(int n = 0; n < 9; n++) {
        for(int i = 0; i < nsd; i++) {
            for(int j = 0; j < nsd; j++) {
                for(int k = 0; k < nsd; k++) {
                    for(int l = 0; l < nsd; l++) {
                        if(n < 5) {  // Only set first 5 tensors initially
                            if(n == 0) {
                                ddInv_data[0][i][j][k][l] = (-1.0/3.0)*(dInv_data[0][i][j]*Ci_data[k][l] + 
                                    Inv[0]*(Ci_data[i][k]*Ci_data[j][l] + Ci_data[i][l]*Ci_data[j][k])/2.0 + 
                                    J2d*Ci_data[i][j]*Idm_data[k][l]);
                            } else if(n == 1) {
                                ddInv_data[1][i][j][k][l] = dInv_data[0][i][j]*dInv_data[0][k][l] + 
                                    Inv[0]*ddInv_data[0][i][j][k][l] + 
                                    (trC2/3.0)*(-Ci_data[i][k]*Ci_data[j][l] - Ci_data[i][l]*Ci_data[j][k])/2.0;
                            } else if(n == 2) {
                                ddInv_data[2][i][j][k][l] = Inv[2]*(Ci_data[i][k]*Ci_data[j][l] + 
                                    Ci_data[i][l]*Ci_data[j][k])/2.0;
                            } else if(n == 3) {
                                ddInv_data[3][i][j][k][l] = (-1.0/3.0)*(dInv_data[3][i][j]*Ci_data[k][l] + 
                                    J2d*Ci_data[i][j]*N1_data[k][l] + 
                                    Inv[3]*(Ci_data[i][k]*Ci_data[j][l] + Ci_data[i][l]*Ci_data[j][k])/2.0);
                            } else if(n == 4) {
                                ddInv_data[4][i][j][k][l] = (-1.0/3.0)*(dInv_data[4][i][j]*Ci_data[k][l] + 
                                    Inv[4]*(Ci_data[i][k]*Ci_data[j][l] + Ci_data[i][l]*Ci_data[j][k])/2.0 + 
                                    2*J4d*Ci_data[i][j]*(N1_data[k][l]*C_data[k][l] + C_data[k][l]*N1_data[k][l]));
                            }
                        } else {
                            ddInv_data[n][i][j][k][l] = 0.0;
                        }
                    }
                }
            }
        }
    }

    if (nfd == 2) {
        // Additional fiber calculations for nfd == 2
        double flC1[3] = {0}; // fl.col(1) * C
        double flC2_1[3] = {0}; // fl.col(1) * C2
        
        for(int i = 0; i < nsd; i++) {
            for(int j = 0; j < nsd; j++) {
                flC1[i] += fl_data[i][1] * C_data[i][j];
                flC2_1[i] += fl_data[i][1] * C2_data[i][j];
            }
        }

        double flCf2 = 0, flC2f2 = 0;
        double flCf12 = 0, flC2f12 = 0;
        
        for(int i = 0; i < nsd; i++) {
            flCf2 += flC1[i] * fl_data[i][1];
            flC2f2 += flC2_1[i] * fl_data[i][1];
            flCf12 += flC[i] * fl_data[i][1];
            flC2f12 += flC2[i] * fl_data[i][1];
        }

        // Set remaining invariants
        Inv[5] = J2d * flCf12;
        Inv[6] = J4d * flC2f12;
        Inv[7] = J2d * flCf2;
        Inv[8] = J4d * flC2f2;

        // Calculate additional derivatives
        for(int i = 0; i < nsd; i++) {
            for(int j = 0; j < nsd; j++) {
                dInv_data[5][i][j] = -Inv[5]/3.0*Ci_data[i][j] + J2d*N12_data[i][j];
                dInv_data[6][i][j] = J4d*(N12_data[i][j]*C_data[i][j] + C_data[i][j]*N12_data[i][j]) - 
                                   Inv[6]/3.0*Ci_data[i][j];
                dInv_data[7][i][j] = -Inv[7]/3.0*Ci_data[i][j] + J2d*N2_data[i][j];
                dInv_data[8][i][j] = J4d*(N2_data[i][j]*C_data[i][j] + C_data[i][j]*N2_data[i][j]) - 
                                   Inv[8]/3.0*Ci_data[i][j];
            }
        }

        // Additional tensor derivatives for nfd == 2
        for(int i = 0; i < nsd; i++) {
            for(int j = 0; j < nsd; j++) {
                for(int k = 0; k < nsd; k++) {
                    for(int l = 0; l < nsd; l++) {
                        ddInv_data[5][i][j][k][l] = (-1.0/3.0)*(dInv_data[5][i][j]*Ci_data[k][l] + 
                            J2d*Ci_data[i][j]*N12_data[k][l] + 
                            Inv[5]*(Ci_data[i][k]*Ci_data[j][l] + Ci_data[i][l]*Ci_data[j][k])/2.0);

                        ddInv_data[6][i][j][k][l] = (-1.0/3.0)*(dInv_data[6][i][j]*Ci_data[k][l] + 
                            Inv[6]*(Ci_data[i][k]*Ci_data[j][l] + Ci_data[i][l]*Ci_data[j][k])/2.0 + 
                            2*J4d*Ci_data[i][j]*(N12_data[k][l]*C_data[k][l] + C_data[k][l]*N12_data[k][l]));

                        ddInv_data[7][i][j][k][l] = (-1.0/3.0)*(dInv_data[7][i][j]*Ci_data[k][l] + 
                            J2d*Ci_data[i][j]*N2_data[k][l] + 
                            Inv[7]*(Ci_data[i][k]*Ci_data[j][l] + Ci_data[i][l]*Ci_data[j][k])/2.0);

                        ddInv_data[8][i][j][k][l] = (-1.0/3.0)*(dInv_data[8][i][j]*Ci_data[k][l] + 
                            Inv[8]*(Ci_data[i][k]*Ci_data[j][l] + Ci_data[i][l]*Ci_data[j][k])/2.0 + 
                            2*J4d*Ci_data[i][j]*(N2_data[k][l]*C_data[k][l] + C_data[k][l]*N2_data[k][l]));
                    }
                }
            }
        }
    }

    // Copy final results back to matrices and tensors
    for(int n = 0; n < 9; n++) {
        for(int i = 0; i < nsd; i++) {
            for(int j = 0; j < nsd; j++) {
                dInv[n](i,j) = dInv_data[n][i][j];
                for(int k = 0; k < nsd; k++) {
                    for(int l = 0; l < nsd; l++) {
                        ddInv[n](i,j,k,l) = ddInv_data[n][i][j][k][l];
                    }
                }
            }
        }
    }
}

// Template instantiations
template void ArtificialNeuralNetMaterial::computeInvariantsAndDerivatives<2>(
const Matrix<2>& C, const Matrix<2>& fl, int nfd, double J2d, double J4d, const Matrix<2>& Ci,
const Matrix<2>& Idm, const double Tfa, Matrix<2>& N1, double& psi, double (&Inv)[9], std::array<Matrix<2>,9>& dInv,
std::array<Tensor<2>,9>& ddInv) const;

template void ArtificialNeuralNetMaterial::computeInvariantsAndDerivatives<3>(
const Matrix<3>& C, const Matrix<3>& fl, int nfd, double J2d, double J4d, const Matrix<3>& Ci,
const Matrix<3>& Idm, const double Tfa, Matrix<3>& N1, double& psi, double (&Inv)[9], std::array<Matrix<3>,9>& dInv,
std::array<Tensor<3>,9>& ddInv) const;