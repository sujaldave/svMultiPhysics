// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FLUID_H 
#define FLUID_H 

#include "ComMod.h"

#include "consts.h"

namespace fluid {

// Maximum size of arrays sized by (3,eNoNw) -> (3,MAX_SIZE).
const int MAX_SIZE = 27;

void b_fluid(ComMod& com_mod, const int eNoN, const double w, const Vector<double>& N, const Vector<double>& y,   
    const double h, const Vector<double>& nV, Array<double>& lR, Array3<double>& lK);

void bw_fluid_2d(ComMod& com_mod, const int eNoNw, const int eNoNq, const double w, const Vector<double>& Nw, 
    const Vector<double>& Nq, const Array<double>& Nwx, const Array<double>& yl, const Vector<double>& ub, 
    const Vector<double>& nV, const Vector<double>& tauB, Array<double>& lR, Array3<double>& lK);

void bw_fluid_3d(ComMod& com_mod, const int eNoNw, const int eNoNq, const double w, const Vector<double>& Nw, 
    const Vector<double>& Nq, const Array<double>& Nwx, const Array<double>& yl, const Vector<double>& ub, 
    const Vector<double>& nV, const Vector<double>& tauB, Array<double>& lR, Array3<double>& lK);

void construct_fluid(ComMod& com_mod, const mshType& lM, const Array<double>& Ag, const Array<double>& Yg);

void fluid_2d_c(ComMod& com_mod, const int vmsFlag, const int eNoNw, const int eNoNq, const double w, const Array<double>& Kxi, 
    const Vector<double>& Nw, const Vector<double>& Nq, const Array<double>& Nwx, const Array<double>& Nqx, 
    const Array<double>& Nwxx, const Array<double>& al, const Array<double>& yl, const Array<double>& bfl, 
    Array<double>& lR, Array3<double>& lK, double K_inverse_darcy_permeabilityx);

void fluid_2d_m(ComMod& com_mod, const int vmsFlag, const int eNoNw, const int eNoNq, const double w, const Array<double>& Kxi, 
    const Vector<double>& Nw, const Vector<double>& Nq, const Array<double>& Nwx, const Array<double>& Nqx, 
    const Array<double>& Nwxx, const Array<double>& al, const Array<double>& yl, const Array<double>& bfl, 
    Array<double>& lR, Array3<double>& lK, double K_inverse_darcy_permeability);

void fluid_3d_c(ComMod& com_mod, const int vmsFlag, const int eNoNw, const int eNoNq, const double w, const Array<double>& Kxi, 
    const Vector<double>& Nw, const Vector<double>& Nq, const Array<double>& Nwx, const Array<double>& Nqx, 
    const Array<double>& Nwxx, const Array<double>& al, const Array<double>& yl, const Array<double>& bfl, 
    Array<double>& lR, Array3<double>& lK, double K_inverse_darcy_permeability, double DDir=0.0);

void fluid_3d_m(ComMod& com_mod, const int vmsFlag, const int eNoNw, const int eNoNq, const double w, const Array<double>& Kxi, 
    const Vector<double>& Nw, const Vector<double>& Nq, const Array<double>& Nwx, const Array<double>& Nqx, 
    const Array<double>& Nwxx, const Array<double>& al, const Array<double>& yl, const Array<double>& bfl, 
    Array<double>& lR, Array3<double>& lK, double K_inverse_darcy_permeability, double DDir=0.0);

void getLocalFluidQuant(bool mvMsh, int eNoNw, int eNoNq, const Vector<double>& Nw, const Vector<double>& Nq,
    const Array<double>& Nwx, const Array<double>& Nqx, const Array<double>& Nwxx, const Array<double>& al, 
    const Array<double>& yl, const Array<double>& bfl, std::array<double,3>& ud, double u[3], double ux[3][3], 
    double uxx[3][3][3], double& divU, std::array<double,3>& d2u2, double& p, double px[3], double es[3][3], 
    double esNx[3][MAX_SIZE], double es_x[3][3][3], std::array<double,3>& mu_x, double& gam);

void getStabilizationTauM( double dt, double K_inverse_darcy_permeability, double mu, double rho, double Res,
   double DDir, const double u[3], const Array<double>& Kxi, double& tauM);

void get_viscosity(const ComMod& com_mod, const dmnType& lDmn, double& gamma, double& mu, double& mu_s, double& mu_x);

};

#endif

