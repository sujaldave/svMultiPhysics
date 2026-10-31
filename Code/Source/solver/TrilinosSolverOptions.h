// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRILINOS_SOLVER_OPTIONS_H
#define TRILINOS_SOLVER_OPTIONS_H

/**
 * @file TrilinosSolverOptions.h
 * @brief Environment-selected Krylov and AMG options shared by every physics.
 *
 * The same choices apply to the NS bipartition solver (Schur AMG, pressure CG)
 * and to the generic Trilinos solve used by FSI, solid, scalar, and mesh
 * equations. Defaults reproduce the original behavior unless noted.
 *
 * | Variable                  | Values                                        | Default    |
 * |---------------------------|-----------------------------------------------|------------|
 * | SV_TRILINOS_CG_FOLD       | 1, 0: fold the CG residual norm into <r,z>    | 1          |
 * | SV_TRILINOS_AMG_REFRESH   | jacobian, step: numeric AMG refresh frequency  | jacobian   |
 * | SV_TRILINOS_MUELU_REUSE   | none, tP, RP: generic MueLu hierarchy reuse    | none       |
 * | SV_TRILINOS_MUELU_SMOOTHER| see set_smoother()                             | gauss-seidel |
 * | SV_TRILINOS_MUELU_SWEEPS  | sweeps, or Chebyshev degree                    | 2          |
 * | SV_TRILINOS_MUELU_COARSE  | klu, tacho, cusolver                           | klu        |
 */

#ifdef WITH_TRILINOS

#include "Teuchos_ParameterList.hpp"

#include <string>
#include <vector>

namespace trilinos_options {

/// @brief Read an environment choice, throwing if it is not in @p allowed.
std::string choice(const char* name, const std::string& fallback,
    const std::vector<std::string>& allowed);

/// @brief Read a positive integer from the environment.
int positive_integer(const char* name, int fallback);

/**
 * @brief Set Belos CG options. With folding, CG computes ||r||^2 and <r,z> in
 *        one reduction and tests convergence on the same unpreconditioned
 *        residual norm, removing one global reduction per iteration.
 */
void configure_cg(Teuchos::ParameterList& belos);

/// @brief Whether an AMG refreshed in this time step is kept for later Newton
///        iterations of the same step (the Krylov operator is always current).
bool refresh_once_per_step();

/**
 * @brief Write "smoother: type" and "smoother: params" into a MueLu list.
 * @param smoother jacobi, gauss-seidel, symmetric-gauss-seidel,
 *        mt-gauss-seidel, mt-symmetric-gauss-seidel, two-stage-gauss-seidel,
 *        two-stage-symmetric-gauss-seidel, or chebyshev. The mt-/two-stage
 *        variants and Chebyshev run on the device; plain Gauss-Seidel does not.
 * @param sweeps Relaxation sweeps, or the Chebyshev polynomial degree.
 * @param damping Jacobi damping factor; ignored unless positive.
 */
void set_smoother(Teuchos::ParameterList& muelu, const std::string& smoother,
    int sweeps, double damping = 0.0);

/// @brief Set a direct MueLu coarse solver: klu, tacho, or cusolver. Throws
///        if Amesos2 was built without it (MueLu would silently use KLU).
void set_direct_coarse(Teuchos::ParameterList& muelu, const std::string& solver);

/// @brief Names accepted by set_smoother().
const std::vector<std::string>& smoother_names();

} // namespace trilinos_options

#endif

#endif
