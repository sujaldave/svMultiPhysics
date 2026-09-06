// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRILINOS_BIPARTITION_NS_H
#define TRILINOS_BIPARTITION_NS_H

/**
 * @file TrilinosBipartitionNS.h
 * @brief Declares the Trilinos Navier-Stokes bi-partition solver.
 */

#ifdef WITH_TRILINOS

#include "ComMod.h"
#include "TrilinosNSBlockOperators.h"

namespace trilinos_bipartition {

/**
 * @struct BelosGmresBudget
 * @brief Belos parameters equivalent to the FSILS restarted-GMRES controls.
 */
struct BelosGmresBudget
{
  int maximum_iterations = 0; ///< Total Krylov iterations across all cycles.
  int num_blocks = 0;         ///< Krylov-space dimension of one cycle.
  int maximum_restarts = 0;   ///< Restarts after the initial cycle.
};

/**
 * @brief Translate FSILS restart-cycle controls into Belos parameters.
 * @param restart_cycles Number of complete GMRES cycles allowed by FSILS.
 * @param krylov_dimension Maximum basis dimension in each cycle.
 * @return Checked Belos iteration, block, and restart limits.
 * @throws std::runtime_error If either input is nonpositive or their product
 *         cannot be represented by an @c int.
 */
BelosGmresBudget make_belos_gmres_budget(
    int restart_cycles,
    int krylov_dimension);

/**
 * @brief Validate and return the pressure-CG iteration limit.
 * @param max_iterations Total number of CG iterations.
 * @return @p max_iterations when it is positive.
 * @throws std::runtime_error If @p max_iterations is nonpositive.
 */
int make_belos_cg_max_iterations(int max_iterations);

/**
 * @class TrilinosBipartitionNSSolver
 * @brief Solves one assembled incompressible NS Jacobian by RI bi-partition.
 *
 * The solver extracts velocity and pressure blocks, constructs one momentum
 * and one pressure preconditioner for the Jacobian, and reuses those handles
 * for every predictor GMRES, Schur CG, and correction GMRES solve in the RI
 * loop. All large vector and sparse-operator operations remain in Tpetra.
 */
class TrilinosBipartitionNSSolver
{
  public:
    /**
     * @brief Construct a solver over equation-level Trilinos state.
     * @param trilinos Full-system Tpetra state for the current equation.
     * @param nsd Number of spatial velocity components.
     * @param dof Number of scalar degrees of freedom per node.
     */
    TrilinosBipartitionNSSolver(
        const Teuchos::RCP<Trilinos>& trilinos,
        int nsd,
        int dof);

    /**
     * @brief Solve a system assembled directly through Tpetra.
     * @param equation Equation and inner-solver configuration/result fields.
     * @param solution Destination in local-plus-ghost node/DOF ordering.
     * @param dirichlet_weights Existing symmetric scaling mask.
     */
    void solve_assembled(
        eqType& equation,
        double* solution,
        const double* dirichlet_weights);

    /**
     * @brief Lift an FSILS-assembled system into Tpetra and solve it.
     * @param equation Equation and inner-solver configuration/result fields.
     * @param values FSILS CSR nodal-block values.
     * @param rhs FSILS residual in local-plus-ghost node/DOF ordering.
     * @param solution Destination in the same node/DOF ordering as @p rhs.
     * @param dirichlet_weights Existing symmetric scaling mask.
     */
    void solve_fsils_assembled(
        eqType& equation,
        const double* values,
        const double* rhs,
        double* solution,
        const double* dirichlet_weights);

  private:
    void assemble_fsils_system(const double* values, const double* rhs) const;

    void solve_tpetra_system(
        eqType& equation,
        double* solution,
        const double* dirichlet_weights,
        bool rhs_needs_add_export);

    Teuchos::RCP<Trilinos> trilinos_; ///< Full-system state for this equation.
    int nsd_;                         ///< Number of velocity components.
    int dof_;                         ///< Degrees of freedom per node.
};

} // namespace trilinos_bipartition

#endif

#endif
