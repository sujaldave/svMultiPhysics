// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRILINOS_PRECONDITIONER_FACTORY_H
#define TRILINOS_PRECONDITIONER_FACTORY_H

/**
 * @file TrilinosPreconditionerFactory.h
 * @brief Declares reusable Trilinos preconditioner construction and attachment.
 */

#include "consts.h"

#ifdef WITH_TRILINOS

#include "trilinos_impl.h"

namespace trilinos_bipartition {

/**
 * @namespace trilinos_bipartition::preconditioners
 * @brief Constructs and owns preconditioners used by BIPN inner solves.
 *
 * Construction is deliberately separate from Belos problem attachment so one
 * expensive Ifpack2 or MueLu object can be reused by every inner solve for the
 * same Jacobian.
 */
namespace preconditioners {

/**
 * @enum SolverRole
 * @brief Identifies the vector space and Krylov method using a preconditioner.
 */
enum class SolverRole
{
  momentum_gmres, ///< Velocity-space GMRES predictor and correction solves.
  pressure_cg     ///< Pressure-space Schur CG solve.
};

class PreconditionerHandle;

/**
 * @brief Construct and compute a reusable preconditioner.
 * @param type Registered Trilinos preconditioner type.
 * @param role Momentum-GMRES or pressure-CG usage.
 * @param matrix Fill-complete square matrix used for map validation and
 *        Ifpack2 or MueLu setup. It may be null only for the diagonal policy.
 * @param resistance_operator Velocity-space resistance transform required by
 *        the resistance policy.
 * @return An owning handle whose left operator can be attached repeatedly.
 * @throws std::runtime_error For unsupported policies, incompatible matrices,
 *         a missing or map-incompatible resistance operator, or resistance
 *         requested for pressure.
 *
 * The diagonal policy returns a handle with a null left operator because BIPN
 * diagonal scaling is applied directly to the assembled system. If diagonal
 * repair is required for an algebraic preconditioner, the factory creates a
 * private matrix copy and never modifies @p matrix.
 */
Teuchos::RCP<PreconditionerHandle> create_preconditioner(
    consts::PreconditionerType type,
    SolverRole role,
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    const Teuchos::RCP<Tpetra_Operator>& resistance_operator = Teuchos::null);

/**
 * @class PreconditionerHandle
 * @brief Immutable owner of a computed Trilinos preconditioner.
 *
 * Concrete Ifpack2, MueLu, resistance, and any repaired setup matrix objects
 * remain alive for as long as the handle is retained. Accessors expose them
 * for later controlled reuse without permitting ownership to escape.
 */
class PreconditionerHandle
{
  public:
    /// @brief Return the configured preconditioner type.
    consts::PreconditionerType type() const;

    /// @brief Return the intended inner-solver role.
    SolverRole role() const;

    /// @brief Return the operator attached as the Belos left preconditioner.
    const Teuchos::RCP<Tpetra_Operator>& left_operator() const;

    /// @brief Return the matrix used to construct the preconditioner.
    const Teuchos::RCP<Tpetra_CrsMatrix>& setup_matrix() const;

    /// @brief Return the concrete Ifpack2 object, or null for other policies.
    const Teuchos::RCP<Ifpack2_Preconditioner>& ifpack_operator() const;

    /// @brief Return the MueLu hierarchy, or null for other policies.
    const Teuchos::RCP<Tpetra_Operator>& muelu_operator() const;

    /// @brief Return the resistance transform, or null for other policies.
    const Teuchos::RCP<Tpetra_Operator>& resistance_operator() const;

  private:
    PreconditionerHandle(
        consts::PreconditionerType type,
        SolverRole role);

    consts::PreconditionerType type_;
    SolverRole role_;
    Teuchos::RCP<Tpetra_CrsMatrix> setup_matrix_;
    Teuchos::RCP<Ifpack2_Preconditioner> ifpack_operator_;
    Teuchos::RCP<Tpetra_Operator> muelu_operator_;
    Teuchos::RCP<Tpetra_Operator> resistance_operator_;
    Teuchos::RCP<Tpetra_Operator> left_operator_;

    friend Teuchos::RCP<PreconditionerHandle> create_preconditioner(
        consts::PreconditionerType,
        SolverRole,
        const Teuchos::RCP<Tpetra_CrsMatrix>&,
        const Teuchos::RCP<Tpetra_Operator>&);
};

/**
 * @brief Attach an already-computed handle to a Belos linear problem.
 * @param preconditioner Handle returned by create_preconditioner(); null clears
 *        the left preconditioner.
 * @param belos_problem Problem receiving the left operator.
 * @throws std::runtime_error If @p belos_problem is null.
 */
void attach_preconditioner(
    const Teuchos::RCP<PreconditionerHandle>& preconditioner,
    const Teuchos::RCP<Belos_LinearProblem>& belos_problem);

} // namespace preconditioners
} // namespace trilinos_bipartition

#endif

#endif
