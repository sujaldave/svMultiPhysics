// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRILINOS_NS_BLOCK_OPERATORS_H
#define TRILINOS_NS_BLOCK_OPERATORS_H

/**
 * @file TrilinosNSBlockOperators.h
 * @brief Declares Tpetra block data and operators for the Navier-Stokes
 *        bi-partition solver.
 */

#ifdef WITH_TRILINOS

#include "trilinos_impl.h"

#include <memory>
#include <vector>

/**
 * @namespace trilinos_bipartition
 * @brief Tpetra data structures and algorithms used by the Trilinos
 *        Navier-Stokes bi-partition solver.
 *
 * The namespace separates velocity and pressure degrees of freedom while
 * retaining their original global scalar IDs. This permits Tpetra import,
 * export, and sparse operator composition without renumbering application
 * vectors.
 */
namespace trilinos_bipartition {

/**
 * @struct TrilinosNSBlockSystem
 * @brief Owns the Tpetra maps and materialized blocks of one NS Jacobian.
 *
 * For the scalar global Jacobian
 * [
 * J = \begin{pmatrix} A & B \\ C & L \end{pmatrix},
 * ]
 * velocity rows and columns use @ref velocity_map, while pressure rows and
 * columns use @ref pressure_map. Boundary vectors are restricted to the
 * velocity map for use by the momentum and resistance operators.
 */
struct TrilinosNSBlockSystem
{
  Teuchos::RCP<const Tpetra_Map> velocity_map; ///< Owned velocity DOFs.
  Teuchos::RCP<const Tpetra_Map> pressure_map; ///< Owned pressure DOFs.

  Teuchos::RCP<Tpetra_CrsMatrix> A; ///< Velocity-to-velocity momentum block.
  Teuchos::RCP<Tpetra_CrsMatrix> B; ///< Pressure-to-momentum coupling block.
  Teuchos::RCP<Tpetra_CrsMatrix> C; ///< Velocity-to-continuity coupling block.
  Teuchos::RCP<Tpetra_CrsMatrix> L; ///< Pressure stabilization block.

  /// Scaled coupled-outlet update vectors restricted to velocity DOFs.
  std::vector<Teuchos::RCP<Tpetra_MultiVector>> boundary_vectors;

  /// Projection-only cap vectors restricted to velocity DOFs.
  std::vector<Teuchos::RCP<Tpetra_MultiVector>> boundary_cap_vectors;
};

/**
 * @class TrilinosNSBlockTopologyCache
 * @brief Caches equation-local NS block maps, graphs, and communication plans.
 *
 * The scalar equation topology is normally invariant across Newton iterations.
 * This cache derives the velocity/pressure topology once for each full-matrix
 * topology generation. Each Jacobian then receives fresh block matrices whose
 * values are copied into the cached local CRS ordering on the active Kokkos
 * execution space. Matrix values and solve vectors are deliberately not
 * retained between Jacobians.
 *
 * One cache must belong to exactly one equation-level Trilinos backend. It is
 * not thread-safe for concurrent mutation, but independent equation caches may
 * be used concurrently.
 */
class TrilinosNSBlockTopologyCache
{
  public:
    TrilinosNSBlockTopologyCache();
    ~TrilinosNSBlockTopologyCache();

    TrilinosNSBlockTopologyCache(
        const TrilinosNSBlockTopologyCache&) = delete;
    TrilinosNSBlockTopologyCache& operator=(
        const TrilinosNSBlockTopologyCache&) = delete;

    /**
     * @brief Ensure that the cached split topology matches a full NS matrix.
     * @param full_matrix Fill-complete scalar Jacobian.
     * @param full_topology_generation Generation of its equation topology.
     * @param nsd Number of velocity components.
     * @param dof Number of scalar degrees of freedom per node.
     * @return @c true when maps, graphs, and plans were rebuilt.
     */
    bool ensure(
        const Tpetra_CrsMatrix& full_matrix,
        std::size_t full_topology_generation,
        int nsd,
        int dof);

    /// @brief Return whether a valid split topology has been built.
    bool initialized() const;

    /// @brief Return the number of split-topology generations built.
    std::size_t generation() const;

    /// @brief Return the cached velocity map.
    const Teuchos::RCP<const Tpetra_Map>& velocity_map() const;

    /// @brief Return the cached pressure map.
    const Teuchos::RCP<const Tpetra_Map>& pressure_map() const;

    /**
     * @brief Create fresh A/B/C/L matrices and refresh their values on device.
     * @param trilinos Full-system matrix and coupled-boundary vectors.
     * @return A per-Jacobian block system backed by cached static graphs.
     */
    TrilinosNSBlockSystem create_system(
        const Teuchos::RCP<Trilinos>& trilinos) const;

    /// @brief Import a full vector through the cached velocity plan.
    Teuchos::RCP<Tpetra_MultiVector> extract_velocity(
        const Tpetra_MultiVector& source) const;

    /// @brief Import a full vector through the cached pressure plan.
    Teuchos::RCP<Tpetra_MultiVector> extract_pressure(
        const Tpetra_MultiVector& source) const;

    /// @brief Export disjoint block corrections through cached plans.
    void scatter(
        const Tpetra_MultiVector& velocity,
        const Tpetra_MultiVector& pressure,
        const Teuchos::RCP<Tpetra_Vector>& full_vector) const;

    /// @brief Release all Tpetra and Kokkos objects before finalization.
    void clear();

  private:
    class Implementation;
    std::unique_ptr<Implementation> implementation_;
};

/**
 * @class MomentumOperator
 * @brief Applies the physical momentum operator including outlet terms.
 *
 * Given scaled outlet vectors $v_f$ and cap vectors $c_f$, apply()
 * evaluates
 * [
 * y = A x + \sum_f v_f(v_f+c_f)^T x.
 * ]
 * Cap vectors contribute to the scalar projection only, matching FSILS.
 */
class MomentumOperator final : public Tpetra_Operator
{
  public:
    /**
     * @brief Construct a momentum operator over the velocity map.
     * @param A Fill-complete velocity block.
     * @param boundary_vectors Scaled outlet update vectors.
     * @param boundary_cap_vectors Optional projection-only cap vectors.
     * @throws std::runtime_error If matrices or vectors are incompatible.
     */
    MomentumOperator(const Teuchos::RCP<Tpetra_CrsMatrix>& A,
        const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_vectors,
        const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_cap_vectors);

    /**
     * @brief Apply $Y=\beta Y+\alpha A_mX$.
     * @param X Input multivector on the velocity map.
     * @param Y Output multivector on the velocity map.
     * @param mode Transpose mode; transpose is unsupported when outlet terms
     *        are present.
     * @param alpha Scale applied to the momentum result.
     * @param beta Scale applied to the existing output.
     */
    void apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        Scalar_d alpha = Teuchos::ScalarTraits<Scalar_d>::one(),
        Scalar_d beta = Teuchos::ScalarTraits<Scalar_d>::zero()) const override;

    /// @brief Return the velocity domain map.
    Teuchos::RCP<const Tpetra_Map> getDomainMap() const override;

    /// @brief Return the velocity range map.
    Teuchos::RCP<const Tpetra_Map> getRangeMap() const override;

  private:
    Teuchos::RCP<Tpetra_CrsMatrix> A_; ///< Materialized momentum block.
    std::vector<Teuchos::RCP<Tpetra_MultiVector>> boundary_vectors_; ///< Outlet updates.
    std::vector<Teuchos::RCP<Tpetra_MultiVector>> boundary_cap_vectors_; ///< Cap projections.
};

/**
 * @class PressureSchurOperator
 * @brief Matrix-free pressure operator used by the inner CG solve.
 *
 * The operator evaluates
 * [
 * S_p x = Lx + B^T Q_R Bx,
 * ]
 * where $Q_R$ is the resistance transform. A null resistance operator is
 * interpreted as identity, yielding $L+B^TB$.
 */
class PressureSchurOperator final : public Tpetra_Operator
{
  public:
    /**
     * @brief Construct the composed pressure operator.
     * @param L Fill-complete pressure stabilization block.
     * @param B Fill-complete pressure-to-momentum block.
     * @param resistance_operator Optional velocity-space resistance transform.
     * @throws std::runtime_error If operator maps are incompatible.
     */
    PressureSchurOperator(const Teuchos::RCP<Tpetra_CrsMatrix>& L,
        const Teuchos::RCP<Tpetra_CrsMatrix>& B,
        const Teuchos::RCP<Tpetra_Operator>& resistance_operator = Teuchos::null);

    /**
     * @brief Apply $Y=\beta Y+\alpha S_pX$.
     * @param X Input multivector on the pressure map.
     * @param Y Output multivector on the pressure map.
     * @param mode Transpose mode; only Teuchos::NO_TRANS is supported.
     * @param alpha Scale applied to the Schur result.
     * @param beta Scale applied to the existing output.
     */
    void apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        Scalar_d alpha = Teuchos::ScalarTraits<Scalar_d>::one(),
        Scalar_d beta = Teuchos::ScalarTraits<Scalar_d>::zero()) const override;

    /// @brief Return the pressure domain map.
    Teuchos::RCP<const Tpetra_Map> getDomainMap() const override;

    /// @brief Return the pressure range map.
    Teuchos::RCP<const Tpetra_Map> getRangeMap() const override;

  private:
    Teuchos::RCP<Tpetra_CrsMatrix> L_; ///< Pressure stabilization block.
    Teuchos::RCP<Tpetra_CrsMatrix> B_; ///< Pressure-to-momentum block.
    Teuchos::RCP<Tpetra_Operator> resistance_operator_; ///< Optional velocity transform.
};

/**
 * @brief Split a fill-complete scalar NS Jacobian into Tpetra blocks.
 * @param trilinos Equation-level Trilinos state containing the full matrix and
 *        coupled-boundary vectors.
 * @param nsd Number of velocity components.
 * @param dof Number of scalar degrees of freedom per node.
 * @return Maps, matrices, and velocity-restricted boundary vectors.
 * @throws std::runtime_error If dimensions are invalid or the full matrix is
 *         unavailable or not fill complete.
 *
 * This initial extraction path copies rows through Tpetra's host interface.
 * Later cached-graph assembly can populate the same block-system contract
 * directly without changing the solver-facing API.
 */
TrilinosNSBlockSystem build_trilinos_ns_block_system(
    const Teuchos::RCP<Trilinos>& trilinos,
    int nsd,
    int dof);

/**
 * @brief Split an NS Jacobian while reusing equation-local block topology.
 * @param trilinos Equation-level Trilinos state.
 * @param nsd Number of velocity components.
 * @param dof Number of scalar degrees of freedom per node.
 * @param full_topology_generation Generation of the full scalar topology.
 * @param cache Equation-owned block topology cache.
 * @return Fresh block values on cached maps and graphs.
 */
TrilinosNSBlockSystem build_trilinos_ns_block_system(
    const Teuchos::RCP<Trilinos>& trilinos,
    int nsd,
    int dof,
    std::size_t full_topology_generation,
    TrilinosNSBlockTopologyCache& cache);

/**
 * @brief Import selected global IDs from a full vector into a submap.
 * @param source Full-map multivector.
 * @param sub_map Velocity or pressure submap retaining original global IDs.
 * @return A multivector with the same number of columns on @p sub_map.
 */
Teuchos::RCP<Tpetra_MultiVector> extract_subvector(
    const Tpetra_MultiVector& source,
    const Teuchos::RCP<const Tpetra_Map>& sub_map);

/**
 * @brief Export disjoint velocity and pressure corrections into a full vector.
 * @param velocity One-column correction on the velocity submap.
 * @param pressure One-column correction on the pressure submap.
 * @param full_vector Destination on the original scalar map.
 *
 * Tpetra export operations perform local or device copies and any required
 * communication; this function does not acquire host views.
 */
void scatter_subvectors_to_full_vector(
    const Tpetra_MultiVector& velocity,
    const Tpetra_MultiVector& pressure,
    const Teuchos::RCP<Tpetra_Vector>& full_vector);

} // namespace trilinos_bipartition

#endif

#endif
