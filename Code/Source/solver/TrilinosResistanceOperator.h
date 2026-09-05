// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRILINOS_RESISTANCE_OPERATOR_H
#define TRILINOS_RESISTANCE_OPERATOR_H

/**
 * @file TrilinosResistanceOperator.h
 * @brief Declares the matrix-free Trilinos resistance transformation.
 */

#ifdef WITH_TRILINOS

#include "trilinos_impl.h"

#include <vector>

/**
 * @class TrilinosResistanceOperator
 * @brief Applies the coupled-outlet resistance inverse on a Tpetra map.
 *
 * Matrix-free Sherman-Morrison transform for coupled resistance outlets.
 *
 * The stored vectors are v_f = sqrt(abs(R_f)) S_f and optional cap vectors
 * vcap_f. The operator applies
 *
 *   Q_R x = x - sum_f sign(R_f) v_f (v_f + vcap_f)^T x
 *                    / (1 + R_f ||S_f||^2).
 *
 * Cap vectors participate in the scalar projection but not in the update
 * vector, matching the FSILS coupled-boundary convention.
 *
 * The operator owns copies of the RCP containers and resistance metadata.
 * The underlying Tpetra vectors remain reference-counted and must use
 * @p map. Call compute() after the vectors have received their final scaling
 * and before the first apply() when any resistance is active.
 */
class TrilinosResistanceOperator final : public Tpetra_Operator
{
  public:
    /**
     * @brief Construct a resistance transform for a full or block velocity map.
     * @param map Domain and range map of the transformation.
     * @param boundary_vectors Scaled outlet vectors
     *        $v_f=\sqrt{|R_f|}S_f$.
     * @param boundary_cap_vectors Optional scaled cap vectors used only in the
     *        scalar projection.
     * @param resistance_faces Per-face identifiers, resistances, and
     *        coefficient storage copied into the operator.
     * @throws std::runtime_error If @p map is null.
     */
    TrilinosResistanceOperator(
        const Teuchos::RCP<const Tpetra_Map>& map,
        const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_vectors,
        const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_cap_vectors,
        const std::vector<Trilinos::ResistanceFaceData>& resistance_faces);

    /**
     * @brief Compute global face norms and Sherman-Morrison coefficients.
     *
     * Tpetra dot products perform the required MPI reductions. This method
     * must be called again if the scaled boundary vectors change.
     *
     * @throws std::runtime_error If a boundary vector has an incompatible map
     *         or shape, or if a resistance denominator is singular.
     */
    void compute();

    /// @brief Return true when at least one nonzero resistance has a face vector.
    bool has_active_resistance() const;

    /// @brief Return the face metadata and coefficients produced by compute().
    const std::vector<Trilinos::ResistanceFaceData>& resistance_faces() const;

    /**
     * @brief Apply $Y=\beta Y+\alpha Q_R X$.
     * @param X Input multivector on the operator domain map.
     * @param Y Output multivector on the operator range map.
     * @param mode Transpose mode; only Teuchos::NO_TRANS is supported.
     * @param alpha Scale applied to the transformed input.
     * @param beta Scale applied to the existing output.
     * @throws std::runtime_error If maps or column counts are incompatible,
     *         transpose mode is requested, or compute() has not been called.
     */
    void apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        Scalar_d alpha = Teuchos::ScalarTraits<Scalar_d>::one(),
        Scalar_d beta = Teuchos::ScalarTraits<Scalar_d>::zero()) const override;

    /// @brief Return the Tpetra domain map supplied at construction.
    Teuchos::RCP<const Tpetra_Map> getDomainMap() const override;

    /// @brief Return the Tpetra range map supplied at construction.
    Teuchos::RCP<const Tpetra_Map> getRangeMap() const override;

  private:
    Teuchos::RCP<const Tpetra_Map> map_; ///< Shared domain and range map.
    std::vector<Teuchos::RCP<Tpetra_MultiVector>> boundary_vectors_; ///< Scaled face vectors.
    std::vector<Teuchos::RCP<Tpetra_MultiVector>> boundary_cap_vectors_; ///< Projection-only cap vectors.
    std::vector<Trilinos::ResistanceFaceData> resistance_faces_; ///< Owned face metadata.
    bool computed_ = false; ///< Whether coefficients are valid for active faces.
};

#endif

#endif
