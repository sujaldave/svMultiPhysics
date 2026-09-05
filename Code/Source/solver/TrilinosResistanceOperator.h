// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRILINOS_RESISTANCE_OPERATOR_H
#define TRILINOS_RESISTANCE_OPERATOR_H

#ifdef WITH_TRILINOS

#include "trilinos_impl.h"

#include <vector>

/**
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
 */
class TrilinosResistanceOperator final : public Tpetra_Operator
{
  public:
    TrilinosResistanceOperator(
        const Teuchos::RCP<const Tpetra_Map>& map,
        const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_vectors,
        const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_cap_vectors,
        const std::vector<Trilinos::ResistanceFaceData>& resistance_faces);

    void compute();

    bool has_active_resistance() const;

    const std::vector<Trilinos::ResistanceFaceData>& resistance_faces() const;

    void apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        Scalar_d alpha = Teuchos::ScalarTraits<Scalar_d>::one(),
        Scalar_d beta = Teuchos::ScalarTraits<Scalar_d>::zero()) const override;

    Teuchos::RCP<const Tpetra_Map> getDomainMap() const override;
    Teuchos::RCP<const Tpetra_Map> getRangeMap() const override;

  private:
    Teuchos::RCP<const Tpetra_Map> map_;
    std::vector<Teuchos::RCP<Tpetra_MultiVector>> boundary_vectors_;
    std::vector<Teuchos::RCP<Tpetra_MultiVector>> boundary_cap_vectors_;
    std::vector<Trilinos::ResistanceFaceData> resistance_faces_;
    bool computed_ = false;
};

#endif

#endif
