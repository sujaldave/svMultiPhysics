// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "TrilinosResistanceOperator.h"

/**
 * @file TrilinosResistanceOperator.cpp
 * @brief Implements the matrix-free Trilinos resistance transformation.
 */

#ifdef WITH_TRILINOS

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

TrilinosResistanceOperator::TrilinosResistanceOperator(
    const Teuchos::RCP<const Tpetra_Map>& map,
    const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_vectors,
    const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_cap_vectors,
    const std::vector<Trilinos::ResistanceFaceData>& resistance_faces) :
  map_(map),
  boundary_vectors_(boundary_vectors),
  boundary_cap_vectors_(boundary_cap_vectors),
  resistance_faces_(resistance_faces)
{
  if (map_ == Teuchos::null) {
    throw std::runtime_error("[TrilinosResistanceOperator] ERROR: map is null.");
  }
}

void TrilinosResistanceOperator::compute()
{
  for (auto& face : resistance_faces_) {
    face.s_tilde_norm2 = 0.0;
    face.alpha = 0.0;
  }

  const size_t num_faces = std::min(boundary_vectors_.size(), resistance_faces_.size());
  for (size_t i = 0; i < num_faces; ++i) {
    auto& face = resistance_faces_[i];
    const double resistance = face.resistance;

    if (resistance == 0.0 || boundary_vectors_[i] == Teuchos::null) {
      continue;
    }

    if (!boundary_vectors_[i]->getMap()->isSameAs(*map_)) {
      throw std::runtime_error(
          "[TrilinosResistanceOperator] ERROR: boundary vector map does not match operator map.");
    }
    if (boundary_vectors_[i]->getNumVectors() != 1) {
      throw std::runtime_error(
          "[TrilinosResistanceOperator] ERROR: boundary vectors must have one column.");
    }

    Teuchos::Array<Scalar_d> scaled_norm2(1);
    boundary_vectors_[i]->dot(*boundary_vectors_[i], scaled_norm2());

    face.s_tilde_norm2 = scaled_norm2[0] / std::abs(resistance);
    const double denominator = 1.0 + resistance * face.s_tilde_norm2;
    if (std::abs(denominator) <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "[TrilinosResistanceOperator] ERROR: singular resistance transform for face " +
          std::to_string(face.face_id) + ".");
    }
    face.alpha = -resistance / denominator;
  }

  std::vector<double> coefficients(boundary_vectors_.size(),0.0);
  for (size_t i=0; i<num_faces; ++i) {
    const auto& face=resistance_faces_[i];
    if (face.resistance!=0.0 && boundary_vectors_[i]!=Teuchos::null)
      coefficients[i]=-(face.resistance>0.0 ? 1.0 : -1.0)/(1.0+face.resistance*face.s_tilde_norm2);
  }
  low_rank_.reset(new trilinos_bipartition::LowRankUpdate(map_,boundary_vectors_,boundary_cap_vectors_,coefficients));
  computed_ = true;
}

bool TrilinosResistanceOperator::has_active_resistance() const
{
  const size_t num_faces = std::min(boundary_vectors_.size(), resistance_faces_.size());
  for (size_t i = 0; i < num_faces; ++i) {
    if (resistance_faces_[i].resistance != 0.0 &&
        boundary_vectors_[i] != Teuchos::null) {
      return true;
    }
  }
  return false;
}

const std::vector<Trilinos::ResistanceFaceData>&
TrilinosResistanceOperator::resistance_faces() const
{
  return resistance_faces_;
}

void TrilinosResistanceOperator::apply(const Tpetra_MultiVector& X,
    Tpetra_MultiVector& Y, Teuchos::ETransp mode, Scalar_d alpha,
    Scalar_d beta) const
{
  if (mode != Teuchos::NO_TRANS) {
    throw std::runtime_error(
        "[TrilinosResistanceOperator] ERROR: transpose apply is not supported.");
  }
  if (X.getNumVectors() != Y.getNumVectors()) {
    throw std::runtime_error(
        "[TrilinosResistanceOperator] ERROR: input and output column counts differ.");
  }
  if (!X.getMap()->isSameAs(*map_) || !Y.getMap()->isSameAs(*map_)) {
    throw std::runtime_error(
        "[TrilinosResistanceOperator] ERROR: input or output map does not match operator map.");
  }
  if (!computed_ && has_active_resistance()) {
    throw std::runtime_error(
        "[TrilinosResistanceOperator] ERROR: compute() must be called before apply().");
  }

  Y.update(alpha, X, beta);

  if (low_rank_) low_rank_->add(X,Y,alpha);
}

Teuchos::RCP<const Tpetra_Map> TrilinosResistanceOperator::getDomainMap() const
{
  return map_;
}

Teuchos::RCP<const Tpetra_Map> TrilinosResistanceOperator::getRangeMap() const
{
  return map_;
}

#endif
