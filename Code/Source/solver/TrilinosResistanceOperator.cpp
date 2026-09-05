// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "TrilinosResistanceOperator.h"

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

  const size_t num_faces = std::min(boundary_vectors_.size(), resistance_faces_.size());
  for (size_t i = 0; i < num_faces; ++i) {
    const auto& face = resistance_faces_[i];
    if (face.resistance == 0.0 || boundary_vectors_[i] == Teuchos::null) {
      continue;
    }

    const auto face_vector = boundary_vectors_[i]->getVector(0);
    Teuchos::RCP<const Tpetra_Vector> cap_vector = Teuchos::null;
    if (i < boundary_cap_vectors_.size() &&
        boundary_cap_vectors_[i] != Teuchos::null) {
      if (!boundary_cap_vectors_[i]->getMap()->isSameAs(*map_)) {
        throw std::runtime_error(
            "[TrilinosResistanceOperator] ERROR: boundary cap vector map does not match operator map.");
      }
      if (boundary_cap_vectors_[i]->getNumVectors() != 1) {
        throw std::runtime_error(
            "[TrilinosResistanceOperator] ERROR: boundary cap vectors must have one column.");
      }
      cap_vector = boundary_cap_vectors_[i]->getVector(0);
    }

    const double denominator = 1.0 + face.resistance * face.s_tilde_norm2;
    const double sign_resistance = face.resistance > 0.0 ? 1.0 : -1.0;
    const double scaled_alpha = -sign_resistance / denominator;

    for (size_t column = 0; column < X.getNumVectors(); ++column) {
      const auto x_column = X.getVector(column);
      auto y_column = Y.getVectorNonConst(column);
      const Scalar_d dot_face = x_column->dot(*face_vector);
      const Scalar_d dot_cap = cap_vector == Teuchos::null ? 0.0 :
        x_column->dot(*cap_vector);
      y_column->update(alpha * scaled_alpha * (dot_face + dot_cap),
          *face_vector, 1.0);
    }
  }
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
