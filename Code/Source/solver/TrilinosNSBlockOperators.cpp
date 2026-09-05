// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "TrilinosNSBlockOperators.h"

/**
 * @file TrilinosNSBlockOperators.cpp
 * @brief Implements Tpetra block extraction and matrix-free NS operators.
 */

#ifdef WITH_TRILINOS

#include "Tpetra_Export.hpp"

#include <stdexcept>

namespace trilinos_bipartition {

namespace {

int dof_component(GO gid, GO index_base, int dof)
{
  int component = static_cast<int>((gid - index_base) % dof);
  if (component < 0) {
    component += dof;
  }
  return component;
}

Teuchos::RCP<const Tpetra_Map> make_submap(
    const Teuchos::RCP<const Tpetra_Map>& full_map,
    const std::vector<GO>& gids)
{
  return Teuchos::rcp(new Tpetra_Map(
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
      Teuchos::ArrayView<const GO>(gids.data(), gids.size()),
      full_map->getIndexBase(),
      full_map->getComm()));
}

void insert_global_values(const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    GO row_gid,
    const Teuchos::Array<GO>& columns,
    const Teuchos::Array<Scalar_d>& values)
{
  if (!columns.empty()) {
    matrix->insertGlobalValues(row_gid, columns(), values());
  }
}

Teuchos::RCP<Tpetra_MultiVector> import_to_submap(
    const Tpetra_MultiVector& source,
    const Teuchos::RCP<const Tpetra_Map>& sub_map)
{
  if (sub_map == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosNSBlockOperators] ERROR: destination submap is null.");
  }

  auto target = Teuchos::rcp(
      new Tpetra_MultiVector(sub_map, source.getNumVectors()));
  Tpetra_Import importer(source.getMap(), sub_map);
  target->doImport(source, importer, Tpetra::INSERT);
  return target;
}

bool same_map(const Teuchos::RCP<const Tpetra_Map>& left,
    const Teuchos::RCP<const Tpetra_Map>& right)
{
  return left != Teuchos::null && right != Teuchos::null &&
    left->isSameAs(*right);
}

} // namespace

MomentumOperator::MomentumOperator(
    const Teuchos::RCP<Tpetra_CrsMatrix>& A,
    const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_vectors,
    const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& boundary_cap_vectors) :
  A_(A),
  boundary_vectors_(boundary_vectors),
  boundary_cap_vectors_(boundary_cap_vectors)
{
  if (A_ == Teuchos::null || !A_->isFillComplete()) {
    throw std::runtime_error(
        "[MomentumOperator] ERROR: A must be non-null and fill complete.");
  }

  const auto map = A_->getDomainMap();
  if (!same_map(map, A_->getRangeMap())) {
    throw std::runtime_error(
        "[MomentumOperator] ERROR: A must have identical domain and range maps.");
  }

  for (const auto& vector : boundary_vectors_) {
    if (vector != Teuchos::null &&
        (!same_map(vector->getMap(), map) || vector->getNumVectors() != 1)) {
      throw std::runtime_error(
          "[MomentumOperator] ERROR: boundary vectors must have one column on the velocity map.");
    }
  }
  for (const auto& vector : boundary_cap_vectors_) {
    if (vector != Teuchos::null &&
        (!same_map(vector->getMap(), map) || vector->getNumVectors() != 1)) {
      throw std::runtime_error(
          "[MomentumOperator] ERROR: boundary cap vectors must have one column on the velocity map.");
    }
  }
}

void MomentumOperator::apply(const Tpetra_MultiVector& X,
    Tpetra_MultiVector& Y, Teuchos::ETransp mode, Scalar_d alpha,
    Scalar_d beta) const
{
  if (mode != Teuchos::NO_TRANS && !boundary_vectors_.empty()) {
    throw std::runtime_error(
        "[MomentumOperator] ERROR: transpose apply is unsupported with coupled boundary vectors.");
  }
  if (X.getNumVectors() != Y.getNumVectors()) {
    throw std::runtime_error(
        "[MomentumOperator] ERROR: input and output column counts differ.");
  }

  A_->apply(X, Y, mode, alpha, beta);
  if (boundary_vectors_.empty()) {
    return;
  }

  for (size_t i = 0; i < boundary_vectors_.size(); ++i) {
    if (boundary_vectors_[i] == Teuchos::null) {
      continue;
    }

    const auto face = boundary_vectors_[i]->getVector(0);
    Teuchos::RCP<const Tpetra_Vector> cap = Teuchos::null;
    if (i < boundary_cap_vectors_.size() &&
        boundary_cap_vectors_[i] != Teuchos::null) {
      cap = boundary_cap_vectors_[i]->getVector(0);
    }

    for (size_t column = 0; column < X.getNumVectors(); ++column) {
      const auto x_column = X.getVector(column);
      auto y_column = Y.getVectorNonConst(column);
      const Scalar_d dot_face = x_column->dot(*face);
      const Scalar_d dot_cap = cap == Teuchos::null ? 0.0 :
        x_column->dot(*cap);
      y_column->update(alpha * (dot_face + dot_cap), *face, 1.0);
    }
  }
}

Teuchos::RCP<const Tpetra_Map> MomentumOperator::getDomainMap() const
{
  return A_->getDomainMap();
}

Teuchos::RCP<const Tpetra_Map> MomentumOperator::getRangeMap() const
{
  return A_->getRangeMap();
}

PressureSchurOperator::PressureSchurOperator(
    const Teuchos::RCP<Tpetra_CrsMatrix>& L,
    const Teuchos::RCP<Tpetra_CrsMatrix>& B,
    const Teuchos::RCP<Tpetra_Operator>& resistance_operator) :
  L_(L),
  B_(B),
  resistance_operator_(resistance_operator)
{
  if (L_ == Teuchos::null || B_ == Teuchos::null ||
      !L_->isFillComplete() || !B_->isFillComplete()) {
    throw std::runtime_error(
        "[PressureSchurOperator] ERROR: L and B must be non-null and fill complete.");
  }
  if (!same_map(L_->getDomainMap(), L_->getRangeMap()) ||
      !same_map(B_->getDomainMap(), L_->getDomainMap())) {
    throw std::runtime_error(
        "[PressureSchurOperator] ERROR: pressure maps of L and B are incompatible.");
  }
  if (resistance_operator_ != Teuchos::null &&
      (!same_map(resistance_operator_->getDomainMap(), B_->getRangeMap()) ||
       !same_map(resistance_operator_->getRangeMap(), B_->getRangeMap()))) {
    throw std::runtime_error(
        "[PressureSchurOperator] ERROR: resistance operator must act on the velocity map.");
  }
}

void PressureSchurOperator::apply(const Tpetra_MultiVector& X,
    Tpetra_MultiVector& Y, Teuchos::ETransp mode, Scalar_d alpha,
    Scalar_d beta) const
{
  if (mode != Teuchos::NO_TRANS) {
    throw std::runtime_error(
        "[PressureSchurOperator] ERROR: transpose apply is not supported.");
  }
  if (X.getNumVectors() != Y.getNumVectors()) {
    throw std::runtime_error(
        "[PressureSchurOperator] ERROR: input and output column counts differ.");
  }

  Tpetra_MultiVector result(getRangeMap(), X.getNumVectors());
  Tpetra_MultiVector velocity(B_->getRangeMap(), X.getNumVectors());
  Tpetra_MultiVector transformed_velocity(
      B_->getRangeMap(), X.getNumVectors());
  Tpetra_MultiVector pressure(getRangeMap(), X.getNumVectors());

  L_->apply(X, result);
  B_->apply(X, velocity);
  if (resistance_operator_ != Teuchos::null) {
    resistance_operator_->apply(velocity, transformed_velocity);
  } else {
    transformed_velocity.update(1.0, velocity, 0.0);
  }
  B_->apply(transformed_velocity, pressure, Teuchos::TRANS);
  result.update(1.0, pressure, 1.0);
  Y.update(alpha, result, beta);
}

Teuchos::RCP<const Tpetra_Map> PressureSchurOperator::getDomainMap() const
{
  return L_->getDomainMap();
}

Teuchos::RCP<const Tpetra_Map> PressureSchurOperator::getRangeMap() const
{
  return L_->getRangeMap();
}

TrilinosNSBlockSystem build_trilinos_ns_block_system(
    const Teuchos::RCP<Trilinos>& trilinos,
    int nsd,
    int dof)
{
  if (trilinos == Teuchos::null || trilinos->K == Teuchos::null ||
      !trilinos->K->isFillComplete()) {
    throw std::runtime_error(
        "[TrilinosNSBlockOperators] ERROR: full matrix must be non-null and fill complete.");
  }
  if (nsd <= 0 || dof <= nsd) {
    throw std::runtime_error(
        "[TrilinosNSBlockOperators] ERROR: require dof > nsd > 0.");
  }

  const auto full_map = trilinos->K->getRowMap();
  const auto local_gids = full_map->getLocalElementList();
  const GO index_base = full_map->getIndexBase();

  std::vector<GO> velocity_gids;
  std::vector<GO> pressure_gids;
  velocity_gids.reserve(local_gids.size());
  pressure_gids.reserve(local_gids.size());

  for (const GO gid : local_gids) {
    const int component = dof_component(gid, index_base, dof);
    if (component < nsd) {
      velocity_gids.push_back(gid);
    } else if (component == nsd) {
      pressure_gids.push_back(gid);
    }
  }

  TrilinosNSBlockSystem system;
  system.velocity_map = make_submap(full_map, velocity_gids);
  system.pressure_map = make_submap(full_map, pressure_gids);

  const size_t max_entries = static_cast<size_t>(
      trilinos->K->getGlobalMaxNumRowEntries());
  system.A = Teuchos::rcp(
      new Tpetra_CrsMatrix(system.velocity_map, max_entries));
  system.B = Teuchos::rcp(
      new Tpetra_CrsMatrix(system.velocity_map, max_entries));
  system.C = Teuchos::rcp(
      new Tpetra_CrsMatrix(system.pressure_map, max_entries));
  system.L = Teuchos::rcp(
      new Tpetra_CrsMatrix(system.pressure_map, max_entries));

  for (const GO row_gid : local_gids) {
    const int row_component = dof_component(row_gid, index_base, dof);
    const bool row_is_velocity = row_component < nsd;
    const bool row_is_pressure = row_component == nsd;
    if (!row_is_velocity && !row_is_pressure) {
      continue;
    }

    const size_t row_entries =
      trilinos->K->getNumEntriesInGlobalRow(row_gid);
    Tpetra_CrsMatrix::nonconst_global_inds_host_view_type columns(
        "svmp_ns_block_columns", row_entries);
    Tpetra_CrsMatrix::nonconst_values_host_view_type values(
        "svmp_ns_block_values", row_entries);
    size_t num_entries = 0;
    trilinos->K->getGlobalRowCopy(
        row_gid, columns, values, num_entries);

    Teuchos::Array<GO> A_columns;
    Teuchos::Array<Scalar_d> A_values;
    Teuchos::Array<GO> B_columns;
    Teuchos::Array<Scalar_d> B_values;
    Teuchos::Array<GO> C_columns;
    Teuchos::Array<Scalar_d> C_values;
    Teuchos::Array<GO> L_columns;
    Teuchos::Array<Scalar_d> L_values;

    for (size_t entry = 0; entry < num_entries; ++entry) {
      const GO column_gid = columns(entry);
      const int column_component =
        dof_component(column_gid, index_base, dof);
      const bool column_is_velocity = column_component < nsd;
      const bool column_is_pressure = column_component == nsd;

      if (row_is_velocity && column_is_velocity) {
        A_columns.push_back(column_gid);
        A_values.push_back(values(entry));
      } else if (row_is_velocity && column_is_pressure) {
        B_columns.push_back(column_gid);
        B_values.push_back(values(entry));
      } else if (row_is_pressure && column_is_velocity) {
        C_columns.push_back(column_gid);
        C_values.push_back(values(entry));
      } else if (row_is_pressure && column_is_pressure) {
        L_columns.push_back(column_gid);
        L_values.push_back(values(entry));
      }
    }

    if (row_is_velocity) {
      insert_global_values(system.A, row_gid, A_columns, A_values);
      insert_global_values(system.B, row_gid, B_columns, B_values);
    } else {
      insert_global_values(system.C, row_gid, C_columns, C_values);
      insert_global_values(system.L, row_gid, L_columns, L_values);
    }
  }

  system.A->fillComplete(system.velocity_map, system.velocity_map);
  system.B->fillComplete(system.pressure_map, system.velocity_map);
  system.C->fillComplete(system.velocity_map, system.pressure_map);
  system.L->fillComplete(system.pressure_map, system.pressure_map);

  system.boundary_vectors.reserve(trilinos->bdryVec_list.size());
  for (const auto& vector : trilinos->bdryVec_list) {
    system.boundary_vectors.push_back(
        vector == Teuchos::null ? Teuchos::null :
        import_to_submap(*vector, system.velocity_map));
  }

  system.boundary_cap_vectors.reserve(trilinos->bdryCapVec_list.size());
  for (const auto& vector : trilinos->bdryCapVec_list) {
    system.boundary_cap_vectors.push_back(
        vector == Teuchos::null ? Teuchos::null :
        import_to_submap(*vector, system.velocity_map));
  }

  return system;
}

Teuchos::RCP<Tpetra_MultiVector> extract_subvector(
    const Tpetra_MultiVector& source,
    const Teuchos::RCP<const Tpetra_Map>& sub_map)
{
  return import_to_submap(source, sub_map);
}

void scatter_subvectors_to_full_vector(
    const Tpetra_MultiVector& velocity,
    const Tpetra_MultiVector& pressure,
    const Teuchos::RCP<Tpetra_Vector>& full_vector)
{
  if (full_vector == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosNSBlockOperators] ERROR: full destination vector is null.");
  }
  if (velocity.getNumVectors() != 1 || pressure.getNumVectors() != 1) {
    throw std::runtime_error(
        "[TrilinosNSBlockOperators] ERROR: scatter requires one-column subvectors.");
  }

  full_vector->putScalar(0.0);
  Tpetra::Export<LO, GO, Node> velocity_exporter(
      velocity.getMap(), full_vector->getMap());
  Tpetra::Export<LO, GO, Node> pressure_exporter(
      pressure.getMap(), full_vector->getMap());
  full_vector->doExport(velocity, velocity_exporter, Tpetra::INSERT);
  full_vector->doExport(pressure, pressure_exporter, Tpetra::INSERT);
}

} // namespace trilinos_bipartition

#endif
