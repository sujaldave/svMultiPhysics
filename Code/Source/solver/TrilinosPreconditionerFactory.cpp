// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "TrilinosPreconditionerFactory.h"
#include "Profiling.h"

/**
 * @file TrilinosPreconditionerFactory.cpp
 * @brief Implements reusable Ifpack2, MueLu, and resistance preconditioners.
 */

#ifdef WITH_TRILINOS

#include "Teuchos_CommHelpers.hpp"

#include <stdexcept>

namespace trilinos_bipartition {
namespace preconditioners {

namespace {

std::string preconditioner_name(consts::PreconditionerType type)
{
  const auto entry = consts::preconditioner_type_to_name.find(type);
  return entry == consts::preconditioner_type_to_name.end() ?
    std::to_string(static_cast<int>(type)) : entry->second;
}

void validate_type(consts::PreconditionerType type)
{
  if (consts::trilinos_preconditioners.count(type) == 0) {
    throw std::runtime_error(
        "[TrilinosPreconditionerFactory] ERROR: '" +
        preconditioner_name(type) +
        "' is not a registered Trilinos preconditioner.");
  }
}

void validate_matrix(
    consts::PreconditionerType type,
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix)
{
  if (matrix == Teuchos::null || !matrix->isFillComplete()) {
    throw std::runtime_error(
        "[TrilinosPreconditionerFactory] ERROR: '" +
        preconditioner_name(type) +
        "' requires a non-null, fill-complete Tpetra::CrsMatrix.");
  }
  if (!matrix->getDomainMap()->isSameAs(*matrix->getRangeMap())) {
    throw std::runtime_error(
        "[TrilinosPreconditionerFactory] ERROR: preconditioner matrix must be square.");
  }
}

bool has_zero_diagonal(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix)
{
  Tpetra_Vector diagonal(matrix->getRowMap());
  matrix->getLocalDiagCopy(diagonal);

  int local_has_zero = 0;
  {
    const auto values =
      diagonal.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t row = 0; row < values.extent(0); ++row) {
      if (values(row, 0) == 0.0) {
        local_has_zero = 1;
        break;
      }
    }
  }

  int global_has_zero = 0;
  Teuchos::reduceAll(*matrix->getRowMap()->getComm(), Teuchos::REDUCE_MAX,
      1, &local_has_zero, &global_has_zero);
  return global_has_zero != 0;
}

Teuchos::RCP<Tpetra_CrsMatrix> copy_with_repaired_diagonal(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix)
{
  if (!has_zero_diagonal(matrix)) {
    return matrix;
  }

  const size_t allocation = static_cast<size_t>(
      matrix->getGlobalMaxNumRowEntries()) + 1;
  auto repaired = Teuchos::rcp(
      new Tpetra_CrsMatrix(matrix->getRowMap(), allocation));

  const auto local_rows = matrix->getRowMap()->getLocalElementList();
  for (const GO row_gid : local_rows) {
    const size_t row_entries =
      matrix->getNumEntriesInGlobalRow(row_gid);
    Tpetra_CrsMatrix::nonconst_global_inds_host_view_type columns(
        "svmp_prec_columns", row_entries);
    Tpetra_CrsMatrix::nonconst_values_host_view_type values(
        "svmp_prec_values", row_entries);
    size_t count = 0;
    matrix->getGlobalRowCopy(row_gid, columns, values, count);

    Teuchos::Array<GO> copied_columns;
    Teuchos::Array<Scalar_d> copied_values;
    copied_columns.reserve(count + 1);
    copied_values.reserve(count + 1);
    bool found_diagonal = false;

    for (size_t entry = 0; entry < count; ++entry) {
      copied_columns.push_back(columns(entry));
      Scalar_d value = values(entry);
      if (columns(entry) == row_gid) {
        found_diagonal = true;
        if (value == 0.0) {
          value = 1.0;
        }
      }
      copied_values.push_back(value);
    }

    if (!found_diagonal) {
      copied_columns.push_back(row_gid);
      copied_values.push_back(1.0);
    }
    repaired->insertGlobalValues(
        row_gid, copied_columns(), copied_values());
  }

  repaired->fillComplete(
      matrix->getDomainMap(), matrix->getRangeMap());
  return repaired;
}

Teuchos::RCP<Tpetra_Operator> create_muelu_preconditioner(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    SolverRole role)
{
  Teuchos::ParameterList parameters;
  parameters.set("verbosity", "none");
  parameters.set("max levels", 6);
  parameters.set("cycle type", "V");
  parameters.set("fuse prolongation and update", true);
  parameters.set("problem: type", "unknown");
  parameters.set("reuse: type", "RAP");

  // Filtered BIPN submaps retain full-system GIDs and are not node-contiguous.
  // Scalar coalescing is therefore the conservative choice for both blocks.
  parameters.set("number of equations", 1);
  parameters.set("aggregation: type", "uncoupled");
  parameters.set("aggregation: min agg size", 2);
  parameters.set("aggregation: max agg size", 8);
  parameters.set("aggregation: ordering", "natural");
#if !defined(KOKKOS_ENABLE_CUDA)
  parameters.set("aggregation: drop scheme", "classical");
  parameters.set(
      "aggregation: strength-of-connection: measure",
      "smoothed aggregation");
#endif
  parameters.set("aggregation: number of random vectors", 5);
  parameters.set(
      "aggregation: number of times to pre or post smooth", 3);
  parameters.set("sa: damping factor", 1.0);
  parameters.set("sa: use filtered matrix", false);
  parameters.set("smoother: type", "RELAXATION");
  parameters.set("smoother: pre or post", "both");
  parameters.set("smoother: overlap", 0);

  auto& smoother = parameters.sublist("smoother: params");
  smoother.set("relaxation: type",
      role == SolverRole::pressure_cg ?
      "Symmetric Gauss-Seidel" : "Gauss-Seidel");
  smoother.set("relaxation: sweeps", 2);

  parameters.set("coarse: type", "KLU");
  parameters.set("coarse: max size", 2000);

  return MueLu::CreateTpetraPreconditioner(
      Teuchos::rcp_static_cast<Tpetra_Operator>(matrix), parameters);
}

} // namespace

bool MueLuReuseCache::matches(
    SolverRole role,
    const Tpetra_CrsMatrix& matrix,
    const PreconditionerReuseContext& reuse) const
{
  return initialized_ && hierarchy_ != Teuchos::null && role_ == role &&
      time_step_ == reuse.time_step &&
      topology_generation_ == reuse.topology_generation &&
      domain_map_ != Teuchos::null && range_map_ != Teuchos::null &&
      matrix.getDomainMap()->isSameAs(*domain_map_) &&
      matrix.getRangeMap()->isSameAs(*range_map_) &&
      matrix.getGlobalNumRows() == global_rows_ &&
      matrix.getGlobalNumCols() == global_columns_ &&
      matrix.getGlobalNumEntries() == global_entries_ &&
      matrix.getLocalNumRows() == local_rows_ &&
      matrix.getLocalNumEntries() == local_entries_;
}

void MueLuReuseCache::store_signature(
    SolverRole role,
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    const PreconditionerReuseContext& reuse)
{
  initialized_ = true;
  role_ = role;
  time_step_ = reuse.time_step;
  topology_generation_ = reuse.topology_generation;
  global_rows_ = matrix->getGlobalNumRows();
  global_columns_ = matrix->getGlobalNumCols();
  global_entries_ = matrix->getGlobalNumEntries();
  local_rows_ = matrix->getLocalNumRows();
  local_entries_ = matrix->getLocalNumEntries();
  domain_map_ = matrix->getDomainMap();
  range_map_ = matrix->getRangeMap();
}

void MueLuReuseCache::clear()
{
  initialized_ = false;
  role_ = SolverRole::momentum_gmres;
  time_step_ = -1;
  topology_generation_ = 0;
  global_rows_ = 0;
  global_columns_ = 0;
  global_entries_ = 0;
  local_rows_ = 0;
  local_entries_ = 0;
  build_count_ = 0;
  reuse_count_ = 0;
  domain_map_ = Teuchos::null;
  range_map_ = Teuchos::null;
  hierarchy_ = Teuchos::null;
}

const Teuchos::RCP<Tpetra_Operator>& MueLuReuseCache::hierarchy() const
{
  return hierarchy_;
}

int MueLuReuseCache::time_step() const
{
  return time_step_;
}

std::size_t MueLuReuseCache::topology_generation() const
{
  return topology_generation_;
}

std::size_t MueLuReuseCache::build_count() const
{
  return build_count_;
}

std::size_t MueLuReuseCache::reuse_count() const
{
  return reuse_count_;
}

PreconditionerHandle::PreconditionerHandle(
    consts::PreconditionerType type,
    SolverRole role) :
  type_(type),
  role_(role)
{
}

consts::PreconditionerType PreconditionerHandle::type() const
{
  return type_;
}

SolverRole PreconditionerHandle::role() const
{
  return role_;
}

const Teuchos::RCP<Tpetra_Operator>&
PreconditionerHandle::left_operator() const
{
  return left_operator_;
}

const Teuchos::RCP<Tpetra_CrsMatrix>&
PreconditionerHandle::setup_matrix() const
{
  return setup_matrix_;
}

const Teuchos::RCP<Ifpack2_Preconditioner>&
PreconditionerHandle::ifpack_operator() const
{
  return ifpack_operator_;
}

const Teuchos::RCP<Tpetra_Operator>&
PreconditionerHandle::muelu_operator() const
{
  return muelu_operator_;
}

const Teuchos::RCP<Tpetra_Operator>&
PreconditionerHandle::resistance_operator() const
{
  return resistance_operator_;
}

Teuchos::RCP<PreconditionerHandle> create_preconditioner(
    consts::PreconditionerType type,
    SolverRole role,
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    const Teuchos::RCP<Tpetra_Operator>& resistance_operator,
    const PreconditionerReuseContext& reuse)
{
  validate_type(type);
  auto handle = Teuchos::rcp(
      new PreconditionerHandle(type, role));

  if (type ==
      consts::PreconditionerType::PREC_TRILINOS_DIAGONAL) {
    return handle;
  }

  if (type ==
      consts::PreconditionerType::PREC_TRILINOS_RESISTANCE) {
    if (role != SolverRole::momentum_gmres) {
      throw std::runtime_error(
          "[TrilinosPreconditionerFactory] ERROR: "
          "'trilinos-resistance' is valid only for momentum GMRES.");
    }
    if (resistance_operator == Teuchos::null) {
      throw std::runtime_error(
          "[TrilinosPreconditionerFactory] ERROR: "
          "'trilinos-resistance' requires a velocity-space resistance operator.");
    }
    validate_matrix(type, matrix);
    if (!resistance_operator->getDomainMap()->isSameAs(
            *matrix->getDomainMap()) ||
        !resistance_operator->getRangeMap()->isSameAs(
            *matrix->getRangeMap())) {
      throw std::runtime_error(
          "[TrilinosPreconditionerFactory] ERROR: resistance operator maps "
          "must match the momentum matrix.");
    }
    handle->setup_matrix_ = matrix;
    handle->resistance_operator_ = resistance_operator;
    handle->left_operator_ = resistance_operator;
    return handle;
  }

  validate_matrix(type, matrix);
  handle->setup_matrix_ =
    copy_with_repaired_diagonal(matrix);

  if (type == consts::PreconditionerType::PREC_TRILINOS_ML) {
    svmp_profiling::ProfilingScope profiling_scope(
        svmp_profiling::stages::MueLuSetup);
    if (reuse.muelu_cache != Teuchos::null) {
      if (reuse.time_step < 0 || reuse.topology_generation == 0) {
        throw std::runtime_error(
            "[TrilinosPreconditionerFactory] ERROR: MueLu reuse requires "
            "a valid time step and topology generation.");
      }

      auto& cache = *reuse.muelu_cache;
      // Refresh only within one time step and for an unchanged block layout.
      if (cache.matches(role, *handle->setup_matrix_, reuse)) {
        auto muelu = Teuchos::rcp_dynamic_cast<
            MueLu::TpetraOperator<Scalar_d, LO, GO, Node>>(
                cache.hierarchy_);
        if (muelu != Teuchos::null) {
          MueLu::ReuseTpetraPreconditioner(handle->setup_matrix_, *muelu);
          ++cache.reuse_count_;
        } else {
          cache.hierarchy_ = Teuchos::null;
        }
      }

      if (cache.hierarchy_ == Teuchos::null ||
          !cache.matches(role, *handle->setup_matrix_, reuse)) {
        // RAP applies only after the first full build in this time step.
        cache.hierarchy_ = Teuchos::null;
        cache.hierarchy_ =
          create_muelu_preconditioner(handle->setup_matrix_, role);
        cache.store_signature(role, handle->setup_matrix_, reuse);
        ++cache.build_count_;
      }
      handle->muelu_operator_ = cache.hierarchy_;
    } else {
      handle->muelu_operator_ =
        create_muelu_preconditioner(handle->setup_matrix_, role);
    }
    handle->left_operator_ = handle->muelu_operator_;
    return handle;
  }

  Ifpack2::Factory factory;
  Teuchos::ParameterList parameters;

  if (type ==
      consts::PreconditionerType::PREC_TRILINOS_BLOCK_JACOBI) {
    handle->ifpack_operator_ =
      factory.create<Tpetra_CrsMatrix>(
          "RELAXATION", handle->setup_matrix_);
    parameters.set("relaxation: type", "Jacobi");
    parameters.set("relaxation: sweeps", 1);
  } else {
    handle->ifpack_operator_ =
      factory.create<Tpetra_CrsMatrix>(
          "SCHWARZ", handle->setup_matrix_);
    parameters.set("schwarz: combine mode", "Add");

    if (type ==
        consts::PreconditionerType::PREC_TRILINOS_ILU) {
      parameters.set("schwarz: inner preconditioner name", "ILUT");
      parameters.set("schwarz: overlap level", 1);
      parameters.set("fact: level-of-fill", 0);
      parameters.set("fact: relax value", 0.0);
    } else if (type ==
        consts::PreconditionerType::PREC_TRILINOS_ILUT) {
      parameters.set("schwarz: inner preconditioner name", "ILUT");
      parameters.set("schwarz: overlap level", 1);
      parameters.set("fact: ilut level-of-fill", 2.0);
      parameters.set("fact: drop tolerance", 1e-2);
      parameters.set("fact: relax value", 0.0);
    } else if (type ==
        consts::PreconditionerType::PREC_TRILINOS_RILUK0) {
      parameters.set("schwarz: inner preconditioner name", "RILUK");
      parameters.set("fact: level-of-fill", 0);
      parameters.set("fact: drop tolerance", 0.0);
      parameters.set("fact: relax value", 0.0);
    } else if (type ==
        consts::PreconditionerType::PREC_TRILINOS_RILUK1) {
      parameters.set("schwarz: inner preconditioner name", "RILUK");
      parameters.set("fact: level-of-fill", 1);
      parameters.set("fact: drop tolerance", 1e-3);
      parameters.set("fact: relax value", 0.0);
    } else {
      throw std::runtime_error(
          "[TrilinosPreconditionerFactory] ERROR: unsupported type '" +
          preconditioner_name(type) + "'.");
    }
  }

  handle->ifpack_operator_->setParameters(parameters);
  handle->ifpack_operator_->initialize();
  handle->ifpack_operator_->compute();
  handle->left_operator_ = handle->ifpack_operator_;
  return handle;
}

void attach_preconditioner(
    const Teuchos::RCP<PreconditionerHandle>& preconditioner,
    const Teuchos::RCP<Belos_LinearProblem>& belos_problem)
{
  if (belos_problem == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosPreconditionerFactory] ERROR: Belos problem is null.");
  }
  belos_problem->setLeftPrec(
      preconditioner == Teuchos::null ?
      Teuchos::null : preconditioner->left_operator());
}

} // namespace preconditioners
} // namespace trilinos_bipartition

#endif
