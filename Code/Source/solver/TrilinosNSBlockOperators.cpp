// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "TrilinosNSBlockOperators.h"

/**
 * @file TrilinosNSBlockOperators.cpp
 * @brief Implements Tpetra block extraction and matrix-free NS operators.
 */

#ifdef WITH_TRILINOS

#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_Export.hpp"

#include <mutex>
#include <stdexcept>

namespace trilinos_bipartition {

namespace {

using NSLocalMatrixDevice = Tpetra_CrsMatrix::local_matrix_device_type;
using NSOffsetView = Kokkos::View<
    std::size_t*, typename NSLocalMatrixDevice::device_type>;

/**
 * @brief Copy selected CRS values between device-local Tpetra matrices.
 *
 * A named namespace-scope functor avoids CUDA's restriction on extended
 * host/device lambdas declared inside private nested implementation classes.
 */
struct CopyCrsValuesFunctor
{
  NSLocalMatrixDevice source_matrix;
  NSLocalMatrixDevice destination_matrix;
  NSOffsetView source_offsets;
  NSOffsetView destination_offsets;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const
  {
    destination_matrix.values(destination_offsets(i)) =
        source_matrix.values(source_offsets(i));
  }
};

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

class TrilinosNSBlockTopologyCache::Implementation
{
  public:
    using offset_view_type = NSOffsetView;

    struct ValuePlan
    {
      offset_view_type source_offsets;
      offset_view_type destination_offsets;
    };

    bool initialized = false;
    std::size_t generation = 0;
    std::size_t full_topology_generation = 0;
    int nsd = 0;
    int dof = 0;
    Teuchos::RCP<const Tpetra_Map> full_map;
    Teuchos::RCP<const Tpetra_Map> velocity_map;
    Teuchos::RCP<const Tpetra_Map> pressure_map;
    Teuchos::RCP<Tpetra_CrsGraph> A_graph;
    Teuchos::RCP<Tpetra_CrsGraph> B_graph;
    Teuchos::RCP<Tpetra_CrsGraph> C_graph;
    Teuchos::RCP<Tpetra_CrsGraph> L_graph;
    Teuchos::RCP<Tpetra_Import> velocity_importer;
    Teuchos::RCP<Tpetra_Import> pressure_importer;
    Teuchos::RCP<Tpetra_Export> velocity_exporter;
    Teuchos::RCP<Tpetra_Export> pressure_exporter;
    ValuePlan A_plan;
    ValuePlan B_plan;
    ValuePlan C_plan;
    ValuePlan L_plan;

    bool matches(const Tpetra_CrsMatrix& matrix,
        std::size_t topology_generation, int new_nsd, int new_dof) const
    {
      return initialized &&
        full_topology_generation == topology_generation &&
        nsd == new_nsd && dof == new_dof &&
        same_map(full_map, matrix.getRowMap());
    }

    static ValuePlan make_value_plan(
        const std::vector<std::size_t>& source,
        const std::vector<std::size_t>& destination,
        const char* label)
    {
      if (source.size() != destination.size()) {
        throw std::runtime_error(
            "[TrilinosNSBlockTopologyCache] ERROR: invalid value-copy plan.");
      }

      ValuePlan plan;
      plan.source_offsets = offset_view_type(
          std::string(label) + "_source", source.size());
      plan.destination_offsets = offset_view_type(
          std::string(label) + "_destination", destination.size());
      auto source_host = Kokkos::create_mirror_view(plan.source_offsets);
      auto destination_host =
          Kokkos::create_mirror_view(plan.destination_offsets);
      for (std::size_t i = 0; i < source.size(); ++i) {
        source_host(i) = source[i];
        destination_host(i) = destination[i];
      }
      Kokkos::deep_copy(plan.source_offsets, source_host);
      Kokkos::deep_copy(plan.destination_offsets, destination_host);
      return plan;
    }

    static std::size_t find_block_offset(
        const Tpetra_CrsGraph& graph, GO row_gid, GO column_gid)
    {
      const LO row_lid = graph.getRowMap()->getLocalElement(row_gid);
      const LO column_lid = graph.getColMap()->getLocalElement(column_gid);
      if (row_lid == Teuchos::OrdinalTraits<LO>::invalid() ||
          column_lid == Teuchos::OrdinalTraits<LO>::invalid()) {
        throw std::runtime_error(
            "[TrilinosNSBlockTopologyCache] ERROR: block graph map lookup failed.");
      }

      const auto local_graph = graph.getLocalGraphHost();
      const auto begin = local_graph.row_map(row_lid);
      const auto end = local_graph.row_map(row_lid + 1);
      for (std::size_t entry = begin; entry < end; ++entry) {
        if (local_graph.entries(entry) == column_lid) {
          return entry;
        }
      }
      throw std::runtime_error(
          "[TrilinosNSBlockTopologyCache] ERROR: block graph entry lookup failed.");
    }

    static void refresh_values(
        const Tpetra_CrsMatrix& source,
        Tpetra_CrsMatrix& destination,
        const ValuePlan& plan,
        const char* label)
    {
      const auto source_matrix = source.getLocalMatrixDevice();
      auto destination_matrix = destination.getLocalMatrixDevice();
      const auto source_offsets = plan.source_offsets;
      const auto destination_offsets = plan.destination_offsets;
      const std::size_t count = source_offsets.extent(0);
      Kokkos::parallel_for(
          label,
          Kokkos::RangePolicy<typename Node::execution_space>(0, count),
          CopyCrsValuesFunctor{
              source_matrix,
              destination_matrix,
              source_offsets,
              destination_offsets});
    }

    void reset()
    {
      initialized = false;
      full_topology_generation = 0;
      nsd = 0;
      dof = 0;
      full_map = Teuchos::null;
      velocity_map = Teuchos::null;
      pressure_map = Teuchos::null;
      A_graph = Teuchos::null;
      B_graph = Teuchos::null;
      C_graph = Teuchos::null;
      L_graph = Teuchos::null;
      velocity_importer = Teuchos::null;
      pressure_importer = Teuchos::null;
      velocity_exporter = Teuchos::null;
      pressure_exporter = Teuchos::null;
      A_plan = ValuePlan{};
      B_plan = ValuePlan{};
      C_plan = ValuePlan{};
      L_plan = ValuePlan{};
    }
};

TrilinosNSBlockTopologyCache::TrilinosNSBlockTopologyCache() :
  implementation_(new Implementation())
{
}

TrilinosNSBlockTopologyCache::~TrilinosNSBlockTopologyCache() = default;

bool TrilinosNSBlockTopologyCache::ensure(
    const Tpetra_CrsMatrix& full_matrix,
    std::size_t full_topology_generation,
    int nsd,
    int dof)
{
  if (!full_matrix.isFillComplete()) {
    throw std::runtime_error(
        "[TrilinosNSBlockTopologyCache] ERROR: full matrix must be fill complete.");
  }
  if (nsd <= 0 || dof <= nsd) {
    throw std::runtime_error(
        "[TrilinosNSBlockTopologyCache] ERROR: require dof > nsd > 0.");
  }
  if (implementation_->matches(
          full_matrix, full_topology_generation, nsd, dof)) {
    return false;
  }

  implementation_->reset();
  auto& cache = *implementation_;
  cache.full_topology_generation = full_topology_generation;
  cache.nsd = nsd;
  cache.dof = dof;
  cache.full_map = full_matrix.getRowMap();
  const auto local_gids = cache.full_map->getLocalElementList();
  const GO index_base = cache.full_map->getIndexBase();

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

  cache.velocity_map = make_submap(cache.full_map, velocity_gids);
  cache.pressure_map = make_submap(cache.full_map, pressure_gids);
  const std::size_t max_entries = static_cast<std::size_t>(
      full_matrix.getGlobalMaxNumRowEntries());
  cache.A_graph = Teuchos::rcp(
      new Tpetra_CrsGraph(cache.velocity_map, max_entries));
  cache.B_graph = Teuchos::rcp(
      new Tpetra_CrsGraph(cache.velocity_map, max_entries));
  cache.C_graph = Teuchos::rcp(
      new Tpetra_CrsGraph(cache.pressure_map, max_entries));
  cache.L_graph = Teuchos::rcp(
      new Tpetra_CrsGraph(cache.pressure_map, max_entries));

  const auto full_local_graph =
      full_matrix.getCrsGraph()->getLocalGraphHost();
  const auto full_column_map = full_matrix.getColMap();
  for (std::size_t local_row = 0;
       local_row < full_matrix.getLocalNumRows(); ++local_row) {
    const GO row_gid = cache.full_map->getGlobalElement(
        static_cast<LO>(local_row));
    const int row_component = dof_component(row_gid, index_base, dof);
    const bool row_is_velocity = row_component < nsd;
    const bool row_is_pressure = row_component == nsd;
    if (!row_is_velocity && !row_is_pressure) {
      continue;
    }

    Teuchos::Array<GO> velocity_columns;
    Teuchos::Array<GO> pressure_columns;
    const std::size_t begin = full_local_graph.row_map(local_row);
    const std::size_t end = full_local_graph.row_map(local_row + 1);
    for (std::size_t entry = begin; entry < end; ++entry) {
      const GO column_gid = full_column_map->getGlobalElement(
          full_local_graph.entries(entry));
      const int component = dof_component(column_gid, index_base, dof);
      if (component < nsd) {
        velocity_columns.push_back(column_gid);
      } else if (component == nsd) {
        pressure_columns.push_back(column_gid);
      }
    }
    if (row_is_velocity) {
      if (!velocity_columns.empty()) {
        cache.A_graph->insertGlobalIndices(row_gid, velocity_columns());
      }
      if (!pressure_columns.empty()) {
        cache.B_graph->insertGlobalIndices(row_gid, pressure_columns());
      }
    } else {
      if (!velocity_columns.empty()) {
        cache.C_graph->insertGlobalIndices(row_gid, velocity_columns());
      }
      if (!pressure_columns.empty()) {
        cache.L_graph->insertGlobalIndices(row_gid, pressure_columns());
      }
    }
  }

  cache.A_graph->fillComplete(cache.velocity_map, cache.velocity_map);
  cache.B_graph->fillComplete(cache.pressure_map, cache.velocity_map);
  cache.C_graph->fillComplete(cache.velocity_map, cache.pressure_map);
  cache.L_graph->fillComplete(cache.pressure_map, cache.pressure_map);

  cache.velocity_importer = Teuchos::rcp(
      new Tpetra_Import(cache.full_map, cache.velocity_map));
  cache.pressure_importer = Teuchos::rcp(
      new Tpetra_Import(cache.full_map, cache.pressure_map));
  cache.velocity_exporter = Teuchos::rcp(
      new Tpetra_Export(cache.velocity_map, cache.full_map));
  cache.pressure_exporter = Teuchos::rcp(
      new Tpetra_Export(cache.pressure_map, cache.full_map));

  std::vector<std::size_t> A_source, A_destination;
  std::vector<std::size_t> B_source, B_destination;
  std::vector<std::size_t> C_source, C_destination;
  std::vector<std::size_t> L_source, L_destination;
  for (std::size_t local_row = 0;
       local_row < full_matrix.getLocalNumRows(); ++local_row) {
    const GO row_gid = cache.full_map->getGlobalElement(
        static_cast<LO>(local_row));
    const int row_component = dof_component(row_gid, index_base, dof);
    const bool row_is_velocity = row_component < nsd;
    const bool row_is_pressure = row_component == nsd;
    if (!row_is_velocity && !row_is_pressure) {
      continue;
    }
    const std::size_t begin = full_local_graph.row_map(local_row);
    const std::size_t end = full_local_graph.row_map(local_row + 1);
    for (std::size_t entry = begin; entry < end; ++entry) {
      const GO column_gid = full_column_map->getGlobalElement(
          full_local_graph.entries(entry));
      const int column_component =
          dof_component(column_gid, index_base, dof);
      const bool column_is_velocity = column_component < nsd;
      const bool column_is_pressure = column_component == nsd;
      if (row_is_velocity && column_is_velocity) {
        A_source.push_back(entry);
        A_destination.push_back(Implementation::find_block_offset(
            *cache.A_graph, row_gid, column_gid));
      } else if (row_is_velocity && column_is_pressure) {
        B_source.push_back(entry);
        B_destination.push_back(Implementation::find_block_offset(
            *cache.B_graph, row_gid, column_gid));
      } else if (row_is_pressure && column_is_velocity) {
        C_source.push_back(entry);
        C_destination.push_back(Implementation::find_block_offset(
            *cache.C_graph, row_gid, column_gid));
      } else if (row_is_pressure && column_is_pressure) {
        L_source.push_back(entry);
        L_destination.push_back(Implementation::find_block_offset(
            *cache.L_graph, row_gid, column_gid));
      }
    }
  }

  cache.A_plan = Implementation::make_value_plan(
      A_source, A_destination, "svmp_ns_A_plan");
  cache.B_plan = Implementation::make_value_plan(
      B_source, B_destination, "svmp_ns_B_plan");
  cache.C_plan = Implementation::make_value_plan(
      C_source, C_destination, "svmp_ns_C_plan");
  cache.L_plan = Implementation::make_value_plan(
      L_source, L_destination, "svmp_ns_L_plan");
  cache.initialized = true;
  ++cache.generation;
  return true;
}

bool TrilinosNSBlockTopologyCache::initialized() const
{
  return implementation_->initialized;
}

std::size_t TrilinosNSBlockTopologyCache::generation() const
{
  return implementation_->generation;
}

const Teuchos::RCP<const Tpetra_Map>&
TrilinosNSBlockTopologyCache::velocity_map() const
{
  return implementation_->velocity_map;
}

const Teuchos::RCP<const Tpetra_Map>&
TrilinosNSBlockTopologyCache::pressure_map() const
{
  return implementation_->pressure_map;
}

TrilinosNSBlockSystem TrilinosNSBlockTopologyCache::create_system(
    const Teuchos::RCP<Trilinos>& trilinos) const
{
  if (!implementation_->initialized || trilinos == Teuchos::null ||
      trilinos->K == Teuchos::null ||
      !same_map(implementation_->full_map, trilinos->K->getRowMap())) {
    throw std::runtime_error(
        "[TrilinosNSBlockTopologyCache] ERROR: cache and Trilinos state are incompatible.");
  }

  const auto& cache = *implementation_;
  TrilinosNSBlockSystem system;
  system.velocity_map = cache.velocity_map;
  system.pressure_map = cache.pressure_map;
  system.A = Teuchos::rcp(new Tpetra_CrsMatrix(cache.A_graph));
  system.B = Teuchos::rcp(new Tpetra_CrsMatrix(cache.B_graph));
  system.C = Teuchos::rcp(new Tpetra_CrsMatrix(cache.C_graph));
  system.L = Teuchos::rcp(new Tpetra_CrsMatrix(cache.L_graph));
  system.A->fillComplete(cache.velocity_map, cache.velocity_map);
  system.B->fillComplete(cache.pressure_map, cache.velocity_map);
  system.C->fillComplete(cache.velocity_map, cache.pressure_map);
  system.L->fillComplete(cache.pressure_map, cache.pressure_map);
  Implementation::refresh_values(
      *trilinos->K, *system.A, cache.A_plan, "svmp_ns_refresh_A");
  Implementation::refresh_values(
      *trilinos->K, *system.B, cache.B_plan, "svmp_ns_refresh_B");
  Implementation::refresh_values(
      *trilinos->K, *system.C, cache.C_plan, "svmp_ns_refresh_C");
  Implementation::refresh_values(
      *trilinos->K, *system.L, cache.L_plan, "svmp_ns_refresh_L");
  Kokkos::fence("svmp_ns_block_value_refresh");

  system.boundary_vectors.reserve(trilinos->bdryVec_list.size());
  for (const auto& vector : trilinos->bdryVec_list) {
    system.boundary_vectors.push_back(
        vector == Teuchos::null ? Teuchos::null : extract_velocity(*vector));
  }
  system.boundary_cap_vectors.reserve(trilinos->bdryCapVec_list.size());
  for (const auto& vector : trilinos->bdryCapVec_list) {
    system.boundary_cap_vectors.push_back(
        vector == Teuchos::null ? Teuchos::null : extract_velocity(*vector));
  }
  return system;
}

Teuchos::RCP<Tpetra_MultiVector>
TrilinosNSBlockTopologyCache::extract_velocity(
    const Tpetra_MultiVector& source) const
{
  if (!implementation_->initialized ||
      !same_map(source.getMap(), implementation_->full_map)) {
    throw std::runtime_error(
        "[TrilinosNSBlockTopologyCache] ERROR: velocity import map mismatch.");
  }
  auto target = Teuchos::rcp(new Tpetra_MultiVector(
      implementation_->velocity_map, source.getNumVectors()));
  target->doImport(
      source, *implementation_->velocity_importer, Tpetra::INSERT);
  return target;
}

Teuchos::RCP<Tpetra_MultiVector>
TrilinosNSBlockTopologyCache::extract_pressure(
    const Tpetra_MultiVector& source) const
{
  if (!implementation_->initialized ||
      !same_map(source.getMap(), implementation_->full_map)) {
    throw std::runtime_error(
        "[TrilinosNSBlockTopologyCache] ERROR: pressure import map mismatch.");
  }
  auto target = Teuchos::rcp(new Tpetra_MultiVector(
      implementation_->pressure_map, source.getNumVectors()));
  target->doImport(
      source, *implementation_->pressure_importer, Tpetra::INSERT);
  return target;
}

void TrilinosNSBlockTopologyCache::scatter(
    const Tpetra_MultiVector& velocity,
    const Tpetra_MultiVector& pressure,
    const Teuchos::RCP<Tpetra_Vector>& full_vector) const
{
  if (!implementation_->initialized || full_vector == Teuchos::null ||
      !same_map(full_vector->getMap(), implementation_->full_map) ||
      !same_map(velocity.getMap(), implementation_->velocity_map) ||
      !same_map(pressure.getMap(), implementation_->pressure_map) ||
      velocity.getNumVectors() != 1 || pressure.getNumVectors() != 1) {
    throw std::runtime_error(
        "[TrilinosNSBlockTopologyCache] ERROR: scatter map or column mismatch.");
  }
  full_vector->putScalar(0.0);
  full_vector->doExport(
      velocity, *implementation_->velocity_exporter, Tpetra::INSERT);
  full_vector->doExport(
      pressure, *implementation_->pressure_exporter, Tpetra::INSERT);
}

void TrilinosNSBlockTopologyCache::clear()
{
  implementation_->reset();
}

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

class PressureSchurOperator::Workspace
{
  public:
    void ensure(const Teuchos::RCP<const Tpetra_Map>& pressure_map,
        const Teuchos::RCP<const Tpetra_Map>& velocity_map,
        std::size_t num_vectors)
    {
      if (result != Teuchos::null &&
          result->getNumVectors() == num_vectors) {
        return;
      }
      result = Teuchos::rcp(
          new Tpetra_MultiVector(pressure_map, num_vectors));
      velocity = Teuchos::rcp(
          new Tpetra_MultiVector(velocity_map, num_vectors));
      transformed_velocity = Teuchos::rcp(
          new Tpetra_MultiVector(velocity_map, num_vectors));
      pressure = Teuchos::rcp(
          new Tpetra_MultiVector(pressure_map, num_vectors));
    }

    std::mutex mutex;
    Teuchos::RCP<Tpetra_MultiVector> result;
    Teuchos::RCP<Tpetra_MultiVector> velocity;
    Teuchos::RCP<Tpetra_MultiVector> transformed_velocity;
    Teuchos::RCP<Tpetra_MultiVector> pressure;
};

PressureSchurOperator::PressureSchurOperator(
    const Teuchos::RCP<Tpetra_CrsMatrix>& L,
    const Teuchos::RCP<Tpetra_CrsMatrix>& B,
    const Teuchos::RCP<Tpetra_Operator>& resistance_operator) :
  L_(L),
  B_(B),
  resistance_operator_(resistance_operator),
  workspace_(new Workspace())
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

PressureSchurOperator::~PressureSchurOperator() = default;

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

  std::lock_guard<std::mutex> lock(workspace_->mutex);
  workspace_->ensure(
      getRangeMap(), B_->getRangeMap(), X.getNumVectors());
  auto& result = *workspace_->result;
  auto& velocity = *workspace_->velocity;
  auto& transformed_velocity = *workspace_->transformed_velocity;
  auto& pressure = *workspace_->pressure;

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
  TrilinosNSBlockTopologyCache cache;
  cache.ensure(*trilinos->K, 1, nsd, dof);
  return cache.create_system(trilinos);
}

TrilinosNSBlockSystem build_trilinos_ns_block_system(
    const Teuchos::RCP<Trilinos>& trilinos,
    int nsd,
    int dof,
    std::size_t full_topology_generation,
    TrilinosNSBlockTopologyCache& cache)
{
  if (trilinos == Teuchos::null || trilinos->K == Teuchos::null ||
      !trilinos->K->isFillComplete()) {
    throw std::runtime_error(
        "[TrilinosNSBlockOperators] ERROR: full matrix must be non-null and fill complete.");
  }
  cache.ensure(
      *trilinos->K, full_topology_generation, nsd, dof);
  return cache.create_system(trilinos);
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
