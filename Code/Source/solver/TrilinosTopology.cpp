// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file TrilinosTopology.cpp
 * @brief Implements equation-local Tpetra topology caching.
 */

#ifdef WITH_TRILINOS

#include "trilinos_impl.h"

#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace trilinos_backend {
namespace {

std::vector<int> copy_prefix(
    const Vector<int>& source,
    std::size_t count,
    const char* field_name)
{
  if (source.size() < count) {
    throw std::runtime_error(
        std::string("[TrilinosTopology] ERROR: ") + field_name +
        " is shorter than the requested topology extent.");
  }

  std::vector<int> copy(count);
  for (std::size_t i = 0; i < count; ++i) {
    copy[i] = source[i];
  }
  return copy;
}

bool equals_prefix(
    const std::vector<int>& cached,
    const Vector<int>& candidate,
    std::size_t count)
{
  if (cached.size() != count || candidate.size() < count) {
    return false;
  }
  for (std::size_t i = 0; i < count; ++i) {
    if (cached[i] != candidate[i]) {
      return false;
    }
  }
  return true;
}

} // namespace

bool TopologyCache::ensure(
    const Teuchos::RCP<const Teuchos::Comm<int>>& communicator,
    int num_global_nodes,
    int num_local_nodes,
    int num_ghost_and_local_nodes,
    int nnz,
    const Vector<int>& local_to_global_sorted,
    const Vector<int>& local_to_global_unsorted,
    const Vector<int>& row_pointer,
    const Vector<int>& column_indices,
    int dof,
    int index_base)
{
  if (signature_matches(
          communicator,
          num_global_nodes,
          num_local_nodes,
          num_ghost_and_local_nodes,
          nnz,
          local_to_global_sorted,
          local_to_global_unsorted,
          row_pointer,
          column_indices,
          dof,
          index_base)) {
    return false;
  }

  if (communicator == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: communicator is null.");
  }
  if (num_global_nodes <= 0 || num_local_nodes < 0 ||
      num_ghost_and_local_nodes < num_local_nodes || nnz < 0 || dof <= 0) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: invalid topology dimensions.");
  }
  if (index_base != 0 && index_base != 1) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: index base must be zero or one.");
  }

  auto sorted = copy_prefix(
      local_to_global_sorted,
      static_cast<std::size_t>(num_ghost_and_local_nodes),
      "local_to_global_sorted");
  auto unsorted = copy_prefix(
      local_to_global_unsorted,
      static_cast<std::size_t>(num_ghost_and_local_nodes),
      "local_to_global_unsorted");
  auto rows = copy_prefix(
      row_pointer,
      static_cast<std::size_t>(num_ghost_and_local_nodes + 1),
      "row_pointer");
  auto columns = copy_prefix(
      column_indices,
      static_cast<std::size_t>(nnz),
      "column_indices");

  std::vector<int> nonzeros_per_row(num_ghost_and_local_nodes);
  if (rows.front() != index_base) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: row-pointer base does not match index base.");
  }
  int counted_nonzeros = 0;
  for (int row = 0; row < num_ghost_and_local_nodes; ++row) {
    const int row_nonzeros = rows[row + 1] - rows[row];
    if (row_nonzeros < 0) {
      throw std::runtime_error(
          "[TrilinosTopology] ERROR: row pointer is not monotone.");
    }
    nonzeros_per_row[row] = row_nonzeros;
    counted_nonzeros += row_nonzeros;
  }
  if (counted_nonzeros != nnz) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: row-pointer nonzeros do not match nnz.");
  }

  std::vector<int> global_columns(nnz);
  for (int entry = 0; entry < nnz; ++entry) {
    const int local_column = columns[entry] - index_base;
    if (local_column < 0 || local_column >= num_ghost_and_local_nodes) {
      throw std::runtime_error(
          "[TrilinosTopology] ERROR: column index is outside the local map.");
    }
    global_columns[entry] = unsorted[local_column];
  }

  std::vector<GO> owned_dof_gids;
  owned_dof_gids.reserve(
      static_cast<std::size_t>(num_local_nodes) * dof);
  for (int node = 0; node < num_local_nodes; ++node) {
    for (int component = 0; component < dof; ++component) {
      owned_dof_gids.push_back(sorted[node] * dof + component);
    }
  }

  std::vector<GO> ghost_dof_gids;
  ghost_dof_gids.reserve(
      static_cast<std::size_t>(num_ghost_and_local_nodes) * dof);
  for (int node = 0; node < num_ghost_and_local_nodes; ++node) {
    for (int component = 0; component < dof; ++component) {
      ghost_dof_gids.push_back(sorted[node] * dof + component);
    }
  }

  auto map = Teuchos::rcp(new Tpetra_Map(
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
      Teuchos::arrayView(owned_dof_gids.data(), owned_dof_gids.size()),
      index_base,
      communicator));
  auto ghost_map = Teuchos::rcp(new Tpetra_Map(
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
      Teuchos::arrayView(ghost_dof_gids.data(), ghost_dof_gids.size()),
      index_base,
      communicator));

  std::unordered_map<GO, std::size_t> unsorted_index;
  unsorted_index.reserve(unsorted.size());
  for (std::size_t i = 0; i < unsorted.size(); ++i) {
    unsorted_index[unsorted[i]] = i;
  }

  std::vector<std::size_t> nonzeros_per_dof_row;
  nonzeros_per_dof_row.reserve(map->getLocalNumElements());
  for (LO local_row = 0;
       local_row < static_cast<LO>(map->getLocalNumElements());
       ++local_row) {
    const GO dof_gid = map->getGlobalElement(local_row);
    const GO node_gid = dof_gid / dof;
    const auto position = unsorted_index.find(node_gid);
    if (position == unsorted_index.end()) {
      throw std::runtime_error(
          "[TrilinosTopology] ERROR: owned node is absent from unsorted map.");
    }
    nonzeros_per_dof_row.push_back(
        static_cast<std::size_t>(nonzeros_per_row[position->second]) * dof);
  }

  std::vector<std::size_t> assembly_nonzeros_per_dof_row;
  assembly_nonzeros_per_dof_row.reserve(ghost_map->getLocalNumElements());
  for (LO local_row = 0;
       local_row < static_cast<LO>(ghost_map->getLocalNumElements());
       ++local_row) {
    const GO dof_gid = ghost_map->getGlobalElement(local_row);
    const GO node_gid = dof_gid / dof;
    const auto position = unsorted_index.find(node_gid);
    if (position == unsorted_index.end()) {
      throw std::runtime_error(
          "[TrilinosTopology] ERROR: overlap node is absent from unsorted map.");
    }
    assembly_nonzeros_per_dof_row.push_back(
        static_cast<std::size_t>(nonzeros_per_row[position->second]) * dof);
  }

  auto graph = Teuchos::rcp(
      new Tpetra_CrsGraph(map, nonzeros_per_dof_row));
  auto assembly_graph = Teuchos::rcp(
      new Tpetra_CrsGraph(ghost_map, assembly_nonzeros_per_dof_row));
  std::size_t nonzero_offset = 0;
  for (int row_node = 0;
       row_node < num_ghost_and_local_nodes;
       ++row_node) {
    const GO row_node_gid = unsorted[row_node];
    const int row_nonzeros = nonzeros_per_row[row_node];
    for (int row_component = 0; row_component < dof; ++row_component) {
      const GO row_gid = row_node_gid * dof + row_component;
      std::vector<GO> row_columns(
          static_cast<std::size_t>(row_nonzeros) * dof);
      for (int entry = 0; entry < row_nonzeros; ++entry) {
        const GO column_node_gid = global_columns[nonzero_offset + entry];
        for (int column_component = 0;
             column_component < dof;
             ++column_component) {
          row_columns[entry * dof + column_component] =
              column_node_gid * dof + column_component;
        }
      }
      graph->insertGlobalIndices(row_gid, row_columns);
      assembly_graph->insertGlobalIndices(row_gid, row_columns);
    }
    nonzero_offset += row_nonzeros;
  }
  graph->fillComplete();
  if (!graph->isFillComplete()) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: graph fillComplete failed.");
  }

  // The overlap graph keeps every local and ghost row so host element
  // contributions can be accumulated without nonlocal insertion. Its domain
  // and range remain the uniquely owned map used by the final linear system.
  assembly_graph->fillComplete(map, map);
  if (!assembly_graph->isFillComplete()) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: assembly graph fillComplete failed.");
  }

  auto importer = Teuchos::rcp(new Tpetra_Import(map, ghost_map));
  auto assembly_exporter = Teuchos::rcp(
      new Tpetra_Export(ghost_map, map));

  std::vector<LO> assembly_vector_lids(
      static_cast<std::size_t>(num_ghost_and_local_nodes) * dof);
  for (int node = 0; node < num_ghost_and_local_nodes; ++node) {
    for (int component = 0; component < dof; ++component) {
      const GO gid = unsorted[node] * dof + component;
      const LO lid = ghost_map->getLocalElement(gid);
      if (lid == Teuchos::OrdinalTraits<LO>::invalid()) {
        throw std::runtime_error(
            "[TrilinosTopology] ERROR: assembly vector GID is absent from "
            "the overlap map.");
      }
      assembly_vector_lids[node * dof + component] = lid;
    }
  }

  const auto local_assembly_graph = assembly_graph->getLocalGraphHost();
  const auto assembly_column_map = assembly_graph->getColMap();
  std::vector<std::size_t> assembly_entry_offsets(
      static_cast<std::size_t>(nnz) * dof * dof);
  std::vector<std::unordered_map<int, std::size_t>>
      assembly_nodal_entry_lookup(num_ghost_and_local_nodes);
  nonzero_offset = 0;
  for (int row_node = 0;
       row_node < num_ghost_and_local_nodes;
       ++row_node) {
    const int row_nonzeros = nonzeros_per_row[row_node];
    assembly_nodal_entry_lookup[row_node].reserve(row_nonzeros);
    for (int entry = 0; entry < row_nonzeros; ++entry) {
      const int local_column =
          columns[nonzero_offset + entry] - index_base;
      const auto inserted = assembly_nodal_entry_lookup[row_node].emplace(
          local_column, nonzero_offset + entry);
      if (!inserted.second) {
        throw std::runtime_error(
            "[TrilinosTopology] ERROR: duplicate nodal column in assembly "
            "graph.");
      }
    }
    for (int row_component = 0; row_component < dof; ++row_component) {
      const GO row_gid = unsorted[row_node] * dof + row_component;
      const LO row_lid = ghost_map->getLocalElement(row_gid);
      const std::size_t row_begin = local_assembly_graph.row_map(row_lid);
      const std::size_t row_end = local_assembly_graph.row_map(row_lid + 1);

      for (int entry = 0; entry < row_nonzeros; ++entry) {
        const GO column_node_gid = global_columns[nonzero_offset + entry];
        for (int column_component = 0;
             column_component < dof;
             ++column_component) {
          const GO column_gid =
              column_node_gid * dof + column_component;
          const LO column_lid =
              assembly_column_map->getLocalElement(column_gid);
          std::size_t scalar_offset = row_end;
          for (std::size_t candidate = row_begin;
               candidate < row_end;
               ++candidate) {
            if (local_assembly_graph.entries(candidate) == column_lid) {
              scalar_offset = candidate;
              break;
            }
          }
          if (scalar_offset == row_end) {
            throw std::runtime_error(
                "[TrilinosTopology] ERROR: nodal entry is absent from the "
                "local assembly graph.");
          }

          const std::size_t block_offset =
              (nonzero_offset + entry) * dof * dof;
          assembly_entry_offsets[
              block_offset + row_component * dof + column_component] =
              scalar_offset;
        }
      }
    }
    nonzero_offset += row_nonzeros;
  }

  communicator_size_ = communicator->getSize();
  communicator_rank_ = communicator->getRank();
  num_global_nodes_ = num_global_nodes;
  num_local_nodes_ = num_local_nodes;
  num_ghost_and_local_nodes_ = num_ghost_and_local_nodes;
  nnz_ = nnz;
  dof_ = dof;
  index_base_ = index_base;
  signature_local_to_global_sorted_ = sorted;
  signature_local_to_global_unsorted_ = unsorted;
  signature_row_pointer_ = rows;
  signature_column_indices_ = columns;
  local_to_global_sorted_ = sorted;
  local_to_global_unsorted_ = unsorted;
  global_column_indices_ = global_columns;
  nonzeros_per_row_ = nonzeros_per_row;
  map_ = map;
  ghost_map_ = ghost_map;
  graph_ = graph;
  assembly_graph_ = assembly_graph;
  importer_ = importer;
  assembly_exporter_ = assembly_exporter;
  assembly_entry_offsets_ = std::move(assembly_entry_offsets);
  assembly_vector_lids_ = std::move(assembly_vector_lids);
  assembly_nodal_entry_lookup_ =
      std::move(assembly_nodal_entry_lookup);
  initialized_ = true;
  ++generation_;
  return true;
}

bool TopologyCache::signature_matches(
    const Teuchos::RCP<const Teuchos::Comm<int>>& communicator,
    int num_global_nodes,
    int num_local_nodes,
    int num_ghost_and_local_nodes,
    int nnz,
    const Vector<int>& local_to_global_sorted,
    const Vector<int>& local_to_global_unsorted,
    const Vector<int>& row_pointer,
    const Vector<int>& column_indices,
    int dof,
    int index_base) const
{
  if (!initialized_ || communicator == Teuchos::null) {
    return false;
  }
  return communicator_size_ == communicator->getSize() &&
      communicator_rank_ == communicator->getRank() &&
      num_global_nodes_ == num_global_nodes &&
      num_local_nodes_ == num_local_nodes &&
      num_ghost_and_local_nodes_ == num_ghost_and_local_nodes &&
      nnz_ == nnz && dof_ == dof && index_base_ == index_base &&
      equals_prefix(
          signature_local_to_global_sorted_,
          local_to_global_sorted,
          static_cast<std::size_t>(num_ghost_and_local_nodes)) &&
      equals_prefix(
          signature_local_to_global_unsorted_,
          local_to_global_unsorted,
          static_cast<std::size_t>(num_ghost_and_local_nodes)) &&
      equals_prefix(
          signature_row_pointer_,
          row_pointer,
          static_cast<std::size_t>(num_ghost_and_local_nodes + 1)) &&
      equals_prefix(
          signature_column_indices_,
          column_indices,
          static_cast<std::size_t>(nnz));
}

bool TopologyCache::initialized() const
{
  return initialized_;
}

void TopologyCache::clear()
{
  importer_ = Teuchos::null;
  assembly_exporter_ = Teuchos::null;
  graph_ = Teuchos::null;
  assembly_graph_ = Teuchos::null;
  map_ = Teuchos::null;
  ghost_map_ = Teuchos::null;
  signature_local_to_global_sorted_.clear();
  signature_local_to_global_unsorted_.clear();
  signature_row_pointer_.clear();
  signature_column_indices_.clear();
  local_to_global_sorted_.clear();
  local_to_global_unsorted_.clear();
  global_column_indices_.clear();
  nonzeros_per_row_.clear();
  assembly_entry_offsets_.clear();
  assembly_vector_lids_.clear();
  assembly_nodal_entry_lookup_.clear();
  initialized_ = false;
  generation_ = 0;
}

std::size_t TopologyCache::generation() const
{
  return generation_;
}

int TopologyCache::dof() const
{
  return dof_;
}

int TopologyCache::local_nodes() const
{
  return num_local_nodes_;
}

int TopologyCache::ghost_and_local_nodes() const
{
  return num_ghost_and_local_nodes_;
}

const Teuchos::RCP<const Tpetra_Map>& TopologyCache::map() const
{
  return map_;
}

const Teuchos::RCP<const Tpetra_Map>& TopologyCache::ghost_map() const
{
  return ghost_map_;
}

const Teuchos::RCP<Tpetra_CrsGraph>& TopologyCache::graph() const
{
  return graph_;
}

const Teuchos::RCP<Tpetra_CrsGraph>& TopologyCache::assembly_graph() const
{
  return assembly_graph_;
}

const Teuchos::RCP<Tpetra_Import>& TopologyCache::importer() const
{
  return importer_;
}

const Teuchos::RCP<Tpetra_Export>& TopologyCache::assembly_exporter() const
{
  return assembly_exporter_;
}

const std::size_t* TopologyCache::assembly_block_offsets(
    int local_row_node,
    int local_column_node) const
{
  if (local_row_node < 0 ||
      local_row_node >= num_ghost_and_local_nodes_ ||
      local_column_node < 0 ||
      local_column_node >= num_ghost_and_local_nodes_) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: assembly node index is out of range.");
  }

  const auto entry =
      assembly_nodal_entry_lookup_[local_row_node].find(local_column_node);
  if (entry != assembly_nodal_entry_lookup_[local_row_node].end()) {
    return assembly_entry_offsets_.data() +
        entry->second * dof_ * dof_;
  }

  throw std::runtime_error(
      "[TrilinosTopology] ERROR: element node pair is absent from the "
      "assembly graph.");
}

LO TopologyCache::assembly_vector_lid(
    int local_node,
    int component) const
{
  if (local_node < 0 || local_node >= num_ghost_and_local_nodes_ ||
      component < 0 || component >= dof_) {
    throw std::runtime_error(
        "[TrilinosTopology] ERROR: assembly vector index is out of range.");
  }
  return assembly_vector_lids_[local_node * dof_ + component];
}

const std::vector<int>& TopologyCache::local_to_global_sorted() const
{
  return local_to_global_sorted_;
}

const std::vector<int>& TopologyCache::local_to_global_unsorted() const
{
  return local_to_global_unsorted_;
}

const std::vector<int>& TopologyCache::global_column_indices() const
{
  return global_column_indices_;
}

const std::vector<int>& TopologyCache::nonzeros_per_row() const
{
  return nonzeros_per_row_;
}

} // namespace trilinos_backend

#endif
