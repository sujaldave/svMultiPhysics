// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file TrilinosAssembly.cpp
 * @brief Implements staged local CRS assembly for the Trilinos backend.
 */

#ifdef WITH_TRILINOS

#include "trilinos_impl.h"

#include <stdexcept>
#include <cstdlib>
#include <limits>
#include "Teuchos_CommHelpers.hpp"

namespace trilinos_backend {

namespace {

// NVCC cannot lower an extended host/device lambda inside a constructor:
// its enclosing function must have an address, and constructors do not.
// A namespace-scope functor has no such enclosing-function requirement.
struct AssemblyEntryIdsFunctor {
  Tpetra_CrsMatrix::local_matrix_device_type matrix;
  GO offset;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    matrix.values(k) = static_cast<double>(offset + k);
  }
};

} // namespace

// A sparse matrix doExport in Trilinos 16.2 copies/permutates rows through host
// APIs. Build entry IDs once, then export only numeric values as a multivector.
// This uses Tpetra's execution-space-aware vector communication for every later
// Jacobian, including GPU-aware MPI when configured by the installed Trilinos.
class NumericAssemblyExport {
  Teuchos::RCP<Tpetra_MultiVector> overlap_values_, owned_values_;
  Teuchos::RCP<Tpetra_Export> exporter_;
public:
  NumericAssemblyExport(const TopologyCache& topology, const Tpetra_CrsMatrix& owned) {
    const auto count = owned.getGlobalNumEntries();
    // This backend currently uses GO=int. Keep a correct generic fallback if
    // the flattened global entry space exceeds that ordinal type.
    if (count > static_cast<Tpetra::global_size_t>(std::numeric_limits<GO>::max())) return;
    auto value_map = Teuchos::rcp(new Tpetra_Map(count,owned.getLocalNumEntries(),0,topology.map()->getComm()));
    auto ids = Teuchos::rcp(new Tpetra_CrsMatrix(topology.graph()));
    ids->fillComplete(topology.map(),topology.map());
    {
      const auto local = ids->getLocalMatrixDevice();
      const GO offset = value_map->getMinGlobalIndex();
      Kokkos::parallel_for("assembly_entry_ids",Kokkos::RangePolicy<typename Node::execution_space>(0,local.values.extent(0)),
          AssemblyEntryIdsFunctor{local,offset});
      Kokkos::fence("assembly_entry_ids_ready");
    }
    auto ghost_ids = Teuchos::rcp(new Tpetra_CrsMatrix(topology.ghost_map(),owned.getGlobalMaxNumRowEntries()));
    ghost_ids->doImport(*ids,*topology.importer(),Tpetra::INSERT);
    ghost_ids->fillComplete(topology.map(),topology.map());
    std::vector<GO> gids;
    {
      const auto imported = ghost_ids->getLocalMatrixHost();
      const auto graph = topology.assembly_graph()->getLocalGraphHost();
      gids.resize(graph.entries.extent(0));
      for (LO row=0; row<static_cast<LO>(topology.ghost_map()->getLocalNumElements()); ++row) {
        std::unordered_map<GO,GO> entry_ids;
        for (auto k=imported.graph.row_map(row); k<imported.graph.row_map(row+1); ++k)
          entry_ids.emplace(ghost_ids->getColMap()->getGlobalElement(imported.graph.entries(k)),static_cast<GO>(imported.values(k)));
        for (auto k=graph.row_map(row); k<graph.row_map(row+1); ++k)
          gids[k]=entry_ids.at(topology.assembly_graph()->getColMap()->getGlobalElement(graph.entries(k)));
      }
    }
    auto overlap_map = Teuchos::rcp(new Tpetra_Map(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
        Teuchos::arrayView(gids.data(),gids.size()),0,topology.map()->getComm()));
    overlap_values_=Teuchos::rcp(new Tpetra_MultiVector(overlap_map,1));
    owned_values_=Teuchos::rcp(new Tpetra_MultiVector(value_map,1));
    exporter_=Teuchos::rcp(new Tpetra_Export(overlap_map,value_map));
  }
  bool apply(Tpetra_CrsMatrix& overlap, Tpetra_CrsMatrix& owned) {
    if (exporter_ == Teuchos::null) return false;
    {
      const auto source=overlap.getLocalMatrixDevice();
      const auto target=overlap_values_->getLocalViewDevice(Tpetra::Access::OverwriteAll);
      Kokkos::parallel_for("assembly_pack_numeric",Kokkos::RangePolicy<typename Node::execution_space>(0,source.values.extent(0)),
          KOKKOS_LAMBDA(const int k) { target(k,0)=source.values(k); });
      Kokkos::fence("assembly_numeric_ready");
    }
    owned_values_->putScalar(0.0);
    owned_values_->doExport(*overlap_values_,*exporter_,Tpetra::ADD);
    {
      const auto source=owned_values_->getLocalViewDevice(Tpetra::Access::ReadOnly);
      const auto target=owned.getLocalMatrixDevice();
      Kokkos::parallel_for("assembly_unpack_numeric",Kokkos::RangePolicy<typename Node::execution_space>(0,target.values.extent(0)),
          KOKKOS_LAMBDA(const int k) { target.values(k)=source(k,0); });
      Kokkos::fence("assembly_owned_values_ready");
    }
    return true;
  }
};

void LocalAssemblyBuffer::reset(const TopologyCache& topology)
{
  if (!topology.initialized() || topology.assembly_graph() == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosAssembly] ERROR: topology is not initialized.");
  }

  if (overlap_matrix_ == Teuchos::null || topology_generation_ != topology.generation() ||
      overlap_matrix_->getCrsGraph().getRawPtr() != topology.assembly_graph().getRawPtr()) {
    numeric_export_.reset();
    overlap_matrix_ = Teuchos::rcp(new Tpetra_CrsMatrix(topology.assembly_graph()));
    overlap_matrix_->fillComplete(topology.map(), topology.map());
  }
  overlap_matrix_->setAllToScalar(0.0);
  topology_generation_ = topology.generation();
  pending_ = false;
  device_assembly_ = false;
  staged_matrix_offsets_.clear();
  staged_matrix_values_.clear();
  staged_rhs_offsets_.clear();
  staged_rhs_values_.clear();
}

void LocalAssemblyBuffer::add_element(
    const TopologyCache& topology,
    Tpetra_MultiVector& ghost_rhs,
    int num_element_nodes,
    const int* equation_nodes,
    const double* element_matrix,
    const double* element_rhs)
{
  if (overlap_matrix_ == Teuchos::null ||
      topology_generation_ != topology.generation()) {
    throw std::runtime_error(
        "[TrilinosAssembly] ERROR: assembly buffer does not match topology.");
  }
  if (num_element_nodes <= 0 || equation_nodes == nullptr ||
      element_matrix == nullptr || element_rhs == nullptr) {
    throw std::runtime_error(
        "[TrilinosAssembly] ERROR: invalid element assembly input.");
  }

  // Small host boundary contributions must not synchronize the full matrix
  // back from a GPU after volume assembly. Upload sparse deltas once in flush.
  if (device_assembly_ &&
      !Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Node::memory_space>::accessible) {
    const int dof = topology.dof();
    for (int a = 0; a < num_element_nodes; ++a) {
      for (int i = 0; i < dof; ++i) {
        const double value = element_rhs[a*dof + i];
        if (value != 0.0) {
          staged_rhs_offsets_.push_back(topology.assembly_vector_lid(equation_nodes[a], i));
          staged_rhs_values_.push_back(value);
        }
      }
      for (int b = 0; b < num_element_nodes; ++b) {
        const auto offsets = topology.assembly_block_offsets(equation_nodes[a], equation_nodes[b]);
        for (int i = 0; i < dof; ++i) for (int j = 0; j < dof; ++j) {
          const double value = element_matrix[b*dof*dof*num_element_nodes + a*dof*dof + i*dof + j];
          if (value != 0.0) {
            staged_matrix_offsets_.push_back(offsets[i*dof + j]);
            staged_matrix_values_.push_back(value);
          }
        }
      }
    }
    pending_ = true;
    return;
  }

  const int dof = topology.dof();
  auto local_matrix = overlap_matrix_->getLocalMatrixHost();
  auto local_rhs = ghost_rhs.getLocalViewHost(Tpetra::Access::ReadWrite);

  for (int row_node = 0; row_node < num_element_nodes; ++row_node) {
    const int local_row_node = equation_nodes[row_node];
    for (int row_component = 0; row_component < dof; ++row_component) {
      const LO rhs_lid = topology.assembly_vector_lid(
          local_row_node, row_component);
      local_rhs(rhs_lid, 0) +=
          element_rhs[row_node * dof + row_component];
    }

    for (int column_node = 0;
         column_node < num_element_nodes;
         ++column_node) {
      const std::size_t* offsets = topology.assembly_block_offsets(
          local_row_node, equation_nodes[column_node]);
      for (int row_component = 0; row_component < dof; ++row_component) {
        for (int column_component = 0;
             column_component < dof;
             ++column_component) {
          // lK stores the nodal indices transposed relative to the mathematical
          // row/column convention used by Tpetra.
          const std::size_t element_offset =
              column_node * dof * dof * num_element_nodes +
              row_node * dof * dof +
              row_component * dof +
              column_component;
          local_matrix.values(
              offsets[row_component * dof + column_component]) +=
              element_matrix[element_offset];
        }
      }
    }
  }

  pending_ = true;
}

Teuchos::RCP<Tpetra_CrsMatrix> LocalAssemblyBuffer::device_assembly_matrix()
{
  if (overlap_matrix_ == Teuchos::null) throw std::runtime_error("Assembly buffer is not initialized.");
  device_assembly_ = true;
  pending_ = true;
  return overlap_matrix_;
}

namespace {
template<class T>
auto upload_deltas(const std::vector<T>& host, const char* label)
{
  Kokkos::View<T*, typename Node::device_type> device(std::string(label), host.size());
  auto mirror = Kokkos::create_mirror_view(device);
  for (size_t i = 0; i < host.size(); ++i) mirror(i) = host[i];
  Kokkos::deep_copy(device, mirror);
  return device;
}
}

bool LocalAssemblyBuffer::flush(
    const TopologyCache& topology,
    Tpetra_CrsMatrix& owned_matrix,
    Tpetra_MultiVector& ghost_rhs)
{
  if (!pending_) {
    return false;
  }
  const double start = MPI_Wtime();
  if (overlap_matrix_ == Teuchos::null ||
      topology_generation_ != topology.generation() ||
      topology.assembly_exporter() == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosAssembly] ERROR: cannot flush stale assembly storage.");
  }

  {
    const auto matrix = overlap_matrix_->getLocalMatrixDevice();
    const auto rhs = ghost_rhs.getLocalViewDevice(Tpetra::Access::ReadWrite);
    const auto mi = upload_deltas(staged_matrix_offsets_, "boundary_matrix_offsets");
    const auto mv = upload_deltas(staged_matrix_values_, "boundary_matrix_values");
    const auto ri = upload_deltas(staged_rhs_offsets_, "boundary_rhs_offsets");
    const auto rv = upload_deltas(staged_rhs_values_, "boundary_rhs_values");
    Kokkos::parallel_for("svmp_boundary_matrix", Kokkos::RangePolicy<typename Node::execution_space>(0, mi.extent(0)),
        KOKKOS_LAMBDA(const int i) { Kokkos::atomic_add(&matrix.values(mi(i)), mv(i)); });
    Kokkos::parallel_for("svmp_boundary_rhs", Kokkos::RangePolicy<typename Node::execution_space>(0, ri.extent(0)),
        KOKKOS_LAMBDA(const int i) { Kokkos::atomic_add(&rhs(ri(i), 0), rv(i)); });
    Kokkos::fence("svmp_assembly_complete");
  }
  staged_matrix_offsets_.clear();
  staged_matrix_values_.clear();
  staged_rhs_offsets_.clear();
  staged_rhs_values_.clear();

  const double stage_seconds = MPI_Wtime()-start;
  const char* transport="local-copy";
  if (topology.map()->getComm()->getSize() == 1 &&
      topology.map()->isSameAs(*topology.ghost_map())) {
    // Both graphs are constructed from the same rows/columns by TopologyCache.
    // With no shared rows, copying numeric CRS values avoids Tpetra's general
    // row pack/unpack (which includes global-column lookup on every Jacobian).
    if (!owned_matrix.isFillComplete()) owned_matrix.fillComplete(topology.map(),topology.map());
    {
      const auto source = overlap_matrix_->getLocalMatrixDevice();
      const auto target = owned_matrix.getLocalMatrixDevice();
      if (source.values.extent(0) != target.values.extent(0))
        throw std::runtime_error("Incompatible serial assembly CRS graphs.");
      Kokkos::deep_copy(target.values,source.values);
    }
  } else {
    if (!owned_matrix.isFillComplete()) owned_matrix.fillComplete(topology.map(),topology.map());
    transport="numeric-values";
    if (!numeric_export_) numeric_export_ = std::make_shared<NumericAssemblyExport>(topology,owned_matrix);
    if (!numeric_export_->apply(*overlap_matrix_,owned_matrix)) {
      transport="matrix-rows";
      if (owned_matrix.isFillComplete()) owned_matrix.resumeFill();
      owned_matrix.setAllToScalar(0.0);
      owned_matrix.doExport(*overlap_matrix_,*topology.assembly_exporter(),Tpetra::ADD);
      owned_matrix.fillComplete(topology.map(),topology.map());
    }
  }

  if (std::getenv("SV_TRILINOS_ASSEMBLY_STATS")) {
    Kokkos::fence("assembly_export_timing");
    const double local[] = {stage_seconds,MPI_Wtime()-start-stage_seconds};
    double global[2];
    Teuchos::reduceAll(*topology.map()->getComm(),Teuchos::REDUCE_MAX,2,local,global);
    if (topology.map()->getComm()->getRank() == 0) std::cout << "ASSEMBLY_EXPORT_STATS stage_s="
        << global[0] << " export_s=" << global[1] << " transport=" << transport << std::endl;
  }
  pending_ = false;
  return true;
}

bool LocalAssemblyBuffer::pending() const
{
  return pending_;
}

void LocalAssemblyBuffer::clear()
{
  numeric_export_.reset();
  overlap_matrix_ = Teuchos::null;
  topology_generation_ = 0;
  pending_ = false;
  device_assembly_ = false;
  staged_matrix_offsets_.clear();
  staged_matrix_values_.clear();
  staged_rhs_offsets_.clear();
  staged_rhs_values_.clear();
}

} // namespace trilinos_backend

#endif
