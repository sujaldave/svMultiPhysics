// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file TrilinosAssembly.cpp
 * @brief Implements staged local CRS assembly for the Trilinos backend.
 */

#ifdef WITH_TRILINOS

#include "trilinos_impl.h"

#include <stdexcept>

namespace trilinos_backend {

void LocalAssemblyBuffer::reset(const TopologyCache& topology)
{
  if (!topology.initialized() || topology.assembly_graph() == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosAssembly] ERROR: topology is not initialized.");
  }

  overlap_matrix_ = Teuchos::rcp(
      new Tpetra_CrsMatrix(topology.assembly_graph()));
  overlap_matrix_->fillComplete(topology.map(), topology.map());

  // Tpetra allocates static-graph values without initializing them. Zero the
  // host view once; subsequent element calls add directly to this CRS array.
  auto local_matrix = overlap_matrix_->getLocalMatrixHost();
  Kokkos::deep_copy(local_matrix.values, Scalar_d(0.0));

  topology_generation_ = topology.generation();
  pending_ = false;
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

bool LocalAssemblyBuffer::flush(
    const TopologyCache& topology,
    Tpetra_CrsMatrix& owned_matrix,
    Tpetra_MultiVector& ghost_rhs)
{
  if (!pending_) {
    return false;
  }
  if (overlap_matrix_ == Teuchos::null ||
      topology_generation_ != topology.generation() ||
      topology.assembly_exporter() == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosAssembly] ERROR: cannot flush stale assembly storage.");
  }

  // These device views trigger one bulk synchronization of the host-written
  // matrix and RHS. Destroy them before calling Tpetra operations that may
  // access the same graph metadata on host; WrappedDualView forbids host
  // access while a device graph view remains alive.
  {
    const auto local_matrix_device = overlap_matrix_->getLocalMatrixDevice();
    const auto local_rhs_device =
        ghost_rhs.getLocalViewDevice(Tpetra::Access::ReadOnly);
    (void)local_matrix_device;
    (void)local_rhs_device;
    Kokkos::fence("svmp_trilinos_assembly_host_to_device");
  }

  if (owned_matrix.isFillComplete()) {
    throw std::runtime_error(
        "[TrilinosAssembly] ERROR: owned matrix is already fill complete.");
  }
  owned_matrix.setAllToScalar(0.0);
  owned_matrix.doExport(
      *overlap_matrix_, *topology.assembly_exporter(), Tpetra::ADD);
  owned_matrix.fillComplete(topology.map(), topology.map());

  pending_ = false;
  return true;
}

bool LocalAssemblyBuffer::pending() const
{
  return pending_;
}

void LocalAssemblyBuffer::clear()
{
  overlap_matrix_ = Teuchos::null;
  topology_generation_ = 0;
  pending_ = false;
}

} // namespace trilinos_backend

#endif
