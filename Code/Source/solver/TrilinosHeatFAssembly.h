// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SV_TRILINOS_HEATF_ASSEMBLY_H
#define SV_TRILINOS_HEATF_ASSEMBLY_H
#ifdef WITH_TRILINOS
#include "trilinos_impl.h"
#include "ComMod.h"
#include "SolutionStates.h"
#include <memory>

namespace trilinos_backend {
// Kokkos volume assembly for the heatF / scalarTransport equation (dof=1).
// Same lifetime rules as FluidAssembly: mesh geometry and CRS scatter offsets
// are cached per mesh and rebuilt when the topology or coordinates change.
// Boundary terms (b_heatf) stay on the host and go through LocalAssemblyBuffer.
class HeatFAssembly {
public:
  HeatFAssembly();
  ~HeatFAssembly();
  bool assemble(ComMod&, const mshType&, const SolutionStates&, Trilinos&);
  void clear();
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}
#endif
#endif
