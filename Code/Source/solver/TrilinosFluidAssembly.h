// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef SV_TRILINOS_FLUID_ASSEMBLY_H
#define SV_TRILINOS_FLUID_ASSEMBLY_H
#ifdef WITH_TRILINOS
#include "trilinos_impl.h"
#include "ComMod.h"
#include "SolutionStates.h"
#include <memory>

namespace trilinos_backend {
// Per-equation ownership: cached mesh geometry, device connectivity, and CRS
// scatter plans outlive Newton iterations, but never the Kokkos runtime.
class FluidAssembly {
public:
  FluidAssembly();
  ~FluidAssembly();
  bool assemble(ComMod&, const mshType&, const SolutionStates&, Trilinos&);
  void clear();
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}
#endif
#endif
