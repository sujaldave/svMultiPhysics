// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause
#include "TrilinosFluidAssembly.h"
#ifdef WITH_TRILINOS
#include "FluidElementKernel.h"
#include "fluid.h"
#include "fs.h"
#include "nn.h"
#include "Teuchos_CommHelpers.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <string>

namespace trilinos_backend {
namespace {
using Device = typename Tpetra_CrsMatrix::local_matrix_device_type::device_type;
using Exec = typename Device::execution_space;
template<class T> using View2 = Kokkos::View<T**, Kokkos::LayoutRight, Device>;
using RhsView = decltype(std::declval<Tpetra_MultiVector&>().getLocalViewDevice(Tpetra::Access::ReadWrite));
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

struct MeshCache {
  std::size_t generation = 0;
  Teuchos::RCP<const Tpetra_CrsGraph> graph;
  int nodes = 0, elements = 0, points = 0, geometric_points = 0;
  std::vector<int> ien_snapshot;
  std::vector<double> coordinates_snapshot, quadrature_snapshot;
  View2<int> ien, rhs_lids;
  View2<std::size_t> matrix_offsets;
  Kokkos::View<double***, Kokkos::LayoutRight, Device> geometry;
  View2<double> shapes, state;
  Kokkos::View<double*, Device> weights;
  typename View2<double>::HostMirror host_state;

  static std::vector<double> quadrature(const mshType& mesh) {
    std::vector<double> data;
    const auto& f = mesh.fs[0];
    for (auto* a : {&f.w}) data.insert(data.end(), a->data(), a->data() + a->size());
    for (auto* a : {&f.N}) data.insert(data.end(), a->data(), a->data() + a->size());
    for (auto* a : {&f.Nx, &f.Nxx}) data.insert(data.end(), a->data(), a->data() + a->size());
    return data;
  }
  bool matches(const ComMod& c, const mshType& m, const TopologyCache& topology) const {
    return generation == topology.generation() && graph.getRawPtr() == topology.assembly_graph().getRawPtr() && nodes == m.eNoN && elements == m.nEl &&
        points == m.fs[0].nG && ien_snapshot.size() == m.IEN.size() &&
        coordinates_snapshot.size() == c.x.size() &&
        std::equal(ien_snapshot.begin(), ien_snapshot.end(), m.IEN.data()) &&
        std::equal(coordinates_snapshot.begin(), coordinates_snapshot.end(), c.x.data()) &&
        quadrature_snapshot == quadrature(m);
  }

  void build(ComMod& c, const mshType& m, const TopologyCache& topology) {
    nodes = m.eNoN; elements = m.nEl; points = m.fs[0].nG;
    geometric_points = m.eType == consts::ElementType::TET4 ? 1 : points;
    generation = topology.generation();
    graph = topology.assembly_graph();
    if (m.IEN.size()) ien_snapshot.assign(m.IEN.data(), m.IEN.data() + m.IEN.size());
    if (c.x.size()) coordinates_snapshot.assign(c.x.data(), c.x.data() + c.x.size());
    quadrature_snapshot = quadrature(m);
    ien = View2<int>("fluid_connectivity", elements, nodes);
    rhs_lids = View2<int>("fluid_rhs_lids", elements, nodes*4);
    // Each scalar CRS row has four adjacent columns per nodal block.
    matrix_offsets = View2<std::size_t>("fluid_crs_offsets", elements, nodes*nodes*4);
    const int geometry_width = 1 + 3*nodes + (nodes == 8 ? 6*nodes : 0) + 9;
    geometry = decltype(geometry)("fluid_geometry", elements, geometric_points, geometry_width);
    shapes = View2<double>("fluid_shapes", points, nodes);
    weights = decltype(weights)("fluid_weights", points);
    state = View2<double>("fluid_state", c.tnNo, 11);
    host_state = Kokkos::create_mirror_view(state);
    auto hi = Kokkos::create_mirror_view(ien);
    auto hr = Kokkos::create_mirror_view(rhs_lids);
    auto ho = Kokkos::create_mirror_view(matrix_offsets);
    auto hg = Kokkos::create_mirror_view(geometry);
    auto hn = Kokkos::create_mirror_view(shapes);
    auto hw = Kokkos::create_mirror_view(weights);
    std::array<fsType,2> f;
    fs::get_thood_fs(c, f, m, true, 1);
    for (int g = 0; g < points; ++g) {
      hw(g) = f[0].w(g);
      for (int a = 0; a < nodes; ++a) hn(g,a) = f[0].N(a,g);
    }
    Array<double> x(3,nodes), nx(3,nodes), nxx(6,nodes), metric(3,3);
    for (int e = 0; e < elements; ++e) {
      for (int a = 0; a < nodes; ++a) {
        const int node = m.IEN(a,e);
        if (node < 0 || node >= c.tnNo) throw std::runtime_error("Invalid fluid connectivity.");
        hi(e,a) = node;
        for (int i = 0; i < 3; ++i) x(i,a) = c.x(i,node);
        for (int i = 0; i < 4; ++i) hr(e,a*4+i) = topology.assembly_vector_lid(node,i);
        for (int b = 0; b < nodes; ++b) {
          const auto offsets = topology.assembly_block_offsets(node,m.IEN(b,e));
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
              if (offsets[4*i+j] != offsets[4*i]+j)
                throw std::runtime_error("Fluid scalar CRS columns are not nodal-block contiguous.");
            }
            ho(e,(a*nodes+b)*4+i) = offsets[4*i];
          }
        }
      }
      for (int g = 0; g < geometric_points; ++g) {
        double jac = 0.0;
        auto nxi = f[0].Nx.rslice(g);
        auto nxixi = f[0].Nxx.rslice(g);
        nx = 0.0; nxx = 0.0; metric = 0.0;
        nn::gnn(nodes,3,3,nxi,x,nx,jac,metric);
        if (!std::isfinite(jac) || jac <= 0.0) throw std::runtime_error("Invalid fluid element Jacobian.");
        nn::gn_nxx(6,nodes,3,3,nxi,nxixi,x,nx,nxx);
        hg(e,g,0) = jac;
        for (int k = 0; k < 3*nodes; ++k) hg(e,g,1+k) = nx.data()[k];
        if (nodes == 8) for (int k = 0; k < 6*nodes; ++k) hg(e,g,1+3*nodes+k) = nxx.data()[k];
        for (int k = 0; k < 9; ++k) hg(e,g,geometry_width-9+k) = metric.data()[k];
      }
    }
    Kokkos::deep_copy(ien,hi); Kokkos::deep_copy(rhs_lids,hr);
    Kokkos::deep_copy(matrix_offsets,ho); Kokkos::deep_copy(geometry,hg);
    Kokkos::deep_copy(shapes,hn); Kokkos::deep_copy(weights,hw);
  }
};

template<int N>
struct AssembleFluid {
  View2<int> ien, lids;
  View2<std::size_t> offsets;
  decltype(MeshCache::geometry) geometry;
  View2<double> shapes, state;
  decltype(MeshCache::weights) weights;
  typename Tpetra_CrsMatrix::local_matrix_device_type matrix;
  RhsView rhs;
  fluid::element::Parameters prm;
  double brinkman;
  int points;
  bool atomics;

  KOKKOS_INLINE_FUNCTION void operator()(const int e) const {
    double al[4*N], yl[4*N], bf[3*N];
    double residual[4*N] = {}, tangent[16*N*N] = {};
    const double zero_hessian[6*N] = {}, zero_uris[3] = {};
    for (int a = 0; a < N; ++a) {
      const int node = ien(e,a);
      for (int i = 0; i < 4; ++i) { al[a*4+i] = state(node,i); yl[a*4+i] = state(node,4+i); }
      for (int i = 0; i < 3; ++i) bf[a*3+i] = state(node,8+i);
    }
    for (int phase = 0; phase < 2; ++phase) for (int g = 0; g < points; ++g) {
      const int q = N == 4 ? 0 : g;
      const double* nx = &geometry(e,q,1);
      // construct_fluid retains the final momentum-quadrature Hessian for
      // its continuity loop. Preserve that existing HEX8 discretization here.
      const int hessian_q = phase == 0 ? q : points-1;
      const double* nxx = N == 4 ? zero_hessian : &geometry(e,hessian_q,1+3*N);
      const double* metric = &geometry(e,q,geometry.extent(2)-9);
      const double w = weights(g)*geometry(e,q,0);
      if (phase == 0) fluid::element::fluid_3d_m<N>(prm,1,N,N,w,
          {metric,3},{&shapes(g,0)},{&shapes(g,0)},{nx,3},{nx,3},{nxx,6},
          {al,4},{yl,4},{bf,3},{residual,4},{tangent,16,N},brinkman,0.0,zero_uris);
      else fluid::element::fluid_3d_c<N>(prm,1,N,N,w,
          {metric,3},{&shapes(g,0)},{&shapes(g,0)},{nx,3},{nx,3},{nxx,6},
          {al,4},{yl,4},{bf,3},{residual,4},{tangent,16,N},brinkman,0.0,zero_uris);
    }
    for (int a = 0; a < N; ++a) {
      for (int i = 0; i < 4; ++i) {
        if (atomics) Kokkos::atomic_add(&rhs(lids(e,a*4+i),0), residual[a*4+i]);
        else rhs(lids(e,a*4+i),0) += residual[a*4+i];
      }
      for (int b = 0; b < N; ++b) for (int i = 0; i < 4; ++i) {
        const auto offset = offsets(e,(a*N+b)*4+i);
        for (int j = 0; j < 4; ++j) {
          const double value = tangent[16*(a+N*b)+4*i+j];
          if (atomics) Kokkos::atomic_add(&matrix.values(offset+j),value);
          else matrix.values(offset+j) += value;
        }
      }
    }
  }
};

template<int N>
void launch(MeshCache& cache, Trilinos& t, const fluid::element::Parameters& prm, double brinkman) {
  // Keep WrappedDualView lifetimes inside this function: later boundary and
  // Tpetra communication calls must not overlap outstanding local graph views.
  const auto matrix = t.local_assembly.device_assembly_matrix()->getLocalMatrixDevice();
  const auto rhs = t.ghostF->getLocalViewDevice(Tpetra::Access::ReadWrite);
  AssembleFluid<N> kernel{cache.ien,cache.rhs_lids,cache.matrix_offsets,cache.geometry,
      cache.shapes,cache.state,cache.weights,matrix,rhs,prm,brinkman,cache.points,Exec().concurrency()>1};
  Kokkos::parallel_for("svmp_fluid_compute_and_scatter",Kokkos::RangePolicy<Exec>(0,cache.elements),kernel);
  Kokkos::fence("svmp_fluid_assembly_complete");
}
} // namespace

class FluidAssembly::Impl { public: std::map<const mshType*,MeshCache> caches; };
FluidAssembly::FluidAssembly() : impl_(new Impl) {}
FluidAssembly::~FluidAssembly() = default;
void FluidAssembly::clear() { impl_->caches.clear(); }

bool FluidAssembly::assemble(ComMod& c, const mshType& m, const SolutionStates& solution, Trilinos& t) {
  const std::string mode = std::getenv("SV_TRILINOS_ASSEMBLY") ? std::getenv("SV_TRILINOS_ASSEMBLY") : "auto";
  if (mode != "auto" && mode != "legacy" && mode != "kokkos")
    throw std::runtime_error("SV_TRILINOS_ASSEMBLY must be auto, legacy, or kokkos.");
  if (mode == "legacy") return false;
  const auto& eq = c.eq[c.cEq];
  const bool supported = c.nsd == 3 && c.dof == 4 && c.tDof >= 4 && m.nFs == 1 && m.fs.size() == 1 &&
      ((m.eNoN == 4 && m.eType == consts::ElementType::TET4 && m.fs[0].lShpF) ||
       (m.eNoN == 8 && m.eType == consts::ElementType::HEX8 && !m.fs[0].lShpF)) &&
      !c.mvMsh && !c.risFlag && !c.urisFlag && eq.nDmn == 1 && eq.dmn.size() == 1 &&
      eq.dmn[0].phys == consts::EquationType::phys_fluid &&
      eq.dmn[0].fluid_visc.viscType == consts::FluidViscosityModelType::viscType_Const;
  if (!supported) {
    if (mode == "kokkos") throw std::runtime_error("Kokkos fluid assembly requires rigid-wall 3D TET4/HEX8 VMS fluid, one domain, constant viscosity, and no RIS/URIS.");
    return false;
  }
  const auto start = Clock::now();
  auto& cache = impl_->caches[&m];
  const bool rebuild = !cache.matches(c,m,t.topology);
  if (rebuild) {
    MeshCache fresh;
    fresh.build(c,m,t.topology);
    cache = std::move(fresh);
  }
  const double setup = seconds(start);
  const auto upload_start = Clock::now();
  const auto& a = solution.intermediate.get_acceleration();
  const auto& y = solution.intermediate.get_velocity();
  if (c.tnNo>0 && (a.nrows()<4 || y.nrows()<4 || c.Bf.nrows()<3 ||
      a.ncols()<c.tnNo || y.ncols()<c.tnNo || c.Bf.ncols()<c.tnNo))
    throw std::runtime_error("Fluid state dimensions do not match the local node layout.");
  for (int node = 0; node < c.tnNo; ++node) {
    for (int i = 0; i < 4; ++i) { cache.host_state(node,i) = a(i,node); cache.host_state(node,4+i) = y(i,node); }
    for (int i = 0; i < 3; ++i) cache.host_state(node,8+i) = c.Bf(i,node);
  }
  Kokkos::deep_copy(cache.state,cache.host_state);
  const double upload = seconds(upload_start);
  const auto kernel_start = Clock::now();
  const auto prm = fluid::element_parameters(c,eq.dmn[0]);
  const double brinkman = eq.dmn[0].prop.at(consts::PhysicalPropertyType::brinkman_inverse_permeability);
  if (m.eNoN == 4) launch<4>(cache,t,prm,brinkman); else launch<8>(cache,t,prm,brinkman);
  c.cDmn = 0;
  const double kernel = seconds(kernel_start);
  if (std::getenv("SV_TRILINOS_ASSEMBLY_STATS")) {
    const double local[] = {seconds(start),setup,upload,kernel};
    double global[4];
    Teuchos::reduceAll(*t.comm,Teuchos::REDUCE_MAX,4,local,global);
    if (t.comm->getRank() == 0) std::cout << "ASSEMBLY_STATS backend=kokkos execution_space=" << Exec::name() << " total_s=" << global[0]
        << " cache_s=" << global[1] << " state_s=" << global[2] << " kernel_s=" << global[3]
        << " rebuilt=" << rebuild << " nodes_per_element=" << m.eNoN << std::endl;
  }
  return true;
}
} // namespace trilinos_backend
#endif
