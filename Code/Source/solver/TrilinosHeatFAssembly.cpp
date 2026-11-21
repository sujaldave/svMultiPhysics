// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause
#include "TrilinosHeatFAssembly.h"
#ifdef WITH_TRILINOS
#include "nn.h"
#include "Teuchos_CommHelpers.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
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

// Per-node state columns: T_dot (acceleration row eq.s), u_x, u_y, u_z, T.
constexpr int kStateWidth = 5;

struct MeshCache {
  std::size_t generation = 0;
  Teuchos::RCP<const Tpetra_CrsGraph> graph;
  int nodes = 0, elements = 0, points = 0;
  std::vector<int> ien_snapshot;
  std::vector<double> coordinates_snapshot, quadrature_snapshot;
  View2<int> ien, rhs_lids;
  View2<std::size_t> matrix_offsets;
  View2<double> geometry, shapes, state;
  Kokkos::View<double*, Device> weights;
  typename View2<double>::host_mirror_type host_state;

  static std::vector<double> quadrature(const mshType& m) {
    std::vector<double> data;
    for (auto* a : {&m.w}) data.insert(data.end(), a->data(), a->data() + a->size());
    for (auto* a : {&m.N}) data.insert(data.end(), a->data(), a->data() + a->size());
    data.insert(data.end(), m.Nx.data(), m.Nx.data() + m.Nx.size());
    return data;
  }
  bool matches(const ComMod& c, const mshType& m, const TopologyCache& topology) const {
    return generation == topology.generation() && graph.getRawPtr() == topology.assembly_graph().getRawPtr() &&
        nodes == m.eNoN && elements == m.nEl && points == m.nG && ien_snapshot.size() == m.IEN.size() &&
        coordinates_snapshot.size() == c.x.size() &&
        std::equal(ien_snapshot.begin(), ien_snapshot.end(), m.IEN.data()) &&
        std::equal(coordinates_snapshot.begin(), coordinates_snapshot.end(), c.x.data()) &&
        quadrature_snapshot == quadrature(m);
  }

  void build(ComMod& c, const mshType& m, const TopologyCache& topology) {
    nodes = m.eNoN; elements = m.nEl; points = m.nG;
    generation = topology.generation();
    graph = topology.assembly_graph();
    if (m.IEN.size()) ien_snapshot.assign(m.IEN.data(), m.IEN.data() + m.IEN.size());
    if (c.x.size()) coordinates_snapshot.assign(c.x.data(), c.x.data() + c.x.size());
    quadrature_snapshot = quadrature(m);
    const int width = 1 + 3*nodes + 9;
    ien = View2<int>("heatf_connectivity", elements, nodes);
    rhs_lids = View2<int>("heatf_rhs_lids", elements, nodes);
    matrix_offsets = View2<std::size_t>("heatf_crs_offsets", elements, nodes*nodes);
    geometry = View2<double>("heatf_geometry", elements, width);
    shapes = View2<double>("heatf_shapes", points, nodes);
    weights = decltype(weights)("heatf_weights", points);
    state = View2<double>("heatf_state", c.tnNo, kStateWidth);
    host_state = Kokkos::create_mirror_view(state);
    auto hi = Kokkos::create_mirror_view(ien);
    auto hr = Kokkos::create_mirror_view(rhs_lids);
    auto ho = Kokkos::create_mirror_view(matrix_offsets);
    auto hg = Kokkos::create_mirror_view(geometry);
    auto hn = Kokkos::create_mirror_view(shapes);
    auto hw = Kokkos::create_mirror_view(weights);
    for (int g = 0; g < points; ++g) {
      hw(g) = m.w(g);
      for (int a = 0; a < nodes; ++a) hn(g,a) = m.N(a,g);
    }
    // Same calls as construct_heatf: TET4 has lShpF, so geometry from g = 0.
    Array<double> x(3,nodes), nx(3,nodes), ksix(3,3);
    auto nxi = m.Nx.slice(0);
    for (int e = 0; e < elements; ++e) {
      for (int a = 0; a < nodes; ++a) {
        const int node = m.IEN(a,e);
        if (node < 0 || node >= c.tnNo) throw std::runtime_error("Invalid heatF connectivity.");
        hi(e,a) = node;
        for (int i = 0; i < 3; ++i) x(i,a) = c.x(i,node);
        hr(e,a) = topology.assembly_vector_lid(node,0);
        for (int b = 0; b < nodes; ++b) ho(e,a*nodes+b) = topology.assembly_block_offsets(node,m.IEN(b,e))[0];
      }
      double jac = 0.0;
      nn::gnn(nodes,3,3,nxi,x,nx,jac,ksix);
      if (!std::isfinite(jac) || jac <= 0.0) throw std::runtime_error("Invalid heatF element Jacobian.");
      hg(e,0) = jac;
      for (int k = 0; k < 3*nodes; ++k) hg(e,1+k) = nx.data()[k];
      for (int k = 0; k < 9; ++k) hg(e,1+3*nodes+k) = ksix.data()[k];
    }
    Kokkos::deep_copy(ien,hi); Kokkos::deep_copy(rhs_lids,hr);
    Kokkos::deep_copy(matrix_offsets,ho); Kokkos::deep_copy(geometry,hg);
    Kokkos::deep_copy(shapes,hn); Kokkos::deep_copy(weights,hw);
  }
};

struct Parameters { double dt, T1, amd, nu, s; };

// Port of heatf::heatf_3d. Operation order is kept identical to the host code
// so Serial results match the legacy loop to rounding.
template<int N>
struct AssembleHeatF {
  View2<int> ien, lids;
  View2<std::size_t> offsets;
  View2<double> geometry, shapes, state;
  Kokkos::View<double*, Device> weights;
  typename Tpetra_CrsMatrix::local_matrix_device_type matrix;
  RhsView rhs;
  Parameters prm;
  int points;
  bool atomics;

  KOKKOS_INLINE_FUNCTION void operator()(const int e) const {
    double ad[N], y0[N], y1[N], y2[N], yT[N];
    double residual[N] = {}, tangent[N*N] = {};
    for (int a = 0; a < N; ++a) {
      const int node = ien(e,a);
      ad[a] = state(node,0); y0[a] = state(node,1); y1[a] = state(node,2); y2[a] = state(node,3); yT[a] = state(node,4);
    }
    const double jac = geometry(e,0);
    const double* nx = &geometry(e,1);              // Nx(i,a) = nx[i+3a]
    const double* k = &geometry(e,1+3*N);           // ksix(i,j) = k[i+3j]
    const double kS = k[0]*k[0] + k[1]*k[1] + k[2]*k[2] + k[3]*k[3] +
                      k[4]*k[4] + k[5]*k[5] + k[6]*k[6] + k[7]*k[7] + k[8]*k[8];
    const double eps = Kokkos::Experimental::epsilon_v<double>;
    for (int g = 0; g < points; ++g) {
      const double w = weights(g)*jac;
      const double wl = w*prm.T1;
      const double* Ng = &shapes(g,0);
      double Td = -prm.s, u[3] = {0.0,0.0,0.0}, Tx[3] = {0.0,0.0,0.0};
      for (int a = 0; a < N; ++a) {
        u[0] = u[0] + Ng[a]*y0[a];
        u[1] = u[1] + Ng[a]*y1[a];
        u[2] = u[2] + Ng[a]*y2[a];
        Td = Td + Ng[a]*ad[a];
        Tx[0] = Tx[0] + nx[3*a]*yT[a];
        Tx[1] = Tx[1] + nx[3*a+1]*yT[a];
        Tx[2] = Tx[2] + nx[3*a+2]*yT[a];
      }
      const double kU = u[0]*u[0]*k[0] + u[1]*u[0]*k[1] + u[2]*u[0]*k[2] + u[0]*u[1]*k[3] +
                        u[1]*u[1]*k[4] + u[2]*u[1]*k[5] + u[0]*u[2]*k[6] + u[1]*u[2]*k[7] +
                        u[2]*u[2]*k[8];
      double nTx = k[0]*Tx[0]*Tx[0] + k[4]*Tx[1]*Tx[1] + k[8]*Tx[2]*Tx[2] +
                   (k[3] + k[1])*Tx[0]*Tx[1] + (k[6] + k[2])*Tx[0]*Tx[2] +
                   (k[7] + k[5])*Tx[1]*Tx[2];
      // utils::is_zero(nTx)
      { const double an = Kokkos::fabs(nTx); if (an/Kokkos::fmax(an,eps) < 10.0*eps) nTx = eps; }
      const double udTx = u[0]*Tx[0] + u[1]*Tx[1] + u[2]*Tx[2];
      double Tp = Kokkos::fabs(Td + udTx);
      const double nu = prm.nu + 0.5 * Tp / Kokkos::sqrt(nTx);
      const double tauM = 1.0 / Kokkos::sqrt((4.0/(prm.dt*prm.dt)) + 1.0*kU + 3.0*nu*nu*kS);
      Tp = -tauM*(Td + udTx);
      double udNx[N];
      for (int a = 0; a < N; ++a) udNx[a] = u[0]*nx[3*a] + u[1]*nx[3*a+1] + u[2]*nx[3*a+2];
      for (int a = 0; a < N; ++a) {
        residual[a] = residual[a] + w*(Ng[a]*(Td + udTx) + (nx[3*a]*Tx[0] + nx[3*a+1]*Tx[1] + nx[3*a+2]*Tx[2])*nu - udNx[a]*Tp);
        for (int b = 0; b < N; ++b) {
          tangent[a*N+b] = tangent[a*N+b] + wl*(nu*(nx[3*a]*nx[3*b] + nx[3*a+1]*nx[3*b+1] + nx[3*a+2]*nx[3*b+2]) +
                           (Ng[a] + tauM*udNx[a])*(Ng[b]*prm.amd + udNx[b]));
        }
      }
    }
    for (int a = 0; a < N; ++a) {
      if (atomics) Kokkos::atomic_add(&rhs(lids(e,a),0), residual[a]);
      else rhs(lids(e,a),0) += residual[a];
      for (int b = 0; b < N; ++b) {
        const auto offset = offsets(e,a*N+b);
        if (atomics) Kokkos::atomic_add(&matrix.values(offset), tangent[a*N+b]);
        else matrix.values(offset) += tangent[a*N+b];
      }
    }
  }
};

void launch(MeshCache& cache, Trilinos& t, const Parameters& prm) {
  const auto matrix = t.local_assembly.device_assembly_matrix()->getLocalMatrixDevice();
  const auto rhs = t.ghostF->getLocalViewDevice(Tpetra::Access::ReadWrite);
  AssembleHeatF<4> kernel{cache.ien,cache.rhs_lids,cache.matrix_offsets,cache.geometry,cache.shapes,cache.state,
      cache.weights,matrix,rhs,prm,cache.points,Exec().concurrency()>1};
  Kokkos::parallel_for("svmp_heatf_compute_and_scatter",Kokkos::RangePolicy<Exec>(0,cache.elements),kernel);
  Kokkos::fence("svmp_heatf_assembly_complete");
}
} // namespace

class HeatFAssembly::Impl { public: std::map<const mshType*,MeshCache> caches; };
HeatFAssembly::HeatFAssembly() : impl_(new Impl) {}
HeatFAssembly::~HeatFAssembly() = default;
void HeatFAssembly::clear() { impl_->caches.clear(); }

bool HeatFAssembly::assemble(ComMod& c, const mshType& m, const SolutionStates& solution, Trilinos& t) {
  const std::string mode = std::getenv("SV_TRILINOS_ASSEMBLY") ? std::getenv("SV_TRILINOS_ASSEMBLY") : "auto";
  if (mode != "auto" && mode != "legacy" && mode != "kokkos")
    throw std::runtime_error("SV_TRILINOS_ASSEMBLY must be auto, legacy, or kokkos.");
  if (mode == "legacy") return false;
  const auto& eq = c.eq[c.cEq];
  const bool supported = c.nsd == 3 && c.dof == 1 && t.topology.dof() == 1 && eq.s >= 3 && c.tDof > eq.s &&
      m.eNoN == 4 && m.eType == consts::ElementType::TET4 && m.lShpF && m.nG > 0 &&
      !c.mvMsh && eq.nDmn == 1 && eq.dmn.size() == 1 && eq.dmn[0].Id == -1 &&
      eq.dmn[0].phys == consts::EquationType::phys_heatF;
  if (!supported) {
    if (mode == "kokkos") throw std::runtime_error("Kokkos heatF assembly requires a 3D TET4 mesh, one domain, and no moving mesh.");
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
  const int s = eq.s;
  if (c.tnNo>0 && (a.nrows()<=s || y.nrows()<=s || a.ncols()<c.tnNo || y.ncols()<c.tnNo))
    throw std::runtime_error("HeatF state dimensions do not match the local node layout.");
  for (int node = 0; node < c.tnNo; ++node) {
    cache.host_state(node,0) = a(s,node);
    for (int i = 0; i < 3; ++i) cache.host_state(node,1+i) = y(i,node);
    cache.host_state(node,4) = y(s,node);
  }
  Kokkos::deep_copy(cache.state,cache.host_state);
  const double upload = seconds(upload_start);
  const auto kernel_start = Clock::now();
  const auto& dmn = eq.dmn[0];
  Parameters prm;
  prm.dt = c.dt;
  prm.T1 = eq.af * eq.gam * c.dt;
  prm.amd = eq.am / prm.T1;
  prm.nu = dmn.prop.at(consts::PhysicalPropertyType::conductivity);
  prm.s = dmn.prop.at(consts::PhysicalPropertyType::source_term);
  launch(cache,t,prm);
  c.cDmn = 0;
  const double kernel = seconds(kernel_start);
  if (std::getenv("SV_TRILINOS_ASSEMBLY_STATS")) {
    const double local[] = {seconds(start),setup,upload,kernel};
    double global[4];
    Teuchos::reduceAll(*t.comm,Teuchos::REDUCE_MAX,4,local,global);
    if (t.comm->getRank() == 0) std::cout << "ASSEMBLY_STATS backend=kokkos-heatf execution_space=" << Exec::name() << " total_s=" << global[0]
        << " cache_s=" << global[1] << " state_s=" << global[2] << " kernel_s=" << global[3]
        << " rebuilt=" << rebuild << " nodes_per_element=" << m.eNoN << std::endl;
  }
  return true;
}
} // namespace trilinos_backend
#endif
