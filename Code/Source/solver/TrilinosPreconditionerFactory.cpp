// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "TrilinosPreconditionerFactory.h"
#include "TrilinosSolverOptions.h"

/**
 * @file TrilinosPreconditionerFactory.cpp
 * @brief Implements reusable Ifpack2, MueLu, and resistance preconditioners.
 */

#ifdef WITH_TRILINOS

#include "Teuchos_CommHelpers.hpp"
#include "TpetraExt_MatrixMatrix.hpp"
#include "KokkosKernels_Handle.hpp"
#include "KokkosSparse_spgemm.hpp"

#include <cmath>
#include <algorithm>
#include <limits>

#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include "Teuchos_FancyOStream.hpp"

namespace trilinos_bipartition {
namespace preconditioners {

namespace {

bool same_local_map(const Tpetra_Map& a,const Tpetra_Map& b) {
  const auto x=a.getLocalElementList(),y=b.getLocalElementList();
  return a.getGlobalNumElements()==b.getGlobalNumElements() &&
      a.getIndexBase()==b.getIndexBase() && x.size()==y.size() &&
      std::equal(x.begin(),x.end(),y.begin());
}

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

// Scale Jacobi damping using a fine-level Gershgorin spectral bound instead
// of a mesh-specific constant (a fixed 0.8 diverged in the CUDA experiment).
// The V-cycle uses matching pre/post sweeps for pressure CG.
double schur_spectral_bound(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix)
{
  Tpetra_Vector diagonal(matrix->getRowMap());
  matrix->getLocalDiagCopy(diagonal);
  double local_bound = 0.0;
  {
    const auto d = diagonal.getLocalViewDevice(Tpetra::Access::ReadOnly);
    const auto a = matrix->getLocalMatrixDevice();
    const double maximum = std::numeric_limits<double>::max();
    const double invalid = std::numeric_limits<double>::infinity();
    Kokkos::parallel_reduce("schur_amg_spectral_bound",
        Kokkos::RangePolicy<typename Node::execution_space>(0,matrix->getLocalNumRows()),
        KOKKOS_LAMBDA(const int row, double& bound) {
          double sum = 0.0;
          for (auto k=a.graph.row_map(row); k<a.graph.row_map(row+1); ++k) sum += fabs(a.values(k));
          double ratio = sum/d(row,0);
          if (!(d(row,0)>0.0 && d(row,0)<=maximum && ratio>=0.0 && ratio<=maximum)) ratio=invalid;
          if (ratio>bound) bound=ratio;
        },Kokkos::Max<double>(local_bound));
  }
  local_bound = std::max(1.0,local_bound);
  double bound = 0.0;
  const auto comm = matrix->getRowMap()->getComm();
  Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,1,&local_bound,&bound);
  if (!std::isfinite(bound)) {
    throw std::runtime_error("[SchurAMG] Expected a finite positive Schur diagonal.");
  }

  return bound;
}

// Schur AMG options; each is a run-wide environment choice. The signature
// invalidates a cached hierarchy if any choice differs from the one it used.
struct SchurAMGOptions {
  std::string coarse,smoother,damping;
  int sweeps,levels,coarse_size,aggregate_min,aggregate_max;
  double drop_tolerance;
  std::string signature() const {
    return coarse+"/"+smoother+"/"+damping+"/"+std::to_string(sweeps)+"/"+std::to_string(drop_tolerance)+
        "/"+std::to_string(levels)+"/"+std::to_string(coarse_size)+"/"+std::to_string(aggregate_min)+
        "-"+std::to_string(aggregate_max);
  }
};

SchurAMGOptions schur_amg_options() {
  SchurAMGOptions options;
  options.coarse=trilinos_options::choice("SV_TRILINOS_SCHUR_COARSE","klu",
      {"klu","chebyshev","tacho","cusolver"});
  // Pressure CG needs a symmetric V-cycle: only symmetric smoothers are offered.
  options.smoother=trilinos_options::choice("SV_TRILINOS_SCHUR_SMOOTHER","jacobi",
      {"jacobi","chebyshev","symmetric-gauss-seidel","mt-symmetric-gauss-seidel",
       "two-stage-symmetric-gauss-seidel"});
  options.damping=trilinos_options::choice("SV_TRILINOS_SCHUR_DAMPING","gershgorin",{"gershgorin","power"});
  options.sweeps=trilinos_options::positive_integer("SV_TRILINOS_SCHUR_SWEEPS",2);
  options.levels=trilinos_options::positive_integer("SV_TRILINOS_SCHUR_LEVELS",10);
  options.coarse_size=trilinos_options::positive_integer("SV_TRILINOS_SCHUR_COARSE_SIZE",200);
  options.aggregate_min=trilinos_options::positive_integer("SV_TRILINOS_SCHUR_AGG_MIN",3);
  options.aggregate_max=trilinos_options::positive_integer("SV_TRILINOS_SCHUR_AGG_MAX",12);
  if (options.levels<2 || options.aggregate_max<options.aggregate_min)
    throw std::runtime_error("Schur AMG needs at least two levels and AGG_MAX >= AGG_MIN");
  const char* drop=std::getenv("SV_TRILINOS_SCHUR_DROP_TOL");
  options.drop_tolerance=drop && *drop ? std::stod(drop) : 0.0;
  if (!(options.drop_tolerance>=0.0 && options.drop_tolerance<1.0))
    throw std::runtime_error("SV_TRILINOS_SCHUR_DROP_TOL must be in [0, 1)");
  return options;
}

// Lower estimate of lambda_max(D^-1 S) from a D-weighted Rayleigh quotient of
// power iteration, raised by 10% and capped by the Gershgorin bound (always an
// upper bound). The eigenvector estimate is kept to warm-start later Jacobians.
double schur_power_bound(const Teuchos::RCP<Tpetra_CrsMatrix>& matrix,
    Teuchos::RCP<Tpetra_Vector>& state,double gershgorin)
{
  const auto map=matrix->getRangeMap();
  if (!matrix->getRowMap()->isSameAs(*map) || !matrix->getDomainMap()->isSameAs(*map))
    return gershgorin;
  Tpetra_Vector diagonal(map),inverse(map),product(map),weighted(map);
  matrix->getLocalDiagCopy(diagonal);
  inverse.reciprocal(diagonal);
  const bool warm=state!=Teuchos::null && state->getMap()->isSameAs(*map);
  if (!warm) {
    state=Teuchos::rcp(new Tpetra_Vector(map));
    const auto x=state->getLocalViewDevice(Tpetra::Access::OverwriteAll);
    const auto local=map->getLocalMap();
    // Partition-independent, non-constant start (constants are near the bottom mode).
    Kokkos::parallel_for("schur_power_start",
      Kokkos::RangePolicy<typename Node::execution_space>(0,x.extent(0)),
      KOKKOS_LAMBDA(const int i) { x(i,0)=1.0+0.5*static_cast<double>((local.getGlobalElement(i)*7919)%17)/17.0; });
  }
  double estimate=0.0;
  for (int iteration=0,count=warm ? 6 : 20;iteration<count;++iteration) {
    matrix->apply(*state,product);
    weighted.elementWiseMultiply(1.0,diagonal,*state,0.0);
    const double denominator=state->dot(weighted);
    if (!(denominator>0.0)) return gershgorin;
    estimate=state->dot(product)/denominator;
    state->elementWiseMultiply(1.0,inverse,product,0.0);
    const double norm=state->norm2();
    if (!(norm>0.0) || !std::isfinite(norm)) return gershgorin;
    state->scale(1.0/norm);
  }
  return std::isfinite(estimate) && estimate>0.0 ? std::min(gershgorin,1.1*estimate) : gershgorin;
}

Teuchos::RCP<Tpetra_Operator> create_schur_amg(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix, double bound, bool reuse_transfers,
    const SchurAMGOptions& options)
{
  Teuchos::ParameterList parameters;
  parameters.set("verbosity", "none");
  parameters.set("max levels", options.levels);
  parameters.set("cycle type", "V");
  parameters.set("number of equations", 1);
  parameters.set("multigrid algorithm", "unsmoothed");
  parameters.set("aggregation: type", "uncoupled");
  parameters.set("aggregation: min agg size", options.aggregate_min);
  parameters.set("aggregation: max agg size", options.aggregate_max);
  if (options.drop_tolerance>0.0) parameters.set("aggregation: drop tol", options.drop_tolerance);
  parameters.set("smoother: pre or post", "both");
  // Jacobi uses 1.8/bound on every level; Chebyshev estimates each level itself.
  trilinos_options::set_smoother(parameters,options.smoother,options.sweeps,
      options.smoother=="jacobi" ? 1.8/bound : 0.0);
  if (options.coarse=="chebyshev") {
    parameters.set("coarse: type", "CHEBYSHEV");
    auto& polynomial=parameters.sublist("coarse: params");
    // For an SPD coarse matrix, D^-1 A has positive eigenvalues and
    // trace equal to its dimension. The coarse cap bounds lambda_max by
    // 200, avoiding random power estimates and preserving scaling on reuse.
    polynomial.set("chebyshev: max eigenvalue", 200.0);
    polynomial.set("chebyshev: degree", 8);
    polynomial.set("chebyshev: ratio eigenvalue", 30.0);
    polynomial.set("chebyshev: zero starting solution", true);
  } else {
    trilinos_options::set_direct_coarse(parameters,options.coarse);
  }
  parameters.set("coarse: max size", options.coarse_size);
  // RP retains transfers, but recomputes every coarse matrix and smoother.
  // RAP would retain coarse numerical matrices and is unsuitable here.
  parameters.set("reuse: type", reuse_transfers ? "RP" : "none");
  auto result=MueLu::CreateTpetraPreconditioner(
      Teuchos::rcp_static_cast<Tpetra_Operator>(matrix), parameters);
  if (options.coarse=="chebyshev") {
    using Matrix=Xpetra::Matrix<double,LO,GO,Node>;
    auto hierarchy=result->GetHierarchy();
    const auto coarse_matrix=hierarchy->GetLevel(hierarchy->GetNumLevels()-1)
        ->Get<Teuchos::RCP<Matrix>>("A");
    if (coarse_matrix->getGlobalNumRows()>200)
      throw std::runtime_error("Polynomial Schur coarse solver requires at most 200 coarse rows "
          "(lower SV_TRILINOS_SCHUR_COARSE_SIZE or raise SV_TRILINOS_SCHUR_LEVELS)");
  }
  if (std::getenv("SV_TRILINOS_AMG_DETAILS")) {
    std::cout << "[SchurAMG] options " << options.signature() << " jacobi_damping="
              << (options.smoother=="jacobi" ? 1.8/bound : 0.0) << std::endl;
    auto stream=Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    result->GetHierarchy()->describe(*stream,Teuchos::VERB_HIGH);
  }
  return result;
}

} // namespace

Teuchos::RCP<Tpetra_CrsMatrix> build_schur_preconditioning_matrix(
    const Teuchos::RCP<Tpetra_CrsMatrix>& L,
    const Teuchos::RCP<Tpetra_CrsMatrix>& B)
{
  validate_matrix(consts::PreconditionerType::PREC_TRILINOS_SCHUR_AMG, L);
  if (B == Teuchos::null || !B->isFillComplete() ||
      !B->getDomainMap()->isSameAs(*L->getDomainMap())) {
    throw std::runtime_error("[SchurAMG] Incompatible pressure maps of L and B.");
  }
  auto product = Teuchos::rcp(new Tpetra_CrsMatrix(L->getRowMap(), 0));
  Tpetra::MatrixMatrix::Multiply(*B, true, *B, false, *product);
  return Tpetra::MatrixMatrix::add(1.0, false, *L, 1.0, false,
      *product, L->getDomainMap(), L->getRangeMap());
}

// The block topology cache supplies stable input graphs. Numeric SpGEMM
// reuses the product graph; the union graph and scatter plans are built once.
class SchurMatrixStorage {
public:
  using LocalMatrix=Tpetra_CrsMatrix::local_matrix_device_type;
  using Device=typename LocalMatrix::device_type;
  using Plan=Kokkos::View<std::size_t*,Device>;
  using Offset=typename LocalMatrix::row_map_type::non_const_value_type;
  using Ordinal=typename LocalMatrix::index_type::non_const_value_type;
  using Value=typename LocalMatrix::values_type::non_const_value_type;
  using Handle=KokkosKernels::Experimental::KokkosKernelsHandle<Offset,Ordinal,Value,
      typename Device::execution_space,typename Device::memory_space,typename Device::memory_space>;
  std::size_t generation=0;
  Teuchos::RCP<const Tpetra_CrsGraph> L_graph,B_graph;
  Teuchos::RCP<Tpetra_CrsMatrix> product,schur;
  Plan L_plan,product_plan;

  // Single-rank cached product C = B^T B. B^T's structure (with the B value
  // index of every entry) and the SpGEMM symbolic phase are computed once;
  // a Jacobian then gathers B^T values and runs only the numeric phase.
  // Tpetra's Multiply instead re-forms the transpose matrix on every call.
  // MPI runs keep Tpetra's distributed product, which includes off-rank rows.
  bool cached=false;
  std::unique_ptr<Handle> handle;
  Ordinal pressure_columns=0,velocity_rows=0;
  Kokkos::View<Offset*,Device> bt_rows,c_rows;
  Kokkos::View<Ordinal*,Device> bt_entries,c_entries;
  Kokkos::View<Offset*,Device> bt_source;
  Kokkos::View<Value*,Device> bt_values,c_values;
  Plan c_plan;

  void cached_product(const Tpetra_CrsMatrix& B) {
    const auto b=B.getLocalMatrixDevice();
    const auto source=bt_source;
    const auto target=bt_values;
    Kokkos::parallel_for("schur_transpose_values",
      Kokkos::RangePolicy<typename Node::execution_space>(0,bt_values.extent(0)),
      KOKKOS_LAMBDA(const int k) { target(k)=b.values(source(k)); });
    KokkosSparse::spgemm_numeric(handle.get(),pressure_columns,velocity_rows,pressure_columns,
        bt_rows,bt_entries,bt_values,false,b.graph.row_map,b.graph.entries,b.values,false,
        c_rows,c_entries,c_values);
  }

  static constexpr std::size_t unmapped_entry=~std::size_t(0);

  // Count product entries outside the S0 graph whose current value is nonzero.
  int unmapped_nonzero() const {
    const auto plan=c_plan;
    const auto values=c_values;
    const std::size_t unmapped=unmapped_entry;  // device lambdas capture locals only
    int count=0;
    Kokkos::parallel_reduce("schur_cached_product_unmapped",
      Kokkos::RangePolicy<typename Node::execution_space>(0,c_plan.extent(0)),
      KOKKOS_LAMBDA(const int k,int& n) { if (plan(k)==unmapped && values(k)!=0.0) ++n; },count);
    return count;
  }

  // Returns false if S0's graph no longer holds the product; S0 must be rebuilt.
  bool refresh_cached(const Tpetra_CrsMatrix& L,const Tpetra_CrsMatrix& B) {
    cached_product(B);
    if (unmapped_nonzero()) return false;
    schur->setAllToScalar(0.0);
    add_values(L,*schur,L_plan);
    const auto s=schur->getLocalMatrixDevice();
    const auto plan=c_plan;
    const auto values=c_values;
    const std::size_t unmapped=unmapped_entry;
    Kokkos::parallel_for("schur_add_cached_product",
      Kokkos::RangePolicy<typename Node::execution_space>(0,c_plan.extent(0)),
      KOKKOS_LAMBDA(const int k) { if (plan(k)!=unmapped) s.values(plan(k))+=values(k); });
    return true;
  }

  // Returns false (Tpetra product retained) unless the cached product is
  // valid and reproduces the Tpetra-built S0 that this call compares it with.
  static bool decline(const char* reason) {
    if (std::getenv("SV_TRILINOS_AMG_DETAILS"))
      std::cout << "[SchurAMG] cached B^T B product not used: " << reason << std::endl;
    return false;
  }

  bool enable_cached(const Tpetra_CrsMatrix& L,const Tpetra_CrsMatrix& B) {
    const char* policy=std::getenv("SV_TRILINOS_SCHUR_PRODUCT");
    const std::string choice=policy ? policy : "auto";
    if (choice!="auto" && choice!="cached" && choice!="tpetra")
      throw std::runtime_error("SV_TRILINOS_SCHUR_PRODUCT must be auto, cached or tpetra");
    if (choice=="tpetra") return decline("SV_TRILINOS_SCHUR_PRODUCT=tpetra");
    // On host execution spaces Tpetra's Gustavson reuse kernel is faster than
    // KokkosKernels SpGEMM (iliac, one M4 core: 16 vs 22 ms per Jacobian). On
    // devices Tpetra re-forms the transpose matrix on every call instead.
    if (choice=="auto" &&
        Kokkos::SpaceAccessibility<Kokkos::HostSpace,typename Device::memory_space>::accessible)
      return decline("host execution space (SV_TRILINOS_SCHUR_PRODUCT=cached forces it)");
    if (B.getComm()->getSize()!=1) return decline("more than one MPI rank");

    // Host transpose of B's graph, ordered by velocity row within each column.
    // The host view is scoped so no host and device views of B coexist.
    {
    const auto bh=B.getLocalMatrixHost();
    velocity_rows=static_cast<Ordinal>(bh.numRows());
    pressure_columns=static_cast<Ordinal>(bh.numCols());
    auto rows_h=Kokkos::View<Offset*,Kokkos::HostSpace>("schur_bt_rows",pressure_columns+1);
    for (Offset k=0;k<bh.graph.entries.extent(0);++k) ++rows_h(bh.graph.entries(k)+1);
    for (Ordinal i=0;i<pressure_columns;++i) rows_h(i+1)+=rows_h(i);
    auto entries_h=Kokkos::View<Ordinal*,Kokkos::HostSpace>("schur_bt_entries",bh.graph.entries.extent(0));
    auto source_h=Kokkos::View<Offset*,Kokkos::HostSpace>("schur_bt_source",bh.graph.entries.extent(0));
    std::vector<Offset> next(rows_h.data(),rows_h.data()+pressure_columns);
    for (Ordinal row=0;row<velocity_rows;++row)
      for (Offset k=bh.graph.row_map(row);k<bh.graph.row_map(row+1);++k) {
        const Offset position=next[bh.graph.entries(k)]++;
        entries_h(position)=row;
        source_h(position)=k;
      }
    bt_rows=Kokkos::create_mirror_view_and_copy(typename Device::memory_space(),rows_h);
    bt_entries=Kokkos::create_mirror_view_and_copy(typename Device::memory_space(),entries_h);
    bt_source=Kokkos::create_mirror_view_and_copy(typename Device::memory_space(),source_h);
    }
    bt_values=Kokkos::View<Value*,Device>("schur_bt_values",bt_entries.extent(0));

    const auto b=B.getLocalMatrixDevice();
    handle=std::make_unique<Handle>();
    const char* algorithm_name=std::getenv("SV_TRILINOS_SCHUR_SPGEMM");
    const std::string algorithm=algorithm_name ? algorithm_name : "default";
    if (algorithm=="default") handle->create_spgemm_handle();
    else if (algorithm=="kk") handle->create_spgemm_handle(KokkosSparse::SPGEMM_KK);
    else if (algorithm=="dense") handle->create_spgemm_handle(KokkosSparse::SPGEMM_KK_DENSE);
    else if (algorithm=="memory") handle->create_spgemm_handle(KokkosSparse::SPGEMM_KK_MEMORY);
    else if (algorithm=="lp") handle->create_spgemm_handle(KokkosSparse::SPGEMM_KK_LP);
    else throw std::runtime_error("SV_TRILINOS_SCHUR_SPGEMM must be default, kk, dense, memory or lp");
    c_rows=Kokkos::View<Offset*,Device>("schur_c_rows",pressure_columns+1);
    KokkosSparse::spgemm_symbolic(handle.get(),pressure_columns,velocity_rows,pressure_columns,
        bt_rows,bt_entries,false,b.graph.row_map,b.graph.entries,false,c_rows);
    const auto c_nnz=handle->get_spgemm_handle()->get_c_nnz();
    c_entries=Kokkos::View<Ordinal*,Device>("schur_c_entries",c_nnz);
    c_values=Kokkos::View<Value*,Device>("schur_c_values",c_nnz);
    cached_product(B);

    // Map each product entry to its S0 offset. One rank owns every pressure
    // GID, so each B column GID is an S0 row and an S0 column.
    auto row_h=Kokkos::View<Ordinal*,Kokkos::HostSpace>("schur_c_row_lid",pressure_columns);
    auto col_h=Kokkos::View<Ordinal*,Kokkos::HostSpace>("schur_c_col_lid",pressure_columns);
    const auto invalid=Teuchos::OrdinalTraits<LO>::invalid();
    for (Ordinal i=0;i<pressure_columns;++i) {
      const GO gid=B.getColMap()->getGlobalElement(i);
      row_h(i)=schur->getRowMap()->getLocalElement(gid);
      col_h(i)=schur->getColMap()->getLocalElement(gid);
      if (row_h(i)==invalid || col_h(i)==invalid) return decline("a B column GID is not an S0 row/column");
    }
    const auto row_lid=Kokkos::create_mirror_view_and_copy(typename Device::memory_space(),row_h);
    const auto col_lid=Kokkos::create_mirror_view_and_copy(typename Device::memory_space(),col_h);
    c_plan=Plan("schur_c_plan",c_nnz);
    const auto s=schur->getLocalMatrixDevice();
    const auto crow=c_rows;
    const auto cent=c_entries;
    const auto plan=c_plan;
    const auto unmapped=unmapped_entry;
    int missing=0;
    Kokkos::parallel_reduce("schur_cached_product_plan",
      Kokkos::RangePolicy<typename Node::execution_space>(0,pressure_columns),
      KOKKOS_LAMBDA(const int i,int& count) {
        const auto row=row_lid(i);
        for (auto k=crow(i);k<crow(i+1);++k) {
          const auto col=col_lid(cent(k));
          auto j=s.graph.row_map(row);
          while (j<s.graph.row_map(row+1) && s.graph.entries(j)!=col) ++j;
          if (j==s.graph.row_map(row+1)) { plan(k)=unmapped; ++count; } else plan(k)=j;
        }
      },missing);
    // Tpetra omits product entries that are numerically zero when S0 is first
    // built (B rows of Dirichlet-scaled velocity DOFs). They must stay zero;
    // refresh_cached() verifies this for every Jacobian.
    if (missing && unmapped_nonzero()) return decline("a nonzero product entry is missing from the S0 graph");
    if (std::getenv("SV_TRILINOS_AMG_DETAILS"))
      std::cout << "[SchurAMG] cached B^T B product: " << c_nnz << " entries, " << missing
                << " structurally zero entries outside S0" << std::endl;

    // Verify against the Tpetra-built S0 currently held in schur.
    Kokkos::View<Value*,Device> reference("schur_reference",s.values.extent(0));
    Kokkos::deep_copy(reference,s.values);
    if (!refresh_cached(L,B)) return decline("product check failed");
    double difference=0.0,scale=0.0;
    const auto current=schur->getLocalMatrixDevice().values;
    Kokkos::parallel_reduce("schur_cached_product_difference",
      Kokkos::RangePolicy<typename Node::execution_space>(0,current.extent(0)),
      KOKKOS_LAMBDA(const int k,double& value) {
        const double d=Kokkos::fabs(current(k)-reference(k)); if (d>value) value=d; },
      Kokkos::Max<double>(difference));
    Kokkos::parallel_reduce("schur_cached_product_scale",
      Kokkos::RangePolicy<typename Node::execution_space>(0,current.extent(0)),
      KOKKOS_LAMBDA(const int k,double& value) {
        const double d=Kokkos::fabs(reference(k)); if (d>value) value=d; },
      Kokkos::Max<double>(scale));
    if (!(difference<=1.e-12*std::max(scale,1.0))) {
      Kokkos::deep_copy(schur->getLocalMatrixDevice().values,reference);
      std::cerr << "[SchurAMG] WARNING: cached B^T B differs from Tpetra by " << difference
                << "; keeping the Tpetra product." << std::endl;
      return false;
    }
    return true;
  }

  static Plan plan(const Tpetra_CrsMatrix& source,const Tpetra_CrsMatrix& target) {
    const auto a=source.getCrsGraph()->getLocalGraphHost();
    const auto b=target.getCrsGraph()->getLocalGraphHost();
    Plan result("schur_value_offsets",a.entries.extent(0));
    auto host=Kokkos::create_mirror_view(result);
    for (std::size_t row=0;row<source.getLocalNumRows();++row) {
      for (auto k=a.row_map(row);k<a.row_map(row+1);++k) {
        const GO gid=source.getColMap()->getGlobalElement(a.entries(k));
        const LO col=target.getColMap()->getLocalElement(gid);
        auto j=b.row_map(row);
        while (j<b.row_map(row+1) && b.entries(j)!=col) ++j;
        if (j==b.row_map(row+1)) throw std::runtime_error("Schur union graph missing an entry");
        host(k)=j;
      }
    }
    Kokkos::deep_copy(result,host);
    return result;
  }
  static void add_values(const Tpetra_CrsMatrix& source,Tpetra_CrsMatrix& target,const Plan& offsets) {
    const auto a=source.getLocalMatrixDevice();
    const auto b=target.getLocalMatrixDevice();
    Kokkos::parallel_for("schur_refresh_values",Kokkos::RangePolicy<typename Node::execution_space>(0,offsets.extent(0)),
      KOKKOS_LAMBDA(const int i) { b.values(offsets(i))+=a.values(i); });
  }
};

Teuchos::RCP<Tpetra_CrsMatrix> MueLuReuseCache::schur_matrix(
    const Teuchos::RCP<Tpetra_CrsMatrix>& L,
    const Teuchos::RCP<Tpetra_CrsMatrix>& B,std::size_t generation) {
  validate_matrix(consts::PreconditionerType::PREC_TRILINOS_SCHUR_AMG,L);
  if (B==Teuchos::null || !B->isFillComplete() || !B->getDomainMap()->isSameAs(*L->getDomainMap()))
    throw std::runtime_error("Schur matrix: incompatible coupling block");
  int changed=!schur_storage_ || schur_storage_->generation!=generation ||
    schur_storage_->L_graph.getRawPtr()!=L->getCrsGraph().getRawPtr() ||
    schur_storage_->B_graph.getRawPtr()!=B->getCrsGraph().getRawPtr();
  int global_changed=0;
  Teuchos::reduceAll(*L->getComm(),Teuchos::REDUCE_MAX,1,&changed,&global_changed);
  if (global_changed) {
    auto storage=std::make_shared<SchurMatrixStorage>();
    storage->generation=generation;
    storage->L_graph=L->getCrsGraph();
    storage->B_graph=B->getCrsGraph();
    storage->product=Teuchos::rcp(new Tpetra_CrsMatrix(L->getRowMap(),0));
    Tpetra::MatrixMatrix::Multiply(*B,true,*B,false,*storage->product);
    storage->schur=Tpetra::MatrixMatrix::add(1.0,false,*L,1.0,false,
      *storage->product,L->getDomainMap(),L->getRangeMap());
    storage->L_plan=SchurMatrixStorage::plan(*L,*storage->schur);
    storage->product_plan=SchurMatrixStorage::plan(*storage->product,*storage->schur);
    storage->cached=storage->enable_cached(*L,*B);
    schur_storage_=storage;
  } else if (schur_storage_->cached) {
    if (!schur_storage_->refresh_cached(*L,*B)) {
      // A new nonzero lies outside S0's graph: rebuild S0 (and, through the
      // graph check, the AMG hierarchy) from the current matrices.
      schur_storage_.reset();
      return schur_matrix(L,B,generation);
    }
  } else {
    // Multiply overwrites the product values on its existing graph.
    Tpetra::MatrixMatrix::Multiply(*B,true,*B,false,*schur_storage_->product);
    schur_storage_->schur->setAllToScalar(0.0);
    SchurMatrixStorage::add_values(*L,*schur_storage_->schur,schur_storage_->L_plan);
    SchurMatrixStorage::add_values(*schur_storage_->product,*schur_storage_->schur,schur_storage_->product_plan);
  }
  return schur_storage_->schur;
}

bool MueLuReuseCache::matches(
    SolverRole role,
    const Tpetra_CrsMatrix& matrix,
    const PreconditionerReuseContext& reuse) const
{
  return !schur_ && initialized_ && hierarchy_ != Teuchos::null && role_ == role &&
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
  schur_=false;
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
  schur_storage_.reset();
  power_vector_=Teuchos::null;
  skip_count_=0;
  schur_=false;
  damping_bound_=0.0;
  schur_graph_=Teuchos::null;
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

bool MueLuReuseCache::schur_product_cached() const
{
  return schur_storage_ && schur_storage_->cached;
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

std::size_t MueLuReuseCache::skip_count() const
{
  return skip_count_;
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
  if (type == consts::PreconditionerType::PREC_TRILINOS_SCHUR_AMG) {
    if (role != SolverRole::pressure_cg) {
      throw std::runtime_error("[SchurAMG] Only valid for pressure CG.");
    }
    handle->setup_matrix_ = matrix;
    const std::string policy=std::getenv("SV_TRILINOS_SCHUR_REUSE") ? std::getenv("SV_TRILINOS_SCHUR_REUSE") : "transfers";
    if (policy!="transfers" && policy!="fresh")
      throw std::runtime_error("SV_TRILINOS_SCHUR_REUSE must be transfers or fresh");
    const auto options=schur_amg_options();
    const auto coarse_policy=options.signature();
    double bound=schur_spectral_bound(matrix);
    if (options.smoother=="jacobi" && options.damping=="power") {
      Teuchos::RCP<Tpetra_Vector> scratch;
      bound=schur_power_bound(matrix,reuse.muelu_cache!=Teuchos::null ?
          reuse.muelu_cache->power_vector_ : scratch,bound);
    }
    if (policy=="fresh" || reuse.muelu_cache==Teuchos::null) {
      handle->muelu_operator_=create_schur_amg(matrix,bound*1.05,false,options);
    } else {
      if (reuse.topology_generation==0)
        throw std::runtime_error("Schur reuse requires a valid topology generation");
      auto& cache=*reuse.muelu_cache;
      // Check complete local CSR and maps on device, not just dimensions/nnz.
      // Rebuild collectively if any rank detects a changed graph or unsafe
      // damping bound. Transfer reuse is valid across time steps on fixed mesh.
      int mismatch=1;
      if (cache.schur_ && cache.coarse_policy_==coarse_policy && cache.hierarchy_!=Teuchos::null &&
          cache.topology_generation_==reuse.topology_generation &&
          cache.schur_graph_!=Teuchos::null && bound<=cache.damping_bound_ &&
          same_local_map(*matrix->getDomainMap(),*cache.domain_map_) &&
          same_local_map(*matrix->getRangeMap(),*cache.range_map_) &&
          same_local_map(*matrix->getColMap(),*cache.schur_graph_->getColMap())) {
        const auto old_graph=cache.schur_graph_->getLocalGraphDevice();
        const auto graph=matrix->getCrsGraph()->getLocalGraphDevice();
        if (graph.entries.extent(0)==old_graph.entries.extent(0) &&
            graph.row_map.extent(0)==old_graph.row_map.extent(0)) {
          mismatch=0;
          int entries_changed=0,rows_changed=0;
          Kokkos::parallel_reduce("schur_graph_entries_match",Kokkos::RangePolicy<typename Node::execution_space>(0,graph.entries.extent(0)),
            KOKKOS_LAMBDA(const int i,int& count) { if (graph.entries(i)!=old_graph.entries(i)) ++count; },entries_changed);
          Kokkos::parallel_reduce("schur_graph_rows_match",Kokkos::RangePolicy<typename Node::execution_space>(0,graph.row_map.extent(0)),
            KOKKOS_LAMBDA(const int i,int& count) { if (graph.row_map(i)!=old_graph.row_map(i)) ++count; },rows_changed);
          mismatch=(entries_changed!=0 || rows_changed!=0);
        }
      }
      int global_mismatch=0;
      Teuchos::reduceAll(*matrix->getComm(),Teuchos::REDUCE_MAX,1,&mismatch,&global_mismatch);
      if (global_mismatch) {
        cache.hierarchy_=Teuchos::null;
        cache.coarse_policy_=coarse_policy;
        cache.damping_bound_=bound*1.05;
        cache.hierarchy_=create_schur_amg(matrix,cache.damping_bound_,true,options);
        cache.store_signature(role,matrix,reuse);
        cache.schur_=true;
        cache.schur_graph_=matrix->getCrsGraph();
        ++cache.build_count_;
      } else if (trilinos_options::refresh_once_per_step() && cache.time_step_==reuse.time_step) {
        // Keep the hierarchy refreshed earlier in this time step. CG still uses
        // the current Schur operator; only its preconditioner is older.
        ++cache.skip_count_;
      } else {
        auto muelu=Teuchos::rcp_dynamic_cast<MueLu::TpetraOperator<Scalar_d,LO,GO,Node>>(cache.hierarchy_,true);
        // Mark invalid until numeric refresh completes; an exception must not
        // leave a partly refreshed hierarchy eligible for future reuse.
        cache.schur_=false;
        MueLu::ReuseTpetraPreconditioner(matrix,*muelu);
        cache.schur_=true;
        cache.time_step_=reuse.time_step;
        ++cache.reuse_count_;
      }
      handle->muelu_operator_=cache.hierarchy_;
    }
    handle->left_operator_ = handle->muelu_operator_;
    return handle;
  }

  handle->setup_matrix_ = copy_with_repaired_diagonal(matrix);

  if (type == consts::PreconditionerType::PREC_TRILINOS_ML) {
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
