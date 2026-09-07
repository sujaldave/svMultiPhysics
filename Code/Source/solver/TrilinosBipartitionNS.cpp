// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "TrilinosBipartitionNS.h"

#ifdef WITH_TRILINOS

#include "Profiling.h"
#include "TrilinosPreconditionerFactory.h"
#include "TrilinosResistanceOperator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace trilinos_bipartition {
namespace {

struct BelosSolveStats
{
  bool converged = false;
  int iterations = 0;
  double elapsed = 0.0;
  double initial_norm = 0.0;
  double final_norm = 0.0;
};

double norm2(const Tpetra_MultiVector& vector)
{
  Teuchos::Array<double> norms(vector.getNumVectors());
  vector.norm2(norms());
  return norms[0];
}

double dot(const Tpetra_MultiVector& left, const Tpetra_MultiVector& right)
{
  Teuchos::Array<Scalar_d> dots(left.getNumVectors());
  left.dot(right, dots());
  return dots[0];
}

Teuchos::RCP<Tpetra_MultiVector> clone_shape(
    const Tpetra_MultiVector& vector)
{
  return Teuchos::rcp(
      new Tpetra_MultiVector(vector.getMap(), vector.getNumVectors()));
}

Teuchos::RCP<Tpetra_MultiVector> residual_from_basis(
    const Tpetra_MultiVector& initial,
    const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& basis,
    const std::vector<double>& coefficients,
    int active_basis)
{
  auto residual = clone_shape(initial);
  residual->update(1.0, initial, 0.0);

  for (int i = 0; i < active_basis; ++i) {
    if (basis[i] != Teuchos::null) {
      residual->update(-coefficients[i], *basis[i], 1.0);
    }
  }

  return residual;
}

bool solve_dense_system(
    int active_size,
    int stride,
    const std::vector<double>& matrix,
    const std::vector<double>& rhs,
    std::vector<double>& solution)
{
  std::vector<double> a(active_size * active_size, 0.0);
  std::vector<double> b(active_size, 0.0);

  for (int row = 0; row < active_size; ++row) {
    b[row] = rhs[row];
    for (int column = 0; column < active_size; ++column) {
      a[row * active_size + column] = matrix[row * stride + column];
    }
  }

  for (int pivot = 0; pivot < active_size; ++pivot) {
    int pivot_row = pivot;
    double pivot_value = std::abs(a[pivot * active_size + pivot]);
    for (int row = pivot + 1; row < active_size; ++row) {
      const double candidate = std::abs(a[row * active_size + pivot]);
      if (candidate > pivot_value) {
        pivot_value = candidate;
        pivot_row = row;
      }
    }

    if (pivot_value <= 1.0e-30) {
      return false;
    }

    if (pivot_row != pivot) {
      for (int column = pivot; column < active_size; ++column) {
        std::swap(a[pivot * active_size + column],
            a[pivot_row * active_size + column]);
      }
      std::swap(b[pivot], b[pivot_row]);
    }

    for (int row = pivot + 1; row < active_size; ++row) {
      const double factor =
          a[row * active_size + pivot] / a[pivot * active_size + pivot];
      a[row * active_size + pivot] = 0.0;
      for (int column = pivot + 1; column < active_size; ++column) {
        a[row * active_size + column] -=
            factor * a[pivot * active_size + column];
      }
      b[row] -= factor * b[pivot];
    }
  }

  std::vector<double> local_solution(active_size, 0.0);
  for (int row = active_size - 1; row >= 0; --row) {
    double value = b[row];
    for (int column = row + 1; column < active_size; ++column) {
      value -= a[row * active_size + column] * local_solution[column];
    }
    local_solution[row] = value / a[row * active_size + row];
  }

  for (int i = 0; i < active_size; ++i) {
    solution[i] = local_solution[i];
  }
  return true;
}

double effective_relative_tolerance(
    double rhs_norm,
    double relative_tolerance,
    double absolute_tolerance)
{
  if (rhs_norm == 0.0) {
    return relative_tolerance;
  }
  return std::max(relative_tolerance, absolute_tolerance / rhs_norm);
}

BelosSolveStats solve_with_belos(
    const std::string& solver_type,
    const Teuchos::RCP<Tpetra_Operator>& operator_,
    const Teuchos::RCP<preconditioners::PreconditionerHandle>& preconditioner,
    const Teuchos::RCP<Tpetra_MultiVector>& solution,
    const Teuchos::RCP<Tpetra_MultiVector>& rhs,
    double relative_tolerance,
    double absolute_tolerance,
    int max_iterations,
    int krylov_dimension)
{
  BelosSolveStats stats;
  stats.initial_norm = norm2(*rhs);
  solution->putScalar(0.0);

  if (stats.initial_norm == 0.0) {
    stats.converged = true;
    return stats;
  }

  auto problem = Teuchos::rcp(
      new Belos_LinearProblem(operator_, solution, rhs));
  preconditioners::attach_preconditioner(preconditioner, problem);
  if (!problem->setProblem()) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: Belos LinearProblem setup failed.");
  }

  Teuchos::ParameterList parameters;
  parameters.set("Verbosity", Belos::Errors + Belos::Warnings);
  parameters.set("Output Frequency", 1);
  parameters.set("Output Style", 1);
  parameters.set("Convergence Tolerance", effective_relative_tolerance(
      stats.initial_norm, relative_tolerance, absolute_tolerance));

  if (solver_type == "Block GMRES") {
    const auto budget =
        make_belos_gmres_budget(max_iterations, krylov_dimension);
    parameters.set("Maximum Iterations", budget.maximum_iterations);
    parameters.set("Num Blocks", budget.num_blocks);
    parameters.set("Maximum Restarts", budget.maximum_restarts);
    parameters.set("Orthogonalization", "DGKS");
  } else {
    parameters.set("Maximum Iterations",
        make_belos_cg_max_iterations(max_iterations));
  }

  Belos_SolverFactory factory;
  auto manager = factory.create(
      solver_type, Teuchos::rcpFromRef(parameters));
  manager->setProblem(problem);

  Teuchos::Time timer("Trilinos NS Belos solve");
  timer.start();
  const Belos::ReturnType result = manager->solve();
  timer.stop();

  stats.elapsed = timer.totalElapsedTime();
  stats.converged = result == Belos::Converged;
  stats.iterations = manager->getNumIters();

  auto residual = clone_shape(*rhs);
  operator_->apply(*solution, *residual);
  residual->update(1.0, *rhs, -1.0);
  stats.final_norm = norm2(*residual);
  return stats;
}

void accumulate_inner_solve(
    fsi_linear_solver::FSILS_subLsType& fields,
    const BelosSolveStats& stats)
{
  fields.suc = fields.suc && stats.converged;
  fields.itr += stats.iterations;
  fields.callD += stats.elapsed;
  fields.iNorm = stats.initial_norm;
  fields.fNorm = stats.final_norm;
  if (stats.initial_norm > 0.0 && stats.final_norm > 0.0) {
    fields.dB = 10.0 * std::log(stats.final_norm / stats.initial_norm);
  } else {
    fields.dB = 0.0;
  }
}

void copy_solution_to_host(
    const Teuchos::RCP<Trilinos>& trilinos,
    double* solution)
{
  svmp_profiling::ProfilingScope profiling_scope(
      svmp_profiling::stages::HostDeviceSynchronization);
  trilinos->ghostX->doImport(
      *trilinos->X, *trilinos->topology.importer(), Tpetra::INSERT);
  const auto local_view =
      trilinos->ghostX->getLocalViewHost(Tpetra::Access::ReadOnly);
  const size_t local_length = trilinos->ghostX->getLocalLength();
  for (size_t i = 0; i < local_length; ++i) {
    solution[i] = local_view(i, 0);
  }
}

void reset_trilinos_state_after_solve(
    const Teuchos::RCP<Trilinos>& trilinos)
{
  trilinos->ghostF->putScalar(0.0);
  trilinos->F->putScalar(0.0);
  for (auto& vector : trilinos->bdryVec_list) {
    vector->putScalar(0.0);
  }
  for (auto& vector : trilinos->bdryCapVec_list) {
    vector->putScalar(0.0);
  }
  trilinos->X->putScalar(0.0);
  trilinos->MueluPrec = Teuchos::null;
  trilinos->ifpackPrec = Teuchos::null;
  trilinos->resistancePrec = Teuchos::null;
  trilinos->resistanceFaces.clear();
  trilinos->K = Teuchos::null;
}

} // namespace

BelosGmresBudget make_belos_gmres_budget(
    int restart_cycles,
    int krylov_dimension)
{
  if (restart_cycles <= 0) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: NS_GM_max_iterations must be positive.");
  }
  if (krylov_dimension <= 0) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: Krylov_space_dimension must be positive.");
  }
  if (restart_cycles >
      std::numeric_limits<int>::max() / krylov_dimension) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: GMRES restart-cycle budget "
        "exceeds the supported integer range.");
  }

  BelosGmresBudget budget;
  budget.maximum_iterations = restart_cycles * krylov_dimension;
  budget.num_blocks = krylov_dimension;
  budget.maximum_restarts = restart_cycles - 1;
  return budget;
}

int make_belos_cg_max_iterations(int max_iterations)
{
  if (max_iterations <= 0) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: NS_CG_max_iterations must be positive.");
  }
  return max_iterations;
}

TrilinosBipartitionNSSolver::TrilinosBipartitionNSSolver(
    const Teuchos::RCP<Trilinos>& trilinos,
    int nsd,
    int dof,
    int time_step,
    const Teuchos::RCP<preconditioners::MueLuReuseCache>&
        momentum_muelu_cache,
    const Teuchos::RCP<preconditioners::MueLuReuseCache>&
        pressure_muelu_cache,
    const Teuchos::RCP<TrilinosNSBlockTopologyCache>&
        block_topology_cache) :
  trilinos_(trilinos),
  nsd_(nsd),
  dof_(dof),
  time_step_(time_step),
  momentum_muelu_cache_(momentum_muelu_cache),
  pressure_muelu_cache_(pressure_muelu_cache),
  block_topology_cache_(block_topology_cache)
{
  if (trilinos_ == Teuchos::null || block_topology_cache_ == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: Trilinos state or block cache is null.");
  }
}

void TrilinosBipartitionNSSolver::solve_assembled(
    eqType& equation,
    double* solution,
    const double* dirichlet_weights)
{
  solve_tpetra_system(equation, solution, dirichlet_weights, true);
}

void TrilinosBipartitionNSSolver::solve_fsils_assembled(
    eqType& equation,
    const double* values,
    const double* rhs,
    double* solution,
    const double* dirichlet_weights)
{
  assemble_fsils_system(values, rhs);
  solve_tpetra_system(equation, solution, dirichlet_weights, false);
}

void TrilinosBipartitionNSSolver::assemble_fsils_system(
    const double* values,
    const double* rhs) const
{
  if (trilinos_->K == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: Trilinos matrix is not allocated.");
  }
  const auto& topology = trilinos_->topology;
  if (topology.dof() != dof_) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: FSILS and equation DOF counts differ.");
  }

  const int ghost_and_local_nodes = topology.ghost_and_local_nodes();
  const auto& nonzeros_per_row = topology.nonzeros_per_row();
  const auto& local_to_global_unsorted =
      topology.local_to_global_unsorted();
  const auto& global_column_indices = topology.global_column_indices();

  int nonzero_offset = 0;
  int value_block = 0;
  std::vector<Scalar_d> row_values(dof_);
  std::vector<GO> column_gids(dof_);

  for (int row_node = 0; row_node < ghost_and_local_nodes; ++row_node) {
    const int row_entries = nonzeros_per_row[row_node];
    const GO row_node_gid = local_to_global_unsorted[row_node];
    const GO* column_nodes = &global_column_indices[nonzero_offset];

    for (int component = 0; component < dof_; ++component) {
      trilinos_->ghostF->replaceGlobalValue(
          row_node_gid * dof_ + component,
          0,
          rhs[row_node * dof_ + component]);
    }

    for (int entry = 0; entry < row_entries; ++entry) {
      for (int row_component = 0; row_component < dof_; ++row_component) {
        const GO matrix_row_gid =
            row_node_gid * dof_ + row_component;
        for (int column_component = 0;
             column_component < dof_;
             ++column_component) {
          column_gids[column_component] =
              column_nodes[entry] * dof_ + column_component;
          row_values[column_component] =
              values[value_block * dof_ * dof_ +
                     row_component * dof_ + column_component];
        }
        trilinos_->K->sumIntoGlobalValues(
            matrix_row_gid,
            dof_,
            row_values.data(),
            column_gids.data());
      }
      ++value_block;
    }
    nonzero_offset += row_entries;
  }
}

void TrilinosBipartitionNSSolver::solve_tpetra_system(
    eqType& equation,
    double* solution,
    const double* dirichlet_weights,
    bool rhs_needs_add_export)
{
  if (dof_ != nsd_ + 1) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: NS block solve expects dof = nsd + 1.");
  }
  if (trilinos_->K == Teuchos::null ||
      trilinos_->F == Teuchos::null ||
      trilinos_->ghostF == Teuchos::null) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: assembled Tpetra system is incomplete.");
  }

  auto& linear_solver = equation.FSILS;
  linear_solver.RI.suc = false;
  linear_solver.GM.suc = true;
  linear_solver.CG.suc = true;
  linear_solver.GM.itr = 0;
  linear_solver.CG.itr = 0;
  linear_solver.GM.callD = 0.0;
  linear_solver.CG.callD = 0.0;
  linear_solver.GM.dB = 0.0;
  linear_solver.CG.dB = 0.0;

  svmp_profiling::begin(svmp_profiling::stages::SystemSetup);
  trilinos_->local_assembly.flush(
      trilinos_->topology, *trilinos_->K, *trilinos_->ghostF);
  if (!trilinos_->K->isFillComplete()) {
    trilinos_->K->fillComplete();
  }
  Tpetra::Export<LO, GO, Node> rhs_exporter(
      trilinos_->ghostF->getMap(), trilinos_->F->getMap());
  trilinos_->F->doExport(
      *trilinos_->ghostF,
      rhs_exporter,
      rhs_needs_add_export ? Tpetra::ADD : Tpetra::REPLACE);

  auto diagonal = Teuchos::rcp(
      new Tpetra_Vector(trilinos_->topology.map()));
  constructJacobiScaling(trilinos_, dirichlet_weights, *diagonal);
  svmp_profiling::end(svmp_profiling::stages::SystemSetup);

  svmp_profiling::begin(svmp_profiling::stages::BlockExtraction);
  auto blocks = build_trilinos_ns_block_system(
      trilinos_, nsd_, dof_, trilinos_->topology.generation(),
      *block_topology_cache_);
  svmp_profiling::end(svmp_profiling::stages::BlockExtraction);

  auto momentum = Teuchos::rcp(new MomentumOperator(
      blocks.A, blocks.boundary_vectors, blocks.boundary_cap_vectors));
  svmp_profiling::begin(svmp_profiling::stages::BoundaryCondition);
  auto resistance = Teuchos::rcp(new TrilinosResistanceOperator(
      blocks.velocity_map,
      blocks.boundary_vectors,
      blocks.boundary_cap_vectors,
      trilinos_->resistanceFaces));
  resistance->compute();
  svmp_profiling::end(svmp_profiling::stages::BoundaryCondition);
  auto resistance_operator =
      Teuchos::rcp_implicit_cast<Tpetra_Operator>(resistance);
  auto pressure = Teuchos::rcp(new PressureSchurOperator(
      blocks.L, blocks.B, resistance_operator));

  Teuchos::RCP<Tpetra_Operator> momentum_resistance = Teuchos::null;
  if (equation.linear_algebra_gmres_preconditioner ==
      consts::PreconditionerType::PREC_TRILINOS_RESISTANCE) {
    momentum_resistance = resistance_operator;
  }

  // These are the only two preconditioner constructions for this Jacobian.
  // Every LinearProblem below attaches one of these existing handles. Any
  // MueLu build/reuse inside create_preconditioner() is additionally timed
  // under "MueLu Setup" as a sub-component of "Preconditioner Setup".
  svmp_profiling::begin(svmp_profiling::stages::PreconditionerSetup);
  preconditioners::PreconditionerReuseContext momentum_reuse;
  momentum_reuse.muelu_cache = momentum_muelu_cache_;
  momentum_reuse.time_step = time_step_;
  momentum_reuse.topology_generation = trilinos_->topology.generation();
  auto momentum_preconditioner = preconditioners::create_preconditioner(
      equation.linear_algebra_gmres_preconditioner,
      preconditioners::SolverRole::momentum_gmres,
      blocks.A,
      momentum_resistance,
      momentum_reuse);

  preconditioners::PreconditionerReuseContext pressure_reuse;
  pressure_reuse.muelu_cache = pressure_muelu_cache_;
  pressure_reuse.time_step = time_step_;
  pressure_reuse.topology_generation = trilinos_->topology.generation();
  auto pressure_preconditioner = preconditioners::create_preconditioner(
      equation.linear_algebra_cg_preconditioner,
      preconditioners::SolverRole::pressure_cg,
      blocks.L,
      Teuchos::null,
      pressure_reuse);
  svmp_profiling::end(svmp_profiling::stages::PreconditionerSetup);

  svmp_profiling::begin(svmp_profiling::stages::TpetraAllocation);
  auto initial_momentum = block_topology_cache_->extract_velocity(*trilinos_->F);
  auto initial_continuity = block_topology_cache_->extract_pressure(*trilinos_->F);
  auto momentum_residual = clone_shape(*initial_momentum);
  auto continuity_residual = clone_shape(*initial_continuity);
  svmp_profiling::end(svmp_profiling::stages::TpetraAllocation);
  momentum_residual->update(1.0, *initial_momentum, 0.0);
  continuity_residual->update(1.0, *initial_continuity, 0.0);

  const double momentum_norm = norm2(*initial_momentum);
  const double continuity_norm = norm2(*initial_continuity);
  const double initial_norm = std::sqrt(
      momentum_norm * momentum_norm + continuity_norm * continuity_norm);
  linear_solver.RI.iNorm = initial_norm;
  linear_solver.RI.fNorm = initial_norm * initial_norm;
  linear_solver.RI.itr = 0;
  linear_solver.RI.dB = 0.0;

  if (initial_norm == 0.0) {
    Tpetra_MultiVector zero_velocity(blocks.velocity_map, 1);
    Tpetra_MultiVector zero_pressure(blocks.pressure_map, 1);
    zero_velocity.putScalar(0.0);
    zero_pressure.putScalar(0.0);
    block_topology_cache_->scatter(
        zero_velocity, zero_pressure, trilinos_->X);
    copy_solution_to_host(trilinos_, solution);
    linear_solver.RI.suc = true;
    reset_trilinos_state_after_solve(trilinos_);
    return;
  }

  if (linear_solver.RI.mItr <= 0) {
    throw std::runtime_error(
        "[TrilinosBipartitionNS] ERROR: Max_iterations must be positive.");
  }

  const int max_ri_iterations = linear_solver.RI.mItr;
  const int max_basis = 2 * max_ri_iterations;
  const double tolerance = std::max(
      linear_solver.RI.absTol,
      linear_solver.RI.relTol * initial_norm);
  const double tolerance_squared = tolerance * tolerance;

  std::vector<Teuchos::RCP<Tpetra_MultiVector>> velocity_basis(
      max_ri_iterations);
  std::vector<Teuchos::RCP<Tpetra_MultiVector>> pressure_basis(
      max_ri_iterations);
  std::vector<Teuchos::RCP<Tpetra_MultiVector>> momentum_images(max_basis);
  std::vector<Teuchos::RCP<Tpetra_MultiVector>> continuity_images(max_basis);
  std::vector<double> dense_matrix(max_basis * max_basis, 0.0);
  std::vector<double> dense_rhs(max_basis, 0.0);
  std::vector<double> coefficients(max_basis, 0.0);
  std::vector<double> old_coefficients(max_basis, 0.0);

  double previous_ri_norm_squared = linear_solver.RI.fNorm;
  double ri_norm_squared = linear_solver.RI.fNorm;
  int active_basis = 0;
  int completed_iteration = 0;

  Teuchos::Time ri_timer("Trilinos NS RI solve");
  ri_timer.start();

  for (int iteration = 0; iteration < max_ri_iterations; ++iteration) {
    const int pressure_index = 2 * iteration;
    const int velocity_index = pressure_index + 1;
    active_basis = velocity_index + 1;
    previous_ri_norm_squared = ri_norm_squared;
    completed_iteration = iteration;

    velocity_basis[iteration] = Teuchos::rcp(
        new Tpetra_MultiVector(blocks.velocity_map, 1));
    svmp_profiling::begin(svmp_profiling::stages::Predictor);
    const auto predictor = solve_with_belos(
        "Block GMRES",
        Teuchos::rcp_implicit_cast<Tpetra_Operator>(momentum),
        momentum_preconditioner,
        velocity_basis[iteration],
        momentum_residual,
        linear_solver.GM.relTol,
        linear_solver.GM.absTol,
        linear_solver.GM.mItr,
        linear_solver.GM.sD);
    svmp_profiling::end(svmp_profiling::stages::Predictor);
    accumulate_inner_solve(linear_solver.GM, predictor);

    auto pressure_rhs = Teuchos::rcp(
        new Tpetra_MultiVector(blocks.pressure_map, 1));
    blocks.C->apply(*velocity_basis[iteration], *pressure_rhs);
    pressure_rhs->update(1.0, *continuity_residual, -1.0);

    pressure_basis[iteration] = Teuchos::rcp(
        new Tpetra_MultiVector(blocks.pressure_map, 1));
    svmp_profiling::begin(svmp_profiling::stages::LinearSolve);
    const auto cg = solve_with_belos(
        "Pseudoblock CG",
        Teuchos::rcp_implicit_cast<Tpetra_Operator>(pressure),
        pressure_preconditioner,
        pressure_basis[iteration],
        pressure_rhs,
        linear_solver.CG.relTol,
        linear_solver.CG.absTol,
        linear_solver.CG.mItr,
        linear_solver.CG.sD);
    svmp_profiling::end(svmp_profiling::stages::LinearSolve);
    accumulate_inner_solve(linear_solver.CG, cg);

    momentum_images[pressure_index] = Teuchos::rcp(
        new Tpetra_MultiVector(blocks.velocity_map, 1));
    blocks.B->apply(
        *pressure_basis[iteration], *momentum_images[pressure_index]);

    momentum_images[velocity_index] = Teuchos::rcp(
        new Tpetra_MultiVector(blocks.velocity_map, 1));
    momentum_images[velocity_index]->update(
        1.0, *momentum_residual, 0.0);
    momentum_images[velocity_index]->update(
        -1.0, *momentum_images[pressure_index], 1.0);

    svmp_profiling::begin(svmp_profiling::stages::LinearSolve);
    const auto correction = solve_with_belos(
        "Block GMRES",
        Teuchos::rcp_implicit_cast<Tpetra_Operator>(momentum),
        momentum_preconditioner,
        velocity_basis[iteration],
        momentum_images[velocity_index],
        linear_solver.GM.relTol,
        linear_solver.GM.absTol,
        linear_solver.GM.mItr,
        linear_solver.GM.sD);
    svmp_profiling::end(svmp_profiling::stages::LinearSolve);
    accumulate_inner_solve(linear_solver.GM, correction);

    momentum->apply(
        *velocity_basis[iteration], *momentum_images[velocity_index]);

    continuity_images[pressure_index] = Teuchos::rcp(
        new Tpetra_MultiVector(blocks.pressure_map, 1));
    blocks.L->apply(
        *pressure_basis[iteration], *continuity_images[pressure_index]);

    continuity_images[velocity_index] = Teuchos::rcp(
        new Tpetra_MultiVector(blocks.pressure_map, 1));
    blocks.C->apply(
        *velocity_basis[iteration], *continuity_images[velocity_index]);

    for (int row = pressure_index; row <= velocity_index; ++row) {
      for (int column = 0; column <= row; ++column) {
        const double value =
            dot(*momentum_images[column], *momentum_images[row]) +
            dot(*continuity_images[column], *continuity_images[row]);
        dense_matrix[column * max_basis + row] = value;
        dense_matrix[row * max_basis + column] = value;
      }
      dense_rhs[row] =
          dot(*momentum_images[row], *initial_momentum) +
          dot(*continuity_images[row], *initial_continuity);
    }

    if (solve_dense_system(
            active_basis,
            max_basis,
            dense_matrix,
            dense_rhs,
            coefficients)) {
      old_coefficients = coefficients;
    } else {
      if (iteration == 0) {
        throw std::runtime_error(
            "[TrilinosBipartitionNS] ERROR: singular RI coefficient matrix.");
      }
      coefficients = old_coefficients;
      active_basis -= 2;
      break;
    }

    double projected_norm = 0.0;
    for (int i = 0; i < active_basis; ++i) {
      projected_norm += coefficients[i] * dense_rhs[i];
    }
    ri_norm_squared = std::max(
        0.0, initial_norm * initial_norm - projected_norm);

    momentum_residual = residual_from_basis(
        *initial_momentum,
        momentum_images,
        coefficients,
        active_basis);
    continuity_residual = residual_from_basis(
        *initial_continuity,
        continuity_images,
        coefficients,
        active_basis);

    if (ri_norm_squared < tolerance_squared) {
      linear_solver.RI.suc = true;
      break;
    }
  }

  ri_timer.stop();
  linear_solver.RI.callD = ri_timer.totalElapsedTime();
  linear_solver.RI.itr = completed_iteration;

  const double final_continuity_norm = norm2(*continuity_residual);
  if (ri_norm_squared > 0.0) {
    linear_solver.Resc = static_cast<int>(
        100.0 * final_continuity_norm * final_continuity_norm /
        ri_norm_squared);
    linear_solver.Resm = 100 - linear_solver.Resc;
  } else {
    linear_solver.Resc = 0;
    linear_solver.Resm = 0;
  }

  if (previous_ri_norm_squared > 0.0 && ri_norm_squared > 0.0) {
    linear_solver.RI.dB =
        5.0 * std::log(ri_norm_squared / previous_ri_norm_squared);
  } else {
    linear_solver.RI.dB = 0.0;
  }

  if (linear_solver.Resc < 0 || linear_solver.Resm < 0) {
    linear_solver.Resc = 0;
    linear_solver.Resm = 0;
    linear_solver.RI.dB = 0.0;
    ri_norm_squared = 0.0;
    if (trilinos_->comm == Teuchos::null ||
        trilinos_->comm->getRank() == 0) {
      std::cout << "[svMultiPhysics] WARNING: The Trilinos NS solver has "
                << "computed an ill-conditioned RI coefficient system."
                << std::endl;
    }
  }
  linear_solver.RI.fNorm = std::sqrt(ri_norm_squared);

  svmp_profiling::begin(svmp_profiling::stages::TpetraAllocation);
  auto velocity_solution = Teuchos::rcp(
      new Tpetra_MultiVector(blocks.velocity_map, 1));
  auto pressure_solution = Teuchos::rcp(
      new Tpetra_MultiVector(blocks.pressure_map, 1));
  velocity_solution->putScalar(0.0);
  pressure_solution->putScalar(0.0);
  svmp_profiling::end(svmp_profiling::stages::TpetraAllocation);

  if (active_basis >= 2) {
    velocity_solution->update(
        coefficients[1], *velocity_basis[0], 0.0);
    pressure_solution->update(
        coefficients[0], *pressure_basis[0], 0.0);
    for (int iteration = 1;
         2 * iteration + 1 < active_basis;
         ++iteration) {
      const int pressure_index = 2 * iteration;
      const int velocity_index = pressure_index + 1;
      velocity_solution->update(
          coefficients[velocity_index], *velocity_basis[iteration], 1.0);
      pressure_solution->update(
          coefficients[pressure_index], *pressure_basis[iteration], 1.0);
    }
  }

  block_topology_cache_->scatter(
      *velocity_solution, *pressure_solution, trilinos_->X);
  trilinos_->X->elementWiseMultiply(
      1.0, *trilinos_->X, *diagonal, 0.0);
  copy_solution_to_host(trilinos_, solution);
  reset_trilinos_state_after_solve(trilinos_);
}

} // namespace trilinos_bipartition

#endif
