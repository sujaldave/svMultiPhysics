// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SVMP_PROFILING_H
#define SVMP_PROFILING_H

#include <string>

/**
 * @file Profiling.h
 * @brief Minimal cross-backend stage timer used to compare the FSILS
 *        (CPU, no Trilinos), Trilinos-CPU, and Trilinos-GPU solves across
 *        equations (NS bi-partition, FSI, struct, mesh, ...).
 *
 * All backends and equations report wall-clock time into the same fixed
 * set of named stages (see @ref stages) so their timings land in one CSV
 * file with identical columns, ready to be pivoted into per-stage
 * histograms. A stage that does not apply to a given backend or equation
 * (e.g. "MueLu Setup" for the FSILS path, or "Predictor" outside the NS
 * bi-partition solve) simply never appears in that row group.
 *
 * Usage: call set_backend_label() once near start-up; call
 * set_equation_label() and set_preconditioner_labels() right before
 * dispatching each equation's linear solve (both are cheap and meant to be
 * called every solve, since which equation/preconditioner is active can
 * change from one solve to the next within the same run -- e.g. an FSI run
 * alternates between the FSI equation's GMRES solve and the mesh
 * equation's CG solve). Time individual stages with the ProfilingScope
 * RAII helper (or begin()/end() directly), and call write_csv() once at
 * shutdown.
 *
 * Each stage timer captures a snapshot of the equation/backend/
 * preconditioner labels at the moment it starts (not when write_csv() runs),
 * so time spent in, say, "System Setup" during the FSI equation's solve and
 * during the mesh equation's solve accumulate into separate CSV rows even
 * though both use the stage name "System Setup".
 */
namespace svmp_profiling {

/// @brief Canonical stage names shared by every backend so CSV rows line up.
namespace stages {
  constexpr const char* ElementAssembly = "Element Assembly";
  constexpr const char* MueLuSetup = "MueLu Setup";
  constexpr const char* LinearSolve = "Linear Solve";
  constexpr const char* BoundaryCondition = "Boundary Condition";
  constexpr const char* TpetraAllocation = "Tpetra Allocation";
  constexpr const char* Predictor = "Predictor";
  constexpr const char* HostDeviceSynchronization =
      "Host & Device Synchronization";
  constexpr const char* SystemSetup = "System Setup";
  constexpr const char* BlockExtraction = "Block Extraction";
  constexpr const char* PreconditionerSetup = "Preconditioner Setup";
} // namespace stages

/// @brief Identify which of the three configurations is producing timings.
/// Safe to call repeatedly (e.g. once per Jacobian); affects stage timers
/// started after this call.
void set_backend_label(const std::string& label);

/**
 * @brief Record which equation is about to solve, using the same short
 * symbol svMultiPhysics prints in histor.dat (e.g. "NS", "FS", "MS", "ST").
 * Call this right before dispatching that equation's linear solve, every
 * time -- a single run may solve several different equations (e.g. FSI's
 * "FS" fluid solve and its coupled "MS" mesh solve), and stage timers
 * started after this call are tagged with this equation until it changes
 * again.
 */
void set_equation_label(const std::string& equation);

/**
 * @brief Record which preconditioner is active for the momentum/GMRES and
 * pressure/CG inner solves, so runs that only differ by preconditioner
 * choice (e.g. trilinos-ml vs. trilinos-diagonal on the same backend) can
 * be told apart in the CSV. For a solver with one preconditioner rather
 * than separate GMRES/CG roles (FSILS, or Trilinos's monolithic path),
 * pass the same name for both arguments.
 *
 * Safe to call repeatedly (e.g. once per equation solve, since the
 * preconditioner choice is read from that equation's XML config); affects
 * stage timers started after this call.
 */
void set_preconditioner_labels(
    const std::string& gmres_preconditioner,
    const std::string& cg_preconditioner);

/// @brief Start (or resume, if already active) timing a named stage. The
/// current equation/backend/preconditioner labels are captured now and
/// used for this timer's row even if those labels change before end().
void begin(const std::string& stage);

/// @brief Stop timing a stage previously started with begin().
/// A call with no matching begin() is ignored.
void end(const std::string& stage);

/// @brief RAII helper: times its enclosing scope under one stage name.
class ProfilingScope
{
  public:
    explicit ProfilingScope(const std::string& stage) : stage_(stage)
    {
      svmp_profiling::begin(stage_);
    }

    ~ProfilingScope() { svmp_profiling::end(stage_); }

    ProfilingScope(const ProfilingScope&) = delete;
    ProfilingScope& operator=(const ProfilingScope&) = delete;

  private:
    std::string stage_;
};

/**
 * @brief Append this rank's accumulated stage totals to a CSV file.
 *
 * Only MPI rank 0 writes. Columns are:
 * backend,equation,gmres_preconditioner,cg_preconditioner,stage,calls,total_seconds,avg_seconds
 *
 * A header row is written only when the file does not already exist, so
 * repeated runs (e.g. one per backend and/or preconditioner combination
 * under comparison) can all target the same path and accumulate into a
 * single file ready for cross-backend, cross-equation, and
 * cross-preconditioner histograms. A CSV written before the "equation"
 * column was added will not have it; append fresh runs to a new file
 * rather than mixing old- and new-schema rows in one file.
 */
void write_csv(const std::string& path);

/**
 * @brief Same as write_csv(path), but resolves the path from the
 * SVMP_PROFILE_CSV environment variable (default: "svmp_profiling.csv" in
 * the current working directory). Call this once per run at shutdown so
 * each of the three backend configurations can point the same environment
 * variable at one shared file and accumulate comparable rows.
 */
void write_csv();

/// @brief Clear all accumulated stage timings. Does not reset the backend,
/// equation, or preconditioner labels.
void reset();

} // namespace svmp_profiling

#endif
