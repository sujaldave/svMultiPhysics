// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SVMP_PROFILING_H
#define SVMP_PROFILING_H

#include <string>

/**
 * @file Profiling.h
 * @brief Minimal cross-backend stage timer used to compare the FSILS
 *        (CPU, no Trilinos), Trilinos-CPU, and Trilinos-GPU bi-partition
 *        Navier-Stokes solves.
 *
 * All three backends report wall-clock time into the same fixed set of
 * named stages (see @ref stages) so their timings land in one CSV file with
 * identical columns, ready to be pivoted into per-stage histograms. A stage
 * that does not apply to a given backend (e.g. "MueLu Setup" for the FSILS
 * path) simply never appears in that run's rows.
 *
 * Usage: call set_backend_label() once near start-up, time individual
 * stages with the ProfilingScope RAII helper (or begin()/end() directly),
 * and call write_csv() once at shutdown. Stage timers nest safely: the same
 * or different stage names may be active concurrently on the call stack.
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
/// Safe to call repeatedly (e.g. once per Jacobian); the last call wins.
void set_backend_label(const std::string& label);

/// @brief Start (or resume, if already active) timing a named stage.
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
      begin(stage_);
    }

    ~ProfilingScope() { end(stage_); }

    ProfilingScope(const ProfilingScope&) = delete;
    ProfilingScope& operator=(const ProfilingScope&) = delete;

  private:
    std::string stage_;
};

/**
 * @brief Append this rank's accumulated stage totals to a CSV file.
 *
 * Only MPI rank 0 writes. Columns are:
 * backend,stage,calls,total_seconds,avg_seconds
 *
 * A header row is written only when the file does not already exist, so
 * repeated runs (e.g. one per backend under comparison) can all target the
 * same path and accumulate into a single file ready for cross-backend
 * histograms.
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

/// @brief Clear all accumulated stage timings. Does not reset the backend
/// label set by set_backend_label().
void reset();

} // namespace svmp_profiling

#endif
