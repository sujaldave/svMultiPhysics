// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "Profiling.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <tuple>

#include <mpi.h>

namespace svmp_profiling {

namespace {

// (equation, backend, gmres_preconditioner, cg_preconditioner, stage).
// Every stage timer is filed under the label snapshot captured when it
// started, so two equations (or two preconditioner choices) that happen to
// share a stage name never blend into one total.
using RowKey = std::tuple<std::string, std::string, std::string, std::string, std::string>;

struct StageTotals
{
  double total_seconds = 0.0;
  long long calls = 0;
};

struct ActiveTimer
{
  std::chrono::steady_clock::time_point start;
  int depth = 0;
  RowKey key;
};

std::mutex& registry_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::map<RowKey, StageTotals>& totals()
{
  static std::map<RowKey, StageTotals> map;
  return map;
}

std::map<std::string, ActiveTimer>& active_timers()
{
  static std::map<std::string, ActiveTimer> map;
  return map;
}

// TEMPORARY diagnostic: remove once the empty-CSV issue is understood.
long long& begin_call_count()
{
  static long long count = 0;
  return count;
}

std::string& backend_label()
{
  static std::string label = "unknown";
  return label;
}

std::string& equation_label()
{
  static std::string label = "unknown";
  return label;
}

std::string& gmres_preconditioner_label()
{
  static std::string label = "n/a";
  return label;
}

std::string& cg_preconditioner_label()
{
  static std::string label = "n/a";
  return label;
}

int mpi_rank()
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return 0;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

} // namespace

void set_backend_label(const std::string& label)
{
  std::lock_guard<std::mutex> lock(registry_mutex());
  backend_label() = label;
}

void set_equation_label(const std::string& equation)
{
  std::lock_guard<std::mutex> lock(registry_mutex());
  equation_label() = equation;
}

void set_preconditioner_labels(
    const std::string& gmres_preconditioner,
    const std::string& cg_preconditioner)
{
  std::lock_guard<std::mutex> lock(registry_mutex());
  gmres_preconditioner_label() = gmres_preconditioner;
  cg_preconditioner_label() = cg_preconditioner;
}

void begin(const std::string& stage)
{
  std::lock_guard<std::mutex> lock(registry_mutex());
  ++begin_call_count(); // TEMPORARY diagnostic.
  auto& timer = active_timers()[stage];
  if (timer.depth == 0) {
    timer.start = std::chrono::steady_clock::now();
    timer.key = RowKey(
        equation_label(), backend_label(), gmres_preconditioner_label(),
        cg_preconditioner_label(), stage);
  }
  ++timer.depth;
}

void end(const std::string& stage)
{
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(registry_mutex());
  auto& timers = active_timers();
  auto it = timers.find(stage);
  if (it == timers.end() || it->second.depth == 0) {
    return;
  }
  --it->second.depth;
  if (it->second.depth > 0) {
    return;
  }
  const std::chrono::duration<double> elapsed = now - it->second.start;
  auto& stage_totals = totals()[it->second.key];
  stage_totals.total_seconds += elapsed.count();
  ++stage_totals.calls;
}

void reset()
{
  std::lock_guard<std::mutex> lock(registry_mutex());
  totals().clear();
  active_timers().clear();
}

void write_csv(const std::string& path)
{
  std::map<RowKey, StageTotals> totals_copy;
  {
    std::lock_guard<std::mutex> lock(registry_mutex());
    totals_copy = totals();
  }

  // TEMPORARY diagnostic: remove once the empty-CSV issue is understood.
  std::cerr << "[svmp_profiling] write_csv('" << path << "'): mpi_rank="
             << mpi_rank() << " rows=" << totals_copy.size()
             << " begin_calls=" << begin_call_count() << std::endl;

  if (mpi_rank() != 0) {
    return;
  }

  std::ifstream existing(path);
  const bool need_header = !existing.good();
  existing.close();

  std::ofstream out(path, std::ios::app);
  if (!out.is_open()) {
    std::cerr << "[svmp_profiling] WARNING: could not open '" << path
               << "' for writing." << std::endl;
    return;
  }

  if (need_header) {
    out << "backend,equation,gmres_preconditioner,cg_preconditioner,stage,"
           "calls,total_seconds,avg_seconds\n";
  }

  for (const auto& entry : totals_copy) {
    const auto& equation = std::get<0>(entry.first);
    const auto& backend = std::get<1>(entry.first);
    const auto& gmres_prec = std::get<2>(entry.first);
    const auto& cg_prec = std::get<3>(entry.first);
    const auto& stage = std::get<4>(entry.first);
    const auto& stage_totals = entry.second;
    const double avg = stage_totals.calls > 0 ?
        stage_totals.total_seconds / stage_totals.calls : 0.0;
    out << backend << "," << equation << "," << gmres_prec << ","
        << cg_prec << "," << stage << "," << stage_totals.calls << ","
        << stage_totals.total_seconds << "," << avg << "\n";
  }
}

void write_csv()
{
  const char* env_path = std::getenv("SVMP_PROFILE_CSV");
  write_csv(env_path != nullptr ? std::string(env_path) : "svmp_profiling.csv");
}

} // namespace svmp_profiling
