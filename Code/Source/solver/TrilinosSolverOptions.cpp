// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "TrilinosSolverOptions.h"

#ifdef WITH_TRILINOS

#include "Amesos2_Factory.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace trilinos_options {

std::string choice(const char* name, const std::string& fallback,
    const std::vector<std::string>& allowed)
{
  const char* value = std::getenv(name);
  const std::string result = value && *value ? value : fallback;
  if (std::find(allowed.begin(), allowed.end(), result) == allowed.end()) {
    std::string list;
    for (const auto& item : allowed) list += (list.empty() ? "" : ", ") + item;
    throw std::runtime_error(std::string(name) + " must be one of: " + list);
  }
  return result;
}

int positive_integer(const char* name, int fallback)
{
  const char* value = std::getenv(name);
  if (!value || !*value) return fallback;
  char* end = nullptr;
  const long result = std::strtol(value, &end, 10);
  if (*end != '\0' || result <= 0 || result > 100000000)
    throw std::runtime_error(std::string(name) + " must be a positive integer");
  return static_cast<int>(result);
}

void configure_cg(Teuchos::ParameterList& belos)
{
  belos.set("Fold Convergence Detection Into Allreduce",
      choice("SV_TRILINOS_CG_FOLD", "1", {"0", "1"}) == "1");
}

bool refresh_once_per_step()
{
  return choice("SV_TRILINOS_AMG_REFRESH", "jacobian", {"jacobian", "step"}) == "step";
}

const std::vector<std::string>& smoother_names()
{
  static const std::vector<std::string> names = {"jacobi", "gauss-seidel",
      "symmetric-gauss-seidel", "mt-gauss-seidel", "mt-symmetric-gauss-seidel",
      "two-stage-gauss-seidel", "two-stage-symmetric-gauss-seidel", "chebyshev"};
  return names;
}

void set_smoother(Teuchos::ParameterList& muelu, const std::string& smoother,
    int sweeps, double damping)
{
  auto& params = muelu.sublist("smoother: params");
  if (smoother == "chebyshev") {
    // Ifpack2 estimates lambda_max(D^-1 A) on every level at each (re)compute.
    muelu.set("smoother: type", "CHEBYSHEV");
    params.set("chebyshev: degree", sweeps);
    params.set("chebyshev: ratio eigenvalue", 20.0);
    // An underestimated lambda_max makes the polynomial amplify the top of the
    // spectrum (an indefinite V-cycle for CG); estimate more conservatively.
    params.set("chebyshev: eigenvalue max iterations", 20);
    params.set("chebyshev: boost factor", 1.2);
    return;
  }
  static const std::vector<std::pair<std::string, std::string>> relaxation = {
      {"jacobi", "Jacobi"}, {"gauss-seidel", "Gauss-Seidel"},
      {"symmetric-gauss-seidel", "Symmetric Gauss-Seidel"},
      {"mt-gauss-seidel", "MT Gauss-Seidel"},
      {"mt-symmetric-gauss-seidel", "MT Symmetric Gauss-Seidel"},
      {"two-stage-gauss-seidel", "Two-stage Gauss-Seidel"},
      {"two-stage-symmetric-gauss-seidel", "Two-stage Symmetric Gauss-Seidel"}};
  const auto entry = std::find_if(relaxation.begin(), relaxation.end(),
      [&](const auto& item) { return item.first == smoother; });
  if (entry == relaxation.end())
    throw std::runtime_error("Unknown MueLu smoother: " + smoother);
  muelu.set("smoother: type", "RELAXATION");
  params.set("relaxation: type", entry->second);
  params.set("relaxation: sweeps", sweeps);
  if (damping > 0.0) params.set("relaxation: damping factor", damping);
}

void set_direct_coarse(Teuchos::ParameterList& muelu, const std::string& solver)
{
  const std::string amesos2 = solver == "klu" ? "klu2" : solver;
  if (!Amesos2::query(amesos2))
    throw std::runtime_error("Coarse solver '" + solver +
        "' is not available in this Trilinos build (Amesos2).");
  muelu.set("coarse: type", solver == "klu" ? "KLU" : solver == "tacho" ? "Tacho" : "cuSOLVER");
}

} // namespace trilinos_options

#endif
