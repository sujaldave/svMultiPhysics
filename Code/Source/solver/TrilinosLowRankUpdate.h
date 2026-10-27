// SPDX-License-Identifier: BSD-3-Clause
#ifndef TRILINOS_LOW_RANK_UPDATE_H
#define TRILINOS_LOW_RANK_UPDATE_H
#ifdef WITH_TRILINOS
#include "trilinos_impl.h"
#include <mutex>

namespace trilinos_bipartition {
// Apply alpha U V^T X to Y. Tpetra batches all outlet projections in one
// distributed multiply/reduction; small projection coefficients stay in Tpetra.
class LowRankUpdate {
public:
  LowRankUpdate(const Teuchos::RCP<const Tpetra_Map>& map,
      const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& updates,
      const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& caps,
      const std::vector<double>& coefficients);
  LowRankUpdate(const Tpetra_CrsMatrix& B, const LowRankUpdate& velocity);
  void add(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y, double alpha) const;
  std::size_t rank() const { return rank_; }
private:
  std::size_t rank_ = 0;
  Teuchos::RCP<Tpetra_MultiVector> U_, V_;
  Teuchos::RCP<const Tpetra_Map> coefficient_map_;
  mutable Teuchos::RCP<Tpetra_MultiVector> projection_;
  mutable std::mutex mutex_;
};
}
#endif
#endif
