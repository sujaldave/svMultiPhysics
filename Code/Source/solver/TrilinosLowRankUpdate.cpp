// SPDX-License-Identifier: BSD-3-Clause
#include "TrilinosLowRankUpdate.h"
#ifdef WITH_TRILINOS
#include <stdexcept>

namespace trilinos_bipartition {
namespace {
void pack_column(Tpetra_MultiVector& U, Tpetra_MultiVector& V,
    const Tpetra_MultiVector& face_vector, const Tpetra_MultiVector* cap_vector,
    int col, double scale) {
  auto u=U.getLocalViewDevice(Tpetra::Access::ReadWrite);
  auto v=V.getLocalViewDevice(Tpetra::Access::ReadWrite);
  auto face=face_vector.getLocalViewDevice(Tpetra::Access::ReadOnly);
  const bool has_cap=cap_vector!=nullptr;
  auto cap=has_cap ? cap_vector->getLocalViewDevice(Tpetra::Access::ReadOnly) : face;
    Kokkos::parallel_for("svmp_pack_outlet_factors",Kokkos::RangePolicy<typename Node::execution_space>(0,U.getLocalLength()),
      KOKKOS_LAMBDA(const int row) {
        u(row,col)=scale*face(row,0);
        v(row,col)=face(row,0)+(has_cap ? cap(row,0) : 0.0);
      });
}
}

LowRankUpdate::LowRankUpdate(const Teuchos::RCP<const Tpetra_Map>& map,
    const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& updates,
    const std::vector<Teuchos::RCP<Tpetra_MultiVector>>& caps,
    const std::vector<double>& coefficients) {
  if (updates.size() != coefficients.size())
    throw std::runtime_error("LowRankUpdate: coefficient count mismatch");
  for (std::size_t i=0; i<updates.size(); ++i) {
    if (updates[i] == Teuchos::null || coefficients[i] == 0.0) continue;
    if (!updates[i]->getMap()->isSameAs(*map) || updates[i]->getNumVectors()!=1 ||
        (i<caps.size() && caps[i]!=Teuchos::null &&
         (!caps[i]->getMap()->isSameAs(*map) || caps[i]->getNumVectors()!=1)))
      throw std::runtime_error("LowRankUpdate: incompatible outlet vector");
    ++rank_;
  }
  if (!rank_) return;
  U_ = Teuchos::rcp(new Tpetra_MultiVector(map,rank_));
  V_ = Teuchos::rcp(new Tpetra_MultiVector(map,rank_));
  coefficient_map_ = Teuchos::rcp(new Tpetra_Map(rank_,0,map->getComm(),Tpetra::LocallyReplicated));
  int column=0;
  for (std::size_t i=0; i<updates.size(); ++i) {
    if (updates[i] == Teuchos::null || coefficients[i] == 0.0) continue;
    const bool has_cap=i<caps.size() && caps[i]!=Teuchos::null;
    const double scale=coefficients[i];
    const int col=column++;
    pack_column(*U_,*V_,*updates[i],has_cap ? caps[i].getRawPtr() : nullptr,col,scale);
  }
}

LowRankUpdate::LowRankUpdate(const Tpetra_CrsMatrix& B, const LowRankUpdate& velocity)
  : rank_(velocity.rank_), coefficient_map_(velocity.coefficient_map_) {
  if (!rank_) return;
  U_=Teuchos::rcp(new Tpetra_MultiVector(B.getDomainMap(),rank_));
  V_=Teuchos::rcp(new Tpetra_MultiVector(B.getDomainMap(),rank_));
  B.apply(*velocity.U_,*U_,Teuchos::TRANS);
  B.apply(*velocity.V_,*V_,Teuchos::TRANS);
}

void LowRankUpdate::add(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y, double alpha) const {
  if (!rank_ || alpha==0.0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  if (projection_==Teuchos::null || projection_->getNumVectors()!=X.getNumVectors())
    projection_=Teuchos::rcp(new Tpetra_MultiVector(coefficient_map_,X.getNumVectors()));
  // Distributed V^T X -> replicated rank-by-numVectors matrix. One collective
  // covers every face and every RHS, including ranks without local face nodes.
  projection_->multiply(Teuchos::TRANS,Teuchos::NO_TRANS,1.0,*V_,X,0.0);
  Y.multiply(Teuchos::NO_TRANS,Teuchos::NO_TRANS,alpha,*U_,*projection_,1.0);
}
}
#endif
