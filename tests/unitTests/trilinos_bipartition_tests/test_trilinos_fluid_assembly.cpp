// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause
#include "gtest/gtest.h"
#ifdef WITH_TRILINOS
#include "TrilinosFluidAssembly.h"
#include "fluid.h"
#include "fs.h"
#include "nn.h"
#include "LinearAlgebra.h"
#include "Teuchos_DefaultSerialComm.hpp"

namespace {
void initialize_runtime() {
  static int argc = 0;
  static char** argv = nullptr;
  static Tpetra::ScopeGuard guard(&argc,&argv);
}

struct ReferenceAssembler : LinearAlgebra {
  Trilinos& t;
  explicit ReferenceAssembler(Trilinos& state) : t(state) {}
  void alloc(ComMod&,eqType&) override {}
  void assemble(ComMod&,int n,const Vector<int>& nodes,const Array3<double>& k,const Array<double>& r) override {
    t.local_assembly.add_element(t.topology,*t.ghostF,n,nodes.data(),k.data(),r.data());
  }
  void check_options(consts::PreconditionerType,consts::LinearAlgebraType) override {}
  void initialize(ComMod&,eqType&) override {}
  void set_assembly(consts::LinearAlgebraType) override {}
  void set_preconditioner(consts::PreconditionerType) override {}
  void solve(ComMod&,eqType&,const Vector<int>&,const Vector<double>&) override {}
  void finalize() override {}
};

struct FluidFixture {
  ComMod c;
  mshType mesh;
  SolutionStates solution;
  Trilinos reference, candidate;
  trilinos_backend::FluidAssembly assembly;
  std::unique_ptr<ReferenceAssembler> assembler;

  explicit FluidFixture(int n) {
    initialize_runtime();
    c.cm.new_cm(MPI_COMM_SELF);
    c.nsd=3; c.nsymd=6; c.dof=4; c.tDof=4; c.tnNo=n; c.gtnNo=n;
    c.cEq=0; c.cDmn=0; c.dt=0.01;
    c.eq.resize(1);
    auto& eq=c.eq[0]; eq.nDmn=1; eq.dmn.resize(1); eq.af=0.8; eq.am=0.9; eq.gam=0.7;
    auto& d=eq.dmn[0]; d.Id=-1; d.phys=consts::EquationType::phys_fluid;
    using P=consts::PhysicalPropertyType;
    d.prop[P::fluid_density]=1.06; d.prop[P::f_x]=0.1; d.prop[P::f_y]=0.2; d.prop[P::f_z]=-0.3;
    d.prop[P::brinkman_inverse_permeability]=0.05;
    d.fluid_visc.viscType=consts::FluidViscosityModelType::viscType_Const; d.fluid_visc.mu_i=0.04;
    mesh.eNoN=n; mesh.nEl=2; mesh.nFs=1;
    mesh.eType=n==4?consts::ElementType::TET4:consts::ElementType::HEX8;
    mesh.fs.resize(1);
    auto& f=mesh.fs[0]; f.eNoN=n; f.eType=mesh.eType; f.nG=n; f.lShpF=n==4;
    fs::init_fs(f,3,3);
    for (int g=0;g<f.nG;++g) nn::get_gn_nxx(3,6,mesh.eType,n,g,f.xi,f.Nxx);
    c.x.resize(3,n); c.Bf.resize(3,n); mesh.IEN.resize(n,2);
    const double tet[4][3]={{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    const double hex[8][3]={{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
                           {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
    for (int a=0;a<n;++a) {
      const double* p=n==4?tet[a]:hex[a];
      c.x(0,a)=p[0]+0.15*p[0]*p[1];
      c.x(1,a)=p[1]+0.10*p[1]*p[2];
      c.x(2,a)=p[2]+0.08*p[0]*p[2];
      for (int e=0;e<2;++e) mesh.IEN(a,e)=a;
      for (int i=0;i<3;++i) c.Bf(i,a)=0.01*(i+1)*(a+1);
    }
    auto nxi=f.Nx.rslice(0); Array<double> nx(3,n),metric(3,3); double jac=0;
    nn::gnn(n,3,3,nxi,c.x,nx,jac,metric);
    if (jac<0) for (int a=0;a<n;++a) std::swap(c.x(0,a),c.x(1,a));
    solution.intermediate.get_acceleration().resize(4,n);
    solution.intermediate.get_velocity().resize(4,n);
    for (int a=0;a<n;++a) for (int i=0;i<4;++i) {
      solution.intermediate.get_acceleration()(i,a)=0.03*(i+1)*(a+1);
      solution.intermediate.get_velocity()(i,a)=0.1*(i+1)+0.05*a;
    }
    for (auto* t:{&reference,&candidate}) {
      t->comm=Teuchos::rcp(new Teuchos::SerialComm<int>());
      Vector<int> order(n),rows(n+1),cols(n*n);
      for (int a=0;a<n;++a) { order(a)=a;rows(a)=a*n;for(int b=0;b<n;++b) cols(a*n+b)=b; }
      rows(n)=n*n;
      t->topology.ensure(t->comm,n,n,n,n*n,order,order,rows,cols,4,0);
      t->ghostF=Teuchos::rcp(new Tpetra_MultiVector(t->topology.ghost_map(),1));
      t->K=Teuchos::rcp(new Tpetra_CrsMatrix(t->topology.graph()));
      t->local_assembly.reset(t->topology);
    }
    assembler=std::make_unique<ReferenceAssembler>(reference);
    c.eq[0].linear_algebra=assembler.get();
  }
  void compare() {
    reference.local_assembly.reset(reference.topology);
    candidate.local_assembly.reset(candidate.topology);
    reference.ghostF->putScalar(0); candidate.ghostF->putScalar(0);
    const int node=0;
    double boundary_matrix[16]={0.1}, boundary_rhs[4]={0.2,0.3,0.4,0.5};
    // Host assembly before and after device assembly must preserve both paths.
    for (auto* t:{&reference,&candidate})
      t->local_assembly.add_element(t->topology,*t->ghostF,1,&node,boundary_matrix,boundary_rhs);
    fluid::construct_fluid(c,mesh,solution);
    ASSERT_TRUE(assembly.assemble(c,mesh,solution,candidate));
    for (auto* t:{&reference,&candidate})
      t->local_assembly.add_element(t->topology,*t->ghostF,1,&node,boundary_matrix,boundary_rhs);
    reference.local_assembly.flush(reference.topology,*reference.K,*reference.ghostF);
    candidate.local_assembly.flush(candidate.topology,*candidate.K,*candidate.ghostF);
    {
      const auto r=reference.ghostF->getLocalViewHost(Tpetra::Access::ReadOnly);
      const auto a=candidate.ghostF->getLocalViewHost(Tpetra::Access::ReadOnly);
      for (size_t i=0;i<r.extent(0);++i) EXPECT_NEAR(a(i,0),r(i,0),1e-11*std::max(1.0,std::abs(r(i,0))));
    }
    const auto r=reference.K->getLocalMatrixHost();
    const auto a=candidate.K->getLocalMatrixHost();
    ASSERT_EQ(r.values.extent(0),a.values.extent(0));
    for (size_t k=0;k<r.values.extent(0);++k) EXPECT_NEAR(a.values(k),r.values(k),1e-11*std::max(1.0,std::abs(r.values(k))));
  }
};
}

TEST(TrilinosFluidAssembly, TetrahedraMatchLegacyAndRefreshState) {
  FluidFixture f(4); f.compare();
  f.solution.intermediate.get_velocity()(0,0)+=0.5;
  f.c.Bf(2,1)-=0.2;
  f.c.eq[0].dmn[0].fluid_visc.mu_i=0.08;
  f.compare();
  f.mesh.fs[0].w(0) *= 1.01;
  f.compare();
  std::swap(f.mesh.IEN(0,0),f.mesh.IEN(1,0));
  EXPECT_THROW(f.assembly.assemble(f.c,f.mesh,f.solution,f.candidate),std::runtime_error);
  EXPECT_THROW(f.assembly.assemble(f.c,f.mesh,f.solution,f.candidate),std::runtime_error);
}
TEST(TrilinosFluidAssembly, DistortedHexahedraMatchLegacyAndRebuildGeometry) {
  FluidFixture f(8); f.compare();
  f.c.x(0,6)+=0.02;
  f.compare();
  for (int a=0;a<8;++a) f.c.x(0,a) *= -1.0;
  EXPECT_THROW(f.assembly.assemble(f.c,f.mesh,f.solution,f.candidate),std::runtime_error);
}
TEST(TrilinosFluidAssembly, UnsupportedPhysicsFallsBack) {
  FluidFixture f(4);
  f.c.mvMsh=true;
  EXPECT_FALSE(f.assembly.assemble(f.c,f.mesh,f.solution,f.candidate));
}

TEST(TrilinosNumericExport, SharedRowsAccumulateAndReuseNumericPlan) {
  initialize_runtime();
  const auto comm=Tpetra::getDefaultComm();
  if (comm->getSize()!=2) GTEST_SKIP() << "Run with mpirun -np 2";
  const int rank=comm->getRank();
  Trilinos t; t.comm=comm;
  Vector<int> order(2),rows={0,2,4},cols={0,1,0,1};
  order(0)=rank==0?0:2; order(1)=1;
  ASSERT_TRUE(t.topology.ensure(comm,3,rank==0?2:1,2,4,order,order,rows,cols,1,0));
  t.ghostF=Teuchos::rcp(new Tpetra_MultiVector(t.topology.ghost_map(),1));
  t.K=Teuchos::rcp(new Tpetra_CrsMatrix(t.topology.graph()));
  const int nodes[2]={0,1};
  const double expected[3][3]={{1,2,0},{3,12,7},{0,6,5}};
  for (int jacobian=1;jacobian<=2;++jacobian) {
    t.local_assembly.reset(t.topology);
    t.ghostF->putScalar(0.0);
    t.local_assembly.device_assembly_matrix();
    const double k[4]={double(jacobian*(1+4*rank)),double(jacobian*(3+4*rank)),
                       double(jacobian*(2+4*rank)),double(jacobian*(4+4*rank))};
    const double r[2]={double(jacobian*(10+20*rank)),double(jacobian*(20+20*rank))};
    t.local_assembly.add_element(t.topology,*t.ghostF,2,nodes,k,r);
    ASSERT_TRUE(t.local_assembly.flush(t.topology,*t.K,*t.ghostF));
    for (LO i=0;i<static_cast<LO>(t.topology.map()->getLocalNumElements());++i) {
      const GO row=t.topology.map()->getGlobalElement(i);
      Tpetra_CrsMatrix::nonconst_global_inds_host_view_type columns("cols",3);
      Tpetra_CrsMatrix::nonconst_values_host_view_type values("vals",3);
      size_t count=0;
      t.K->getGlobalRowCopy(row,columns,values,count);
      for (size_t k=0;k<count;++k) EXPECT_DOUBLE_EQ(values(k),jacobian*expected[row][columns(k)]);
    }
    Tpetra_MultiVector rhs(t.topology.map(),1);
    rhs.doExport(*t.ghostF,*t.topology.assembly_exporter(),Tpetra::ADD);
    const double expected_rhs[]={10,60,30};
    const auto v=rhs.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (LO i=0;i<static_cast<LO>(t.topology.map()->getLocalNumElements());++i)
      EXPECT_DOUBLE_EQ(v(i,0),jacobian*expected_rhs[t.topology.map()->getGlobalElement(i)]);
  }
}
#endif
