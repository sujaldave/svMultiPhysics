// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

//--------------------------------------------------------------------
//
// Interface to Trilinos linear solver library.
//
//--------------------------------------------------------------------

/*!
  \file    trilinos_linear_solver.cpp
  \brief   wrap Trilinos solver functions
*/

#include "trilinos_impl.h"
#include "ComMod.h"
#include "TrilinosBipartitionNS.h"
#include "TrilinosPreconditionerFactory.h"
#include "TrilinosResistanceOperator.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#define NOOUTPUT

namespace {

std::mutex& trilinos_runtime_mutex()
{
  static std::mutex mutex;
  return mutex;
}

std::size_t& trilinos_runtime_users()
{
  static std::size_t users = 0;
  return users;
}

} // namespace

// ----------------------------------------------------------------------------
/**
 * Define the matrix vector multiplication operation to do at each iteration of
 * an iterative linear solver. This function is called by the AztecOO
 * (K + v1*v1' + v2*v2' + ...) *x = K*x + v1*(v1'*x) + v2*(v2'*x), 
 * where [v1, v2, ...] = bdryVec_list
 *
 * For coupled Neumann boundary terms (v1*v1', v2*v2'), we use efficient an vectorized 
 * operation rather than explicitly forming the rank 1 outer product matrix and 
 * adding it to the global stiffness matrix K.
 *
 * \param x vector to be applied on the operator
 * \param y result of sprase matrix vector multiplication
 */
void TrilinosMatVec::apply(const Tpetra_MultiVector& x, Tpetra_MultiVector& y,
           Teuchos::ETransp mode, Scalar_d alpha, Scalar_d beta) const
{
  // Store initial matrix vector product result in y (y = K*x if alpha = 1.0, beta = 0.0) 
  trilinos_->K->apply(x, y, mode, alpha, beta); //Y = beta*Y + alpha*K*X 

  // Now add on the coupled Neumann boundary contribution y += v1*(v1'*x) + v2*(v2'*x) + ...
  if (trilinos_->coupled_boundary)
  {
    // Cap terms contribute to the flux scalar term only, not to pressure update rows.
    // So we compute (v_face^T x + v_cap^T x) and apply it with v_face.
    for (size_t i = 0; i < trilinos_->bdryVec_list.size(); ++i)
    {
      Scalar_d dot_face = 0.0;
      Scalar_d dot_cap = 0.0;
      Teuchos::Array<Scalar_d> dots(1);

      auto bdryVec = trilinos_->bdryVec_list[i];
      x.dot(*bdryVec, dots());
      dot_face = dots[0];

      if (i < trilinos_->bdryCapVec_list.size()) {
        auto bdryCapVec = trilinos_->bdryCapVec_list[i];
        x.dot(*bdryCapVec, dots());
        dot_cap = dots[0];
      }

      y.update(dot_face + dot_cap, *bdryVec, 1.0);
    }
  }
}

// ----------------------------------------------------------------------------
/**
 * make the graph global and optimize storage on it
 * this function creates the LHS structure topology and RHS vector based on the
 * block map
 * \param numGlobalNodes total/global number of nodes each node can have dof
 *                       for the spatial dim
 * \param numLocalNodes  number of nodes owned by this proc in block coordiantes
 * \param numGhostAndLocalNodes number of nodes owned and shared by this
 *                              processor includes ghost nodes
 * \param nnz            number of nonzeros in LHS matrix of calling proc
 * \param ltgSorted      integer pointer of size numLocalNodes returns global
 *                       node number of local node a to give blockrow
 * \param ltgUnsorted    unsorted/not-reordered version
 * \param rowPtr         CSR row ptr of size numLocalNodes + 1 to block rows
 * \param colInd         CSR column indices ptr (size nnz points) to block rows
 * \param Dof            size of each block element to give dim of each block
 * 
 * \param numCoupledNeumannBC number of coupled Neumann BCs

 */

void trilinos_lhs_create(const Teuchos::RCP<Trilinos> &trilinos_, const int numGlobalNodes, const int numLocalNodes,
        const int numGhostAndLocalNodes, const int nnz, const Vector<int>& ltgSorted,
        const Vector<int>& ltgUnsorted, const Vector<int>& rowPtr, const Vector<int>& colInd,
        const int Dof, const int cpp_index, const int proc_id, const int numCoupledNeumannBC)
{
  #ifdef debug_trilinos_lhs_create
  std::string msg_prefix;
  msg_prefix = std::string("[trilinos_lhs_create:") + std::to_string(proc_id) + "] ";
  std::cout << msg_prefix << std::endl;
  std::cout << msg_prefix << "========== trilinos_lhs_create ==========" << std::endl;
  std::cout << msg_prefix << "Dof: " << Dof << std::endl;
  std::cout << msg_prefix << "cpp_index: " << cpp_index << std::endl;
  #endif

  int indexBase = 1; //0 based indexing for C/C++ / 1 for Fortran
  if (cpp_index == 1) {
    indexBase = 0; 
  }

  trilinos_->comm = Tpetra::getDefaultComm();

  #ifdef debug_trilinos_lhs_create
  std::cout <<  msg_prefix << "indexBase: " << indexBase << std::endl;
  std::cout << msg_prefix << "dof: " << Dof << std::endl;
  std::cout << msg_prefix << "ghostAndLocalNodes: " << numGhostAndLocalNodes << std::endl;
  std::cout << msg_prefix << "localNodes: " << numLocalNodes << std::endl;
  #endif

  // Maps, importer, and sparsity are equation topology. Reuse them across
  // Newton iterations; only the matrix values and vectors below are fresh.
  trilinos_->topology.ensure(
      trilinos_->comm,
      numGlobalNodes,
      numLocalNodes,
      numGhostAndLocalNodes,
      nnz,
      ltgSorted,
      ltgUnsorted,
      rowPtr,
      colInd,
      Dof,
      indexBase);

  // --- Create block finite element matrix from graph with fillcomplete ------
  // construct matrix from filled graph
  trilinos_->K = Teuchos::rcp(
      new Tpetra_CrsMatrix(trilinos_->topology.graph()));

  //construct RHS force vector F topology
  trilinos_->F = Teuchos::rcp(
      new Tpetra_MultiVector(trilinos_->topology.map(), 1));

  //construct RHS force vector ghostF topology
  trilinos_->ghostF = Teuchos::rcp(
      new Tpetra_MultiVector(trilinos_->topology.ghost_map(), 1));

  // Construct a boundary vector for each coupled Neumann boundary condition
  trilinos_->bdryVec_list.clear();
  trilinos_->bdryCapVec_list.clear();
  for (int i = 0; i < numCoupledNeumannBC; ++i)
  {
    trilinos_->bdryVec_list.push_back(Teuchos::rcp(
        new Tpetra_MultiVector(trilinos_->topology.map(), 1)));
    trilinos_->bdryCapVec_list.push_back(Teuchos::rcp(
        new Tpetra_MultiVector(trilinos_->topology.map(), 1)));
  }

  // Initialize solution vector which is unique and does not include the ghost
  // indices using the unique map
  trilinos_->X = Teuchos::rcp(
      new Tpetra_Vector(trilinos_->topology.map()));

  //initialize vector which will import the ghost nodes using the ghost map
  trilinos_->ghostX = Teuchos::rcp(
      new Tpetra_Vector(trilinos_->topology.ghost_map()));

} // trilinos_lhs_create_

// ----------------------------------------------------------------------------
/**
 * This function assembles the dense element block stiffness matrix and element
 * block force vector into the global block stiffness K and global block force
 * vector F.
 * Happens on a single element and will be called repeatedly as we loop over
 * all the elements.
 *
 * \param numNodesPerElement number of nodes (d) for element e
 * \param eqN                converts local element node number to local proc
 *                           equation number-size is numNodesPerElement
 * \param lK                 element local stiffness of size
 *                           (dof*dof, numNodesPerElement, numNodesPerElement)
 * \param lR                 element local force vector of size
 *                           (dof, numNodesPerElement)
 */
void trilinos_doassem_(const Teuchos::RCP<Trilinos> &trilinos_, int &numNodesPerElement, const int *eqN, const double *lK, double *lR)
{
  #ifdef debug_trilinos_doassem
  std::cout << "[trilinos_doassem_] ========== trilinos_doassem_ ===========" << std::endl;
  std::cout << "[trilinos_doassem_] dof: " << trilinos_->topology.dof() << std::endl;
  std::cout << "[trilinos_doassem_] numNodesPerElement: " << numNodesPerElement << std::endl;
  #endif

  trilinos_->local_assembly.add_element(
      trilinos_->topology,
      *trilinos_->ghostF,
      numNodesPerElement,
      eqN,
      lK,
      lR);
} // trilinos_doassem_

// ----------------------------------------------------------------------------
/**
 * Tthis function creates the LHS structure topology and RHS vector based on
 * the block map using the global stiffness matrix and global force vector
 * assuming the assembly part has been done in svFSI
 *
 * \param Val         nonzero values of assembled global stiffness
 * \param RHS         global force vector
 * \param x           solution vector
 * \param dirW        gives information about the Dirichlet BC
 * \param resNorm     norm of solution
 * \param initNorm    initial norm of RHS
 * \param numIters    linear solver number of iterations
 * \param solverTime  time in linear solver
 * \param dB          log ratio for output
 * \param converged   neither or not converged in the max number of iterations
 * \param lsType      defines type of linear solver to use
 * \param relTol      default relative tolerance or as set in input file
 * \param maxIters    default max number of iterations for gmres per restart
 * \param kspace      specific for gmres dim of the stored Krylov space vectors
 * \param precondType defines type of preconditioner to use
 */

void trilinos_global_solve_(const Teuchos::RCP<Trilinos> &trilinos_, const double *Val, const double *RHS, double *x,
        const double *dirW, double &resNorm, double &initNorm, int &numIters,
        double &solverTime, double &dB, bool &converged, int &lsType,
        double &relTol, int &maxIters, int &kspace, int &precondType)
{
  const auto& topology = trilinos_->topology;
  const int dof = topology.dof();
  const int ghostAndLocalNodes = topology.ghost_and_local_nodes();
  const auto& nnzPerRow = topology.nonzeros_per_row();
  const auto& localToGlobalUnsorted =
      topology.local_to_global_unsorted();
  const auto& globalColInd = topology.global_column_indices();
  int nnzCount = 0; //cumulate count of block nnz per rows
  int count = 0;
  std::vector<Scalar_d> values(dof); // holds local matrix entries
  std::vector<GO> colGIDK(dof); // holds global column indices for K

  // loop over block rows owned by current proc using localToGlobal index pointer
  //
  for (int i = 0; i < ghostAndLocalNodes; ++i) {
    int numEntries = nnzPerRow[i]; //block per of entries per row
    const GO rowGID = localToGlobalUnsorted[i];
    const GO* colGIDs = &globalColInd[nnzCount];

    // Copy RHS values into ghostF 
    for (int j = 0; j < dof; ++j) {
        trilinos_->ghostF->replaceGlobalValue(rowGID * dof + j, 0, RHS[i * dof + j]);
    }
    // Tpetra assembly from FSILS assembly
    for (int j = 0; j < numEntries; ++j) {
      for (int l = 0; l < dof; ++l) { //loop over dof for bool to contruct
        GO rowGIDK = rowGID * dof + l; // global row index for K
        for (int m = 0; m < dof; ++m) {
          colGIDK[m] = colGIDs[j] * dof + m;
          values[m] = Val[count*dof*dof + l*dof + m];
        }
        // even though Val contains already assembled values we use sumIntoGlobalValues
        // because Tpetra can only store rows owned by the process and NOT ghost rows
        // when assemblying with FillComplete across processors missing contributions happens.
        // In this way we ensure all contributions are accounted for across processors, 
        // without duplicating contributions.
        trilinos_->K->sumIntoGlobalValues(rowGIDK, dof, values.data(), colGIDK.data());
      }
      count++;
    }

    nnzCount += numEntries;
  }
  // Call solver code which assembles K and F for shared processors
  bool flagFassem = false;

  trilinos_solve_(trilinos_, x, dirW, resNorm, initNorm, numIters,
          solverTime, dB, converged, lsType,
          relTol, maxIters, kspace, precondType, flagFassem);

} // trilinos_global_solve_

// ----------------------------------------------------------------------------
/**
 * This function uses the established data structure to solve the block linear
 * system and passes the solution vector back to fortran with the ghost nodes
 * filled in
 *
 * \param x           solution vector
 * \param dirW        gives information about Dirichlet BC
 * \param resNorm     norm of the final residual
 * \param initNorm    norm of initial resiual x_init = 0 ||b||
 * \param numIters    number of iterations in the linear solver
 * \param solverTime  time in the linear solver
 * \param dB          log ratio for output
 * \param converged   can pass in solver and preconditioner type too
 * \param lsType      defines type of linear solver to use
 * \param relTol      default relative tolerance or as set in input file
 * \param maxIters    default max number of iterations for gmres per restart
 * \param kspace      specific for gmres dim of the stored Krylov space vectors
 * \param precondType defines type of preconditioner to use
 * \param isFassem    determines if F is already assembled at ghost nodes
 */
void trilinos_solve_(const Teuchos::RCP<Trilinos> &trilinos_, double *x, const double *dirW, double &resNorm,
        double &initNorm, int &numIters, double &solverTime, double &dB,
        bool &converged, int &lsType, double &relTol, int &maxIters,
        int &kspace, int &precondType, bool &isFassem)
{
  #define n_debug_trilinos_solve
  #ifdef debug_trilinos_solve
  std::cout << "[trilinos_solve] ========== trilinos_solve ==========" << std::endl;
  std::cout << "[trilinos_solve] resNorm: " << resNorm << std::endl;
  std::cout << "[trilinos_solve] initNorm: " << initNorm << std::endl;
  std::cout << "[trilinos_solve] lsType: " << lsType << std::endl;
  std::cout << "[trilinos_solve] precondType: " << precondType << std::endl;
  std::cout << "[trilinos_solve] isFassem: " << isFassem << std::endl;
  #endif
  bool flagFassem = isFassem;

  // Native assembly stages local and ghost rows in one overlap matrix. Flush
  // it once so Tpetra performs a bulk device synchronization and MPI export.
  trilinos_->local_assembly.flush(
      trilinos_->topology, *trilinos_->K, *trilinos_->ghostF);
  if (!trilinos_->K->isFillComplete()) {
    trilinos_->K->fillComplete();
  }

  if (flagFassem) {
    Tpetra::Export exporter(trilinos_->ghostF->getMap(), trilinos_->F->getMap());
    trilinos_->F->doExport(*trilinos_->ghostF, exporter, Tpetra::ADD);
  } else { // RHS when using fsils assembly is already assembled and communicated
           // correctly among processors. REPLACE allows to create the correct
           // RHS vector of owned nodes only (no ghosts nodes)
    Tpetra::Export exporter(trilinos_->ghostF->getMap(), trilinos_->F->getMap());
    trilinos_->F->doExport(*trilinos_->ghostF, exporter, Tpetra::REPLACE);
  }

  // Construct Jacobi scaling vector which uses dirW to take the Dirichlet BC
  // into account
  //
  Teuchos::RCP<Tpetra_Vector> diagonal = Teuchos::rcp(
      new Tpetra_Vector(trilinos_->topology.map()));
  constructJacobiScaling(trilinos_, dirW, *diagonal);

  // Compute norm of preconditioned multivector F
  Teuchos::Array<double> norms(1);
  trilinos_->F->norm2(norms());
  initNorm = norms[0];

  Teuchos::RCP<TrilinosMatVec> K_bdry = Teuchos::rcp(new TrilinosMatVec(trilinos_));

  // Define Belos linear problem if v is 0 does standard matvec product with K
  /*
    K_bdry -> Tpetra operator to include resistance boundary terms in LHS
    X -> Tpetra vector for the solution
    F -> Tpetra vector for the RHS
  */
  auto BelosProblem = Teuchos::rcp(new Belos_LinearProblem(K_bdry, trilinos_->X, trilinos_->F));

  setPreconditioner(trilinos_, precondType, BelosProblem);

  bool set = BelosProblem->setProblem();
  if (!set) {
    std::cerr << "ERROR: Belos LinearProblem setup failed!" << std::endl;
    exit(1);
  }
  
  Teuchos::ParameterList belosParams;
  int verbosity = Belos::Errors 
                + Belos::Warnings   
                + Belos::TimingDetails   
                + Belos::StatusTestDetails  
                + Belos::IterationDetails;         

  int maxRestarts = int(maxIters / kspace)+1;

  belosParams.set("Verbosity", verbosity);
  belosParams.set("Output Frequency", 1);
  belosParams.set("Output Style", 1);  
  belosParams.set("Convergence Tolerance", relTol);
  belosParams.set("Maximum Iterations", maxIters); 
  belosParams.set("Maximum Restarts", maxRestarts);
  
  std::string solverType;

  #ifdef NOOUTPUT
    belosParams.set("Verbosity", Belos::Errors);
  #endif

  // Solver is GMRES by default
  if (lsType == TRILINOS_GMRES_SOLVER) {
    solverType = "Block GMRES";
    belosParams.set("Num Blocks", kspace); 
    belosParams.set("Orthogonalization", "DGKS"); // DGKS orthogonalization is the most general
                                                  // and is recommended for most problems
  }
  else if (lsType == TRILINOS_BICGSTAB_SOLVER) {
    solverType = "BiCGStab";
  }
  else if (lsType == TRILINOS_CG_SOLVER) {
    solverType = "Pseudoblock CG";
  }
  else {
    throw std::runtime_error("Unknown solver type requested");
  }
  
  //checkStatus to calculate residual norm
  Belos_SolverFactory factory;
  auto solverManager = factory.create(solverType, Teuchos::rcpFromRef(belosParams));
  solverManager->setProblem(BelosProblem);

  // Run the solver (solve() handles restarts internally)
  converged = false;

  Teuchos::Time timer("Belos Solve Timer");
  timer.start();

  Belos::ReturnType result = solverManager->solve();
  
  timer.stop();

  solverTime = timer.totalElapsedTime();

  if (result == Belos::Converged) {
    converged = true;
  }

  // Get number of iterations performed
  numIters = solverManager->getNumIters();

  // Get relative residual norm from the convergence test
  // Compute residual r = b - A*x manually since Belos does not provide it directly
  auto r = Teuchos::rcp(new Tpetra_MultiVector(trilinos_->F->getMap(), trilinos_->F->getNumVectors()));
  K_bdry->apply(*trilinos_->X, *r);
  r->update(1.0, *trilinos_->F, -1.0); // r = b - A*x

  // Compute residual norm
  Teuchos::Array<double> normR(1);
  r->norm2(normR());
  resNorm = normR[0];

  // Compute relative residual norm using precomputed initNorm
  double relRes = resNorm / initNorm;

  // Convert to decibel scale
  dB = 10.0 * log10(relRes);

  //Right scaling so need to multiply x by diagonal
  trilinos_->X->elementWiseMultiply(1.0, *trilinos_->X, *diagonal, 0.0);

  //Fill ghost X with x communicating ghost nodes amongst processors
  trilinos_->ghostX->doImport(
      *trilinos_->X, *trilinos_->topology.importer(), Tpetra::INSERT);

  auto localView = trilinos_->ghostX->getLocalViewHost(Tpetra::Access::ReadOnly);
  size_t localLength = trilinos_->ghostX->getLocalLength();
  if (localLength == 0) {
    std::cout << "ERROR: Extracting copy of solution vector!" << std::endl;
    exit(1);
  }
  for (size_t i = 0; i < localLength; ++i) {
    x[i] = localView(i, 0);
  }

  // Zero out residual and solution vectors, lhs and, if used,
  // any preconditioner objects
  trilinos_->ghostF->putScalar(0.0);
  trilinos_->F->putScalar(0.0);
  if (trilinos_->coupled_boundary) {
    for (auto bdryVec : trilinos_->bdryVec_list)
      bdryVec->putScalar(0.0);
    for (auto bdryCapVec : trilinos_->bdryCapVec_list)
      bdryCapVec->putScalar(0.0);

  }
  trilinos_->X->putScalar(0.0);

  if (trilinos_->MueluPrec != Teuchos::null) trilinos_->MueluPrec = Teuchos::null;
  if (trilinos_->ifpackPrec != Teuchos::null) trilinos_->ifpackPrec = Teuchos::null;
  if (trilinos_->resistancePrec != Teuchos::null) trilinos_->resistancePrec = Teuchos::null;
  trilinos_->resistanceFaces.clear();

  trilinos_->K = Teuchos::null;

} // trilinos_solve_

// ----------------------------------------------------------------------------
void setPreconditioner(const Teuchos::RCP<Trilinos> &trilinos_, int precondType, 
  Teuchos::RCP<Belos_LinearProblem>& BelosProblem)
{
  if (precondType == TRILINOS_DIAGONAL_PRECONDITIONER ||
      precondType == NO_PRECONDITIONER) {
    BelosProblem->setLeftPrec(Teuchos::null);
    return;
  }

  if (precondType == TRILINOS_RESISTANCE_PRECONDITIONER) {
    auto resistancePrec = Teuchos::rcp(new TrilinosResistanceOperator(
        trilinos_->K->getDomainMap(), trilinos_->bdryVec_list,
        trilinos_->bdryCapVec_list, trilinos_->resistanceFaces));

    // Precompute the MPI-invariant per-face quantities used by apply():
    //   ||S_f||^2 and alpha_f = -R_f / (1 + R_f ||S_f||^2).
    // These are logged immediately so the Trilinos and FSILS paths can be
    // compared before Belos starts iterating.
    resistancePrec->compute();
    trilinos_->resistanceFaces = resistancePrec->resistance_faces();
    trilinos_->resistancePrec = resistancePrec;

    // Left preconditioning makes Belos solve approximately
    //   P^{-1} A x = P^{-1} b
    // using the outlet inverse implemented in TrilinosResistanceOperator::apply().
    BelosProblem->setLeftPrec(trilinos_->resistancePrec);
    logResistancePreconditioner(trilinos_, "trilinos_resistance.log");
    return;
  }

  // Create Ifpack2 preconditioner
  Ifpack2::Factory factory;
  Teuchos::ParameterList precParams;    // Parameter list setup

  if (precondType == TRILINOS_BLOCK_JACOBI_PRECONDITIONER) {
    checkDiagonalIsZero(trilinos_);
    trilinos_->ifpackPrec = factory.create<Tpetra_CrsMatrix>("RELAXATION", trilinos_->K);
    precParams.set("relaxation: type", "Jacobi");
    precParams.set("relaxation: sweeps", 1);

  } else if (precondType == TRILINOS_ILU_PRECONDITIONER) {
    checkDiagonalIsZero(trilinos_);
    trilinos_->ifpackPrec = factory.create<Tpetra_CrsMatrix>("SCHWARZ", trilinos_->K);
    precParams.set("schwarz: inner preconditioner name", "ILUT");
    precParams.set("schwarz: combine mode", "Add");
    precParams.set("schwarz: overlap level", 1);
    precParams.set("fact: level-of-fill", 0); // Classic ILU(0)
     precParams.set("fact: relax value", 0.0);

  } else if (precondType == TRILINOS_ILUT_PRECONDITIONER) {
    checkDiagonalIsZero(trilinos_);
    trilinos_->ifpackPrec = factory.create<Tpetra_CrsMatrix>("SCHWARZ", trilinos_->K);
    precParams.set("schwarz: overlap level", 1);
    precParams.set("schwarz: inner preconditioner name", "ILUT");
    precParams.set("schwarz: combine mode", "Add");
    precParams.set("fact: ilut level-of-fill", 2.0);
    precParams.set("fact: drop tolerance", 1e-2);
    precParams.set("fact: relax value", 0.0);

  } else if (precondType == TRILINOS_RILUK0_PRECONDITIONER) {
    checkDiagonalIsZero(trilinos_);
    trilinos_->ifpackPrec = factory.create<Tpetra_CrsMatrix>("SCHWARZ", trilinos_->K);
    precParams.set("schwarz: inner preconditioner name", "RILUK"); 
    precParams.set("fact: level-of-fill", 0);
    precParams.set("fact: drop tolerance", 0.0);                   
    precParams.set("fact: relax value", 0.0);

  } else if (precondType == TRILINOS_RILUK1_PRECONDITIONER) {
    checkDiagonalIsZero(trilinos_);
    trilinos_->ifpackPrec = factory.create<Tpetra_CrsMatrix>("SCHWARZ", trilinos_->K);
    precParams.set("schwarz: inner preconditioner name", "RILUK"); 
    precParams.set("fact: level-of-fill", 1); 
    precParams.set("fact: drop tolerance", 1e-3);
    precParams.set("fact: relax value", 0.0);

  } else if (precondType == TRILINOS_ML_PRECONDITIONER) {
    checkDiagonalIsZero(trilinos_);
    setMueLuPreconditioner(trilinos_->MueluPrec, trilinos_->K);
    BelosProblem->setLeftPrec(trilinos_->MueluPrec);
    return;
  } else {
    throw std::runtime_error("[ERROR Trilinos] Unsupported preconditioner type.");
  }

  trilinos_->ifpackPrec->setParameters(precParams);
  trilinos_->ifpackPrec->initialize();
  trilinos_->ifpackPrec->compute();

  BelosProblem->setLeftPrec(trilinos_->ifpackPrec);

} // setPreconditioner

// ----------------------------------------------------------------------------
/*
 * Tune parameters for ML preconditioner using MueLu package
 * For a complete guide, refer to the MueLu documentation:
 * https://trilinos.github.io/pdfs/mueluguide.pdf
 */
void setMueLuPreconditioner(Teuchos::RCP<MueLu_Preconditioner> &MueLuPrec,
                            const Teuchos::RCP<Tpetra_CrsMatrix> &A)
{
  // MueLuPrec is now a Tpetra::Operator that can be plug into BelosProblem
  std::string optionsFile = "mueluOptions.xml";

  // The following built-in parameters proved to be generally good for big FSI problems
  Teuchos::ParameterList mueluParams;

  mueluParams.set("verbosity", "none");
  mueluParams.set("max levels", 6);   // number of multigrid levels
  mueluParams.set("cycle type", "V"); // V-cycle is standard
  mueluParams.set("fuse prolongation and update", true);

  // Problem type
  mueluParams.set("problem: type", "unknown"); // FSI is generally nonsymmetric
  mueluParams.set("number of equations", 4);   // dof

  // Aggregation
  mueluParams.set("aggregation: type", "uncoupled");
  mueluParams.set("aggregation: min agg size", 2);
  mueluParams.set("aggregation: max agg size", 8);
  mueluParams.set("aggregation: ordering", "natural");
  #if !defined(KOKKOS_ENABLE_CUDA)
    mueluParams.set("aggregation: drop scheme", "classical");
    mueluParams.set("aggregation: strength-of-connection: measure", "smoothed aggregation");
  #else
    // GPU compatibility: these MueLu aggregation options have triggered CUDA
    // build/runtime issues in Trilinos GPU builds, so keep the CUDA path on
    // MueLu's safer defaults.
  #endif
  mueluParams.set("aggregation: number of random vectors", 5);
  mueluParams.set("aggregation: number of times to pre or post smooth", 3);

  // Smoothed Aggregation-specific parameters
  mueluParams.set("sa: damping factor", 1.0);
  mueluParams.set("sa: use filtered matrix", false);

  // Smoother
  mueluParams.set("smoother: type", "RELAXATION");
  mueluParams.set("smoother: pre or post", "both");
  mueluParams.set("smoother: overlap", 0);

  Teuchos::ParameterList &smootherParams = mueluParams.sublist("smoother: params");
  smootherParams.set("relaxation: type", "Gauss-Seidel");
  smootherParams.set("relaxation: sweeps", 2);

  // Coarse solver
  mueluParams.set("coarse: type", "KLU"); // robust direct solver for FSI
  mueluParams.set("coarse: max size", 2000);

  // Create MueLu preconditioner from matrix and parameter list, as Tpetra::Operator
  std::ifstream ifs(optionsFile.c_str());
  if (ifs.good())
  {
    try
    {
      MueLuPrec = MueLu::CreateTpetraPreconditioner(
          Teuchos::rcp_static_cast<Tpetra_Operator>(A), optionsFile);
    }
    catch (std::exception &e)
    {
      std::cerr << "[MueLu Warning]: failed to create MueLu from file '" << optionsFile
                << "': " << e.what() << "\n  Falling back to built-in parameters.\n";
      MueLuPrec = MueLu::CreateTpetraPreconditioner(
          Teuchos::rcp_static_cast<Tpetra_Operator>(A), mueluParams);
    }
  }
  else
  {
    MueLuPrec = MueLu::CreateTpetraPreconditioner(
        Teuchos::rcp_static_cast<Tpetra_Operator>(A), mueluParams);
  }

}

// ----------------------------------------------------------------------------
/**
 * This routine is to be used with preconditioners such as BlockJacobi/ILU/ILUT
 * which require 1s on the diagonal
 */
void checkDiagonalIsZero(const Teuchos::RCP<Trilinos> &trilinos_)
{
  Teuchos::RCP<const Tpetra_Map> rowMap = trilinos_->K->getRowMap();
  Tpetra_Vector diagonal(rowMap);
  trilinos_->K->getLocalDiagCopy(diagonal);
  bool isZeroDiag = false;
  // Not Compatible with CUDA builds: 
  //  getLocalViewHost creates potential build/runtime issues in Trilinos CUDA builds. 
  // auto diagData = diagonal.getLocalViewHost(Tpetra::Access::ReadWrite);
  // for (size_t i = 0; i < diagData.extent(0); ++i)
  // {
  //   if (diagData(i, 0) == 0.0)
  //   {
  //     diagData(i, 0) = 1.0;
  //     isZeroDiag = true;
  //   }
  // }
  // New scope to ensure diagData dies before replaceDiagonalCrsMatrix()
  {
    // GPU compatibility: release the host view before calling back into Tpetra
    // to avoid DualView "modified on both host/device" lifetime conflicts.
    auto diagData = diagonal.getLocalViewHost(Tpetra::Access::ReadWrite);
    for (size_t i = 0; i < diagData.extent(0); ++i)
    {
      if (diagData(i, 0) == 0.0)
      {
        diagData(i, 0) = 1.0;
        isZeroDiag = true;
      }
    }
  } // diagData dies here; host view released

  Tpetra::replaceDiagonalCrsMatrix(*trilinos_->K, diagonal);

} // void checkDiagonalIsZero()

// ----------------------------------------------------------------------------
/**
 * To be called within solve-output diagonal from here
 *
 * \param dirW    pass in array with Dirichlet boundary face nodes marked
 * \paramdiagonal diagonal scaling vector need to output to multiply solution by
 */
void constructJacobiScaling(const Teuchos::RCP<Trilinos> &trilinos_, const double *dirW, Tpetra_Vector& diagonal)
{
  const auto& topology = trilinos_->topology;
  const int localNodes = topology.local_nodes();
  const int dof = topology.dof();
  const auto& localToGlobalSorted = topology.local_to_global_sorted();
  Teuchos::RCP<const Tpetra_Map> map = diagonal.getMap();

  // Start from the Dirichlet mask W_D supplied by svMultiPhysics.  Entries on
  // constrained rows/columns are zeroed in the same spirit as FSILS
  // precond_diag(), while unconstrained entries remain one.
  { // New scope to ensure any local variables die before later Tpetra operations
  for (int i = 0; i < localNodes; ++i) {
    for (int j = 0; j < dof; ++j) {
      GO gid = localToGlobalSorted[i] * dof + j;
      if (map->isNodeGlobalElement(gid)) {
        size_t lid = map->getLocalElement(gid);
        diagonal.replaceLocalValue(lid, dirW[i * dof + j]);
      } else {
        std::cerr << "[ERROR] Setting Dirichlet diagonal scaling value failed at GID " 
                  << gid << std::endl;
        exit(1);
      }
    }
  }
} // host writes to 'diagonal' (local replaceLocalValue) are now done

  // Extract diag(K) and compute the Jacobi factor:
  //   W_J(i) = 1 / sqrt(abs(K_ii)).
  // The Kokkos kernel keeps this large diagonal operation on the active device.
  Tpetra_Vector Kdiag(trilinos_->K->getRowMap());
  trilinos_->K->getLocalDiagCopy(Kdiag);

  // Not compatible with GPU builds
  // auto KdiagView = Kdiag.getLocalViewDevice(Tpetra::Access::ReadWrite);
  // using execution_space = typename Node::execution_space;
  // Kokkos::parallel_for("svmp_trilinos_jacobi_scaling",
  //     Kokkos::RangePolicy<execution_space>(0, KdiagView.extent(0)),
  //     KOKKOS_LAMBDA(const int i) {
  //       double diag = KdiagView(i, 0);
  //       if (diag == 0.0) {
  //         diag = 1.0;
  //       }
  //       KdiagView(i, 0) = 1.0 / sqrt(fabs(diag));
  //     });
  // Kokkos::fence();

  { // New scope to ensure KdiagView dies before any later Tpetra operations

    // GPU compatibility: keep the diagonal transform on the active Kokkos
    // execution space, but end the device view lifetime before later Tpetra
    // vector/matrix operations touch Kdiag.
    auto KdiagView = Kdiag.getLocalViewDevice(Tpetra::Access::ReadWrite);
    using execution_space = typename Node::execution_space;
    Kokkos::parallel_for("svmp_trilinos_jacobi_scaling",
        Kokkos::RangePolicy<execution_space>(0, KdiagView.extent(0)),
        KOKKOS_LAMBDA(const int i) {
          double diag = KdiagView(i, 0);
          if (diag == 0.0) {
            diag = 1.0;
          }
          KdiagView(i, 0) = 1.0 / sqrt(fabs(diag));
        });
    Kokkos::fence();
  } // Kdiagview dies here; device view released

  // Final symmetric scaling vector:
  //   W = W_D * W_J.
  diagonal.elementWiseMultiply(1.0, diagonal, Kdiag, 0.0);

  // Apply the same symmetric scaling as FSILS:
  //   K_tilde = W K W,    F_tilde = W F.
  trilinos_->K->leftScale(diagonal);
  trilinos_->F->elementWiseMultiply(1.0, diagonal, *trilinos_->F, 0.0);
  trilinos_->K->rightScale(diagonal);

  // Scale each outlet vector by the same W:
  //   S_f <- W S_f.
  // This is the Trilinos equivalent of FSILS face.valM = face.val * W.  The
  // norms used by the resistance preconditioner must be computed after this
  // step to match FSILS face.nS.
  if (trilinos_->coupled_boundary) {
    for (auto bdryVec : trilinos_->bdryVec_list) {
      bdryVec->elementWiseMultiply(1.0, diagonal, *bdryVec, 0.0);
    }
    for (auto bdryCapVec : trilinos_->bdryCapVec_list) {
      bdryCapVec->elementWiseMultiply(1.0, diagonal, *bdryCapVec, 0.0);
    }
  }
} // void constructJacobiScaling()

// ----------------------------------------------------------------------------
void logResistancePreconditioner(const Teuchos::RCP<Trilinos> &trilinos_,
    const std::string& file_name)
{
  if (trilinos_->comm != Teuchos::null && trilinos_->comm->getRank() != 0) {
    return;
  }

  std::ofstream log(file_name);
  log << std::scientific << std::setprecision(17);

  // One line per coupled face.  These columns intentionally match the FSILS
  // logger in gmres.cpp/ns_solver.cpp for a direct diff:
  //   resistance      = R_f
  //   s_tilde_norm2   = ||W S_f||^2
  //   s_tilde_norm    = sqrt(||W S_f||^2)
  //   alpha           = -R_f / (1 + R_f ||W S_f||^2)
  log << "# face_id resistance s_tilde_norm2 s_tilde_norm alpha\n";

  for (const auto& face : trilinos_->resistanceFaces) {
    if (face.resistance == 0.0) {
      continue;
    }

    log << face.face_id << " "
        << face.resistance << " "
        << face.s_tilde_norm2 << " "
        << std::sqrt(face.s_tilde_norm2) << " "
        << face.alpha << "\n";
  }
}

// ----------------------------------------------------------------------------
/**
 * \param  v            coupled boundary vector
 * \param  isCoupledBC  determines if coupled resistance BC is turned on
 */
void trilinos_bc_create_(const Teuchos::RCP<Trilinos> &trilinos_,
  const std::vector<Array<double>> &v_list, const std::vector<Array<double>> &vcap_list, bool &isCoupledBC)
{
  trilinos_->coupled_boundary = isCoupledBC;

  if (isCoupledBC)
  {
    const auto& topology = trilinos_->topology;
    const int dof = topology.dof();
    const int ghostAndLocalNodes = topology.ghost_and_local_nodes();
    const auto& localToGlobalSorted = topology.local_to_global_sorted();
    // v_list/vcap_list are first built on the local+ghost node layout because
    // outlet faces may be split by MPI partitions.  Exporting ghostMap -> Map
    // with ADD globally sums shared face contributions onto the unique Tpetra
    // map.  This is what makes the later dot products and norms independent of
    // the number of MPI ranks.
    Tpetra::Export<LO, GO, Node> exporter(
        topology.ghost_map(), topology.map());

    for (int k = 0; k < v_list.size(); ++k) {
      const double* v = v_list[k].data();
      const double* vcap = vcap_list[k].data();

      Tpetra_MultiVector ghostBdryVec(topology.ghost_map(), 1);
      Tpetra_MultiVector ghostBdryCapVec(topology.ghost_map(), 1);
      ghostBdryVec.putScalar(0.0);
      ghostBdryCapVec.putScalar(0.0);

      for (int i = 0; i < ghostAndLocalNodes; ++i) {
        auto globalRow = static_cast<GO>(localToGlobalSorted[i]);

        for (int j = 0; j < dof; ++j)
        {
          ghostBdryVec.replaceGlobalValue(globalRow * dof + j, 0, v[i * dof + j]);
          ghostBdryCapVec.replaceGlobalValue(globalRow * dof + j, 0, vcap[i * dof + j]);
        }
      }

      trilinos_->bdryVec_list[k]->putScalar(0.0);
      trilinos_->bdryCapVec_list[k]->putScalar(0.0);

      // Global face assembly:
      //   S_f(global owned dofs) = sum_r S_f(local/ghost dofs on rank r).
      trilinos_->bdryVec_list[k]->doExport(ghostBdryVec, exporter, Tpetra::ADD);
      trilinos_->bdryCapVec_list[k]->doExport(ghostBdryCapVec, exporter, Tpetra::ADD);
    }
  }
}

// ----------------------------------------------------------------------------
/**
 * for debugging purposes here are routines to print the matrix and RHS vector
 */
void printMatrixToFile(const Teuchos::RCP<Trilinos> &trilinos_)
{
  const auto& topology = trilinos_->topology;
  const int dof = topology.dof();
  const int ghostAndLocalNodes = topology.ghost_and_local_nodes();
  const auto& localToGlobalUnsorted =
      topology.local_to_global_unsorted();
  std::ofstream Kfile("K.txt");
  Kfile << std::scientific;
  Kfile.precision(17);

  Teuchos::RCP<const Tpetra_CrsMatrix> K = trilinos_->K;
  Teuchos::RCP<const Tpetra_Map> rowMap = K->getRowMap();

  for (int i = 0; i < ghostAndLocalNodes; ++i)
  {
    int nodeGlobalID = localToGlobalUnsorted[i];

    // Loop over each DoF in the block row
    for (int d = 0; d < dof; ++d)
    {
      int rowGID = nodeGlobalID * dof + d;

      // Get number of entries in the row
      size_t maxEntries = K->getNumEntriesInGlobalRow(rowGID);
      Tpetra_CrsMatrix::nonconst_global_inds_host_view_type indices("indices", maxEntries);
      Tpetra_CrsMatrix::nonconst_values_host_view_type values("values", maxEntries);

      size_t numEntries = 0;

      // Now get the actual data
      K->getGlobalRowCopy(rowGID, indices, values, numEntries);

      for (size_t j = 0; j < numEntries; ++j)
      {
        int colGID = indices(j);
        double val = values(j);
        Kfile << "Row " << rowGID << " Col " << colGID << " Val " << val << std::endl;
      }
    }
  }

  Kfile.close();
}

void printRHSToFile(const Teuchos::RCP<Trilinos> &trilinos_)
{
  std::ofstream Ffile("F.txt");
  Ffile.precision(17);

  // Get the local length of the Tpetra::MultiVector (assumes single vector)
  Teuchos::RCP<const Tpetra_MultiVector> F = trilinos_->F;
  std::size_t localLength = F->getLocalLength();

  // Create a view of the local data
  Teuchos::ArrayRCP<const double> F_local = F->getData(0); // 0 = first vector in MultiVector

  // Print each value to file
  for (std::size_t i = 0; i < localLength; ++i)
    Ffile << F_local[i] << std::endl;

  Ffile.close();
}

// ----------------------------------------------------------------------------
/**
 */
void printSolutionToFile(const Teuchos::RCP<Trilinos> &trilinos_)
{
  std::ofstream Xfile("X.txt");
  Xfile.precision(17);

  // Get the local length of the Tpetra::MultiVector (assumes single vector)
  Teuchos::RCP<const Tpetra_MultiVector> X = trilinos_->X;
  std::size_t localLength = X->getLocalLength();

  // Create a view of the local data
  Teuchos::ArrayRCP<const double> X_local = X->getData(0); // 0 = first vector in MultiVector

  // Print each value to file
  for (std::size_t i = 0; i < localLength; ++i)
    Xfile << X_local[i] << std::endl;

  Xfile.close();
}

/////////////////////////////////////////////////////////////////
//                  T r i l i n o s I m p l                    //
/////////////////////////////////////////////////////////////////

//--------------
// TrilinosImpl 
//--------------
// The TrilinosImpl private class hides Trilinos data structures
// and functions.
//
class TrilinosLinearAlgebra::TrilinosImpl {
  public:
    TrilinosImpl():
      trilinos_(Teuchos::rcp(new Trilinos())),
      momentum_muelu_cache_(Teuchos::rcp(
          new trilinos_bipartition::preconditioners::MueLuReuseCache())),
      pressure_muelu_cache_(Teuchos::rcp(
          new trilinos_bipartition::preconditioners::MueLuReuseCache())) {}
    void alloc(ComMod& com_mod, eqType& lEq, bool native_assembly);
    void assemble(ComMod& com_mod, const int num_elem_nodes, const Vector<int>& eqN,
        const Array3<double>& lK, const Array<double>& lR);
    void initialize(ComMod& com_mod);
    void solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, const Vector<double>& res);
    void solve_assembled(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, const Vector<double>& res);
    void init_dir_and_coup_neu(ComMod& com_mod, const Vector<int>& incL, const Vector<double>& res);
    void set_preconditioner(consts::PreconditionerType preconditioner);
    void finalize();

    consts::PreconditionerType preconditioner_;

    /// @brief Local to global mapping
    Vector<int> ltg_;

    /// @brief Factor for Dirichlet BCs
    Array<double> W_;

    /// @brief Residual
    Array<double> R_;
  
  private:
    Teuchos::RCP<Trilinos> trilinos_;
    Teuchos::RCP<trilinos_bipartition::preconditioners::MueLuReuseCache>
        momentum_muelu_cache_;
    Teuchos::RCP<trilinos_bipartition::preconditioners::MueLuReuseCache>
        pressure_muelu_cache_;
    bool runtime_registered_ = false;
};

/// @brief Allocate Trilinos arrays.
void TrilinosLinearAlgebra::TrilinosImpl::alloc(
    ComMod& com_mod,
    eqType& lEq,
    bool native_assembly)
{
  int dof = com_mod.dof;
  int tnNo = com_mod.tnNo;
  int gtnNo = com_mod.gtnNo;
  auto& lhs = com_mod.lhs;
  #ifdef n_debug_alloc
  std::cout << "[TrilinosImpl.alloc] dof: " << dof << std::endl;
  std::cout << "[TrilinosImpl.alloc] tnNo: " << tnNo << std::endl;
  std::cout << "[TrilinosImpl.alloc] gtnNo: " << gtnNo << std::endl;
  std::cout << "[TrilinosImpl.alloc] ltg_.size(): " << ltg_.size() << std::endl;
  #endif

  if (W_.size() != 0) {
    W_.clear();
    R_.clear();
  }

  W_.resize(dof,tnNo); 
  R_.resize(dof,tnNo);

  int cpp_index = 1;
  int task_id = com_mod.cm.idcm();

  trilinos_lhs_create(trilinos_, gtnNo, lhs.mynNo, tnNo, lhs.nnz, ltg_, com_mod.ltg, com_mod.rowPtr, 
      com_mod.colPtr, dof, cpp_index, task_id, com_mod.lhs.nFaces);

  if (native_assembly) {
    trilinos_->local_assembly.reset(trilinos_->topology);
  }
}

/// @brief Assemble local element arrays.
void TrilinosLinearAlgebra::TrilinosImpl::assemble(ComMod& com_mod, const int num_elem_nodes, const Vector<int>& eqN,
        const Array3<double>& lK, const Array<double>& lR)
{
  trilinos_doassem_(trilinos_, const_cast<int&>(num_elem_nodes), eqN.data(), lK.data(), lR.data());
}

/// @brief Set data for Dirichlet and coupled Neumann boundary conditions.
void TrilinosLinearAlgebra::TrilinosImpl::init_dir_and_coup_neu(ComMod& com_mod, const Vector<int>& incL, const Vector<double>& res)
{
  using namespace consts;
  using namespace fsi_linear_solver;

  int dof = com_mod.dof;
  int gtnNo = com_mod.gtnNo;
  int tnNo = com_mod.tnNo;
  auto& lhs = com_mod.lhs;

  if (lhs.nFaces != 0) {
    for (auto& face : lhs.face) {
      face.incFlag = true;
    }

    for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
      if (incL(faIn) == 0)  {
        lhs.face[faIn].incFlag = false;
      }
    }

    for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
      auto& face = lhs.face[faIn];
      face.coupledFlag = false;
      if (!face.incFlag) {
        continue;
      }

      bool flag = (face.bGrp == BcType::BC_TYPE_Neu);
      if (flag && res(faIn) != 0.0) {
        face.res = res(faIn);
        face.coupledFlag = true;
      }
    }
  }

  W_ = 1.0;

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    auto& face = lhs.face[faIn];
    if (!face.incFlag) {
      continue;
    }

    int faDof = std::min(face.dof,dof);

    if (face.bGrp == BcType::BC_TYPE_Dir) {
      for (int a = 0; a < face.nNo; a++) {
        int Ac = face.glob(a);
        for (int i = 0; i < faDof; i++) {
          W_(i,Ac) = W_(i,Ac) * face.val(i,a);
        }
      }
    }
  }

  std::vector<Array<double>> v_list;
  std::vector<Array<double>> vcap_list;
  bool isCoupledBC = false;
  trilinos_->resistanceFaces.clear();

  for (int faIn = 0; faIn < lhs.nFaces; faIn++) {
    // Extract face
    auto& face = lhs.face[faIn]; 

    // Create a new array for each face and add it to the list
    v_list.push_back(Array<double>(dof,tnNo));
    auto& v = v_list.back();
    vcap_list.push_back(Array<double>(dof,tnNo));
    auto& vcap = vcap_list.back();

    Trilinos::ResistanceFaceData face_data;
    face_data.face_id = faIn;
    face_data.resistance = face.coupledFlag ? res(faIn) : 0.0;
    trilinos_->resistanceFaces.push_back(face_data);
    
    if (face.coupledFlag) {
      isCoupledBC = true;
      int faDof = std::min(face.dof,dof);

      // Build the unassembled outlet vector on this rank:
      //   v_f = sqrt(abs(R_f)) * S_f,
      // where S_f contains the integrated outlet shape-function normal
      // contributions, int_Gamma N_A n_i dGamma.  The sqrt(abs(R_f)) scaling
      // lets the matrix-free operator represent R_f S_f S_f^T as
      // sign(R_f) v_f v_f^T.
      for (int a = 0; a < face.nNo; a++) {
        int Ac = face.glob(a);
        for (int i = 0; i < faDof; i++) {
          v(i,Ac) = v(i,Ac) + sqrt(fabs(res(faIn))) * face.val(i,a);
        }
      }
      
      // Cap surfaces contribute to the scalar flow projection
      // (S_f + C_f)^T x, but not to the update vector that receives pressure
      // coupling.  Store C_f separately so TrilinosResistanceOperator::apply() can
      // add it only to the dot product.
      if (face.cap_val.size() != 0 && face.cap_glob.size() != 0) {
        int cap_nNo = face.cap_val.ncols();
        for (int a = 0; a < cap_nNo; a++) {
          int Ac = face.cap_glob(a);
          if (Ac < 0) continue;  // cap node not on this rank
          for (int i = 0; i < faDof; i++) {
            vcap(i,Ac) = vcap(i,Ac) + sqrt(fabs(res(faIn))) * face.cap_val(i,a);
          }
        }
      }
    }
  }

  // Add the v vectors to global bdryVec_list
  trilinos_bc_create_(trilinos_, v_list, vcap_list, isCoupledBC);

}

/// @brief Initialze an array used for something.
void TrilinosLinearAlgebra::TrilinosImpl::initialize(ComMod& com_mod)
{
  #ifdef n_debug_initialize
  std::cout << "[TrilinosImpl] ---------- initialize ---------- " << std::endl;
  std::cout << "[TrilinosImpl.initialize] com_mod.tnNo: " << com_mod.tnNo << std::endl;
  #endif

  if (!runtime_registered_) {
    std::lock_guard<std::mutex> lock(trilinos_runtime_mutex());
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
    ++trilinos_runtime_users();
    runtime_registered_ = true;
  }

  ltg_.resize(com_mod.tnNo);

  for (int a = 0; a < com_mod.tnNo; a++) {
    ltg_(com_mod.lhs.map(a)) = com_mod.ltg(a);
  }
}

void TrilinosLinearAlgebra::TrilinosImpl::finalize()
{
  if (!runtime_registered_) {
    return;
  }

  // Every Tpetra object must be released before the final equation shuts down
  // the process-wide Kokkos runtime.
  trilinos_->MueluPrec = Teuchos::null;
  trilinos_->ifpackPrec = Teuchos::null;
  trilinos_->resistancePrec = Teuchos::null;
  trilinos_->K = Teuchos::null;
  trilinos_->F = Teuchos::null;
  trilinos_->ghostF = Teuchos::null;
  trilinos_->X = Teuchos::null;
  trilinos_->ghostX = Teuchos::null;
  trilinos_->bdryVec_list.clear();
  trilinos_->bdryCapVec_list.clear();
  trilinos_->resistanceFaces.clear();
  momentum_muelu_cache_->clear();
  pressure_muelu_cache_->clear();
  momentum_muelu_cache_ = Teuchos::null;
  pressure_muelu_cache_ = Teuchos::null;
  trilinos_->local_assembly.clear();
  trilinos_->topology.clear();
  trilinos_->comm = Teuchos::null;

  std::lock_guard<std::mutex> lock(trilinos_runtime_mutex());
  auto& users = trilinos_runtime_users();
  if (users == 0) {
    throw std::runtime_error(
        "[TrilinosLinearAlgebra] ERROR: invalid runtime user count.");
  }
  --users;
  runtime_registered_ = false;
  if (users == 0 && Kokkos::is_initialized()) {
    Kokkos::finalize();
  }
}

/// @brief Set the preconditioner.
void TrilinosLinearAlgebra::TrilinosImpl::set_preconditioner(consts::PreconditionerType prec_type)
{
  preconditioner_ = prec_type;
}

/// @brief Solve a system of linear equations assembled by fsils.
void TrilinosLinearAlgebra::TrilinosImpl::solve(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, 
    const Vector<double>& res)
{
  init_dir_and_coup_neu(com_mod, incL, res);

  auto& Val = com_mod.Val;
  auto& R = com_mod.R;
  int solver_type = static_cast<int>(lEq.ls.LS_type);
  int prec_type = static_cast<int>(preconditioner_);
  #define n_debug_solve
  #ifdef debug_solve
  std::cout << "[TrilinosImpl::solve] ---------- solve ---------- " << std::endl;
  auto prec_name = consts::preconditioner_type_to_name.at(preconditioner_); 
  std::cout << "[TrilinosImpl::solve] solver_type: " << solver_type << std::endl;
  std::cout << "[TrilinosImpl::solve] prec_type: " << prec_name << std::endl;
  std::cout << "[TrilinosImpl::solve] Val.size(): " << Val.size() << std::endl;
  std::cout << "[TrilinosImpl::solve] R_.size(): " << R_.size() << std::endl;
  std::cout << "[TrilinosImpl::solve] W_.size(): " << W_.size() << std::endl;
  std::cout << "[TrilinosImpl::solve] Call trilinos_global_solve_ " << std::endl;
  #endif

  if (consts::trilinos_preconditioners.count(preconditioner_) == 0) {
    auto prec_name = consts::preconditioner_type_to_name.at(preconditioner_); 
    throw std::runtime_error("[TrilinosLinearAlgebra::solve] ERROR: '" + prec_name + "' is not a valid Trilinos preconditioner.");
  }

  if (lEq.ls.LS_type == consts::SolverType::lSolver_NS) {
    trilinos_bipartition::TrilinosBipartitionNSSolver ns_solver(
        trilinos_, com_mod.nsd, com_mod.dof, com_mod.cTS,
        momentum_muelu_cache_, pressure_muelu_cache_);
    ns_solver.solve_fsils_assembled(
        lEq, Val.data(), R.data(), R_.data(), W_.data());

    for (int a = 0; a < com_mod.tnNo; ++a) {
      for (int i = 0; i < com_mod.R.nrows(); ++i) {
        com_mod.R(i, a) = R_(i, com_mod.lhs.map(a));
      }
    }
    return;
  }

  trilinos_global_solve_(trilinos_, Val.data(), R.data(), R_.data(), W_.data(), lEq.FSILS.RI.fNorm,
      lEq.FSILS.RI.iNorm, lEq.FSILS.RI.itr, lEq.FSILS.RI.callD, lEq.FSILS.RI.dB, lEq.FSILS.RI.suc,
      solver_type, lEq.FSILS.RI.relTol, lEq.FSILS.RI.mItr, lEq.FSILS.RI.sD, prec_type);

  for (int a = 0; a < com_mod.tnNo; a++) {
    for (int i = 0; i < com_mod.R.nrows(); i++) {
      com_mod.R(i,a) = R_(i,com_mod.lhs.map(a));
    }
  } 
}

/// @brief Solve a system of linear equations assembled by Trilinos.
void TrilinosLinearAlgebra::TrilinosImpl::solve_assembled(ComMod& com_mod, eqType& lEq, const Vector<int>& incL, const Vector<double>& res)
{
  lEq.FSILS.RI.suc = false; 
  int solver_type = static_cast<int>(lEq.ls.LS_type);
  int prec_type = static_cast<int>(preconditioner_);
  bool assembled = true;
  #define n_debug_solve_assembled
  #ifdef debug_solve_assembled
  auto prec_name = consts::preconditioner_type_to_name.at(preconditioner_); 
  std::cout << "[TrilinosImpl::solve_assembled] ---------- solve_assembled ---------- " << std::endl;
  std::cout << "[TrilinosImpl::solve_assembled] solver_type: " << solver_type << std::endl;
  std::cout << "[TrilinosImpl::solve_assembled] prec_type: " << prec_name << std::endl;
  std::cout << "[TrilinosImpl::solve_assembled] R_.size(): " << R_.size() << std::endl;
  std::cout << "[TrilinosImpl::solve_assembled] W_.size(): " << W_.size() << std::endl;
  std::cout << "[TrilinosImpl::solve_assembled] lEq.FSILS.RI.suc: " << lEq.FSILS.RI.suc << std::endl;
  #endif

  if (consts::trilinos_preconditioners.count(preconditioner_) == 0) {
    auto prec_name = consts::preconditioner_type_to_name.at(preconditioner_);
    throw std::runtime_error("[TrilinosLinearAlgebra::solve_assembled] ERROR: '" + prec_name + "' is not a valid Trilinos preconditioner.");
  }

  init_dir_and_coup_neu(com_mod, incL, res);

  if (lEq.ls.LS_type == consts::SolverType::lSolver_NS) {
    trilinos_bipartition::TrilinosBipartitionNSSolver ns_solver(
        trilinos_, com_mod.nsd, com_mod.dof, com_mod.cTS,
        momentum_muelu_cache_, pressure_muelu_cache_);
    ns_solver.solve_assembled(lEq, R_.data(), W_.data());

    for (int a = 0; a < com_mod.tnNo; ++a) {
      for (int i = 0; i < com_mod.R.nrows(); ++i) {
        com_mod.R(i, a) = R_(i, com_mod.lhs.map(a));
      }
    }
    return;
  }

  trilinos_solve_(trilinos_, R_.data(), W_.data(), lEq.FSILS.RI.fNorm, lEq.FSILS.RI.iNorm, 
      lEq.FSILS.RI.itr, lEq.FSILS.RI.callD, lEq.FSILS.RI.dB, lEq.FSILS.RI.suc, 
      solver_type, lEq.FSILS.RI.relTol, lEq.FSILS.RI.mItr, lEq.FSILS.RI.sD, 
      prec_type, assembled);

  for (int a = 0; a < com_mod.tnNo; a++) {
    for (int i = 0; i < com_mod.R.nrows(); i++) {
      com_mod.R(i,a) = R_(i,com_mod.lhs.map(a));
    }
  }
}
