// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRILINOS_LINEAR_SOLVER_H
#define TRILINOS_LINEAR_SOLVER_H
/*!
  \file    trilinos_linear_solver.h
  \brief   wrap Trilinos solver functions
*/

/**************************************************************/
/*                          Includes                          */
/**************************************************************/

#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include "mpi.h"
#include <time.h>
#include <numeric>
#include <unordered_map>

#include "Array.h"
#include "Vector.h"

// Theuchos includes
#include "Teuchos_RCP.hpp"
#include "Teuchos_DefaultComm.hpp"
#include <Teuchos_Time.hpp> 

#include "Kokkos_Core.hpp" 
#include "Tpetra_KokkosCompat_ClassicNodeAPI_Wrapper.hpp"
#include "Xpetra_TpetraCrsMatrix.hpp" 
#include "MueLu_TpetraOperator.hpp"  
#include "Xpetra_Matrix.hpp"

// Kokkos includes
// #include "KokkosCompat_KokkosSerialWrapperNode.hpp"  // For Node type

// Tpetra includes
#include "Tpetra_Core.hpp"                           // For Tpetra::initialize, finalize
#include "Tpetra_Map.hpp"                            // For Tpetra::Map
#include "Tpetra_CrsMatrix.hpp"                      // If you use Tpetra::CrsMatrix
#include "Tpetra_MultiVector.hpp" 
#include "Tpetra_Map_decl.hpp"
#include "NOX_TpetraTypedefs.hpp"

//Belos includes
#include "BelosSolverFactory_Tpetra.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosBiCGStabSolMgr.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"
#include "BelosSolverFactory.hpp"
#include "BelosStatusTestResNorm.hpp"
#include "BelosStatusTestMaxIters.hpp"
#include "BelosStatusTestGenResNorm.hpp"
#include <BelosStatusTestImpResNorm.hpp>
#include "BelosStatusTestCombo.hpp"

//Ifpack2 includes
#include "Ifpack2_Factory.hpp"   

// MueLu includes
#include "MueLu_CreateTpetraPreconditioner.hpp"

/**************************************************************/
/*                      Types Definitions                     */
/**************************************************************/
/* Scalar types aliases */
using Scalar_d = double;
using Scalar_i = int;
using Scalar_c = std::complex<double>;

/* Ordinals and node aliases */
using LO = int;
using GO = int;
using Node = Tpetra::Map<>::node_type;

/* Tpetra type aliases */
using Tpetra_Map            = Tpetra::Map<LO, GO, Node>;
using Tpetra_CrsMatrix      = Tpetra::CrsMatrix<Scalar_d, LO, GO, Node>;
using Tpetra_BlockCrsMatrix = Tpetra::BlockCrsMatrix<Scalar_d, LO, GO, Node>;
using Tpetra_MultiVector    = Tpetra::MultiVector<Scalar_d, LO, GO, Node>; 
using Tpetra_Vector         = Tpetra::Vector<Scalar_d, LO, GO, Node>;
using Tpetra_Import         = Tpetra::Import<LO, GO, Node>;
using Tpetra_Export         = Tpetra::Export<LO, GO, Node>;
using Tpetra_CrsGraph       = Tpetra::CrsGraph<LO, GO, Node>;
using Tpetra_Operator       = Tpetra::Operator<Scalar_d, LO, GO, Node>;

/* Belos aliases */
using Belos_LinearProblem = Belos::LinearProblem<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;
using Belos_SolverFactory =  Belos::TpetraSolverFactory<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;
using Belos_SolverManager = Belos::SolverManager<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;
using Belos_StatusTestGenResNorm = Belos::StatusTestGenResNorm<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;
using Belos_StatusTestCombo = Belos::StatusTestCombo<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;
using Belos_StatusTestMaxIters = Belos::StatusTestMaxIters<Scalar_d, Tpetra_MultiVector, Tpetra_Operator>;

/* IFPACK2 preconditioner aliases */  
using Ifpack2_Preconditioner = Ifpack2::Preconditioner<Scalar_d, LO, GO, Node>;

/* MueLu preconditioner aliases */
using MueLu_Preconditioner = Tpetra_Operator;

/**
 * @namespace trilinos_backend
 * @brief Internal Tpetra ownership and lifecycle support for svMultiPhysics.
 */
namespace trilinos_backend {

/**
 * @class TopologyCache
 * @brief Owns equation-local Tpetra maps, communication plans, and graphs.
 *
 * An equation's node distribution and connectivity normally remain unchanged
 * across Newton iterations and time steps. ensure() compares the complete
 * structural signature and reuses these expensive Tpetra objects on a match.
 * Matrix values and solve vectors are intentionally not cached here.
 */
class TopologyCache
{
  public:
    /**
     * @brief Ensure that cached topology matches the supplied equation layout.
     * @return @c true when the topology was rebuilt, or @c false when reused.
     * @throws std::runtime_error If dimensions or connectivity are invalid.
     */
    bool ensure(
        const Teuchos::RCP<const Teuchos::Comm<int>>& communicator,
        int num_global_nodes,
        int num_local_nodes,
        int num_ghost_and_local_nodes,
        int nnz,
        const Vector<int>& local_to_global_sorted,
        const Vector<int>& local_to_global_unsorted,
        const Vector<int>& row_pointer,
        const Vector<int>& column_indices,
        int dof,
        int index_base);

    /// @brief Return whether ensure() has built a valid topology.
    bool initialized() const;

    /// @brief Release all Tpetra topology objects before Kokkos finalization.
    void clear();

    /// @brief Return the number of structural graph generations built.
    std::size_t generation() const;

    /// @brief Return the nodal degree-of-freedom count in the signature.
    int dof() const;

    /// @brief Return the number of owned nodes on this rank.
    int local_nodes() const;

    /// @brief Return the number of local-plus-ghost nodes on this rank.
    int ghost_and_local_nodes() const;

    /// @brief Return the owned scalar-DOF map.
    const Teuchos::RCP<const Tpetra_Map>& map() const;

    /// @brief Return the local-plus-ghost scalar-DOF map.
    const Teuchos::RCP<const Tpetra_Map>& ghost_map() const;

    /// @brief Return the fill-complete static scalar sparsity graph.
    const Teuchos::RCP<Tpetra_CrsGraph>& graph() const;

    /// @brief Return the overlapping graph used for element assembly.
    const Teuchos::RCP<Tpetra_CrsGraph>& assembly_graph() const;

    /// @brief Return the owned-to-ghost solution importer.
    const Teuchos::RCP<Tpetra_Import>& importer() const;

    /// @brief Return the overlap-to-owned matrix exporter.
    const Teuchos::RCP<Tpetra_Export>& assembly_exporter() const;

    /**
     * @brief Return local CRS offsets for one nodal matrix block.
     * @throws std::runtime_error If the node pair is absent from the graph.
     *
     * The returned array contains @c dof()*dof() offsets in row-major
     * component order and remains valid until the topology is rebuilt.
     */
    const std::size_t* assembly_block_offsets(
        int local_row_node,
        int local_column_node) const;

    /**
     * @brief Return the overlapping vector LID for a node component.
     * @throws std::runtime_error If either index is outside the topology.
     */
    LO assembly_vector_lid(int local_node, int component) const;

    /// @brief Return sorted local-to-global node IDs.
    const std::vector<int>& local_to_global_sorted() const;

    /// @brief Return unsorted local-to-global node IDs.
    const std::vector<int>& local_to_global_unsorted() const;

    /// @brief Return global node IDs for each nodal CSR column entry.
    const std::vector<int>& global_column_indices() const;

    /// @brief Return nodal nonzero counts in unsorted local row order.
    const std::vector<int>& nonzeros_per_row() const;

  private:
    bool signature_matches(
        const Teuchos::RCP<const Teuchos::Comm<int>>& communicator,
        int num_global_nodes,
        int num_local_nodes,
        int num_ghost_and_local_nodes,
        int nnz,
        const Vector<int>& local_to_global_sorted,
        const Vector<int>& local_to_global_unsorted,
        const Vector<int>& row_pointer,
        const Vector<int>& column_indices,
        int dof,
        int index_base) const;

    bool initialized_ = false;
    std::size_t generation_ = 0;
    int communicator_size_ = 0;
    int communicator_rank_ = 0;
    int num_global_nodes_ = 0;
    int num_local_nodes_ = 0;
    int num_ghost_and_local_nodes_ = 0;
    int nnz_ = 0;
    int dof_ = 0;
    int index_base_ = 0;

    std::vector<int> signature_local_to_global_sorted_;
    std::vector<int> signature_local_to_global_unsorted_;
    std::vector<int> signature_row_pointer_;
    std::vector<int> signature_column_indices_;

    std::vector<int> local_to_global_sorted_;
    std::vector<int> local_to_global_unsorted_;
    std::vector<int> global_column_indices_;
    std::vector<int> nonzeros_per_row_;

    Teuchos::RCP<const Tpetra_Map> map_;
    Teuchos::RCP<const Tpetra_Map> ghost_map_;
    Teuchos::RCP<Tpetra_CrsGraph> graph_;
    Teuchos::RCP<Tpetra_CrsGraph> assembly_graph_;
    Teuchos::RCP<Tpetra_Import> importer_;
    Teuchos::RCP<Tpetra_Export> assembly_exporter_;
    std::vector<std::size_t> assembly_entry_offsets_;
    std::vector<LO> assembly_vector_lids_;
    std::vector<std::unordered_map<int, std::size_t>>
        assembly_nodal_entry_lookup_;
};

/**
 * @class LocalAssemblyBuffer
 * @brief Accumulates host element tensors in an overlapping local CRS matrix.
 *
 * Element kernels currently produce small dense arrays in host memory. This
 * buffer writes them directly into the final local CRS ordering and performs
 * one bulk host-to-device synchronization before exporting shared rows to
 * their owning MPI ranks. The class owns only per-Jacobian values; its graph
 * and communication plan come from TopologyCache.
 */
class LocalAssemblyBuffer
{
  public:
    /// @brief Allocate and zero values for a new Jacobian assembly.
    void reset(const TopologyCache& topology);

    /// @brief Add one dense element matrix and residual to overlapping storage.
    void add_element(
        const TopologyCache& topology,
        Tpetra_MultiVector& ghost_rhs,
        int num_element_nodes,
        const int* equation_nodes,
        const double* element_matrix,
        const double* element_rhs);

    /**
     * @brief Synchronize values to device and export shared rows to owners.
     * @return @c true when pending native assembly was flushed.
     */
    bool flush(
        const TopologyCache& topology,
        Tpetra_CrsMatrix& owned_matrix,
        Tpetra_MultiVector& ghost_rhs);

    /// @brief Return whether element contributions await a flush.
    bool pending() const;

    /// @brief Release current assembly values before Kokkos finalization.
    void clear();

  private:
    Teuchos::RCP<Tpetra_CrsMatrix> overlap_matrix_;
    std::size_t topology_generation_ = 0;
    bool pending_ = false;
};

} // namespace trilinos_backend

/**************************************************************/
/*                      Macro Definitions                     */
/**************************************************************/

// Define linear solver as following naming in FSILS_struct
#define TRILINOS_CG_SOLVER 798
#define TRILINOS_GMRES_SOLVER 797
#define TRILINOS_BICGSTAB_SOLVER 795

// Define preconditioners as following naming in FSILS_struct
#define NO_PRECONDITIONER 700
#define TRILINOS_DIAGONAL_PRECONDITIONER 702
#define TRILINOS_BLOCK_JACOBI_PRECONDITIONER 703
#define TRILINOS_ILU_PRECONDITIONER 704
#define TRILINOS_ILUT_PRECONDITIONER 705
#define TRILINOS_RILUK0_PRECONDITIONER 706
#define TRILINOS_RILUK1_PRECONDITIONER 707
#define TRILINOS_ML_PRECONDITIONER 708
#define TRILINOS_RESISTANCE_PRECONDITIONER 712

/// @brief Initialize all Epetra types we need separate from Fortran
struct Trilinos
{
  struct ResistanceFaceData {
    int face_id = -1;
    double resistance = 0.0;
    double s_tilde_norm2 = 0.0;
    double alpha = 0.0;
  };

  /// Equation-local maps, importer, graph, and assembly metadata.
  trilinos_backend::TopologyCache topology;

  /// Per-Jacobian native Trilinos element assembly storage.
  trilinos_backend::LocalAssemblyBuffer local_assembly;

  Teuchos::RCP<Tpetra_MultiVector> F;
  Teuchos::RCP<Tpetra_MultiVector> ghostF;
  Teuchos::RCP<Tpetra_CrsMatrix> K;
  Teuchos::RCP<Tpetra_Vector> X;
  Teuchos::RCP<Tpetra_Vector> ghostX;

  // One pair of vectors per coupled outlet face.  After construction and
  // Jacobi scaling, bdryVec_list[f] stores sqrt(abs(R_f)) * S_f, where S_f is
  // the scaled outlet surface integral vector.  bdryCapVec_list[f] stores the
  // corresponding cap contribution used only in the scalar flow projection.
  std::vector<Teuchos::RCP<Tpetra_MultiVector>> bdryVec_list;
  std::vector<Teuchos::RCP<Tpetra_MultiVector>> bdryCapVec_list;
  Teuchos::RCP<const Teuchos::Comm<int>> comm;

  /// Whether the current Jacobian has active coupled Neumann terms.
  bool coupled_boundary = false;

  Teuchos::RCP<Tpetra_Operator> MueluPrec;
  Teuchos::RCP<Ifpack2_Preconditioner> ifpackPrec;
  Teuchos::RCP<Tpetra_Operator> resistancePrec;

  // Per-face diagnostic and coefficient data for the resistance
  // Sherman-Morrison inverse:
  // alpha_f = -R_f / (1 + R_f * ||S_f||^2).
  std::vector<ResistanceFaceData> resistanceFaces;
  Trilinos() : MueluPrec(nullptr), ifpackPrec(nullptr), resistancePrec(nullptr) {}
};

/**
 * \class TrilinosMatVec
 * \brief This class implements the pure virtual class Epetra_Operator for the
 *        AztecOO iterative solve which only uses the Apply() method to compute
 *        the matrix vector product
 */
class TrilinosMatVec: public Tpetra_Operator
{
public:

  /** Define matrix vector operation at each iteration of the linear solver
   *  adds on the coupled neuman boundary contribution to the matrix
   *
   *  \param x vector to be applied on the operator
   *  \param y result of sparse matrix vector multiplication
   */
  TrilinosMatVec(const Teuchos::RCP<Trilinos>& trilinos) : trilinos_(trilinos) {}

  /* Y = beta * Y + alpha * A^mode * X */
  void apply(const Tpetra_MultiVector& X, Tpetra_MultiVector& Y,
           Teuchos::ETransp mode = Teuchos::NO_TRANS,
           Scalar_d alpha = Teuchos::ScalarTraits< Scalar_d >::one(), 
           Scalar_d beta  = Teuchos::ScalarTraits<Scalar_d>::zero()) const override;

  /*  
    Returns the map describing the layout of the domain vector space.
    This map defines the distribution of the input vectors to the operator.
  */
  Teuchos::RCP<const Tpetra_Map> getDomainMap() const override
  {
    return trilinos_->K->getDomainMap();
  }

  /* 
    Returns the map describing the layout of the range vector space.
    This map defines the distribution of the output vectors from the operator.
  */
  Teuchos::RCP<const Tpetra_Map> getRangeMap() const override
  {
    return trilinos_->K->getRangeMap();
  }

  private:
    Teuchos::RCP<Trilinos> trilinos_;
};// class TrilinosMatVec

//  --- Functions to be called in fortran -------------------------------------

#ifdef __cplusplus
  extern "C"
  {
#endif
  /// Give function definitions which will be called through fortran
  void trilinos_lhs_create(const Teuchos::RCP<Trilinos> &trilinos_, const int numGlobalNodes, const int numLocalNodes,
          const int numGhostAndLocalNodes, const int nnz, const Vector<int>& ltgSorted,
          const Vector<int>& ltgUnsorted, const Vector<int>& rowPtr, const Vector<int>& colInd,
          const int dof, const int cpp_index, const int proc_id, const int numCoupledNeumannBC);

  /**
   * \param v           coeff in the scalar product
   * \param isCoupledBC determines if coupled resistance BC is turned on
   */
  void trilinos_bc_create_(const Teuchos::RCP<Trilinos> &trilinos_, const std::vector<Array<double>> &v_list,
    const std::vector<Array<double>> &vcap_list, bool &isCoupledBC);

  void trilinos_doassem_(const Teuchos::RCP<Trilinos> &trilinos_, int &numNodesPerElement, const int *eqN,
          const double *lK, double *lR);

  void trilinos_global_solve_(const Teuchos::RCP<Trilinos> &trilinos_, const double *Val, const double *RHS,
          double *x, const double *dirW, double &resNorm, double &initNorm,
          int &numIters, double &solverTime, double &dB, bool &converged,
          int &lsType, double &relTol, int &maxIters, int &kspace,
          int &precondType);

  void trilinos_solve_(const Teuchos::RCP<Trilinos> &trilinos_, double *x, const double *dirW, double &resNorm,
          double &initNorm, int &numIters, double &solverTime,
          double &dB, bool &converged, int &lsType, double &relTol,
          int &maxIters, int &kspace, int &precondType, bool &isFassem);

#ifdef __cplusplus  /* this brace matches the one on the extern "C" line */
  }
#endif

// --- Define functions to only be called in C++ ------------------------------
void setPreconditioner(const Teuchos::RCP<Trilinos> &trilinos_, int precondType, 
  Teuchos::RCP<Belos_LinearProblem>& BelosProblem);

void setMueLuPreconditioner(Teuchos::RCP<MueLu_Preconditioner>& MueLuPrec, 
  const Teuchos::RCP<Tpetra_CrsMatrix>& A);

void checkDiagonalIsZero(const Teuchos::RCP<Trilinos> &trilinos_);

void constructJacobiScaling(const Teuchos::RCP<Trilinos> &trilinos_, const double *dirW,
              Tpetra_Vector &diagonal);

void logResistancePreconditioner(const Teuchos::RCP<Trilinos> &trilinos_,
              const std::string& file_name);

// --- Debugging functions ----------------------------------------------------
void printMatrixToFile(const Teuchos::RCP<Trilinos> &trilinos_);

void printRHSToFile(const Teuchos::RCP<Trilinos> &trilinos_);

void printSolutionToFile(const Teuchos::RCP<Trilinos> &trilinos_);

#endif //TRILINOS_LINEAR_SOLVER_H
