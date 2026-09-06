// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "gtest/gtest.h"

#ifdef WITH_TRILINOS

#include "TrilinosNSBlockOperators.h"
#include "TrilinosBipartitionNS.h"
#include "TrilinosPreconditionerFactory.h"
#include "TrilinosResistanceOperator.h"

#include "Teuchos_DefaultSerialComm.hpp"

#include <limits>
#include <utility>
#include <vector>

namespace {

class TpetraBlockTestScope
{
  public:
    TpetraBlockTestScope() :
      scope_(&argc_, &argv_) {}

  private:
    int argc_ = 0;
    char** argv_ = nullptr;
    Tpetra::ScopeGuard scope_;
};

Teuchos::RCP<const Tpetra_Map> make_serial_map(
    Tpetra::global_size_t size)
{
  static TpetraBlockTestScope tpetra_scope;
  auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>());
  return Teuchos::rcp(new Tpetra_Map(size, 0, comm));
}

Teuchos::RCP<Tpetra_CrsMatrix> make_dense_matrix(
    const Teuchos::RCP<const Tpetra_Map>& map,
    int dimension)
{
  auto matrix = Teuchos::rcp(new Tpetra_CrsMatrix(map, dimension));
  Teuchos::Array<GO> columns(dimension);
  Teuchos::Array<Scalar_d> values(dimension);
  for (int row = 0; row < dimension; ++row) {
    for (int column = 0; column < dimension; ++column) {
      columns[column] = column;
      values[column] = 100.0 * row + column + 1.0;
    }
    matrix->insertGlobalValues(row, columns(), values());
  }
  matrix->fillComplete(map, map);
  return matrix;
}

Teuchos::RCP<Tpetra_CrsMatrix> make_three_by_three_ns_matrix(
    const Teuchos::RCP<const Tpetra_Map>& map)
{
  auto matrix = Teuchos::rcp(new Tpetra_CrsMatrix(map, 3));
  const double entries[3][3] = {
    {2.0, 0.0, 1.0},
    {0.0, 3.0, 1.0},
    {0.0, 0.0, 3.0}
  };
  Teuchos::Array<GO> columns = {0, 1, 2};
  for (GO row = 0; row < 3; ++row) {
    Teuchos::Array<Scalar_d> values = {
      entries[row][0], entries[row][1], entries[row][2]
    };
    matrix->insertGlobalValues(row, columns(), values());
  }
  matrix->fillComplete(map, map);
  return matrix;
}

Teuchos::RCP<Tpetra_CrsMatrix> make_diagonal_matrix(
    const Teuchos::RCP<const Tpetra_Map>& map,
    const std::vector<double>& diagonal)
{
  auto matrix = Teuchos::rcp(new Tpetra_CrsMatrix(map, 1));
  for (size_t row = 0; row < diagonal.size(); ++row) {
    Teuchos::Array<GO> column(1, static_cast<GO>(row));
    Teuchos::Array<Scalar_d> value(1, diagonal[row]);
    matrix->insertGlobalValues(
        static_cast<GO>(row), column(), value());
  }
  matrix->fillComplete(map, map);
  return matrix;
}

Teuchos::RCP<Tpetra_MultiVector> make_vector(
    const Teuchos::RCP<const Tpetra_Map>& map,
    const std::vector<std::pair<GO, double>>& entries)
{
  auto vector = Teuchos::rcp(new Tpetra_MultiVector(map, 1));
  vector->putScalar(0.0);
  for (const auto& entry : entries) {
    vector->replaceGlobalValue(entry.first, 0, entry.second);
  }
  return vector;
}

double vector_value(const Tpetra_MultiVector& vector, GO gid)
{
  const LO lid = vector.getMap()->getLocalElement(gid);
  const auto view = vector.getLocalViewHost(Tpetra::Access::ReadOnly);
  return view(lid, 0);
}

double matrix_value(const Tpetra_CrsMatrix& matrix, GO row, GO column)
{
  const size_t row_entries = matrix.getNumEntriesInGlobalRow(row);
  Tpetra_CrsMatrix::nonconst_global_inds_host_view_type columns(
      "test_columns", row_entries);
  Tpetra_CrsMatrix::nonconst_values_host_view_type values(
      "test_values", row_entries);
  size_t count = 0;
  matrix.getGlobalRowCopy(row, columns, values, count);
  for (size_t i = 0; i < count; ++i) {
    if (columns(i) == column) {
      return values(i);
    }
  }
  return 0.0;
}

Teuchos::RCP<Trilinos> make_block_test_state(
    const Teuchos::RCP<Tpetra_CrsMatrix>& matrix)
{
  auto state = Teuchos::rcp(new Trilinos());
  state->K = matrix;
  return state;
}

} // namespace

TEST(TrilinosNSBlockOperators, SplitsTwoDimensionalNodalBlocks)
{
  const auto map = make_serial_map(6);
  auto state = make_block_test_state(make_dense_matrix(map, 6));
  state->bdryVec_list.push_back(make_vector(map, {
    {0, 10.0}, {1, 20.0}, {2, 30.0},
    {3, 40.0}, {4, 50.0}, {5, 60.0}
  }));

  const auto blocks =
    trilinos_bipartition::build_trilinos_ns_block_system(state, 2, 3);

  EXPECT_EQ(blocks.velocity_map->getGlobalNumElements(), 4);
  EXPECT_EQ(blocks.pressure_map->getGlobalNumElements(), 2);
  EXPECT_DOUBLE_EQ(matrix_value(*blocks.A, 0, 4), 5.0);
  EXPECT_DOUBLE_EQ(matrix_value(*blocks.B, 0, 5), 6.0);
  EXPECT_DOUBLE_EQ(matrix_value(*blocks.C, 2, 3), 204.0);
  EXPECT_DOUBLE_EQ(matrix_value(*blocks.L, 2, 5), 206.0);
  ASSERT_EQ(blocks.boundary_vectors.size(), 1);
  EXPECT_DOUBLE_EQ(vector_value(*blocks.boundary_vectors[0], 0), 10.0);
  EXPECT_DOUBLE_EQ(vector_value(*blocks.boundary_vectors[0], 4), 50.0);
}

TEST(TrilinosNSBlockOperators, SplitsThreeDimensionalNodalBlocks)
{
  const auto map = make_serial_map(8);
  const auto state = make_block_test_state(make_dense_matrix(map, 8));

  const auto blocks =
    trilinos_bipartition::build_trilinos_ns_block_system(state, 3, 4);

  EXPECT_EQ(blocks.velocity_map->getGlobalNumElements(), 6);
  EXPECT_EQ(blocks.pressure_map->getGlobalNumElements(), 2);
  EXPECT_DOUBLE_EQ(matrix_value(*blocks.B, 4, 7), 408.0);
  EXPECT_DOUBLE_EQ(matrix_value(*blocks.C, 7, 5), 706.0);
}

TEST(TrilinosNSBlockOperators, AppliesMomentumAndResistanceSchurOperators)
{
  const auto map = make_serial_map(3);
  auto state = make_block_test_state(make_three_by_three_ns_matrix(map));
  const auto blocks =
    trilinos_bipartition::build_trilinos_ns_block_system(state, 2, 3);

  const std::vector<Teuchos::RCP<Tpetra_MultiVector>> faces = {
    make_vector(blocks.velocity_map, {{0, 1.0}, {1, 2.0}})
  };
  const std::vector<Teuchos::RCP<Tpetra_MultiVector>> caps = {
    make_vector(blocks.velocity_map, {{0, 0.5}})
  };

  trilinos_bipartition::MomentumOperator momentum(blocks.A, faces, caps);
  const auto velocity_input =
    make_vector(blocks.velocity_map, {{0, 1.0}, {1, 2.0}});
  Tpetra_MultiVector momentum_output(blocks.velocity_map, 1);
  momentum.apply(*velocity_input, momentum_output);
  EXPECT_NEAR(vector_value(momentum_output, 0), 7.5, 1.0e-13);
  EXPECT_NEAR(vector_value(momentum_output, 1), 17.0, 1.0e-13);

  const auto pressure_input =
    make_vector(blocks.pressure_map, {{2, 2.0}});
  trilinos_bipartition::PressureSchurOperator schur_without_resistance(
      blocks.L, blocks.B);
  Tpetra_MultiVector untransformed_output(blocks.pressure_map, 1);
  schur_without_resistance.apply(*pressure_input, untransformed_output);
  EXPECT_NEAR(vector_value(untransformed_output, 2), 10.0, 1.0e-13);

  std::vector<Trilinos::ResistanceFaceData> face_data(1);
  face_data[0].face_id = 0;
  face_data[0].resistance = 1.0;
  auto resistance = Teuchos::rcp(new TrilinosResistanceOperator(
      blocks.velocity_map, faces, caps, face_data));
  resistance->compute();

  trilinos_bipartition::PressureSchurOperator schur(
      blocks.L, blocks.B,
      Teuchos::rcp_implicit_cast<Tpetra_Operator>(resistance));
  Tpetra_MultiVector pressure_output(blocks.pressure_map, 1);
  schur.apply(*pressure_input, pressure_output);
  EXPECT_NEAR(vector_value(pressure_output, 2), 6.5, 1.0e-13);
}

TEST(TrilinosNSBlockOperators, ExtractsAndScattersWithoutHostAssembly)
{
  const auto map = make_serial_map(3);
  const auto state = make_block_test_state(make_three_by_three_ns_matrix(map));
  const auto blocks =
    trilinos_bipartition::build_trilinos_ns_block_system(state, 2, 3);
  const auto full = make_vector(map, {{0, 4.0}, {1, 5.0}, {2, 6.0}});

  const auto velocity =
    trilinos_bipartition::extract_subvector(*full, blocks.velocity_map);
  const auto pressure =
    trilinos_bipartition::extract_subvector(*full, blocks.pressure_map);
  EXPECT_DOUBLE_EQ(vector_value(*velocity, 0), 4.0);
  EXPECT_DOUBLE_EQ(vector_value(*velocity, 1), 5.0);
  EXPECT_DOUBLE_EQ(vector_value(*pressure, 2), 6.0);

  auto scattered = Teuchos::rcp(new Tpetra_Vector(map));
  trilinos_bipartition::scatter_subvectors_to_full_vector(
      *velocity, *pressure, scattered);
  EXPECT_DOUBLE_EQ(vector_value(*scattered, 0), 4.0);
  EXPECT_DOUBLE_EQ(vector_value(*scattered, 1), 5.0);
  EXPECT_DOUBLE_EQ(vector_value(*scattered, 2), 6.0);
}

TEST(TrilinosPreconditionerFactory, RepresentsDiagonalScalingAsIdentity)
{
  namespace factory = trilinos_bipartition::preconditioners;
  const auto handle = factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_DIAGONAL,
      factory::SolverRole::momentum_gmres,
      Teuchos::null);

  EXPECT_EQ(handle->type(),
      consts::PreconditionerType::PREC_TRILINOS_DIAGONAL);
  EXPECT_EQ(handle->left_operator(), Teuchos::null);
  EXPECT_EQ(handle->setup_matrix(), Teuchos::null);
}

TEST(TrilinosPreconditionerFactory, BuildsAllIfpack2Policies)
{
  namespace factory = trilinos_bipartition::preconditioners;
  const auto map = make_serial_map(4);
  const auto matrix =
    make_diagonal_matrix(map, {2.0, 3.0, 4.0, 5.0});
  const std::vector<consts::PreconditionerType> types = {
    consts::PreconditionerType::PREC_TRILINOS_BLOCK_JACOBI,
    consts::PreconditionerType::PREC_TRILINOS_ILU,
    consts::PreconditionerType::PREC_TRILINOS_ILUT,
    consts::PreconditionerType::PREC_TRILINOS_RILUK0,
    consts::PreconditionerType::PREC_TRILINOS_RILUK1
  };

  for (const auto type : types) {
    const auto handle = factory::create_preconditioner(
        type, factory::SolverRole::momentum_gmres, matrix);
    EXPECT_NE(handle->left_operator(), Teuchos::null)
      << consts::preconditioner_type_to_name.at(type);
  }
}

TEST(TrilinosPreconditionerFactory, KeepsDiagonalRepairPrivate)
{
  namespace factory = trilinos_bipartition::preconditioners;
  const auto map = make_serial_map(2);
  const auto matrix = make_diagonal_matrix(map, {0.0, 2.0});

  const auto handle = factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_BLOCK_JACOBI,
      factory::SolverRole::pressure_cg, matrix);

  EXPECT_NE(handle->setup_matrix().getRawPtr(), matrix.getRawPtr());
  EXPECT_DOUBLE_EQ(matrix_value(*matrix, 0, 0), 0.0);
  EXPECT_DOUBLE_EQ(matrix_value(*handle->setup_matrix(), 0, 0), 1.0);
}

TEST(TrilinosPreconditionerFactory, RestrictsResistanceToMomentum)
{
  namespace factory = trilinos_bipartition::preconditioners;
  const auto map = make_serial_map(2);
  const auto matrix = make_diagonal_matrix(map, {1.0, 1.0});
  const auto resistance =
    Teuchos::rcp_implicit_cast<Tpetra_Operator>(matrix);

  const auto momentum_handle = factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_RESISTANCE,
      factory::SolverRole::momentum_gmres,
      matrix,
      resistance);
  EXPECT_EQ(momentum_handle->left_operator().getRawPtr(),
      resistance.getRawPtr());
  EXPECT_EQ(momentum_handle->resistance_operator().getRawPtr(),
      resistance.getRawPtr());

  EXPECT_THROW(factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_RESISTANCE,
      factory::SolverRole::pressure_cg,
      matrix,
      resistance), std::runtime_error);

  const auto other_map = make_serial_map(3);
  const auto other_resistance = Teuchos::rcp_implicit_cast<Tpetra_Operator>(
      make_diagonal_matrix(other_map, {1.0, 1.0, 1.0}));
  EXPECT_THROW(factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_RESISTANCE,
      factory::SolverRole::momentum_gmres,
      matrix,
      other_resistance), std::runtime_error);
}

TEST(TrilinosPreconditionerFactory, AttachesExistingHandleToBelos)
{
  namespace factory = trilinos_bipartition::preconditioners;
  const auto map = make_serial_map(2);
  const auto matrix = make_diagonal_matrix(map, {2.0, 3.0});
  const auto handle = factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_BLOCK_JACOBI,
      factory::SolverRole::momentum_gmres,
      matrix);
  const auto solution = make_vector(map, {});
  const auto rhs = make_vector(map, {{0, 1.0}, {1, 1.0}});
  auto problem = Teuchos::rcp(new Belos_LinearProblem(
      Teuchos::rcp_implicit_cast<Tpetra_Operator>(matrix),
      solution, rhs));

  factory::attach_preconditioner(handle, problem);

  EXPECT_EQ(problem->getLeftPrec().getRawPtr(),
      handle->left_operator().getRawPtr());
}

TEST(TrilinosPreconditionerFactory, ValidatesMueLuReuseContext)
{
  static TpetraBlockTestScope tpetra_scope;
  namespace factory = trilinos_bipartition::preconditioners;
  const auto map = make_serial_map(2);
  const auto matrix = make_diagonal_matrix(map, {2.0, 3.0});
  const auto cache = Teuchos::rcp(new factory::MueLuReuseCache());
  factory::PreconditionerReuseContext reuse;
  reuse.muelu_cache = cache;
  EXPECT_THROW(factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_ML,
      factory::SolverRole::momentum_gmres,
      matrix,
      Teuchos::null,
      reuse), std::runtime_error);

  reuse.time_step = 7;
  EXPECT_THROW(factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_ML,
      factory::SolverRole::momentum_gmres,
      matrix,
      Teuchos::null,
      reuse), std::runtime_error);

  cache->clear();
  EXPECT_EQ(cache->hierarchy(), Teuchos::null);
  EXPECT_EQ(cache->build_count(), 0);
  EXPECT_EQ(cache->reuse_count(), 0);
}

TEST(TrilinosBipartitionBudget, PreservesFsilsRestartCycles)
{
  const auto budget =
      trilinos_bipartition::make_belos_gmres_budget(10, 300);

  EXPECT_EQ(budget.maximum_iterations, 3000);
  EXPECT_EQ(budget.num_blocks, 300);
  EXPECT_EQ(budget.maximum_restarts, 9);
}

TEST(TrilinosBipartitionBudget, SupportsSingleOneDimensionalCycle)
{
  const auto budget =
      trilinos_bipartition::make_belos_gmres_budget(1, 1);

  EXPECT_EQ(budget.maximum_iterations, 1);
  EXPECT_EQ(budget.num_blocks, 1);
  EXPECT_EQ(budget.maximum_restarts, 0);
}

TEST(TrilinosBipartitionBudget, RejectsInvalidAndOverflowingBudgets)
{
  EXPECT_THROW(
      trilinos_bipartition::make_belos_gmres_budget(0, 10),
      std::runtime_error);
  EXPECT_THROW(
      trilinos_bipartition::make_belos_gmres_budget(10, 0),
      std::runtime_error);
  EXPECT_THROW(
      trilinos_bipartition::make_belos_gmres_budget(
          std::numeric_limits<int>::max(), 2),
      std::runtime_error);
}

TEST(TrilinosBipartitionBudget, LeavesCgAsATotalIterationLimit)
{
  EXPECT_EQ(trilinos_bipartition::make_belos_cg_max_iterations(300), 300);
  EXPECT_THROW(
      trilinos_bipartition::make_belos_cg_max_iterations(0),
      std::runtime_error);
}

TEST(TrilinosTopologyCache, ReusesAnUnchangedStaticGraph)
{
  static TpetraBlockTestScope tpetra_scope;
  auto communicator = Teuchos::rcp(new Teuchos::SerialComm<int>());
  Vector<int> sorted = {0, 1};
  Vector<int> unsorted = {0, 1};
  Vector<int> row_pointer = {0, 2, 4};
  Vector<int> columns = {0, 1, 0, 1};
  trilinos_backend::TopologyCache topology;

  EXPECT_TRUE(topology.ensure(
      communicator, 2, 2, 2, 4,
      sorted, unsorted, row_pointer, columns, 2, 0));
  ASSERT_TRUE(topology.initialized());
  ASSERT_NE(topology.map(), Teuchos::null);
  ASSERT_NE(topology.graph(), Teuchos::null);
  ASSERT_NE(topology.importer(), Teuchos::null);
  const auto* first_map = topology.map().getRawPtr();
  const auto* first_graph = topology.graph().getRawPtr();
  const auto* first_importer = topology.importer().getRawPtr();

  EXPECT_FALSE(topology.ensure(
      communicator, 2, 2, 2, 4,
      sorted, unsorted, row_pointer, columns, 2, 0));
  EXPECT_EQ(topology.generation(), 1);
  EXPECT_EQ(topology.map().getRawPtr(), first_map);
  EXPECT_EQ(topology.graph().getRawPtr(), first_graph);
  EXPECT_EQ(topology.importer().getRawPtr(), first_importer);
}

TEST(TrilinosTopologyCache, RebuildsWhenTheDofLayoutChanges)
{
  static TpetraBlockTestScope tpetra_scope;
  auto communicator = Teuchos::rcp(new Teuchos::SerialComm<int>());
  Vector<int> sorted = {0, 1};
  Vector<int> unsorted = {0, 1};
  Vector<int> row_pointer = {0, 2, 4};
  Vector<int> columns = {0, 1, 0, 1};
  trilinos_backend::TopologyCache topology;

  ASSERT_TRUE(topology.ensure(
      communicator, 2, 2, 2, 4,
      sorted, unsorted, row_pointer, columns, 2, 0));
  const auto* first_graph = topology.graph().getRawPtr();

  EXPECT_TRUE(topology.ensure(
      communicator, 2, 2, 2, 4,
      sorted, unsorted, row_pointer, columns, 3, 0));
  EXPECT_EQ(topology.generation(), 2);
  EXPECT_EQ(topology.dof(), 3);
  EXPECT_EQ(topology.map()->getGlobalNumElements(), 6);
  EXPECT_NE(topology.graph().getRawPtr(), first_graph);
}

TEST(TrilinosTopologyCache, AllocatesOverlapRowsInTpetraMapOrder)
{
  static TpetraBlockTestScope tpetra_scope;
  auto communicator = Teuchos::rcp(new Teuchos::SerialComm<int>());
  Vector<int> sorted = {0, 1, 2};
  Vector<int> unsorted = {2, 0, 1};
  Vector<int> row_pointer = {0, 1, 3, 6};
  Vector<int> columns = {0, 0, 1, 0, 1, 2};
  trilinos_backend::TopologyCache topology;

  EXPECT_TRUE(topology.ensure(
      communicator, 3, 3, 3, 6,
      sorted, unsorted, row_pointer, columns, 1, 0));
  ASSERT_NE(topology.assembly_graph(), Teuchos::null);

  const auto graph = topology.assembly_graph()->getLocalGraphHost();
  EXPECT_EQ(graph.row_map(1) - graph.row_map(0), 2);
  EXPECT_EQ(graph.row_map(2) - graph.row_map(1), 3);
  EXPECT_EQ(graph.row_map(3) - graph.row_map(2), 1);
}

TEST(TrilinosTopologyCache, RejectsInvalidConnectivity)
{
  static TpetraBlockTestScope tpetra_scope;
  auto communicator = Teuchos::rcp(new Teuchos::SerialComm<int>());
  Vector<int> sorted = {0, 1};
  Vector<int> unsorted = {0, 1};
  Vector<int> row_pointer = {0, 1, 2};
  Vector<int> columns = {0, 2};
  trilinos_backend::TopologyCache topology;

  EXPECT_THROW(topology.ensure(
      communicator, 2, 2, 2, 2,
      sorted, unsorted, row_pointer, columns, 2, 0),
      std::runtime_error);
  EXPECT_FALSE(topology.initialized());
  EXPECT_EQ(topology.generation(), 0);
}

TEST(TrilinosLocalAssembly, AccumulatesElementValuesByLocalCrsOffset)
{
  static TpetraBlockTestScope tpetra_scope;
  auto communicator = Teuchos::rcp(new Teuchos::SerialComm<int>());
  Vector<int> sorted = {0, 1};
  Vector<int> unsorted = {0, 1};
  Vector<int> row_pointer = {0, 2, 4};
  Vector<int> columns = {0, 1, 0, 1};
  trilinos_backend::TopologyCache topology;
  ASSERT_TRUE(topology.ensure(
      communicator, 2, 2, 2, 4,
      sorted, unsorted, row_pointer, columns, 2, 0));

  trilinos_backend::LocalAssemblyBuffer assembly;
  assembly.reset(topology);
  Tpetra_MultiVector ghost_rhs(topology.ghost_map(), 1);
  ghost_rhs.putScalar(0.0);
  auto owned_matrix = Teuchos::rcp(
      new Tpetra_CrsMatrix(topology.graph()));

  const int equation_nodes[2] = {0, 1};
  double element_matrix[16] = {};
  double element_rhs[4] = {1.0, 2.0, 3.0, 4.0};
  for (int row_node = 0; row_node < 2; ++row_node) {
    for (int column_node = 0; column_node < 2; ++column_node) {
      for (int row_component = 0; row_component < 2; ++row_component) {
        for (int column_component = 0;
             column_component < 2;
             ++column_component) {
          const std::size_t element_offset =
              column_node * 2 * 2 * 2 + row_node * 2 * 2 +
              row_component * 2 + column_component;
          element_matrix[element_offset] =
              1000.0 * row_node + 100.0 * column_node +
              10.0 * row_component + column_component + 1.0;
        }
      }
    }
  }

  assembly.add_element(
      topology,
      ghost_rhs,
      2,
      equation_nodes,
      element_matrix,
      element_rhs);
  assembly.add_element(
      topology,
      ghost_rhs,
      2,
      equation_nodes,
      element_matrix,
      element_rhs);
  EXPECT_TRUE(assembly.pending());
  EXPECT_TRUE(assembly.flush(topology, *owned_matrix, ghost_rhs));
  EXPECT_FALSE(assembly.pending());
  EXPECT_FALSE(assembly.flush(topology, *owned_matrix, ghost_rhs));

  for (int row_node = 0; row_node < 2; ++row_node) {
    for (int column_node = 0; column_node < 2; ++column_node) {
      for (int row_component = 0; row_component < 2; ++row_component) {
        for (int column_component = 0;
             column_component < 2;
             ++column_component) {
          const GO row = row_node * 2 + row_component;
          const GO column = column_node * 2 + column_component;
          const double expected = 2.0 * (
              1000.0 * row_node + 100.0 * column_node +
              10.0 * row_component + column_component + 1.0);
          EXPECT_DOUBLE_EQ(
              matrix_value(*owned_matrix, row, column), expected);
        }
      }
    }
  }

  const auto rhs = ghost_rhs.getLocalViewHost(Tpetra::Access::ReadOnly);
  EXPECT_DOUBLE_EQ(rhs(topology.assembly_vector_lid(0, 0), 0), 2.0);
  EXPECT_DOUBLE_EQ(rhs(topology.assembly_vector_lid(0, 1), 0), 4.0);
  EXPECT_DOUBLE_EQ(rhs(topology.assembly_vector_lid(1, 0), 0), 6.0);
  EXPECT_DOUBLE_EQ(rhs(topology.assembly_vector_lid(1, 1), 0), 8.0);
}

TEST(TrilinosLocalAssembly, ResetDiscardsPreviousJacobianValues)
{
  static TpetraBlockTestScope tpetra_scope;
  auto communicator = Teuchos::rcp(new Teuchos::SerialComm<int>());
  Vector<int> sorted = {0};
  Vector<int> unsorted = {0};
  Vector<int> row_pointer = {0, 1};
  Vector<int> columns = {0};
  trilinos_backend::TopologyCache topology;
  ASSERT_TRUE(topology.ensure(
      communicator, 1, 1, 1, 1,
      sorted, unsorted, row_pointer, columns, 1, 0));

  trilinos_backend::LocalAssemblyBuffer assembly;
  Tpetra_MultiVector ghost_rhs(topology.ghost_map(), 1);
  ghost_rhs.putScalar(0.0);
  const int equation_node = 0;
  const double first_matrix = 7.0;
  const double first_rhs = 11.0;
  assembly.reset(topology);
  assembly.add_element(
      topology,
      ghost_rhs,
      1,
      &equation_node,
      &first_matrix,
      &first_rhs);
  ASSERT_TRUE(assembly.pending());

  assembly.reset(topology);
  EXPECT_FALSE(assembly.pending());
  ghost_rhs.putScalar(0.0);
  const double second_matrix = 3.0;
  const double second_rhs = 5.0;
  assembly.add_element(
      topology,
      ghost_rhs,
      1,
      &equation_node,
      &second_matrix,
      &second_rhs);
  auto owned_matrix = Teuchos::rcp(
      new Tpetra_CrsMatrix(topology.graph()));
  ASSERT_TRUE(assembly.flush(topology, *owned_matrix, ghost_rhs));

  EXPECT_DOUBLE_EQ(matrix_value(*owned_matrix, 0, 0), second_matrix);
  const auto rhs = ghost_rhs.getLocalViewHost(Tpetra::Access::ReadOnly);
  EXPECT_DOUBLE_EQ(rhs(0, 0), second_rhs);
}

#endif
