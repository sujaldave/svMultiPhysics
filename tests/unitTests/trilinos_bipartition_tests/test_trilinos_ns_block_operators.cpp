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

TEST(TrilinosNSBlockTopologyCache, ReusesTopologyAndRefreshesValues)
{
  const auto map = make_serial_map(6);
  auto state = make_block_test_state(make_dense_matrix(map, 6));
  trilinos_bipartition::TrilinosNSBlockTopologyCache cache;

  EXPECT_TRUE(cache.ensure(*state->K, 4, 2, 3));
  const auto first = cache.create_system(state);
  const auto* velocity_map = first.velocity_map.getRawPtr();
  const auto* pressure_map = first.pressure_map.getRawPtr();
  const auto* A_graph = first.A->getCrsGraph().getRawPtr();
  const auto* B_graph = first.B->getCrsGraph().getRawPtr();
  EXPECT_DOUBLE_EQ(matrix_value(*first.A, 0, 4), 5.0);
  EXPECT_DOUBLE_EQ(matrix_value(*first.B, 0, 5), 6.0);

  state->K->scale(2.0);
  EXPECT_FALSE(cache.ensure(*state->K, 4, 2, 3));
  const auto second = cache.create_system(state);

  EXPECT_EQ(cache.generation(), 1);
  EXPECT_EQ(second.velocity_map.getRawPtr(), velocity_map);
  EXPECT_EQ(second.pressure_map.getRawPtr(), pressure_map);
  EXPECT_EQ(second.A->getCrsGraph().getRawPtr(), A_graph);
  EXPECT_EQ(second.B->getCrsGraph().getRawPtr(), B_graph);
  EXPECT_DOUBLE_EQ(matrix_value(*second.A, 0, 4), 10.0);
  EXPECT_DOUBLE_EQ(matrix_value(*second.B, 0, 5), 12.0);
  EXPECT_DOUBLE_EQ(matrix_value(*second.C, 2, 3), 408.0);
  EXPECT_DOUBLE_EQ(matrix_value(*second.L, 2, 5), 412.0);
}

TEST(TrilinosNSBlockTopologyCache, RebuildsForNewFullTopologyGeneration)
{
  const auto map = make_serial_map(6);
  auto state = make_block_test_state(make_dense_matrix(map, 6));
  trilinos_bipartition::TrilinosNSBlockTopologyCache cache;

  ASSERT_TRUE(cache.ensure(*state->K, 1, 2, 3));
  const auto first = cache.create_system(state);
  const auto* first_graph = first.A->getCrsGraph().getRawPtr();

  EXPECT_TRUE(cache.ensure(*state->K, 2, 2, 3));
  const auto second = cache.create_system(state);
  EXPECT_EQ(cache.generation(), 2);
  EXPECT_NE(second.A->getCrsGraph().getRawPtr(), first_graph);
}

TEST(TrilinosNSBlockTopologyCache, ReusesImportAndExportSemantics)
{
  const auto map = make_serial_map(3);
  auto state = make_block_test_state(make_three_by_three_ns_matrix(map));
  trilinos_bipartition::TrilinosNSBlockTopologyCache cache;
  ASSERT_TRUE(cache.ensure(*state->K, 1, 2, 3));
  const auto full = make_vector(map, {{0, 4.0}, {1, 5.0}, {2, 6.0}});

  const auto velocity = cache.extract_velocity(*full);
  const auto pressure = cache.extract_pressure(*full);
  EXPECT_DOUBLE_EQ(vector_value(*velocity, 0), 4.0);
  EXPECT_DOUBLE_EQ(vector_value(*velocity, 1), 5.0);
  EXPECT_DOUBLE_EQ(vector_value(*pressure, 2), 6.0);

  auto scattered = Teuchos::rcp(new Tpetra_Vector(map));
  cache.scatter(*velocity, *pressure, scattered);
  EXPECT_DOUBLE_EQ(vector_value(*scattered, 0), 4.0);
  EXPECT_DOUBLE_EQ(vector_value(*scattered, 1), 5.0);
  EXPECT_DOUBLE_EQ(vector_value(*scattered, 2), 6.0);
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
  const auto second_pressure_input =
    make_vector(blocks.pressure_map, {{2, 3.0}});
  schur_without_resistance.apply(
      *second_pressure_input, untransformed_output);
  EXPECT_NEAR(vector_value(untransformed_output, 2), 15.0, 1.0e-13);

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

TEST(TrilinosSchurAMG, AssembledMatrixMatchesMatrixFreeAction)
{
  const auto map = make_serial_map(6);
  const auto blocks = trilinos_bipartition::build_trilinos_ns_block_system(
      make_block_test_state(make_dense_matrix(map, 6)), 2, 3);
  const auto matrix = trilinos_bipartition::preconditioners::
      build_schur_preconditioning_matrix(blocks.L, blocks.B);
  trilinos_bipartition::PressureSchurOperator reference(blocks.L, blocks.B);
  Tpetra_MultiVector input(blocks.pressure_map, 2);
  input.randomize();
  Tpetra_MultiVector expected(blocks.pressure_map, 2);
  Tpetra_MultiVector actual(blocks.pressure_map, 2);
  reference.apply(input, expected);
  matrix->apply(input, actual);
  actual.update(-1.0, expected, 1.0);
  Teuchos::Array<double> errors(2);
  actual.norm2(errors());
  EXPECT_LT(errors[0], 1.e-8);
  EXPECT_LT(errors[1], 1.e-8);
}

TEST(TrilinosSchurAMG, VCycleReducesResidualAndIsSymmetric)
{
  namespace factory = trilinos_bipartition::preconditioners;
  make_serial_map(1);
  // MueLu 16.2 describe() assumes an MpiComm in an MPI-enabled build.
  const auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(
      Teuchos::opaqueWrapper(MPI_COMM_SELF)));
  const auto map = Teuchos::rcp(new Tpetra_Map(256, 0, comm));
  auto matrix = Teuchos::rcp(new Tpetra_CrsMatrix(map, 3));
  for (GO row = 0; row < 256; ++row) {
    Teuchos::Array<GO> cols;
    Teuchos::Array<double> vals;
    for (GO col = std::max(0, row - 1); col <= std::min(255, row + 1); ++col) {
      cols.push_back(col);
      vals.push_back(col == row ? 2.0 : -1.0);
    }
    matrix->insertGlobalValues(row, cols(), vals());
  }
  matrix->fillComplete();
  const auto handle = factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_SCHUR_AMG,
      factory::SolverRole::pressure_cg, matrix);
  Tpetra_MultiVector x(map, 1), y(map, 1), mx(map, 1), my(map, 1), residual(map, 1);
  x.randomize();
  y.randomize();
  handle->left_operator()->apply(x, mx);
  handle->left_operator()->apply(y, my);
  Teuchos::Array<double> lhs(1), rhs(1), energy(1), before(1), after(1);
  x.dot(my, lhs());
  mx.dot(y, rhs());
  x.dot(mx, energy());
  EXPECT_NEAR(lhs[0], rhs[0], 1.e-9 * std::max(1.0, std::abs(lhs[0])));
  EXPECT_GT(energy[0], 0.0);
  matrix->apply(mx, residual);
  residual.update(1.0, x, -1.0);
  x.norm2(before());
  residual.norm2(after());
  EXPECT_LT(after[0], before[0]);
  EXPECT_THROW(factory::create_preconditioner(
      consts::PreconditionerType::PREC_TRILINOS_SCHUR_AMG,
      factory::SolverRole::momentum_gmres, matrix), std::runtime_error);
}

TEST(TrilinosSchurAMG, RejectsInvalidSetupWithoutRepairingTheSolveMatrix)
{
  namespace factory = trilinos_bipartition::preconditioners;
  const auto map = make_serial_map(2);
  for (const double diagonal : {0.0, -1.0,
       std::numeric_limits<double>::quiet_NaN()}) {
    const auto matrix = make_diagonal_matrix(map, {diagonal, 2.0});
    EXPECT_THROW(factory::create_preconditioner(
        consts::PreconditionerType::PREC_TRILINOS_SCHUR_AMG,
        factory::SolverRole::pressure_cg, matrix), std::runtime_error);
  }
  EXPECT_THROW(factory::build_schur_preconditioning_matrix(
      Teuchos::null, Teuchos::null), std::runtime_error);
}

TEST(TrilinosSchurAMG, DistributedProductIncludesOffRankContributions)
{
  // Initialize MPI/Kokkos through the same process-lifetime guard as serial tests.
  make_serial_map(1);
  const auto comm = Tpetra::getDefaultComm();
  const GO n = 4 * comm->getSize();
  auto map = Teuchos::rcp(new Tpetra_Map(n, 0, comm));
  auto b = Teuchos::rcp(new Tpetra_CrsMatrix(map, 2));
  auto l = Teuchos::rcp(new Tpetra_CrsMatrix(map, 1));
  for (LO i = 0; i < static_cast<LO>(map->getLocalNumElements()); ++i) {
    const GO row = map->getGlobalElement(i);
    Teuchos::Array<GO> cols = {row, (row + 1) % n};
    Teuchos::Array<double> vals = {2.0, -1.0};
    b->insertGlobalValues(row, cols(), vals());
    l->insertGlobalValues(row, Teuchos::tuple(row), Teuchos::tuple(0.5));
  }
  b->fillComplete();
  l->fillComplete();
  trilinos_bipartition::preconditioners::MueLuReuseCache cache;
  const auto matrix=cache.schur_matrix(l,b,1);
  const auto graph=matrix->getCrsGraph();
  b->scale(0.8);l->scale(1.2);
  const auto refreshed=cache.schur_matrix(l,b,1);
  EXPECT_EQ(matrix.getRawPtr(),refreshed.getRawPtr());
  EXPECT_EQ(graph.getRawPtr(),refreshed->getCrsGraph().getRawPtr());
  trilinos_bipartition::PressureSchurOperator reference(l, b);
  Tpetra_MultiVector x(map, 1), expected(map, 1), actual(map, 1);
  x.randomize();
  reference.apply(x, expected);
  matrix->apply(x, actual);
  actual.update(-1.0, expected, 1.0);
  Teuchos::Array<double> error(1);
  actual.norm2(error());
  EXPECT_LT(error[0], 1.e-12);
}

TEST(TrilinosNSBlockOperators, ResizesSchurWorkspaceForMultipleVectors)
{
  const auto map = make_serial_map(3);
  auto state = make_block_test_state(make_three_by_three_ns_matrix(map));
  const auto blocks =
    trilinos_bipartition::build_trilinos_ns_block_system(state, 2, 3);
  trilinos_bipartition::PressureSchurOperator schur(blocks.L, blocks.B);

  Tpetra_MultiVector input(blocks.pressure_map, 2);
  input.getVectorNonConst(0)->replaceGlobalValue(2, 1.0);
  input.getVectorNonConst(1)->replaceGlobalValue(2, 4.0);
  Tpetra_MultiVector output(blocks.pressure_map, 2);
  schur.apply(input, output);

  EXPECT_NEAR(vector_value(*output.getVector(0), 2), 5.0, 1.0e-13);
  EXPECT_NEAR(vector_value(*output.getVector(1), 2), 20.0, 1.0e-13);
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

TEST(TrilinosSchurAMG, TransferReuseRefreshesEveryNumericLevelAcrossTimeSteps)
{
  namespace factory=trilinos_bipartition::preconditioners;
  make_serial_map(1);
  auto comm=Teuchos::rcp(new Teuchos::MpiComm<int>(Teuchos::opaqueWrapper(MPI_COMM_SELF)));
  auto map=Teuchos::rcp(new Tpetra_Map(256,0,comm));
  auto matrix=Teuchos::rcp(new Tpetra_CrsMatrix(map,3));
  for (GO row=0;row<256;++row) {
    Teuchos::Array<GO> cols;
    Teuchos::Array<double> vals;
    for (GO col=std::max(0,row-1);col<=std::min(255,row+1);++col) {
      cols.push_back(col); vals.push_back(col==row ? 2.1 : -1.0);
    }
    matrix->insertGlobalValues(row,cols(),vals());
  }
  matrix->fillComplete();
  factory::PreconditionerReuseContext context;
  context.muelu_cache=Teuchos::rcp(new factory::MueLuReuseCache());
  context.time_step=1;context.topology_generation=1;
  const auto type=consts::PreconditionerType::PREC_TRILINOS_SCHUR_AMG;
  auto first=factory::create_preconditioner(type,factory::SolverRole::pressure_cg,matrix,Teuchos::null,context);
  Tpetra_MultiVector x(map,1),before(map,1),after(map,1);
  x.randomize(); first->left_operator()->apply(x,before);
  // With all numerical levels refreshed, doubling A must halve M^-1 x.
  // This fails if reuse retains stale coarse matrices or factors.
  matrix->scale(2.0);context.time_step=2;
  auto second=factory::create_preconditioner(type,factory::SolverRole::pressure_cg,matrix,Teuchos::null,context);
  EXPECT_EQ(first->muelu_operator().getRawPtr(),second->muelu_operator().getRawPtr());
  EXPECT_EQ(context.muelu_cache->build_count(),1u);
  EXPECT_EQ(context.muelu_cache->reuse_count(),1u);
  second->left_operator()->apply(x,after);
  after.update(-0.5,before,1.0);
  Teuchos::Array<double> errors(1); after.norm2(errors());
  EXPECT_LT(errors[0],1.e-10);
  context.topology_generation=2;
  factory::create_preconditioner(type,factory::SolverRole::pressure_cg,matrix,Teuchos::null,context);
  EXPECT_EQ(context.muelu_cache->build_count(),2u);
  // Same dimensions and nnz, but a permuted graph: dimensions alone cannot
  // authorize reuse of the old aggregation/communication plans.
  auto changed=Teuchos::rcp(new Tpetra_CrsMatrix(map,3));
  auto permutation=[](GO gid) {return gid==1 ? 2 : (gid==2 ? 1 : gid);};
  for (GO row=0;row<256;++row) {
    Teuchos::Array<GO> cols;Teuchos::Array<double> vals;
    for (GO col=std::max(0,row-1);col<=std::min(255,row+1);++col) {
      cols.push_back(permutation(col));vals.push_back(col==row ? 4.2 : -2.0);
    }
    changed->insertGlobalValues(permutation(row),cols(),vals());
  }
  changed->fillComplete();
  ASSERT_EQ(changed->getGlobalNumEntries(),matrix->getGlobalNumEntries());
  factory::create_preconditioner(type,factory::SolverRole::pressure_cg,changed,Teuchos::null,context);
  EXPECT_EQ(context.muelu_cache->build_count(),3u);
  // SPD congruence scaling can increase the row-sum bound. Rebuild with safe
  // damping rather than silently retaining an unsafe relaxation parameter.
  auto scaling=Teuchos::rcp(new Tpetra_Vector(map));scaling->putScalar(1.0);
  scaling->replaceLocalValue(0,10.0);
  changed->leftScale(*scaling);changed->rightScale(*scaling);
  factory::create_preconditioner(type,factory::SolverRole::pressure_cg,changed,Teuchos::null,context);
  EXPECT_EQ(context.muelu_cache->build_count(),4u);
  context.muelu_cache->clear();
  EXPECT_EQ(context.muelu_cache->hierarchy(),Teuchos::null);
}

TEST(TrilinosSchurAMG, CachedSchurGraphRefreshesNumericValues)
{
  const auto map=make_serial_map(3);
  auto L=make_diagonal_matrix(map,{1,2,3});
  auto B=make_dense_matrix(map,3);
  trilinos_bipartition::preconditioners::MueLuReuseCache cache;
  auto first=cache.schur_matrix(L,B,1);
  const auto graph=first->getCrsGraph();
  B->scale(0.5);L->scale(3.0);
  auto second=cache.schur_matrix(L,B,1);
  EXPECT_EQ(first.getRawPtr(),second.getRawPtr());
  EXPECT_EQ(graph.getRawPtr(),second->getCrsGraph().getRawPtr());
  trilinos_bipartition::PressureSchurOperator reference(L,B);
  Tpetra_MultiVector x(map,2),expected(map,2),actual(map,2);
  x.randomize();reference.apply(x,expected);second->apply(x,actual);
  actual.update(-1.0,expected,1.0);
  Teuchos::Array<double> error(2);actual.norm2(error());
  for (double value:error) EXPECT_LT(value,1.e-9);
}

TEST(TrilinosSchurAMG, CachedProductMatchesTpetraProductAfterValueChanges)
{
  make_serial_map(1);
  auto comm=Tpetra::getDefaultComm();
  const GO n=5*comm->getSize();
  auto pressure=Teuchos::rcp(new Tpetra_Map(n,0,comm));
  auto velocity=Teuchos::rcp(new Tpetra_Map(3*n,0,comm));
  // Rectangular B (velocity rows, pressure columns) with off-diagonal coupling.
  auto B=Teuchos::rcp(new Tpetra_CrsMatrix(velocity,3));
  auto fill_b=[&](double scale,bool insert) {
    for (LO i=0;i<static_cast<LO>(velocity->getLocalNumElements());++i) {
      const GO row=velocity->getGlobalElement(i),p=row/3;
      const auto columns=Teuchos::tuple<GO>(p,(p+1)%n,(p+n-2)%n);
      const auto values=Teuchos::tuple(scale*(1.0+row),-0.5*scale,0.25+0.1*scale*row);
      if (insert) B->insertGlobalValues(row,columns,values);
      else B->replaceGlobalValues(row,columns,values);
    }
    B->fillComplete(pressure,velocity);
  };
  fill_b(1.0,true);
  auto L=make_diagonal_matrix(pressure,std::vector<double>(pressure->getLocalNumElements(),2.0));
  trilinos_bipartition::preconditioners::MueLuReuseCache cached,reference;
  setenv("SV_TRILINOS_SCHUR_PRODUCT","cached",1);
  cached.schur_matrix(L,B,1);
  setenv("SV_TRILINOS_SCHUR_PRODUCT","tpetra",1);
  reference.schur_matrix(L,B,1);
  unsetenv("SV_TRILINOS_SCHUR_PRODUCT");
  EXPECT_EQ(cached.schur_product_cached(),comm->getSize()==1);
  EXPECT_FALSE(reference.schur_product_cached());

  // New numerical values on the same graphs exercise the numeric-only path.
  B->resumeFill();
  fill_b(-1.7,false);
  L->scale(0.3);
  const auto a=cached.schur_matrix(L,B,1);
  const auto b=reference.schur_matrix(L,B,1);
  Tpetra_MultiVector x(pressure,2),ya(pressure,2),yb(pressure,2);
  x.randomize();
  a->apply(x,ya);
  b->apply(x,yb);
  ya.update(-1.0,yb,1.0);
  Teuchos::Array<double> error(2),scale(2);
  ya.norm2(error());
  yb.norm2(scale());
  for (int j=0;j<2;++j) EXPECT_LT(error[j],1.e-13*scale[j]);
}

TEST(TrilinosSchurAMG, ExplicitCorrectionsMatchFactoredWithDistributedCapsAndMultipleRHS)
{
  make_serial_map(1);
  auto comm=Tpetra::getDefaultComm();
  const GO n=4*comm->getSize();
  auto map=Teuchos::rcp(new Tpetra_Map(n,0,comm));
  auto B=Teuchos::rcp(new Tpetra_CrsMatrix(map,2));
  auto L=Teuchos::rcp(new Tpetra_CrsMatrix(map,1));
  for (LO i=0;i<static_cast<LO>(map->getLocalNumElements());++i) {
    const GO row=map->getGlobalElement(i);
    B->insertGlobalValues(row,Teuchos::tuple(row,(row+1)%n),Teuchos::tuple(2.0,-1.0));
    L->insertGlobalValues(row,Teuchos::tuple(row),Teuchos::tuple(0.5));
  }
  B->fillComplete();L->fillComplete();
  std::vector<Teuchos::RCP<Tpetra_MultiVector>> faces,caps;
  std::vector<Trilinos::ResistanceFaceData> data(2);
  for (int face=0;face<2;++face) {
    auto v=Teuchos::rcp(new Tpetra_MultiVector(map,1));
    auto c=Teuchos::rcp(new Tpetra_MultiVector(map,1));
    v->putScalar(0.0);c->putScalar(0.0);
    // Face 0 exists only on rank 0; all ranks must still join the projection.
    {auto vh=v->getLocalViewHost(Tpetra::Access::ReadWrite);
     auto ch=c->getLocalViewHost(Tpetra::Access::ReadWrite);
     for (LO i=0;i<static_cast<LO>(map->getLocalNumElements());++i) {
       const auto gid=map->getGlobalElement(i);
       vh(i,0)=face==0 ? (gid==0 ? 0.3 : 0.0) : 0.1*(gid+1);
       ch(i,0)=0.01*(gid+face);
     }}
    data[face].resistance=face==0 ? 2.0 : -0.1;
    faces.push_back(v);caps.push_back(c);
  }
  auto resistance=Teuchos::rcp(new TrilinosResistanceOperator(map,faces,caps,data));
  resistance->compute();
  auto S=trilinos_bipartition::preconditioners::build_schur_preconditioning_matrix(L,B);
  trilinos_bipartition::ExplicitPressureSchurOperator explicit_op(S,*B,resistance->low_rank_update());
  trilinos_bipartition::PressureSchurOperator reference(L,B,resistance);
  for (const int columns:{2,1,3}) {
    Tpetra_MultiVector x(map,columns),expected(map,columns),actual(map,columns);
    x.randomize();expected.putScalar(2.0);actual.putScalar(2.0);
    reference.apply(x,expected,Teuchos::NO_TRANS,0.7,0.3);
    explicit_op.apply(x,actual,Teuchos::NO_TRANS,0.7,0.3);
    actual.update(-1.0,expected,1.0);
    Teuchos::Array<double> errors(columns);actual.norm2(errors());
    for (double error:errors) EXPECT_LT(error,1.e-11);
  }
}

#endif
