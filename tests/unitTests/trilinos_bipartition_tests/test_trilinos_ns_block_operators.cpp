// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "gtest/gtest.h"

#ifdef WITH_TRILINOS

#include "TrilinosNSBlockOperators.h"
#include "TrilinosResistanceOperator.h"

#include "Teuchos_DefaultSerialComm.hpp"

#include <utility>
#include <vector>

namespace {

class KokkosBlockTestScope
{
  public:
    KokkosBlockTestScope()
    {
      if (!Kokkos::is_initialized()) {
        Kokkos::initialize();
        owns_kokkos_ = true;
      }
    }

    ~KokkosBlockTestScope()
    {
      if (owns_kokkos_ && Kokkos::is_initialized()) {
        Kokkos::finalize();
      }
    }

  private:
    bool owns_kokkos_ = false;
};

Teuchos::RCP<const Tpetra_Map> make_serial_map(
    Tpetra::global_size_t size)
{
  static KokkosBlockTestScope kokkos_scope;
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
  state->Map = matrix->getRowMap();
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

#endif
