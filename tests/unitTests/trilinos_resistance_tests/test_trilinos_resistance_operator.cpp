// SPDX-FileCopyrightText: Copyright (c) Stanford University, The Regents of the University of California, and others.
// SPDX-License-Identifier: BSD-3-Clause

#include "gtest/gtest.h"

#ifdef WITH_TRILINOS

#include "TrilinosResistanceOperator.h"

#include "Teuchos_DefaultSerialComm.hpp"

#include <vector>

namespace {

class TpetraTestScope
{
  public:
    TpetraTestScope() :
      scope_(&argc_, &argv_) {}

  private:
    int argc_ = 0;
    char** argv_ = nullptr;
    Tpetra::ScopeGuard scope_;
};

Teuchos::RCP<const Tpetra_Map> make_serial_map(Tpetra::global_size_t size)
{
  static TpetraTestScope tpetra_scope;
  auto comm = Teuchos::rcp(new Teuchos::SerialComm<int>());
  return Teuchos::rcp(new Tpetra_Map(size, 0, comm));
}

Teuchos::RCP<Tpetra_MultiVector> make_vector(
    const Teuchos::RCP<const Tpetra_Map>& map,
    const std::vector<double>& values)
{
  auto vector = Teuchos::rcp(new Tpetra_MultiVector(map, 1));
  for (size_t i = 0; i < values.size(); ++i) {
    vector->replaceGlobalValue(static_cast<GO>(i), 0, values[i]);
  }
  return vector;
}

double vector_value(const Tpetra_MultiVector& vector, size_t index)
{
  const auto view = vector.getLocalViewHost(Tpetra::Access::ReadOnly);
  return view(index, 0);
}

} // namespace

TEST(TrilinosResistanceOperator, AppliesMultipleFaceAndCapUpdates)
{
  const auto map = make_serial_map(2);
  const std::vector<Teuchos::RCP<Tpetra_MultiVector>> faces = {
    make_vector(map, {2.0, 4.0}),
    make_vector(map, {0.5, -0.5})
  };
  const std::vector<Teuchos::RCP<Tpetra_MultiVector>> caps = {
    make_vector(map, {1.0, 0.0}),
    make_vector(map, {0.0, 0.0})
  };
  std::vector<Trilinos::ResistanceFaceData> face_data(2);
  face_data[0].face_id = 0;
  face_data[0].resistance = 4.0;
  face_data[1].face_id = 1;
  face_data[1].resistance = 1.0;

  TrilinosResistanceOperator resistance(map, faces, caps, face_data);
  resistance.compute();

  const auto input = make_vector(map, {3.0, -1.0});
  Tpetra_MultiVector output(map, 1);
  resistance.apply(*input, output);

  const double first_projection = 5.0;
  const double second_projection = 2.0;
  const double expected_0 = 3.0 - 2.0 * first_projection / 21.0 -
    0.5 * second_projection / 1.5;
  const double expected_1 = -1.0 - 4.0 * first_projection / 21.0 +
    0.5 * second_projection / 1.5;
  EXPECT_NEAR(vector_value(output, 0), expected_0, 1.0e-13);
  EXPECT_NEAR(vector_value(output, 1), expected_1, 1.0e-13);

  const auto& computed_faces = resistance.resistance_faces();
  EXPECT_NEAR(computed_faces[0].s_tilde_norm2, 5.0, 1.0e-13);
  EXPECT_NEAR(computed_faces[0].alpha, -4.0 / 21.0, 1.0e-13);
}

TEST(TrilinosResistanceOperator, IsIdentityWithoutActiveResistance)
{
  const auto map = make_serial_map(2);
  const std::vector<Teuchos::RCP<Tpetra_MultiVector>> faces = {
    make_vector(map, {1.0, 2.0})
  };
  const std::vector<Teuchos::RCP<Tpetra_MultiVector>> caps = {
    make_vector(map, {0.0, 0.0})
  };
  const std::vector<Trilinos::ResistanceFaceData> face_data(1);

  TrilinosResistanceOperator resistance(map, faces, caps, face_data);
  resistance.compute();

  const auto input = make_vector(map, {3.0, -1.0});
  Tpetra_MultiVector output(map, 1);
  resistance.apply(*input, output);

  EXPECT_FALSE(resistance.has_active_resistance());
  EXPECT_DOUBLE_EQ(vector_value(output, 0), 3.0);
  EXPECT_DOUBLE_EQ(vector_value(output, 1), -1.0);
}

#endif
