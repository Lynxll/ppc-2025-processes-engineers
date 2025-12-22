#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "kamalagin_a_mat_mult_strassen/common/include/common.hpp"
#include "kamalagin_a_mat_mult_strassen/mpi/include/ops_mpi.hpp"
#include "kamalagin_a_mat_mult_strassen/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace kamalagin_a_mat_mult_strassen {

class KamalaginARunFuncTestsMatMult : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const auto params = std::get<1>(GetParam());
    const int n = std::get<0>(params);

    input_data_.n = n;
    input_data_.A.assign(static_cast<std::size_t>(n) * n, 0.0);
    input_data_.B.assign(static_cast<std::size_t>(n) * n, 0.0);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        input_data_.A[static_cast<std::size_t>(i) * n + j] = static_cast<double>((i + 1) * (j + 2));
        input_data_.B[static_cast<std::size_t>(i) * n + j] = static_cast<double>((i + 3) - (j + 1));
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
      return true;
    }

    const int n = input_data_.n;

    std::vector<double> expected(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        const double aik =
            input_data_.A[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(k)];
        for (int j = 0; j < n; ++j) {
          expected[(static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(j)] +=
              aik *
              input_data_.B[(static_cast<std::size_t>(k) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(j)];
        }
      }
    }

    return output_data == expected;
  }

  InType GetTestInputData() final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return InType{};
    }
    return input_data_;
  }

 private:
  InType input_data_{};
};

namespace {

TEST_P(KamalaginARunFuncTestsMatMult, DenseMatMultStrassen) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(0, "n0"), std::make_tuple(1, "n1"),
                                            std::make_tuple(2, "n2"), std::make_tuple(3, "n3"),
                                            std::make_tuple(5, "n5"), std::make_tuple(8, "n8")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<KamalaginAMatMultStrassenMPI, InType>(
                                               kTestParam, PPC_SETTINGS_kamalagin_a_mat_mult_strassen),
                                           ppc::util::AddFuncTask<KamalaginAMatMultStrassenSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_kamalagin_a_mat_mult_strassen));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = KamalaginARunFuncTestsMatMult::PrintFuncTestName<KamalaginARunFuncTestsMatMult>;

INSTANTIATE_TEST_SUITE_P(StrassenDenseMatMult, KamalaginARunFuncTestsMatMult, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace kamalagin_a_mat_mult_strassen
