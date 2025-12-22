#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kamalagin_a_vec_mat_mult/common/include/common.hpp"
#include "kamalagin_a_vec_mat_mult/mpi/include/ops_mpi.hpp"
#include "kamalagin_a_vec_mat_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kamalagin_a_vec_mat_mult {

class KamalaginAVecMatMultTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "x" + std::to_string(std::get<1>(test_param)) + "_" +
           std::get<2>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int n = std::get<0>(params);
    const int m = std::get<1>(params);

    std::mt19937 gen(12345U + (static_cast<unsigned>(n) * 1009U) + static_cast<unsigned>(m));
    std::uniform_int_distribution<int> dist(-10, 10);

    std::vector<int> a_flat(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
    std::vector<int> x(static_cast<std::size_t>(m));

    for (auto &v : a_flat) {
      v = dist(gen);
    }
    for (auto &v : x) {
      v = dist(gen);
    }

    input_data_ = std::make_tuple(n, m, std::move(a_flat), std::move(x));

    expected_.assign(static_cast<std::size_t>(n), 0);
    const auto &[nn, mm, A, X] = input_data_;
    for (int i = 0; i < nn; ++i) {
      std::int64_t sum = 0;
      for (int j = 0; j < mm; ++j) {
        sum += static_cast<std::int64_t>(
                   A[(static_cast<std::size_t>(i) * static_cast<std::size_t>(mm)) + static_cast<std::size_t>(j)]) *
               static_cast<std::int64_t>(X[static_cast<std::size_t>(j)]);
      }
      expected_[static_cast<std::size_t>(i)] = static_cast<int>(sum);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_;
};

namespace {

TEST_P(KamalaginAVecMatMultTestsProcesses, MatVec) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 8> kTestParam = {
    std::make_tuple(0, 0, "empty"), std::make_tuple(0, 5, "n0"),        std::make_tuple(5, 0, "m0"),
    std::make_tuple(1, 1, "1x1"),   std::make_tuple(1, 10, "1x10"),     std::make_tuple(10, 1, "10x1"),
    std::make_tuple(5, 3, "5x3"),   std::make_tuple(257, 10, "257x10"),
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<KamalaginAVecMatMultMPI, InType>(kTestParam, PPC_SETTINGS_kamalagin_a_vec_mat_mult),
    ppc::util::AddFuncTask<KamalaginAVecMatMultSEQ, InType>(kTestParam, PPC_SETTINGS_kamalagin_a_vec_mat_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = KamalaginAVecMatMultTestsProcesses::PrintFuncTestName<KamalaginAVecMatMultTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatVecTests, KamalaginAVecMatMultTestsProcesses, kGtestValues, kTestName);

}  // namespace

}  // namespace kamalagin_a_vec_mat_mult
