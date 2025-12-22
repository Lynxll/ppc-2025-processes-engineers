#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "kamalagin_a_vec_mat_mult/common/include/common.hpp"
#include "kamalagin_a_vec_mat_mult/mpi/include/ops_mpi.hpp"
#include "kamalagin_a_vec_mat_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kamalagin_a_vec_mat_mult {

class KamalaginAVecMatMultPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr int kN = 400;
  static constexpr int kM = 400;

 protected:
  void SetUp() override {
    const int n = kN;
    const int m = kM;

    std::mt19937 gen(777U);  // NOLINT(cert-msc51-cpp)
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

    const auto mm_sz = static_cast<std::size_t>(mm);
    for (int i = 0; i < nn; ++i) {
      std::int64_t sum = 0;
      for (int j = 0; j < mm; ++j) {
        sum += static_cast<std::int64_t>(A[(static_cast<std::size_t>(i) * mm_sz) + static_cast<std::size_t>(j)]) *
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

TEST_P(KamalaginAVecMatMultPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, KamalaginAVecMatMultMPI, KamalaginAVecMatMultSEQ>(
    PPC_SETTINGS_kamalagin_a_vec_mat_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KamalaginAVecMatMultPerfTestsProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KamalaginAVecMatMultPerfTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace kamalagin_a_vec_mat_mult
