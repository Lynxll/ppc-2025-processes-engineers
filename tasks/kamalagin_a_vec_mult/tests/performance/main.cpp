#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "kamalagin_a_vec_mult/common/include/common.hpp"
#include "kamalagin_a_vec_mult/mpi/include/ops_mpi.hpp"
#include "kamalagin_a_vec_mult/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kamalagin_a_vec_mult {

class KamalaginAVecMultRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr int kN = 200000;

 protected:
  void SetUp() override {
    unsigned seed = 0U;

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      std::random_device rd;
      seed = rd();
    }

    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(-100, 100);

    std::vector<int> a(static_cast<std::size_t>(kN));
    std::vector<int> b(static_cast<std::size_t>(kN));
    for (int i = 0; i < kN; ++i) {
      a[static_cast<std::size_t>(i)] = dist(gen);
      b[static_cast<std::size_t>(i)] = dist(gen);
    }

    input_data_ = {std::move(a), std::move(b)};

    expected_ = 0;
    const auto &[aa, bb] = input_data_;
    for (std::size_t i = 0; i < aa.size(); ++i) {
      expected_ += static_cast<std::int64_t>(aa[i]) * static_cast<std::int64_t>(bb[i]);
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
  OutType expected_{0};
};

TEST_P(KamalaginAVecMultRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KamalaginAVecMultMPI, KamalaginAVecMultSEQ>(PPC_SETTINGS_kamalagin_a_vec_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KamalaginAVecMultRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, KamalaginAVecMultRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace kamalagin_a_vec_mult
