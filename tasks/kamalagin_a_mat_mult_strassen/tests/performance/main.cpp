#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <string>

#include "kamalagin_a_mat_mult_strassen/common/include/common.hpp"
#include "kamalagin_a_mat_mult_strassen/mpi/include/ops_mpi.hpp"
#include "kamalagin_a_mat_mult_strassen/seq/include/ops_seq.hpp"
#include "performance/include/performance.hpp"
#include "util/include/perf_test_util.hpp"

namespace kamalagin_a_mat_mult_strassen {

class KamalaginARunPerfTestsMatMultStrassenProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType input_data{};

  void SetUp() override {
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const std::string task_name = std::get<1>(GetParam());
    if (task_name.find("_seq_enabled") != std::string::npos && world_size != 1) {
      GTEST_SKIP() << "SEQ perf should be executed with 1 MPI process (mpiexec -n 1).";
    }

    constexpr int kN = 32;
    input_data.n = kN;

    const std::size_t size = static_cast<std::size_t>(kN) * static_cast<std::size_t>(kN);

    input_data.A.assign(size, 1.0);
    input_data.B.assign(size, 1.0);
  }

  void SetPerfAttributes(ppc::performance::PerfAttr &perf_attrs) override {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const double start_time = (rank == 0) ? MPI_Wtime() : 0.0;

    perf_attrs.current_timer = [rank, start_time] {
      if (rank != 0) {
        return 0.0;
      }
      return MPI_Wtime() - start_time;
    };

    perf_attrs.num_running = 5;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    const std::size_t expected_size = static_cast<std::size_t>(input_data.n) * static_cast<std::size_t>(input_data.n);

    return output_data.size() == expected_size;
  }

  InType GetTestInputData() final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return InType{};
    }

    return input_data;
  }
};

TEST_P(KamalaginARunPerfTestsMatMultStrassenProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KamalaginAMatMultStrassenMPI, KamalaginAMatMultStrassenSEQ>(
        PPC_SETTINGS_kamalagin_a_mat_mult_strassen);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KamalaginARunPerfTestsMatMultStrassenProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTestsKamalaginAMatMultStrassen, KamalaginARunPerfTestsMatMultStrassenProcesses,
                         kGtestValues, kPerfTestName);

}  // namespace kamalagin_a_mat_mult_strassen
