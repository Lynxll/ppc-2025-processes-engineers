#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kamalagin_a_vec_mult/common/include/common.hpp"
#include "kamalagin_a_vec_mult/mpi/include/ops_mpi.hpp"
#include "kamalagin_a_vec_mult/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace kamalagin_a_vec_mult {

class KamalaginAVecMultTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int n = std::get<0>(params);

    std::mt19937 gen(12345U + static_cast<unsigned>(n));
    std::uniform_int_distribution<int> dist(-100, 100);

    std::vector<int> a(static_cast<std::size_t>(n));
    std::vector<int> b(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
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

namespace {

TEST_P(KamalaginAVecMultTestsProcesses, DotProduct) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {
    std::make_tuple(0, "empty"),  std::make_tuple(1, "n1"),       std::make_tuple(10, "n10"),
    std::make_tuple(257, "n257"), std::make_tuple(1000, "n1000"),
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<KamalaginAVecMultMPI, InType>(kTestParam, PPC_SETTINGS_kamalagin_a_vec_mult),
                   ppc::util::AddFuncTask<KamalaginAVecMultSEQ, InType>(kTestParam, PPC_SETTINGS_kamalagin_a_vec_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = KamalaginAVecMultTestsProcesses::PrintFuncTestName<KamalaginAVecMultTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(DotProductTests, KamalaginAVecMultTestsProcesses, kGtestValues, kTestName);

}  // namespace

}  // namespace kamalagin_a_vec_mult
