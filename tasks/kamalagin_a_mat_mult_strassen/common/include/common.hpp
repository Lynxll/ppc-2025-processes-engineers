#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kamalagin_a_mat_mult_strassen {

struct InType {
  int n{};
  std::vector<double> A;
  std::vector<double> B;
};

using OutType = std::vector<double>;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kamalagin_a_mat_mult_strassen
