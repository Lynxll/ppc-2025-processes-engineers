#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace kamalagin_a_vec_mat_mult {

using InType = std::tuple<int, int, std::vector<int>, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kamalagin_a_vec_mat_mult
