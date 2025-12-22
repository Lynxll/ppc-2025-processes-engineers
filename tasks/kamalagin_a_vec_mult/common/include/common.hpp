#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace kamalagin_a_vec_mult {

using InType = std::pair<std::vector<int>, std::vector<int>>;
using OutType = std::int64_t;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace kamalagin_a_vec_mult
