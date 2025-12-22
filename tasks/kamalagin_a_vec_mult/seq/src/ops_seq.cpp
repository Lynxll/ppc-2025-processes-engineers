#include "kamalagin_a_vec_mult/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "kamalagin_a_vec_mult/common/include/common.hpp"

namespace kamalagin_a_vec_mult {

KamalaginAVecMultSEQ::KamalaginAVecMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KamalaginAVecMultSEQ::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return a.size() == b.size();
}

bool KamalaginAVecMultSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KamalaginAVecMultSEQ::RunImpl() {
  const auto &[a, b] = GetInput();

  std::int64_t sum = 0;
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i) {
    sum += static_cast<std::int64_t>(a[i]) * static_cast<std::int64_t>(b[i]);
  }

  GetOutput() = sum;
  return true;
}

bool KamalaginAVecMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_vec_mult
