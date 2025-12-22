#include "kamalagin_a_vec_mat_mult/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kamalagin_a_vec_mat_mult/common/include/common.hpp"

namespace kamalagin_a_vec_mat_mult {

KamalaginAVecMatMultSEQ::KamalaginAVecMatMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KamalaginAVecMatMultSEQ::ValidationImpl() {
  const auto &[n, m, a_flat, x] = GetInput();
  if (n < 0 || m < 0) {
    return false;
  }
  if (static_cast<std::size_t>(n) * static_cast<std::size_t>(m) != a_flat.size()) {
    return false;
  }
  if (static_cast<std::size_t>(m) != x.size()) {
    return false;
  }
  return true;
}

bool KamalaginAVecMatMultSEQ::PreProcessingImpl() {
  const auto &[n, m, a_flat, x] = GetInput();
  (void)m;
  (void)a_flat;
  (void)x;

  GetOutput().assign(static_cast<std::size_t>(n), 0);
  return true;
}

bool KamalaginAVecMatMultSEQ::RunImpl() {
  const auto &[n, m, a_flat, x] = GetInput();
  if (n == 0) {
    return true;
  }

  auto &y = GetOutput();
  const auto nn = static_cast<std::size_t>(n);
  const auto mm = static_cast<std::size_t>(m);

  for (std::size_t i = 0; i < nn; ++i) {
    std::int64_t sum = 0;
    const std::size_t row_off = i * mm;
    for (std::size_t j = 0; j < mm; ++j) {
      sum += static_cast<std::int64_t>(a_flat[row_off + j]) * static_cast<std::int64_t>(x[j]);
    }
    y[i] = static_cast<int>(sum);
  }

  return true;
}

bool KamalaginAVecMatMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_vec_mat_mult
