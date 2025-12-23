#include "kamalagin_a_mat_mult_strassen/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "kamalagin_a_mat_mult_strassen/common/include/common.hpp"

namespace kamalagin_a_mat_mult_strassen {

namespace {

inline std::size_t Idx(int i, int j, int n) {
  return (static_cast<std::size_t>(i) * static_cast<std::size_t>(n)) + static_cast<std::size_t>(j);
}

int NextPow2(int n) {
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

std::vector<double> Add(const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> c(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
  return c;
}

std::vector<double> Sub(const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> c(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] - b[i];
  }
  return c;
}

std::vector<double> NaiveMul(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      const double av = a[Idx(i, k, n)];
      for (int j = 0; j < n; ++j) {
        c[Idx(i, j, n)] += av * b[Idx(k, j, n)];
      }
    }
  }
  return c;
}

void Split(const std::vector<double> &a, int n, std::vector<double> *a11, std::vector<double> *a12,
           std::vector<double> *a21, std::vector<double> *a22) {
  const int h = n / 2;
  a11->assign(static_cast<std::size_t>(h) * static_cast<std::size_t>(h), 0.0);
  a12->assign(static_cast<std::size_t>(h) * static_cast<std::size_t>(h), 0.0);
  a21->assign(static_cast<std::size_t>(h) * static_cast<std::size_t>(h), 0.0);
  a22->assign(static_cast<std::size_t>(h) * static_cast<std::size_t>(h), 0.0);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      (*a11)[Idx(i, j, h)] = a[Idx(i, j, n)];
      (*a12)[Idx(i, j, h)] = a[Idx(i, j + h, n)];
      (*a21)[Idx(i, j, h)] = a[Idx(i + h, j, n)];
      (*a22)[Idx(i, j, h)] = a[Idx(i + h, j + h, n)];
    }
  }
}

std::vector<double> Join(const std::vector<double> &c11, const std::vector<double> &c12, const std::vector<double> &c21,
                         const std::vector<double> &c22, int n) {
  const int h = n / 2;
  std::vector<double> c(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      c[Idx(i, j, n)] = c11[Idx(i, j, h)];
      c[Idx(i, j + h, n)] = c12[Idx(i, j, h)];
      c[Idx(i + h, j, n)] = c21[Idx(i, j, h)];
      c[Idx(i + h, j + h, n)] = c22[Idx(i, j, h)];
    }
  }
  return c;
}

// NOLINTNEXTLINE(misc-no-recursion)
std::vector<double> StrassenRec(const std::vector<double> &a, const std::vector<double> &b, int n) {
  constexpr int k_threshold = 64;
  if (n <= k_threshold) {
    return NaiveMul(a, b, n);
  }

  const int h = n / 2;

  std::vector<double> a11;
  std::vector<double> a12;
  std::vector<double> a21;
  std::vector<double> a22;

  std::vector<double> b11;
  std::vector<double> b12;
  std::vector<double> b21;
  std::vector<double> b22;

  Split(a, n, &a11, &a12, &a21, &a22);
  Split(b, n, &b11, &b12, &b21, &b22);

  const auto m1 = StrassenRec(Add(a11, a22), Add(b11, b22), h);
  const auto m2 = StrassenRec(Add(a21, a22), b11, h);
  const auto m3 = StrassenRec(a11, Sub(b12, b22), h);
  const auto m4 = StrassenRec(a22, Sub(b21, b11), h);
  const auto m5 = StrassenRec(Add(a11, a12), b22, h);
  const auto m6 = StrassenRec(Sub(a21, a11), Add(b11, b12), h);
  const auto m7 = StrassenRec(Sub(a12, a22), Add(b21, b22), h);

  const auto c11 = Add(Sub(Add(m1, m4), m5), m7);
  const auto c12 = Add(m3, m5);
  const auto c21 = Add(m2, m4);
  const auto c22 = Add(Add(Sub(m1, m2), m3), m6);

  return Join(c11, c12, c21, c22, n);
}

std::vector<double> Pad(const std::vector<double> &a, int n, int p) {
  std::vector<double> out(static_cast<std::size_t>(p) * static_cast<std::size_t>(p), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      out[Idx(i, j, p)] = a[Idx(i, j, n)];
    }
  }
  return out;
}

std::vector<double> Unpad(const std::vector<double> &c, int n, int p) {
  std::vector<double> out(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      out[Idx(i, j, n)] = c[Idx(i, j, p)];
    }
  }
  return out;
}

}  // namespace

KamalaginAMatMultStrassenSEQ::KamalaginAMatMultStrassenSEQ(const InType &input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput().clear();
}

bool KamalaginAMatMultStrassenSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.n < 0) {
    return false;
  }

  const std::size_t expected = static_cast<std::size_t>(input.n) * static_cast<std::size_t>(input.n);
  return input.A.size() == expected && input.B.size() == expected;
}

bool KamalaginAMatMultStrassenSEQ::PreProcessingImpl() {
  return true;
}

bool KamalaginAMatMultStrassenSEQ::RunImpl() {
  const auto &input = GetInput();
  const int n = input.n;

  if (n == 0) {
    GetOutput().clear();
    return true;
  }

  const int p = NextPow2(n);
  const auto a_pad = (p == n) ? input.A : Pad(input.A, n, p);
  const auto b_pad = (p == n) ? input.B : Pad(input.B, n, p);

  const auto c_pad = StrassenRec(a_pad, b_pad, p);
  GetOutput() = (p == n) ? c_pad : Unpad(c_pad, n, p);

  return true;
}

bool KamalaginAMatMultStrassenSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_mat_mult_strassen
