#include "kamalagin_a_mat_mult_strassen/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "kamalagin_a_mat_mult_strassen/common/include/common.hpp"

namespace kamalagin_a_mat_mult_strassen {

namespace {

inline std::size_t Idx(int i, int j, int n) {
  return static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
         static_cast<std::size_t>(j);
}

int NextPow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

std::vector<double> Add(const std::vector<double>& A,
                        const std::vector<double>& B) {
  std::vector<double> C(A.size());
  for (std::size_t i = 0; i < A.size(); ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

std::vector<double> Sub(const std::vector<double>& A,
                        const std::vector<double>& B) {
  std::vector<double> C(A.size());
  for (std::size_t i = 0; i < A.size(); ++i) {
    C[i] = A[i] - B[i];
  }
  return C;
}

std::vector<double> NaiveMul(const std::vector<double>& A,
                             const std::vector<double>& B, int n) {
  std::vector<double> C(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      const double a = A[Idx(i, k, n)];
      for (int j = 0; j < n; ++j) {
        C[Idx(i, j, n)] += a * B[Idx(k, j, n)];
      }
    }
  }
  return C;
}

void Split(const std::vector<double>& A, int n, std::vector<double>* A11,
           std::vector<double>* A12, std::vector<double>* A21,
           std::vector<double>* A22) {
  const int h = n / 2;
  A11->assign(static_cast<std::size_t>(h) * h, 0.0);
  A12->assign(static_cast<std::size_t>(h) * h, 0.0);
  A21->assign(static_cast<std::size_t>(h) * h, 0.0);
  A22->assign(static_cast<std::size_t>(h) * h, 0.0);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      (*A11)[Idx(i, j, h)] = A[Idx(i, j, n)];
      (*A12)[Idx(i, j, h)] = A[Idx(i, j + h, n)];
      (*A21)[Idx(i, j, h)] = A[Idx(i + h, j, n)];
      (*A22)[Idx(i, j, h)] = A[Idx(i + h, j + h, n)];
    }
  }
}

std::vector<double> Join(const std::vector<double>& C11,
                         const std::vector<double>& C12,
                         const std::vector<double>& C21,
                         const std::vector<double>& C22, int n) {
  const int h = n / 2;
  std::vector<double> C(static_cast<std::size_t>(n) * n, 0.0);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      C[Idx(i, j, n)] = C11[Idx(i, j, h)];
      C[Idx(i, j + h, n)] = C12[Idx(i, j, h)];
      C[Idx(i + h, j, n)] = C21[Idx(i, j, h)];
      C[Idx(i + h, j + h, n)] = C22[Idx(i, j, h)];
    }
  }
  return C;
}

std::vector<double> StrassenRec(const std::vector<double>& A,
                                const std::vector<double>& B, int n) {
  constexpr int kThreshold = 64;
  if (n <= kThreshold) {
    return NaiveMul(A, B, n);
  }

  const int h = n / 2;

  std::vector<double> A11, A12, A21, A22;
  std::vector<double> B11, B12, B21, B22;

  Split(A, n, &A11, &A12, &A21, &A22);
  Split(B, n, &B11, &B12, &B21, &B22);

  auto M1 = StrassenRec(Add(A11, A22), Add(B11, B22), h);
  auto M2 = StrassenRec(Add(A21, A22), B11, h);
  auto M3 = StrassenRec(A11, Sub(B12, B22), h);
  auto M4 = StrassenRec(A22, Sub(B21, B11), h);
  auto M5 = StrassenRec(Add(A11, A12), B22, h);
  auto M6 = StrassenRec(Sub(A21, A11), Add(B11, B12), h);
  auto M7 = StrassenRec(Sub(A12, A22), Add(B21, B22), h);

  auto C11 = Add(Sub(Add(M1, M4), M5), M7);
  auto C12 = Add(M3, M5);
  auto C21 = Add(M2, M4);
  auto C22 = Add(Add(Sub(M1, M2), M3), M6);

  return Join(C11, C12, C21, C22, n);
}

std::vector<double> Pad(const std::vector<double>& A, int n, int p) {
  std::vector<double> P(static_cast<std::size_t>(p) * p, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      P[Idx(i, j, p)] = A[Idx(i, j, n)];
    }
  }
  return P;
}

std::vector<double> Unpad(const std::vector<double>& C, int n, int p) {
  std::vector<double> R(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      R[Idx(i, j, n)] = C[Idx(i, j, p)];
    }
  }
  return R;
}

}  // namespace

KamalaginAMatMultStrassenSEQ::KamalaginAMatMultStrassenSEQ(const InType& input) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = input;
  GetOutput().clear();
}

bool KamalaginAMatMultStrassenSEQ::ValidationImpl() {
  const auto& input = GetInput();
  if (input.n < 0) return false;

  const std::size_t expected =
      static_cast<std::size_t>(input.n) * static_cast<std::size_t>(input.n);
  return input.A.size() == expected && input.B.size() == expected;
}

bool KamalaginAMatMultStrassenSEQ::PreProcessingImpl() {
  return true;
}

bool KamalaginAMatMultStrassenSEQ::RunImpl() {
  const auto& input = GetInput();
  const int n = input.n;

  if (n == 0) {
    GetOutput().clear();
    return true;
  }

  const int p = NextPow2(n);
  const auto A_pad = (p == n) ? input.A : Pad(input.A, n, p);
  const auto B_pad = (p == n) ? input.B : Pad(input.B, n, p);

  const auto C_pad = StrassenRec(A_pad, B_pad, p);
  GetOutput() = (p == n) ? C_pad : Unpad(C_pad, n, p);

  return true;
}


bool KamalaginAMatMultStrassenSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_mat_mult_strassen
