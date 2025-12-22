#include "kamalagin_a_mat_mult_strassen/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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
  for (std::size_t i = 0; i < A.size(); ++i) C[i] = A[i] + B[i];
  return C;
}

std::vector<double> Sub(const std::vector<double>& A,
                        const std::vector<double>& B) {
  std::vector<double> C(A.size());
  for (std::size_t i = 0; i < A.size(); ++i) C[i] = A[i] - B[i];
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

void Split(const std::vector<double>& A, int n,
           std::vector<double>* A11, std::vector<double>* A12,
           std::vector<double>* A21, std::vector<double>* A22) {
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
  if (n <= kThreshold) return NaiveMul(A, B, n);

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

void SendMatrix(int dst, int tag, int n, const std::vector<double>& M) {
  MPI_Send(&n, 1, MPI_INT, dst, tag, MPI_COMM_WORLD);
  if (n > 0) {
    MPI_Send(M.data(), n * n, MPI_DOUBLE, dst, tag + 1, MPI_COMM_WORLD);
  }
}

void RecvMatrix(int src, int tag, int* n, std::vector<double>* M) {
  MPI_Recv(n, 1, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (*n > 0) {
    M->assign(static_cast<std::size_t>(*n) * (*n), 0.0);
    MPI_Recv(M->data(), (*n) * (*n), MPI_DOUBLE, src, tag + 1,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    M->clear();
  }
}

}  // namespace

KamalaginAMatMultStrassenMPI::KamalaginAMatMultStrassenMPI(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KamalaginAMatMultStrassenMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto& in = GetInput();
    if (in.n < 0) return false;

    const std::size_t expected =
        static_cast<std::size_t>(in.n) * static_cast<std::size_t>(in.n);
    if (in.A.size() != expected) return false;
    if (in.B.size() != expected) return false;
  }
  return true;
}

bool KamalaginAMatMultStrassenMPI::PreProcessingImpl() {
  return true;
}

bool KamalaginAMatMultStrassenMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  constexpr int kThreshold = 64;
  const int kTagTask = 100;
  const int kTagX = 200;
  const int kTagY = 300;
  const int kTagResId = 400;
  const int kTagResMat = 410;

  if (rank != 0) {
    int task_id = 0;
    MPI_Recv(&task_id, 1, MPI_INT, 0, kTagTask,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (task_id == 0) return true;

    int nblock = 0;
    std::vector<double> X, Y;
    RecvMatrix(0, kTagX, &nblock, &X);
    RecvMatrix(0, kTagY, &nblock, &Y);

    auto M = StrassenRec(X, Y, nblock);

    MPI_Send(&task_id, 1, MPI_INT, 0, kTagResId, MPI_COMM_WORLD);
    SendMatrix(0, kTagResMat, nblock, M);

    return true;
  }

  const auto& in = GetInput();
  const int n = in.n;

  if (n == 0) {
    GetOutput().clear();
    for (int r = 1; r < size; ++r) {
      int zero = 0;
      MPI_Send(&zero, 1, MPI_INT, r, kTagTask, MPI_COMM_WORLD);
    }
    return true;
  }

  const int p = NextPow2(n);
  const auto A_pad = (p == n) ? in.A : Pad(in.A, n, p);
  const auto B_pad = (p == n) ? in.B : Pad(in.B, n, p);

  if (p < 2 || p <= kThreshold || size < 2) {
    for (int r = 1; r < size; ++r) {
      int zero = 0;
      MPI_Send(&zero, 1, MPI_INT, r, kTagTask, MPI_COMM_WORLD);
    }

    const auto C_pad = StrassenRec(A_pad, B_pad, p);
    GetOutput() = (p == n) ? C_pad : Unpad(C_pad, n, p);
    return true;
  }
  std::vector<double> A11, A12, A21, A22;
  std::vector<double> B11, B12, B21, B22;
  Split(A_pad, p, &A11, &A12, &A21, &A22);
  Split(B_pad, p, &B11, &B12, &B21, &B22);

  const int h = p / 2;

  std::vector<std::vector<double>> X(8), Y(8);
  X[1] = Add(A11, A22);  Y[1] = Add(B11, B22);
  X[2] = Add(A21, A22);  Y[2] = B11;
  X[3] = A11;            Y[3] = Sub(B12, B22);
  X[4] = A22;            Y[4] = Sub(B21, B11);
  X[5] = Add(A11, A12);  Y[5] = B22;
  X[6] = Sub(A21, A11);  Y[6] = Add(B11, B12);
  X[7] = Sub(A12, A22);  Y[7] = Add(B21, B22);

  std::vector<std::vector<double>> M(8);
  std::vector<int> owner(8, 0);

  int next_task = 1;
  for (int r = 1; r < size && next_task <= 7; ++r, ++next_task) {
    owner[next_task] = r;
    MPI_Send(&next_task, 1, MPI_INT, r, kTagTask, MPI_COMM_WORLD);
    SendMatrix(r, kTagX, h, X[next_task]);
    SendMatrix(r, kTagY, h, Y[next_task]);
  }

  for (int r = next_task; r < size; ++r) {
    int zero = 0;
    MPI_Send(&zero, 1, MPI_INT, r, kTagTask, MPI_COMM_WORLD);
  }

  for (int t = 1; t <= 7; ++t) {
    if (owner[t] == 0) {
      M[t] = StrassenRec(X[t], Y[t], h);
    }
  }

  for (int t = 1; t <= 7; ++t) {
    if (owner[t] != 0) {
      int got_id = 0;
      MPI_Recv(&got_id, 1, MPI_INT, owner[t], kTagResId,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int nb = 0;
      RecvMatrix(owner[t], kTagResMat, &nb, &M[got_id]);
    }
  }

  auto C11 = Add(Sub(Add(M[1], M[4]), M[5]), M[7]);
  auto C12 = Add(M[3], M[5]);
  auto C21 = Add(M[2], M[4]);
  auto C22 = Add(Add(Sub(M[1], M[2]), M[3]), M[6]);

  const auto C_pad = Join(C11, C12, C21, C22, p);
  GetOutput() = (p == n) ? C_pad : Unpad(C_pad, n, p);

  return true;
}

bool KamalaginAMatMultStrassenMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_mat_mult_strassen
