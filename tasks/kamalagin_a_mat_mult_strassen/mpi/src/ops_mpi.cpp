#include "kamalagin_a_mat_mult_strassen/mpi/include/ops_mpi.hpp"

#include <mpi.h>

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
  std::vector<double> c(static_cast<std::size_t>(n) * n, 0.0);
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
  a11->assign(static_cast<std::size_t>(h) * h, 0.0);
  a12->assign(static_cast<std::size_t>(h) * h, 0.0);
  a21->assign(static_cast<std::size_t>(h) * h, 0.0);
  a22->assign(static_cast<std::size_t>(h) * h, 0.0);

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
  std::vector<double> c(static_cast<std::size_t>(n) * n, 0.0);

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
  constexpr int kThreshold = 64;
  if (n <= kThreshold) {
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
  std::vector<double> out(static_cast<std::size_t>(p) * p, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      out[Idx(i, j, p)] = a[Idx(i, j, n)];
    }
  }
  return out;
}

std::vector<double> Unpad(const std::vector<double> &c, int n, int p) {
  std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      out[Idx(i, j, n)] = c[Idx(i, j, p)];
    }
  }
  return out;
}

void SendMatrix(int dst, int tag, int n, const std::vector<double> &m) {
  MPI_Send(&n, 1, MPI_INT, dst, tag, MPI_COMM_WORLD);
  if (n > 0) {
    MPI_Send(m.data(), n * n, MPI_DOUBLE, dst, tag + 1, MPI_COMM_WORLD);
  }
}

void RecvMatrix(int src, int tag, int *n, std::vector<double> *m) {
  MPI_Recv(n, 1, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (*n > 0) {
    m->assign(static_cast<std::size_t>(*n) * (*n), 0.0);
    MPI_Recv(m->data(), (*n) * (*n), MPI_DOUBLE, src, tag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    m->clear();
  }
}

}  // namespace

KamalaginAMatMultStrassenMPI::KamalaginAMatMultStrassenMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KamalaginAMatMultStrassenMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &in = GetInput();
    if (in.n < 0) {
      return false;
    }

    const std::size_t expected = static_cast<std::size_t>(in.n) * static_cast<std::size_t>(in.n);
    if (in.A.size() != expected) {
      return false;
    }
    if (in.B.size() != expected) {
      return false;
    }
  }
  return true;
}

bool KamalaginAMatMultStrassenMPI::PreProcessingImpl() {
  return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool KamalaginAMatMultStrassenMPI::RunImpl() {
  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  constexpr int k_threshold = 64;
  constexpr int k_tag_task = 100;
  constexpr int k_tag_x = 200;
  constexpr int k_tag_y = 300;
  constexpr int k_tag_res_id = 400;
  constexpr int k_tag_res_mat = 410;

  if (rank != 0) {
    int task_id = 0;
    MPI_Recv(&task_id, 1, MPI_INT, 0, k_tag_task, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (task_id == 0) {
      return true;
    }

    int nblock = 0;
    std::vector<double> x;
    std::vector<double> y;

    RecvMatrix(0, k_tag_x, &nblock, &x);
    RecvMatrix(0, k_tag_y, &nblock, &y);

    const auto m = StrassenRec(x, y, nblock);

    MPI_Send(&task_id, 1, MPI_INT, 0, k_tag_res_id, MPI_COMM_WORLD);
    SendMatrix(0, k_tag_res_mat, nblock, m);

    return true;
  }

  const auto &in = GetInput();
  const int n = in.n;

  if (n == 0) {
    GetOutput().clear();
    for (int proc = 1; proc < comm_size; ++proc) {
      int zero = 0;
      MPI_Send(&zero, 1, MPI_INT, proc, k_tag_task, MPI_COMM_WORLD);
    }
    return true;
  }

  const int p = NextPow2(n);
  const auto a_pad = (p == n) ? in.A : Pad(in.A, n, p);
  const auto b_pad = (p == n) ? in.B : Pad(in.B, n, p);

  if (p <= k_threshold || comm_size < 2) {
    for (int proc = 1; proc < comm_size; ++proc) {
      int zero = 0;
      MPI_Send(&zero, 1, MPI_INT, proc, k_tag_task, MPI_COMM_WORLD);
    }

    const auto c_pad = StrassenRec(a_pad, b_pad, p);
    GetOutput() = (p == n) ? c_pad : Unpad(c_pad, n, p);
    return true;
  }

  std::vector<double> a11;
  std::vector<double> a12;
  std::vector<double> a21;
  std::vector<double> a22;

  std::vector<double> b11;
  std::vector<double> b12;
  std::vector<double> b21;
  std::vector<double> b22;

  Split(a_pad, p, &a11, &a12, &a21, &a22);
  Split(b_pad, p, &b11, &b12, &b21, &b22);

  const int h = p / 2;

  std::vector<std::vector<double>> x(8);
  std::vector<std::vector<double>> y(8);

  x[1] = Add(a11, a22);
  y[1] = Add(b11, b22);

  x[2] = Add(a21, a22);
  y[2] = b11;

  x[3] = a11;
  y[3] = Sub(b12, b22);

  x[4] = a22;
  y[4] = Sub(b21, b11);

  x[5] = Add(a11, a12);
  y[5] = b22;

  x[6] = Sub(a21, a11);
  y[6] = Add(b11, b12);

  x[7] = Sub(a12, a22);
  y[7] = Add(b21, b22);

  std::vector<std::vector<double>> m(8);
  std::vector<int> owner(8, 0);

  int next_task = 1;
  for (int proc = 1; proc < comm_size && next_task <= 7; ++proc, ++next_task) {
    owner[next_task] = proc;
    MPI_Send(&next_task, 1, MPI_INT, proc, k_tag_task, MPI_COMM_WORLD);
    SendMatrix(proc, k_tag_x, h, x[next_task]);
    SendMatrix(proc, k_tag_y, h, y[next_task]);
  }

  for (int proc = next_task; proc < comm_size; ++proc) {
    int zero = 0;
    MPI_Send(&zero, 1, MPI_INT, proc, k_tag_task, MPI_COMM_WORLD);
  }

  for (int task = 1; task <= 7; ++task) {
    if (owner[task] == 0) {
      m[task] = StrassenRec(x[task], y[task], h);
    }
  }

  for (int task = 1; task <= 7; ++task) {
    if (owner[task] != 0) {
      int got_id = 0;
      MPI_Recv(&got_id, 1, MPI_INT, owner[task], k_tag_res_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int nb = 0;
      RecvMatrix(owner[task], k_tag_res_mat, &nb, &m[got_id]);
    }
  }

  const auto c11 = Add(Sub(Add(m[1], m[4]), m[5]), m[7]);
  const auto c12 = Add(m[3], m[5]);
  const auto c21 = Add(m[2], m[4]);
  const auto c22 = Add(Add(Sub(m[1], m[2]), m[3]), m[6]);

  const auto c_pad = Join(c11, c12, c21, c22, p);
  GetOutput() = (p == n) ? c_pad : Unpad(c_pad, n, p);

  return true;
}

bool KamalaginAMatMultStrassenMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_mat_mult_strassen
