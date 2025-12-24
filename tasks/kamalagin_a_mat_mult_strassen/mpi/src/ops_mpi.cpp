#include "kamalagin_a_mat_mult_strassen/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
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

std::vector<double> StrassenIter(const std::vector<double> &a, const std::vector<double> &b, int n) {
  constexpr int kThreshold = 64;

  struct Frame {
    int n{};
    int stage{};
    std::vector<double> a;
    std::vector<double> b;

    std::vector<double> a11;
    std::vector<double> a12;
    std::vector<double> a21;
    std::vector<double> a22;

    std::vector<double> b11;
    std::vector<double> b12;
    std::vector<double> b21;
    std::vector<double> b22;

    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;

    Frame(int n_val, int stage_val, std::vector<double> a_val, std::vector<double> b_val)
        : n(n_val), stage(stage_val), a(std::move(a_val)), b(std::move(b_val)), x(8), y(8) {}
  };

  std::vector<Frame> frames;
  std::vector<std::vector<double>> results;

  frames.emplace_back(n, 0, a, b);

  while (!frames.empty()) {
    Frame &f = frames.back();

    if (f.stage == 0) {
      if (f.n <= kThreshold) {
        results.push_back(NaiveMul(f.a, f.b, f.n));
        frames.pop_back();
        continue;
      }

      const int h = f.n / 2;

      Split(f.a, f.n, &f.a11, &f.a12, &f.a21, &f.a22);
      Split(f.b, f.n, &f.b11, &f.b12, &f.b21, &f.b22);

      f.x[1] = Add(f.a11, f.a22);
      f.y[1] = Add(f.b11, f.b22);

      f.x[2] = Add(f.a21, f.a22);
      f.y[2] = f.b11;

      f.x[3] = f.a11;
      f.y[3] = Sub(f.b12, f.b22);

      f.x[4] = f.a22;
      f.y[4] = Sub(f.b21, f.b11);

      f.x[5] = Add(f.a11, f.a12);
      f.y[5] = f.b22;

      f.x[6] = Sub(f.a21, f.a11);
      f.y[6] = Add(f.b11, f.b12);

      f.x[7] = Sub(f.a12, f.a22);
      f.y[7] = Add(f.b21, f.b22);

      f.stage = 1;

      frames.emplace_back(h, 0, f.x[7], f.y[7]);
      frames.emplace_back(h, 0, f.x[6], f.y[6]);
      frames.emplace_back(h, 0, f.x[5], f.y[5]);
      frames.emplace_back(h, 0, f.x[4], f.y[4]);
      frames.emplace_back(h, 0, f.x[3], f.y[3]);
      frames.emplace_back(h, 0, f.x[2], f.y[2]);
      frames.emplace_back(h, 0, f.x[1], f.y[1]);
      continue;
    }

    auto m7 = std::move(results.back());
    results.pop_back();
    auto m6 = std::move(results.back());
    results.pop_back();
    auto m5 = std::move(results.back());
    results.pop_back();
    auto m4 = std::move(results.back());
    results.pop_back();
    auto m3 = std::move(results.back());
    results.pop_back();
    auto m2 = std::move(results.back());
    results.pop_back();
    auto m1 = std::move(results.back());
    results.pop_back();

    const auto c11 = Add(Sub(Add(m1, m4), m5), m7);
    const auto c12 = Add(m3, m5);
    const auto c21 = Add(m2, m4);
    const auto c22 = Add(Add(Sub(m1, m2), m3), m6);

    results.push_back(Join(c11, c12, c21, c22, f.n));
    frames.pop_back();
  }

  return results.empty() ? std::vector<double>{} : std::move(results.back());
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

void SendMatrix(int dst, int tag, int n, const std::vector<double> &m) {
  MPI_Send(&n, 1, MPI_INT, dst, tag, MPI_COMM_WORLD);
  if (n > 0) {
    MPI_Send(m.data(), n * n, MPI_DOUBLE, dst, tag + 1, MPI_COMM_WORLD);
  }
}

void RecvMatrix(int src, int tag, int *n, std::vector<double> *m) {
  MPI_Recv(n, 1, MPI_INT, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  if (*n > 0) {
    m->assign(static_cast<std::size_t>(*n) * static_cast<std::size_t>(*n), 0.0);
    MPI_Recv(m->data(), (*n) * (*n), MPI_DOUBLE, src, tag + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    m->clear();
  }
}

struct MpiTags {
  int tag_task{};
  int tag_x{};
  int tag_y{};
  int tag_res_id{};
  int tag_res_mat{};
};

void SendStopToAll(int comm_size, int tag_task) {
  for (int proc = 1; proc < comm_size; ++proc) {
    int zero = 0;
    MPI_Send(&zero, 1, MPI_INT, proc, tag_task, MPI_COMM_WORLD);
  }
}

bool RunWorker(const MpiTags &tags) {
  int task_id = 0;
  MPI_Recv(&task_id, 1, MPI_INT, 0, tags.tag_task, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (task_id == 0) {
    return true;
  }

  int nblock = 0;
  std::vector<double> x;
  std::vector<double> y;

  RecvMatrix(0, tags.tag_x, &nblock, &x);
  RecvMatrix(0, tags.tag_y, &nblock, &y);

  const auto m = StrassenIter(x, y, nblock);

  MPI_Send(&task_id, 1, MPI_INT, 0, tags.tag_res_id, MPI_COMM_WORLD);
  SendMatrix(0, tags.tag_res_mat, nblock, m);
  return true;
}

struct RootPrepared {
  int n{};
  int p{};
  std::vector<double> a_pad;
  std::vector<double> b_pad;
};

RootPrepared PrepareRootInput(const InType &in) {
  RootPrepared prep{};
  prep.n = in.n;

  if (prep.n == 0) {
    prep.p = 0;
    return prep;
  }

  prep.p = NextPow2(prep.n);
  prep.a_pad = (prep.p == prep.n) ? in.A : Pad(in.A, prep.n, prep.p);
  prep.b_pad = (prep.p == prep.n) ? in.B : Pad(in.B, prep.n, prep.p);

  return prep;
}

OutType RunRootTrivialOrSingleProc(const RootPrepared &prep, int comm_size, int k_threshold, int tag_task) {
  if (prep.n == 0) {
    SendStopToAll(comm_size, tag_task);
    return {};
  }

  if (prep.p <= k_threshold || comm_size < 2) {
    SendStopToAll(comm_size, tag_task);
    const auto c_pad = StrassenIter(prep.a_pad, prep.b_pad, prep.p);
    return (prep.p == prep.n) ? c_pad : Unpad(c_pad, prep.n, prep.p);
  }

  return {};
}

struct StrassenOperands {
  int h{};
  std::vector<std::vector<double>> x;
  std::vector<std::vector<double>> y;
};

StrassenOperands BuildRootOperands(const std::vector<double> &a_pad, const std::vector<double> &b_pad, int p) {
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

  StrassenOperands ops{};
  ops.h = p / 2;
  ops.x.assign(8, {});
  ops.y.assign(8, {});

  ops.x[1] = Add(a11, a22);
  ops.y[1] = Add(b11, b22);

  ops.x[2] = Add(a21, a22);
  ops.y[2] = b11;

  ops.x[3] = a11;
  ops.y[3] = Sub(b12, b22);

  ops.x[4] = a22;
  ops.y[4] = Sub(b21, b11);

  ops.x[5] = Add(a11, a12);
  ops.y[5] = b22;

  ops.x[6] = Sub(a21, a11);
  ops.y[6] = Add(b11, b12);

  ops.x[7] = Sub(a12, a22);
  ops.y[7] = Add(b21, b22);

  return ops;
}

std::vector<int> DispatchTasks(int comm_size, const MpiTags &tags, int h, const std::vector<std::vector<double>> &x,
                               const std::vector<std::vector<double>> &y) {
  std::vector<int> owner(8, 0);
  int next_task = 1;

  for (int proc = 1; proc < comm_size && next_task <= 7; ++proc, ++next_task) {
    owner[next_task] = proc;
    MPI_Send(&next_task, 1, MPI_INT, proc, tags.tag_task, MPI_COMM_WORLD);
    SendMatrix(proc, tags.tag_x, h, x[next_task]);
    SendMatrix(proc, tags.tag_y, h, y[next_task]);
  }

  for (int proc = next_task; proc < comm_size; ++proc) {
    int zero = 0;
    MPI_Send(&zero, 1, MPI_INT, proc, tags.tag_task, MPI_COMM_WORLD);
  }

  return owner;
}

std::vector<std::vector<double>> ComputeLocalTasks(const std::vector<int> &owner, int h,
                                                   const std::vector<std::vector<double>> &x,
                                                   const std::vector<std::vector<double>> &y) {
  std::vector<std::vector<double>> m(8);
  for (int task = 1; task <= 7; ++task) {
    if (owner[task] == 0) {
      m[task] = StrassenIter(x[task], y[task], h);
    }
  }
  return m;
}

void CollectRemoteTasks(const std::vector<int> &owner, const MpiTags &tags, std::vector<std::vector<double>> *m) {
  for (int task = 1; task <= 7; ++task) {
    if (owner[task] != 0) {
      int got_id = 0;
      MPI_Recv(&got_id, 1, MPI_INT, owner[task], tags.tag_res_id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      int nb = 0;
      RecvMatrix(owner[task], tags.tag_res_mat, &nb, &(*m)[got_id]);
    }
  }
}

OutType CombineRootResult(const std::vector<std::vector<double>> &m, int p, int n) {
  const auto c11 = Add(Sub(Add(m[1], m[4]), m[5]), m[7]);
  const auto c12 = Add(m[3], m[5]);
  const auto c21 = Add(m[2], m[4]);
  const auto c22 = Add(Add(Sub(m[1], m[2]), m[3]), m[6]);

  const auto c_pad = Join(c11, c12, c21, c22, p);
  return (p == n) ? c_pad : Unpad(c_pad, n, p);
}

}  // namespace

KamalaginAMatMultStrassenMPI::KamalaginAMatMultStrassenMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KamalaginAMatMultStrassenMPI::ValidationImpl() {
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

  return true;
}
bool KamalaginAMatMultStrassenMPI::PreProcessingImpl() {
  return true;
}

bool KamalaginAMatMultStrassenMPI::RunImpl() {
  int initialized = 0;
  MPI_Initialized(&initialized);

  if (!initialized) {
    const auto &in = GetInput();
    const auto prep = PrepareRootInput(in);

    if (prep.n == 0) {
      GetOutput().clear();
      return true;
    }

    const auto c_pad = StrassenIter(prep.a_pad, prep.b_pad, prep.p);
    GetOutput() = (prep.p == prep.n) ? c_pad : Unpad(c_pad, prep.n, prep.p);
    return true;
  }

  int rank = 0;
  int comm_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  constexpr int kThreshold = 64;
  const MpiTags tags{.tag_task = 100, .tag_x = 200, .tag_y = 300, .tag_res_id = 400, .tag_res_mat = 410};

  if (rank != 0) {
    return RunWorker(tags);
  }

  const auto &in = GetInput();
  const auto prep = PrepareRootInput(in);

  auto out = RunRootTrivialOrSingleProc(prep, comm_size, kThreshold, tags.tag_task);
  if (!out.empty() || prep.n == 0) {
    GetOutput() = std::move(out);
    return true;
  }

  const auto ops = BuildRootOperands(prep.a_pad, prep.b_pad, prep.p);

  const auto owner = DispatchTasks(comm_size, tags, ops.h, ops.x, ops.y);

  auto m = ComputeLocalTasks(owner, ops.h, ops.x, ops.y);
  CollectRemoteTasks(owner, tags, &m);

  GetOutput() = CombineRootResult(m, prep.p, prep.n);
  return true;
}

bool KamalaginAMatMultStrassenMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_mat_mult_strassen
