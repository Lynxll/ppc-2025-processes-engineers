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

std::vector<double> StrassenIter(const std::vector<double> &a, const std::vector<double> &b, int n) {
  constexpr int kThreshold = 64;
  struct Frame {
    int n{};
    int stage{};
    std::vector<double> a;
    std::vector<double> b;

    std::vector<double> a11, a12, a21, a22;
    std::vector<double> b11, b12, b21, b22;

    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;

    Frame(int n_, int stage_, std::vector<double> a_, std::vector<double> b_)
        : n(n_),
          stage(stage_),
          a(std::move(a_)),
          b(std::move(b_)),
          a11(),
          a12(),
          a21(),
          a22(),
          b11(),
          b12(),
          b21(),
          b22(),
          x(8),
          y(8) {}
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

    {
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

  const auto c_pad = StrassenIter(a_pad, b_pad, p);
  GetOutput() = (p == n) ? c_pad : Unpad(c_pad, n, p);

  return true;
}

bool KamalaginAMatMultStrassenSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_mat_mult_strassen
