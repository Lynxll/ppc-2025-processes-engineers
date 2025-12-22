#include "kamalagin_a_vec_mat_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kamalagin_a_vec_mat_mult/common/include/common.hpp"

namespace kamalagin_a_vec_mat_mult {

namespace {

void BuildCountsDispls(int n, int size, std::vector<int> *counts, std::vector<int> *displs) {
  counts->assign(size, 0);
  displs->assign(size, 0);

  const int base = n / size;
  const int rem = n % size;

  int offset = 0;
  for (int proc = 0; proc < size; ++proc) {
    const int cnt = base + ((proc < rem) ? 1 : 0);
    (*counts)[proc] = cnt;
    (*displs)[proc] = offset;
    offset += cnt;
  }
}

}  // namespace

KamalaginAVecMatMultMPI::KamalaginAVecMatMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KamalaginAVecMatMultMPI::ValidationImpl() {
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

bool KamalaginAVecMatMultMPI::PreProcessingImpl() {
  const auto &[n, m, a_flat, x] = GetInput();
  (void)m;
  (void)a_flat;
  (void)x;
  GetOutput().assign(static_cast<std::size_t>(n), 0);
  return true;
}

bool KamalaginAVecMatMultMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  int m = 0;
  std::vector<int> a_root;
  std::vector<int> x;

  if (rank == 0) {
    const auto &input = GetInput();
    n = std::get<0>(input);
    m = std::get<1>(input);
    a_root = std::get<2>(input);
    x = std::get<3>(input);
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (m > 0) {
    if (rank != 0) {
      x.resize(static_cast<std::size_t>(m));
    }
    MPI_Bcast(x.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
  }

  std::vector<int> rows_counts;
  std::vector<int> rows_displs;
  BuildCountsDispls(n, size, &rows_counts, &rows_displs);

  const int local_rows = rows_counts[rank];
  std::vector<int> a_local(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(m));

  std::vector<int> sendcounts_a(static_cast<std::size_t>(size));
  std::vector<int> displs_a(static_cast<std::size_t>(size));
  for (int proc = 0; proc < size; ++proc) {
    sendcounts_a[proc] = rows_counts[proc] * m;
    displs_a[proc] = rows_displs[proc] * m;
  }

  MPI_Scatterv(rank == 0 ? a_root.data() : nullptr, sendcounts_a.data(), displs_a.data(), MPI_INT, a_local.data(),
               local_rows * m, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> local_y(static_cast<std::size_t>(local_rows), 0);
  for (int i = 0; i < local_rows; ++i) {
    std::int64_t sum = 0;
    for (int j = 0; j < m; ++j) {
      sum +=
          static_cast<std::int64_t>(a_local[(i * m) + j]) * static_cast<std::int64_t>(x[static_cast<std::size_t>(j)]);
    }
    local_y[static_cast<std::size_t>(i)] = static_cast<int>(sum);
  }

  const std::vector<int> recvcounts_y = rows_counts;
  const std::vector<int> displs_y = rows_displs;

  MPI_Gatherv(local_y.data(), local_rows, MPI_INT, rank == 0 ? GetOutput().data() : nullptr, recvcounts_y.data(),
              displs_y.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (n > 0) {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool KamalaginAVecMatMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_vec_mat_mult
