#include "kamalagin_a_vec_mat_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

namespace kamalagin_a_vec_mat_mult {

static void BuildCountsDispls(int n, int size,
                              std::vector<int>* counts,
                              std::vector<int>* displs) {
  counts->assign(size, 0);
  displs->assign(size, 0);

  const int base = n / size;
  const int rem  = n % size;

  int offset = 0;
  for (int r = 0; r < size; ++r) {
    const int cnt = base + (r < rem ? 1 : 0);
    (*counts)[r] = cnt;
    (*displs)[r] = offset;
    offset += cnt;
  }
}

KamalaginAVecMatMultMPI::KamalaginAVecMatMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KamalaginAVecMatMultMPI::ValidationImpl() {
  const auto& [n, m, a_flat, x] = GetInput();
  if (n < 0 || m < 0) return false;
  if (static_cast<std::size_t>(n) * static_cast<std::size_t>(m) != a_flat.size()) return false;
  if (static_cast<std::size_t>(m) != x.size()) return false;
  return true;
}

bool KamalaginAVecMatMultMPI::PreProcessingImpl() {
  const auto& [n, m, a_flat, x] = GetInput();
  (void)m; (void)a_flat; (void)x;
  GetOutput().assign(static_cast<std::size_t>(n), 0);
  return true;
}

bool KamalaginAVecMatMultMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0, m = 0;
  std::vector<int> a_root;
  std::vector<int> x;

  if (rank == 0) {
    const auto& input = GetInput();
    n = std::get<0>(input);
    m = std::get<1>(input);
    a_root = std::get<2>(input);
    x = std::get<3>(input);
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (m > 0) {
    if (rank != 0) x.resize(m);
    MPI_Bcast(x.data(), m, MPI_INT, 0, MPI_COMM_WORLD);
  }

  std::vector<int> rows_counts, rows_displs;
  BuildCountsDispls(n, size, &rows_counts, &rows_displs);

  const int local_rows = rows_counts[rank];
  std::vector<int> a_local(static_cast<std::size_t>(local_rows * m));

  std::vector<int> sendcountsA(size), displsA(size);
  for (int r = 0; r < size; ++r) {
    sendcountsA[r] = rows_counts[r] * m;
    displsA[r]     = rows_displs[r] * m;
  }

  MPI_Scatterv(rank == 0 ? a_root.data() : nullptr,
               sendcountsA.data(), displsA.data(), MPI_INT,
               a_local.data(), local_rows * m, MPI_INT,
               0, MPI_COMM_WORLD);

  std::vector<int> local_y(static_cast<std::size_t>(local_rows), 0);
  for (int i = 0; i < local_rows; ++i) {
    long long sum = 0;
    for (int j = 0; j < m; ++j) {
      sum += static_cast<long long>(a_local[i * m + j]) *
             static_cast<long long>(x[j]);
    }
    local_y[i] = static_cast<int>(sum);
  }

  std::vector<int> recvcountsY = rows_counts;
  std::vector<int> displsY     = rows_displs;

  MPI_Gatherv(local_y.data(), local_rows, MPI_INT,
            rank == 0 ? GetOutput().data() : nullptr,
            recvcountsY.data(), displsY.data(), MPI_INT,
            0, MPI_COMM_WORLD);

  if (n > 0) {
    MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool KamalaginAVecMatMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_vec_mat_mult
