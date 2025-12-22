#include "kamalagin_a_vec_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "kamalagin_a_vec_mult/common/include/common.hpp"

namespace kamalagin_a_vec_mult {

KamalaginAVecMultMPI::KamalaginAVecMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KamalaginAVecMultMPI::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return a.size() == b.size();
}

bool KamalaginAVecMultMPI::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

namespace {
void BuildCountsDispls(int n, int size, std::vector<int> *counts, std::vector<int> *displs) {
  counts->assign(size, 0);
  displs->assign(size, 0);

  const int base = n / size;
  const int rem = n % size;

  int offset = 0;
  for (int proc = 0; proc < size; ++proc) {
    const int cnt = base + (proc < rem ? 1 : 0);
    (*counts)[proc] = cnt;
    (*displs)[proc] = offset;
    offset += cnt;
  }
}
}  // namespace

bool KamalaginAVecMultMPI::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> a_root;
  std::vector<int> b_root;
  int n = 0;

  if (rank == 0) {
    const auto &[a, b] = GetInput();
    a_root = a;
    b_root = b;
    n = static_cast<int>(a_root.size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  BuildCountsDispls(n, size, &counts, &displs);

  const int local_n = counts[rank];
  std::vector<int> a_local(local_n);
  std::vector<int> b_local(local_n);

  MPI_Scatterv(rank == 0 ? a_root.data() : nullptr, counts.data(), displs.data(), MPI_INT, a_local.data(), local_n,
               MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? b_root.data() : nullptr, counts.data(), displs.data(), MPI_INT, b_local.data(), local_n,
               MPI_INT, 0, MPI_COMM_WORLD);

  std::int64_t local_sum = 0;
  for (int i = 0; i < local_n; ++i) {
    local_sum += static_cast<std::int64_t>(a_local[i]) * static_cast<std::int64_t>(b_local[i]);
  }

  std::int64_t global_sum = 0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = global_sum;

  return true;
}

bool KamalaginAVecMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace kamalagin_a_vec_mult
