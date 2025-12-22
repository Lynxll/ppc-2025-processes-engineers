#pragma once

#include "kamalagin_a_vec_mat_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamalagin_a_vec_mat_mult {

class KamalaginAVecMatMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KamalaginAVecMatMultMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kamalagin_a_vec_mat_mult
