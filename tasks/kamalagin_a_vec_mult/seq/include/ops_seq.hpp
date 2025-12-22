#pragma once

#include "kamalagin_a_vec_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamalagin_a_vec_mult {

class KamalaginAVecMultSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KamalaginAVecMultSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kamalagin_a_vec_mult
