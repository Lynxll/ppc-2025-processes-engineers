#pragma once

#include "kamalagin_a_mat_mult_strassen/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamalagin_a_mat_mult_strassen {

class KamalaginAMatMultStrassenSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KamalaginAMatMultStrassenSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kamalagin_a_mat_mult_strassen
