#include "kamalagin_a_mat_mult_strassen/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamalagin_a_mat_mult_strassen {

class KamalaginAMatMultStrassenMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KamalaginAMatMultStrassenMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kamalagin_a_mat_mult_strassen
