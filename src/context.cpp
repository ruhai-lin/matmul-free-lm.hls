#include "context.hpp"

namespace matmulfree {

void ResetContext(Context& context) {
  context.state.fill(0.0f);
  context.reset_pending = true;
}

}  // namespace matmulfree
