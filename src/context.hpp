#ifndef MATMULFREE_CONTEXT_HPP_
#define MATMULFREE_CONTEXT_HPP_

#include "tensor.hpp"

namespace matmulfree {

struct Context {
  Tensor1d input{};
  Tensor1d branch{};
  Tensor1d mixer_input{};
  Tensor1d forget_gate{};
  Tensor1d output_gate{};
  Tensor1dFFN ffn{};
  std::array<float, kFFNDim> normalized{};
  std::array<int8_t, kFFNDim> quantized{};
  Tensor1dLogits logits{};
  TensorState state{};
  bool reset_pending = true;
  uint64_t forward_calls = 0;
  uint64_t kernel_calls = 0;
  uint64_t kernel_profile_ns = 0;
};

void ResetContext(Context& context);

}  // namespace matmulfree

#endif  // MATMULFREE_CONTEXT_HPP_
