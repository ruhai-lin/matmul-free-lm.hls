#ifndef MATMULFREE_DECODE_HPP_
#define MATMULFREE_DECODE_HPP_

#include <memory>
#include <string>

#include "context.hpp"
#include "weight.hpp"

namespace matmulfree {

// One Device and one Context form a single recurrent sequence. Model buffers
// persist for the Device lifetime; ResetContext resets on the next token call.
class Device {
 public:
  Device(const Weights& weights, const std::string& xclbin);
  ~Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  int DecodeToken(int token, Context& context, bool logits, bool state);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// With device == nullptr, execute the ordered FP32/A8 CPU reference.
// With a Device, launch the complete decode kernel exactly once per token.
int Decode(int token, const Weights& weights, Context& context,
           Device* device = nullptr, bool logits = false, bool state = false);

}  // namespace matmulfree

#endif  // MATMULFREE_DECODE_HPP_
