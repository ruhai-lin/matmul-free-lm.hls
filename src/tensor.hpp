#ifndef MATMULFREE_TENSOR_HPP_
#define MATMULFREE_TENSOR_HPP_

#include <array>
#include <cstdint>

namespace matmulfree {

constexpr int kDim = 128;
constexpr int kFFNDim = 272;
constexpr int kNumLayers = 5;
constexpr int kVocabSize = 512;
constexpr int kSeqLen = 512;
constexpr int kNumProjections = 31;
constexpr int kFloatParameters = 72943;
constexpr int kTernaryParameters = 915456;
constexpr int kPackedWeightWords = 30528;
constexpr int kWeightBankWords = 10000;
constexpr int kOutputWords = 1 + kVocabSize + kNumLayers * kDim;
constexpr float kNormEpsilon = 1e-6f;
constexpr int kLowerBoundOffset = kVocabSize * kDim;
constexpr int kMixerNormOffset = kLowerBoundOffset + kNumLayers * kDim;
constexpr int kGateNormOffset = kMixerNormOffset + kNumLayers * kDim;
constexpr int kFFNNormOffset = kGateNormOffset + kNumLayers * kDim;
constexpr int kFinalNormOffset = kFFNNormOffset + kNumLayers * kDim;
constexpr int kProjectionParametersOffset = kFinalNormOffset + kDim;

using Tensor1d = std::array<float, kDim>;
using Tensor1dFFN = std::array<float, 2 * kFFNDim>;
using Tensor1dLogits = std::array<float, kVocabSize>;
using TensorState = std::array<float, kNumLayers * kDim>;

struct MatrixShape {
  int columns;
  int rows;
};

constexpr MatrixShape ProjectionShape(int projection) {
  return {projection < 30 && projection % 6 == 5 ? kFFNDim : kDim,
          projection == 30 ? kVocabSize
                           : (projection % 6 == 4 ? 2 * kFFNDim : kDim)};
}

}  // namespace matmulfree

// Keep the HLS top in global scope, matching llama2.hls and Vitis cosim stubs.
// The C ABI is unchanged from the measured 2026-10-23 bitstream. The first
// result word contains a float-encoded token ID or a negative error status.
extern "C" void decode(const uint64_t* packed, const float* parameters,
                       float* result, unsigned opcode, unsigned token,
                       unsigned reset, unsigned output_mode, unsigned words);

#endif  // MATMULFREE_TENSOR_HPP_
