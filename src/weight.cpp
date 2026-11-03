#include "weight.hpp"

#include <fstream>
#include <stdexcept>

namespace matmulfree {
namespace {

void PackParameters(Weights& weights) {
  weights.packed.clear();
  weights.packed.reserve(kPackedWeightWords);
  for (const Matrix& matrix : weights.matrices) {
    for (int tile = 0; tile < (matrix.rows + 127) / 128; ++tile) {
      for (int column = 0; column < matrix.columns; ++column) {
        for (int bank = 0; bank < 4; ++bank) {
          uint64_t word = 0;
          for (int lane = 0; lane < 32; ++lane) {
            const int row = tile * 128 + bank * 32 + lane;
            const int value =
                row < matrix.rows
                    ? weights.ternary[matrix.weight_offset +
                                      row * matrix.columns + column]
                    : 0;
            const uint64_t code = value == 1 ? 1 : (value == -1 ? 2 : 0);
            word |= code << (2 * lane);
          }
          weights.packed.push_back(word);
        }
      }
    }
  }
  if (weights.packed.size() != kPackedWeightWords) {
    throw std::runtime_error("packed parameter length mismatch");
  }
}

}  // namespace

void LoadWeights(Weights& weights, const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() != 1207260) {
    throw std::runtime_error("expected published stories1M checkpoint: " +
                             path);
  }
  file.seekg(0);
  int32_t header[8];
  file.read(reinterpret_cast<char*>(header), sizeof(header));
  const int32_t expected[] = {
      0x4c464d4d, 1,          kDim,    kFFNDim,
      kNumLayers, kVocabSize, kSeqLen, kFloatParameters};
  for (int index = 0; index < 8; ++index) {
    if (header[index] != expected[index]) {
      throw std::runtime_error("unsupported checkpoint header");
    }
  }
  weights.parameters.resize(kFloatParameters);
  weights.ternary.resize(kTernaryParameters);
  file.read(reinterpret_cast<char*>(weights.parameters.data()),
            kFloatParameters * sizeof(float));
  file.read(reinterpret_cast<char*>(weights.ternary.data()),
            kTernaryParameters);
  if (!file)
    throw std::runtime_error("truncated checkpoint");
  for (int8_t value : weights.ternary) {
    if (value < -1 || value > 1) {
      throw std::runtime_error("checkpoint contains a non-ternary weight");
    }
  }
  int parameter_offset = kProjectionParametersOffset;
  int weight_offset = 0;
  int packed_offset = 0;
  for (int index = 0; index < kNumProjections; ++index) {
    const MatrixShape shape = ProjectionShape(index);
    weights.matrices[index] = {parameter_offset, weight_offset, packed_offset,
                               shape.columns, shape.rows};
    parameter_offset += shape.columns + 1;
    weight_offset += shape.columns * shape.rows;
    packed_offset += ((shape.rows + 127) / 128) * shape.columns;
  }
  if (parameter_offset != kFloatParameters ||
      weight_offset != kTernaryParameters ||
      packed_offset * 4 != kPackedWeightWords) {
    throw std::runtime_error("model layout mismatch");
  }
  PackParameters(weights);
}

}  // namespace matmulfree
