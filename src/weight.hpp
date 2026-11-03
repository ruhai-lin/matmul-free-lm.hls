#ifndef MATMULFREE_WEIGHT_HPP_
#define MATMULFREE_WEIGHT_HPP_

#include <string>
#include <vector>

#include "tensor.hpp"

namespace matmulfree {

struct Matrix {
  int parameter_offset;
  int weight_offset;
  int packed_offset;
  int columns;
  int rows;
};

struct Weights {
  std::vector<float> parameters;
  std::vector<int8_t> ternary;
  std::vector<uint64_t> packed;
  std::array<Matrix, kNumProjections> matrices{};
};

void LoadWeights(Weights& weights, const std::string& path);

}  // namespace matmulfree

#endif  // MATMULFREE_WEIGHT_HPP_
