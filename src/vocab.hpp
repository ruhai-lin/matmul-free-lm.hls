#ifndef MATMULFREE_VOCAB_HPP_
#define MATMULFREE_VOCAB_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.hpp"

namespace matmulfree {

struct Vocab {
  std::vector<std::string> dict;
  std::vector<float> scores;
  std::unordered_map<std::string, int> token_ids;
};

void LoadVocab(Vocab& vocab, const std::string& path);
std::vector<int> Encode(const Vocab& vocab, const std::string& text,
                        bool add_bos = true, bool add_eos = false);
std::string DecodePiece(const Vocab& vocab, int previous, int token);
void PrintPiece(const std::string& piece);

class Sampler {
 public:
  Sampler(float temperature, float top_p, uint64_t seed);
  int Sample(Tensor1dLogits logits);

 private:
  float temperature_;
  float top_p_;
  uint64_t state_;
};

}  // namespace matmulfree

#endif  // MATMULFREE_VOCAB_HPP_
