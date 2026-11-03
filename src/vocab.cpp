#include "vocab.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace matmulfree {
namespace {

struct Probability {
  float probability;
  int index;
};

static int CompareProbability(const void* left, const void* right) {
  const auto* a = static_cast<const Probability*>(left);
  const auto* b = static_cast<const Probability*>(right);
  return a->probability > b->probability   ? -1
         : a->probability < b->probability ? 1
                                           : 0;
}

static void Softmax(Tensor1dLogits& values) {
  float maximum = values[0];
  for (int index = 1; index < kVocabSize; ++index) {
    if (values[index] > maximum)
      maximum = values[index];
  }
  float sum = 0.0f;
  for (float& value : values) {
    value = ::expf(value - maximum);
    sum += value;
  }
  for (float& value : values)
    value /= sum;
}

}  // namespace

void LoadVocab(Vocab& vocab, const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("cannot read tokenizer: " + path);
  uint32_t max_length = 0;
  file.read(reinterpret_cast<char*>(&max_length), sizeof(max_length));
  if (!file || max_length == 0 || max_length > 1048576) {
    throw std::runtime_error("invalid tokenizer header");
  }
  vocab.dict.resize(kVocabSize);
  vocab.scores.resize(kVocabSize);
  vocab.token_ids.clear();
  for (int index = 0; index < kVocabSize; ++index) {
    int32_t length = 0;
    file.read(reinterpret_cast<char*>(&vocab.scores[index]), sizeof(float));
    file.read(reinterpret_cast<char*>(&length), sizeof(length));
    if (!file || length < 0 || static_cast<uint32_t>(length) > max_length) {
      throw std::runtime_error("invalid tokenizer entry");
    }
    vocab.dict[index].resize(length);
    file.read(vocab.dict[index].data(), length);
    if (!file)
      throw std::runtime_error("truncated tokenizer");
    vocab.token_ids.emplace(vocab.dict[index], index);
  }
}

std::vector<int> Encode(const Vocab& vocab, const std::string& text,
                        bool add_bos, bool add_eos) {
  std::vector<int> tokens;
  if (add_bos)
    tokens.push_back(1);
  if (!text.empty())
    tokens.push_back(vocab.token_ids.at(" "));
  std::string piece;
  for (size_t index = 0; index < text.size(); ++index) {
    const auto byte = static_cast<unsigned char>(text[index]);
    if ((byte & 0xc0) != 0x80)
      piece.clear();
    piece.push_back(text[index]);
    const unsigned char next = index + 1 < text.size() ? text[index + 1] : 0;
    if ((next & 0xc0) == 0x80 && piece.size() < 4)
      continue;
    const auto found = vocab.token_ids.find(piece);
    if (found != vocab.token_ids.end()) {
      tokens.push_back(found->second);
    } else {
      for (unsigned char value : piece)
        tokens.push_back(value + 3);
    }
    piece.clear();
  }
  while (true) {
    float best_score = -1e10f;
    int best_token = -1, best_index = -1;
    for (int index = 0; index + 1 < static_cast<int>(tokens.size()); ++index) {
      const auto found = vocab.token_ids.find(vocab.dict[tokens[index]] +
                                              vocab.dict[tokens[index + 1]]);
      if (found != vocab.token_ids.end() &&
          vocab.scores[found->second] > best_score) {
        best_score = vocab.scores[found->second];
        best_token = found->second;
        best_index = index;
      }
    }
    if (best_index < 0)
      break;
    tokens[best_index] = best_token;
    tokens.erase(tokens.begin() + best_index + 1);
  }
  if (add_eos)
    tokens.push_back(2);
  return tokens;
}

std::string DecodePiece(const Vocab& vocab, int previous, int token) {
  std::string piece = vocab.dict.at(token);
  if (previous == 1 && !piece.empty() && piece[0] == ' ')
    piece.erase(0, 1);
  unsigned char byte = 0;
  if (std::sscanf(piece.c_str(), "<0x%02hhX>", &byte) == 1) {
    return byte ? std::string(1, static_cast<char>(byte)) : std::string();
  }
  return piece;
}

void PrintPiece(const std::string& piece) {
  if (piece.size() == 1) {
    const auto byte = static_cast<unsigned char>(piece[0]);
    if (!std::isprint(byte) && !std::isspace(byte))
      return;
  }
  std::fwrite(piece.data(), 1, piece.size(), stdout);
}

Sampler::Sampler(float temperature, float top_p, uint64_t seed)
    : temperature_(temperature), top_p_(top_p), state_(seed) {}

int Sampler::Sample(Tensor1dLogits logits) {
  if (temperature_ == 0.0f) {
    int next = 0;
    for (int index = 1; index < kVocabSize; ++index) {
      if (logits[index] > logits[next])
        next = index;
    }
    return next;
  }
  for (float& value : logits)
    value /= temperature_;
  Softmax(logits);
  state_ ^= state_ >> 12;
  state_ ^= state_ << 25;
  state_ ^= state_ >> 27;
  const uint32_t random = (state_ * 0x2545f4914f6cdd1dULL) >> 32;
  const float coin = (random >> 8) / 16777216.0f;
  if (top_p_ <= 0.0f || top_p_ >= 1.0f) {
    float sum = 0.0f;
    for (int index = 0; index < kVocabSize; ++index) {
      sum += logits[index];
      if (coin < sum)
        return index;
    }
    return kVocabSize - 1;
  }
  std::array<Probability, kVocabSize> sorted;
  int count = 0;
  const float cutoff = (1.0f - top_p_) / (kVocabSize - 1);
  for (int index = 0; index < kVocabSize; ++index) {
    if (logits[index] >= cutoff)
      sorted[count++] = {logits[index], index};
  }
  std::qsort(sorted.data(), count, sizeof(Probability), CompareProbability);
  float sum = 0.0f;
  int last = count - 1;
  for (int index = 0; index < count; ++index) {
    sum += sorted[index].probability;
    if (sum > top_p_) {
      last = index;
      break;
    }
  }
  const float target = coin * sum;
  float cumulative = 0.0f;
  for (int index = 0; index <= last; ++index) {
    cumulative += sorted[index].probability;
    if (target < cumulative)
      return sorted[index].index;
  }
  return sorted[last].index;
}

}  // namespace matmulfree
