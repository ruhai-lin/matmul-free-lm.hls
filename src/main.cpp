#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "context.hpp"
#include "decode.hpp"
#include "vocab.hpp"
#include "weight.hpp"

namespace matmulfree {
namespace {

struct Options {
  std::string weights = "model/stories1M.bin";
  std::string tokenizer = "model/tok512.bin";
  std::string xclbin = "decode.xclbin";
  std::string prompt;
  std::string system_prompt;
  std::string mode = "generate";
#ifdef USE_HLS_SIM
  std::string backend = "sim";
#elif defined(USE_CPU_ONLY)
  std::string backend = "cpu";
#else
  std::string backend = "fpga";
#endif
  bool have_prompt = false;
  bool have_system_prompt = false;
  int steps = 256;
  float temperature = 1.0f;
  float top_p = 0.9f;
  uint64_t seed = 0;
};

static void Usage() {
  std::cout << "Usage: matmulfree [checkpoint] [options]\n"
            << "  --weights PATH             stories1M checkpoint\n"
            << "  --tokenizer, -z PATH       tok512.bin\n"
            << "  --prompt, -i TEXT          input prompt\n"
            << "  --max_seq, -n N            forward limit (default 256)\n"
            << "  --temp, -t T               temperature (default 1)\n"
            << "  --top_p, -p P              nucleus threshold (default .9)\n"
            << "  --seed, -s N               RNG seed\n"
            << "  --mode, -m generate|chat   host mode\n"
            << "  --system, -y TEXT          chat system prompt\n"
            << "  --backend cpu|fpga|sim     explicitly choose execution\n"
            << "  --xclbin PATH              matching decode.xclbin\n";
}

static Options ParseOptions(int argc, char** argv) {
  Options options;
  int index = 1;
  if (index < argc && argv[index][0] != '-')
    options.weights = argv[index++];
  while (index < argc) {
    const std::string flag = argv[index++];
    if (flag == "--help" || flag == "-h") {
      Usage();
      std::exit(0);
    }
    if (index == argc)
      throw std::runtime_error("missing value for " + flag);
    const std::string value = argv[index++];
    if (flag == "--weights")
      options.weights = value;
    else if (flag == "--tokenizer" || flag == "-z")
      options.tokenizer = value;
    else if (flag == "--prompt" || flag == "-i") {
      options.prompt = value;
      options.have_prompt = true;
    } else if (flag == "--max_seq" || flag == "-n")
      options.steps = std::stoi(value);
    else if (flag == "--temp" || flag == "-t")
      options.temperature = std::stof(value);
    else if (flag == "--top_p" || flag == "-p")
      options.top_p = std::stof(value);
    else if (flag == "--seed" || flag == "-s")
      options.seed = std::stoull(value);
    else if (flag == "--mode" || flag == "-m")
      options.mode = value;
    else if (flag == "--system" || flag == "-y") {
      options.system_prompt = value;
      options.have_system_prompt = true;
    } else if (flag == "--backend")
      options.backend = value;
    else if (flag == "--xclbin")
      options.xclbin = value;
    else
      throw std::runtime_error("unknown option: " + flag);
  }
  if (options.mode != "generate" && options.mode != "chat")
    throw std::runtime_error("unsupported host mode");
  if (options.backend != "cpu" && options.backend != "fpga" &&
      options.backend != "sim")
    throw std::runtime_error("unsupported backend");
#ifdef USE_HLS_SIM
  if (options.backend == "fpga")
    throw std::runtime_error("simulation build has no XRT backend");
#elif defined(USE_CPU_ONLY)
  if (options.backend != "cpu")
    throw std::runtime_error("CPU build has no FPGA backend");
#else
  if (options.backend == "sim")
    throw std::runtime_error("XRT build has no simulation backend");
#endif
  if (!options.seed)
    options.seed = static_cast<uint64_t>(std::time(nullptr));
  options.temperature = std::max(0.0f, options.temperature);
  if (options.top_p < 0 || options.top_p > 1)
    options.top_p = 0.9f;
  if (options.steps <= 0 || options.steps > kSeqLen)
    options.steps = kSeqLen;
  return options;
}

static int NextToken(int token, const Options& options, const Weights& weights,
                     Context& context, Device* device, Sampler& sampler) {
  const int next =
      Decode(token, weights, context, device, options.temperature != 0);
  return options.temperature == 0 ? next : sampler.Sample(context.logits);
}

static void Generate(const Options& options, const Weights& weights,
                     const Vocab& vocab, Context& context, Device* device,
                     Sampler& sampler) {
  const std::vector<int> prompt = Encode(vocab, options.prompt);
  int token = prompt[0];
  for (int position = 0; position < options.steps; ++position) {
    const int prediction =
        Decode(token, weights, context, device, options.temperature != 0);
    const int next =
        position + 1 < static_cast<int>(prompt.size())
            ? prompt[position + 1]
            : (options.temperature == 0 ? prediction
                                        : sampler.Sample(context.logits));
    if (next == 1)
      break;
    PrintPiece(DecodePiece(vocab, token, next));
    std::fflush(stdout);
    token = next;
  }
  std::putchar('\n');
}

static void Chat(const Options& options, const Weights& weights,
                 const Vocab& vocab, Context& context, Device* device,
                 Sampler& sampler) {
  std::string system = options.system_prompt;
  if (!options.have_system_prompt) {
    std::cout << "Enter system prompt (optional): ";
    std::getline(std::cin, system);
  }
  std::vector<int> prompt;
  size_t user_index = 0;
  bool user_turn = true;
  int next = 1;
  for (int position = 0; position < options.steps; ++position) {
    if (user_turn) {
      std::string user = options.prompt;
      if (position != 0 || !options.have_prompt) {
        std::cout << "User: ";
        if (!std::getline(std::cin, user))
          break;
      }
      const std::string rendered = position == 0 && !system.empty()
                                       ? "[INST] <<SYS>>\n" + system +
                                             "\n<</SYS>>\n\n" + user +
                                             " [/INST]"
                                       : "[INST] " + user + " [/INST]";
      prompt = Encode(vocab, rendered);
      user_index = 0;
      user_turn = false;
      std::cout << "Assistant: ";
    }
    const int token = user_index < prompt.size() ? prompt[user_index++] : next;
    if (token == 2)
      user_turn = true;
    next = NextToken(token, options, weights, context, device, sampler);
    if (user_index >= prompt.size() && next != 2) {
      PrintPiece(DecodePiece(vocab, token, next));
      std::fflush(stdout);
    }
    if (next == 2)
      std::putchar('\n');
  }
  std::putchar('\n');
}

}  // namespace
}  // namespace matmulfree

int main(int argc, char** argv) {
  try {
    const auto options = matmulfree::ParseOptions(argc, argv);
    matmulfree::Weights weights;
    matmulfree::LoadWeights(weights, options.weights);
    matmulfree::Vocab vocab;
    matmulfree::LoadVocab(vocab, options.tokenizer);
    matmulfree::Context context;
    std::unique_ptr<matmulfree::Device> device;
    if (options.backend != "cpu") {
      device = std::make_unique<matmulfree::Device>(weights, options.xclbin);
    }
    matmulfree::Sampler sampler(options.temperature, options.top_p,
                                options.seed);
    if (options.mode == "chat") {
      matmulfree::Chat(options, weights, vocab, context, device.get(), sampler);
    } else {
      matmulfree::Generate(options, weights, vocab, context, device.get(),
                           sampler);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "matmulfree: " << error.what() << '\n';
    return 1;
  }
}
