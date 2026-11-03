#ifdef BUILD_DECODE_KERNEL

#include <hls_stream.h>

#include "tensor.hpp"
#ifdef USE_NATIVE_MATH
#include <cmath>
namespace hls {
using ::expf;
using ::fabsf;
using ::roundf;
using ::sqrtf;
}  // namespace hls
#else
#include <hls_math.h>
#endif

// HGRN-Bit stories1M. All tensors and recurrent state remain on chip between
// invocations. The scheduler reuses one normalization/quantize/MVM pipeline.
namespace matmulfree {
namespace {

// Arm expf table/polynomial adapted for HLS. FP64 intermediates preserve the
// validated libc behavior across A8 rounding boundaries.
// Copyright (c) 1999-2022, Arm Limited.
// Copyright (c) 2017-2025, Arm Limited.
// Used under the MIT license; see LICENSE and docs/arch.md for provenance.
inline float arm_exp(float x) {
  static const uint64_t table[32] = {
      0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL,
      0x3fef9301d0125b51ULL, 0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL,
      0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL, 0x3fef06fe0a31b715ULL,
      0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
      0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL,
      0x3feea47eb03a5585ULL, 0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL,
      0x3feea11473eb0187ULL, 0x3feea589994cce13ULL, 0x3feeace5422aa0dbULL,
      0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
      0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL,
      0x3fef3720dcef9069ULL, 0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL,
      0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL};
  if (x != x)
    return x + x;
  if (x > 0x1.62e42ep6f) {
    union {
      uint32_t u;
      float f;
    } inf;
    inf.u = 0x7f800000;
    return inf.f;
  }
  if (x < -0x1.9fe368p6f)
    return 0;
  double z = (0x1.71547652b82fep0 * 32.0) * (double)x;
  int k = (int)z;
  double fraction = z - (double)k;
  if (fraction >= 0.5)
    ++k;
  else if (fraction <= -0.5)
    --k;
  double r = z - (double)k;
  union {
    uint64_t u;
    double d;
  } scale;
  scale.u = table[(unsigned)k & 31] + ((uint64_t)(int64_t)k << 47);
#ifdef USE_NATIVE_MATH
  double a =
      ::fma(0x1.c6af84b912394p-5 / 32768.0, r, 0x1.ebfce50fac4f3p-3 / 1024.0);
  double b = ::fma(0x1.62e42ff0c52d6p-1 / 32.0, r, 1.0);
  double y = ::fma(a, r * r, b);
#else
  double a = hls::fma(0x1.c6af84b912394p-5 / 32768.0, r,
                      0x1.ebfce50fac4f3p-3 / 1024.0);
  double b = hls::fma(0x1.62e42ff0c52d6p-1 / 32.0, r, 1.0);
  double y = hls::fma(a, r * r, b);
#endif
  return (float)(y * scale.d);
}

struct Projection {
  unsigned fp, offset, n, d;
};
static float sigmoid(float x) {
#pragma HLS INLINE
  return 1.0f / (1.0f + arm_exp(-x));
}
static float silu(float x) {
#pragma HLS INLINE
  return x * sigmoid(x);
}
static void rmsnorm(float* out, const float* x, const float* fp,
                    unsigned weight, unsigned n) {
#pragma HLS INLINE off
  float ss = 0;
  for (unsigned j = 0; j < n; ++j) {
#pragma HLS PIPELINE
    ss = ss + x[j] * x[j];
  }
  ss = ss / (float)n;
  ss = ss + 1e-6f;
  ss = 1.0f / hls::sqrtf(ss);
  for (unsigned j = 0; j < n; ++j) {
#pragma HLS PIPELINE II = 1
    out[j] = fp[weight + j] * (ss * x[j]);
  }
}
static void quantize(const float* x, const float* fp, Projection p,
                     hls::stream<int8_t>& q, hls::stream<float>& scale) {
#pragma HLS INLINE off
  float xn[272];
  rmsnorm(xn, x, fp, p.fp, p.n);
  float amax = 1e-5f;
  for (unsigned j = 0; j < p.n; ++j) {
#pragma HLS PIPELINE II = 1
    float a = hls::fabsf(xn[j]);
    if (a > amax)
      amax = a;
  }
  float xs = 127.0f / amax;
  scale.write(fp[p.fp + p.n] / xs);
  for (unsigned j = 0; j < p.n; ++j) {
#pragma HLS PIPELINE II = 1
    float v = hls::roundf(xn[j] * xs);
    q.write((int8_t)(v > 127.f ? 127.f : (v < -128.f ? -128.f : v)));
  }
}
static void ternary_gemv_engine(const uint64_t cache[4][kWeightBankWords],
                                Projection p, hls::stream<int8_t>& in,
                                hls::stream<int32_t>& out) {
#pragma HLS INLINE off
  int8_t x[272];
  for (unsigned j = 0; j < p.n; ++j) {
#pragma HLS PIPELINE II = 1
    x[j] = in.read();
  }
  for (unsigned tile = 0; tile < (p.d + 127) / 128; ++tile) {
    int32_t sums[128];
#pragma HLS ARRAY_PARTITION variable = sums complete
    for (int lane = 0; lane < 128; ++lane) {
#pragma HLS UNROLL
      sums[lane] = 0;
    }
    for (unsigned j = 0; j < p.n; ++j) {
#pragma HLS PIPELINE II = 1
      int32_t v = x[j];
      for (int bank = 0; bank < 4; ++bank) {
#pragma HLS UNROLL
        uint64_t w = cache[bank][p.offset + tile * p.n + j];
        for (int lane = 0; lane < 32; ++lane) {
#pragma HLS UNROLL
          unsigned code = (w >> (2 * lane)) & 3;
          if (code == 1)
            sums[bank * 32 + lane] += v;
          if (code == 2)
            sums[bank * 32 + lane] -= v;
        }
      }
    }
    for (unsigned lane = 0; lane < 128; ++lane) {
#pragma HLS PIPELINE II = 1
      if (tile * 128 + lane < p.d)
        out.write(sums[lane]);
    }
  }
}
static void dequantize(float* y, unsigned d, hls::stream<int32_t>& sums,
                       hls::stream<float>& scales) {
#pragma HLS INLINE off
  float scale = scales.read();
  for (unsigned j = 0; j < d; ++j) {
#pragma HLS PIPELINE II = 1
    y[j] = (float)sums.read() * scale;
  }
}
static void bitlinear(float* y, const float* x, const float* fp,
                      const uint64_t cache[4][kWeightBankWords], Projection p) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
  hls::stream<int8_t> q("quantized");
  hls::stream<int32_t> sums("sums");
  hls::stream<float> scales("scales");
#pragma HLS STREAM variable = q depth = 32
#pragma HLS STREAM variable = sums depth = 32
#pragma HLS STREAM variable = scales depth = 2
  quantize(x, fp, p, q, scales);
  ternary_gemv_engine(cache, p, q, sums);
  dequantize(y, p.d, sums, scales);
}
}  // namespace
}  // namespace matmulfree

using namespace matmulfree;

extern "C" void decode(const uint64_t* packed, const float* parameters,
                       float* result, unsigned opcode, unsigned token,
                       unsigned reset, unsigned output_mode, unsigned words) {
#pragma HLS INTERFACE m_axi port = packed offset = slave bundle = \
    model depth = 40000 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = parameters offset = slave bundle = \
    model depth = 72943 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = result offset = slave bundle = \
    output_mem depth = 1153 max_write_burst_length = 64
#pragma HLS INTERFACE s_axilite port = packed bundle = control
#pragma HLS INTERFACE s_axilite port = parameters bundle = control
#pragma HLS INTERFACE s_axilite port = result bundle = control
#pragma HLS INTERFACE s_axilite port = opcode bundle = control
#pragma HLS INTERFACE s_axilite port = token bundle = control
#pragma HLS INTERFACE s_axilite port = reset bundle = control
#pragma HLS INTERFACE s_axilite port = output_mode bundle = control
#pragma HLS INTERFACE s_axilite port = words bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS ALLOCATION function instances = bitlinear limit = 1
#pragma HLS ALLOCATION function instances = rmsnorm limit = 1
  static uint64_t cache[4][kWeightBankWords];
#pragma HLS ARRAY_PARTITION variable = cache complete dim = 1
#pragma HLS BIND_STORAGE variable = cache type = ram_1p impl = bram
  static float fp[kFloatParameters];
#pragma HLS BIND_STORAGE variable = fp type = ram_2p impl = uram
  static float state[640];
#pragma HLS BIND_STORAGE variable = state type = ram_2p impl = bram
  static Projection projections[31];
  static bool loaded = false;
  float x[128], xb[128], i[128], f[128], g[128], hb[272];
  float input[272], projected[544];
  if (opcode == 0) {
    loaded = false;
    if (words == 0 || words > 40000 || words % 4) {
      result[0] = -1;
      return;
    }
    for (unsigned j = 0; j < words; ++j) {
#pragma HLS PIPELINE II = 1
      cache[j % 4][j / 4] = packed[j];
    }
    for (unsigned j = 0; j < kFloatParameters; ++j) {
#pragma HLS PIPELINE II = 1
      fp[j] = parameters[j];
    }
    unsigned woff = 0, foff = kProjectionParametersOffset;
    for (unsigned k = 0; k < 31; ++k) {
      unsigned kind = k % 6;
      unsigned n = kind == 5 && k < 30 ? 272 : 128;
      unsigned d = k == 30 ? 512 : (kind == 4 ? 544 : 128);
      projections[k] = {foff, woff, n, d};
      foff += n + 1;
      woff += ((d + 127) / 128) * n;
    }
    if (woff * 4 != words) {
      result[0] = -2;
      return;
    }
    for (int j = 0; j < 640; ++j) {
#pragma HLS PIPELINE II = 1
      state[j] = 0;
    }
    loaded = true;
    result[0] = 0;
    return;
  }
  if (!loaded || opcode != 1 || token >= 512 || output_mode > 2) {
    result[0] = -3;
    return;
  }
  if (reset)
    for (int j = 0; j < 640; ++j) {
#pragma HLS PIPELINE II = 1
      state[j] = 0;
    }
  for (int j = 0; j < 128; ++j) {
#pragma HLS PIPELINE II = 1
    x[j] = fp[token * 128 + j];
  }
  // Exactly one syntactic call site prevents HLS from specializing and
  // duplicating the dataflow engine for different layer roles.
  for (unsigned k = 0; k < 31; ++k) {
    unsigned layer = k / 6, stage = k % 6;
    if (k == 30)
      rmsnorm(x, x, fp, kFinalNormOffset, 128);
    else if (stage == 0)
      rmsnorm(xb, x, fp, kMixerNormOffset + layer * 128, 128);
    unsigned n = projections[k].n;
    for (unsigned j = 0; j < n; ++j) {
#pragma HLS PIPELINE II = 1
      input[j] =
          k == 30 ? x[j] : (stage == 3 ? g[j] : (stage == 5 ? hb[j] : xb[j]));
    }
    bitlinear(projected, input, fp, cache, projections[k]);
    if (k == 30)
      break;
    if (stage < 3)
      for (int j = 0; j < 128; ++j) {
#pragma HLS PIPELINE II = 1
        if (stage == 0)
          i[j] = projected[j];
        else if (stage == 1)
          f[j] = projected[j];
        else
          g[j] = projected[j];
      }
    if (stage == 2) {
      for (int j = 0; j < 128; ++j) {
#pragma HLS PIPELINE
        float forget = sigmoid(f[j]);
        if (layer > 0) {
          float lb = fp[kLowerBoundOffset + layer * 128 + j];
          forget = lb + (1.0f - lb) * forget;
        }
        state[layer * 128 + j] =
            forget * state[layer * 128 + j] + (1.0f - forget) * silu(i[j]);
      }
      rmsnorm(g, g, fp, kGateNormOffset + layer * 128, 128);
      for (int j = 0; j < 128; ++j) {
#pragma HLS PIPELINE
        g[j] = g[j] * silu(state[layer * 128 + j]);
      }
    }
    if (stage == 3 || stage == 5)
      for (int j = 0; j < 128; ++j) {
#pragma HLS PIPELINE II = 1
        x[j] = x[j] + projected[j];
      }
    if (stage == 3)
      rmsnorm(xb, x, fp, kFFNNormOffset + layer * 128, 128);
    if (stage == 4)
      for (int j = 0; j < 272; ++j) {
#pragma HLS PIPELINE
        hb[j] = silu(projected[j]) * projected[j + 272];
      }
  }
  unsigned next = 0;
  float maximum = projected[0];
  for (unsigned j = 1; j < 512; ++j) {
#pragma HLS PIPELINE II = 1
    if (projected[j] > maximum) {
      maximum = projected[j];
      next = j;
    }
  }
  result[0] = (float)next;
  if (output_mode >= 1)
    for (int j = 0; j < 512; ++j) {
#pragma HLS PIPELINE II = 1
      result[1 + j] = projected[j];
    }
  if (output_mode == 2)
    for (int j = 0; j < 640; ++j) {
#pragma HLS PIPELINE II = 1
      result[513 + j] = state[j];
    }
}

#else  // BUILD_DECODE_KERNEL

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "decode.hpp"

#ifndef USE_CPU_ONLY
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#endif

namespace matmulfree {
namespace {

static void rmsnorm(float* out, const float* input, const float* weight,
                    int size) {
  float sum = 0.0f;
  for (int index = 0; index < size; ++index) {
    sum += input[index] * input[index];
  }
  sum /= size;
  sum += kNormEpsilon;
  sum = 1.0f / ::sqrtf(sum);
  for (int index = 0; index < size; ++index) {
    out[index] = weight[index] * (sum * input[index]);
  }
}

static float sigmoid(float value) {
  return 1.0f / (1.0f + ::expf(-value));
}

static float silu(float value) {
  return value * sigmoid(value);
}

static void bitlinear(float* out, const float* input, const Weights& weights,
                      int projection, Context& context) {
  const Matrix& matrix = weights.matrices[projection];
  const float* norm_weight =
      weights.parameters.data() + matrix.parameter_offset;
  rmsnorm(context.normalized.data(), input, norm_weight, matrix.columns);
  float maximum = 1e-5f;
  for (int column = 0; column < matrix.columns; ++column) {
    const float value = ::fabsf(context.normalized[column]);
    if (value > maximum)
      maximum = value;
  }
  const float input_scale = 127.0f / maximum;
  for (int column = 0; column < matrix.columns; ++column) {
    const float value = ::roundf(context.normalized[column] * input_scale);
    context.quantized[column] = static_cast<int8_t>(value > 127.0f    ? 127.0f
                                                    : value < -128.0f ? -128.0f
                                                                      : value);
  }
  const float output_scale = norm_weight[matrix.columns] / input_scale;
  for (int row = 0; row < matrix.rows; ++row) {
    int32_t sum = 0;
    for (int column = 0; column < matrix.columns; ++column) {
      sum +=
          weights
              .ternary[matrix.weight_offset + row * matrix.columns + column] *
          context.quantized[column];
    }
    out[row] = sum * output_scale;
  }
}

static int decode_cpu(int token, const Weights& weights, Context& context) {
  const float* parameters = weights.parameters.data();
  float* input = context.input.data();
  float* branch = context.branch.data();
  float* gate = context.output_gate.data();
  std::copy_n(parameters + token * kDim, kDim, input);
  for (int layer = 0; layer < kNumLayers; ++layer) {
    rmsnorm(branch, input, parameters + kMixerNormOffset + layer * kDim, kDim);
    bitlinear(context.mixer_input.data(), branch, weights, layer * 6, context);
    bitlinear(context.forget_gate.data(), branch, weights, layer * 6 + 1,
              context);
    bitlinear(gate, branch, weights, layer * 6 + 2, context);
    float* state = context.state.data() + layer * kDim;
    const float* lower_bound = parameters + kLowerBoundOffset + layer * kDim;
    for (int index = 0; index < kDim; ++index) {
      float forget = sigmoid(context.forget_gate[index]);
      if (layer > 0) {
        forget = lower_bound[index] + (1.0f - lower_bound[index]) * forget;
      }
      state[index] = forget * state[index] +
                     (1.0f - forget) * silu(context.mixer_input[index]);
    }
    rmsnorm(gate, gate, parameters + kGateNormOffset + layer * kDim, kDim);
    for (int index = 0; index < kDim; ++index)
      gate[index] *= silu(state[index]);
    bitlinear(branch, gate, weights, layer * 6 + 3, context);
    for (int index = 0; index < kDim; ++index)
      input[index] += branch[index];
    rmsnorm(branch, input, parameters + kFFNNormOffset + layer * kDim, kDim);
    bitlinear(context.ffn.data(), branch, weights, layer * 6 + 4, context);
    for (int index = 0; index < kFFNDim; ++index) {
      context.ffn[index] =
          silu(context.ffn[index]) * context.ffn[index + kFFNDim];
    }
    bitlinear(branch, context.ffn.data(), weights, layer * 6 + 5, context);
    for (int index = 0; index < kDim; ++index)
      input[index] += branch[index];
  }
  rmsnorm(input, input, parameters + kFinalNormOffset, kDim);
  bitlinear(context.logits.data(), input, weights, 30, context);
  int next_token = 0;
  for (int index = 1; index < kVocabSize; ++index) {
    if (context.logits[index] > context.logits[next_token])
      next_token = index;
  }
  return next_token;
}

#ifndef USE_CPU_ONLY
static void CheckOpenCL(cl_int status, const char* operation) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + ": OpenCL " +
                             std::to_string(status));
  }
}
#endif

}  // namespace

struct Device::Impl {
  std::array<float, kOutputWords> output{};
  unsigned words = 0;
  uint64_t profile_ns = 0;
#ifdef USE_HLS_SIM
  const Weights* weights = nullptr;
#elif !defined(USE_CPU_ONLY)
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_mem packed = nullptr;
  cl_mem parameters = nullptr;
  cl_mem result = nullptr;

  ~Impl() {
    if (queue)
      clFinish(queue);
    if (packed)
      clReleaseMemObject(packed);
    if (parameters)
      clReleaseMemObject(parameters);
    if (result)
      clReleaseMemObject(result);
    if (kernel)
      clReleaseKernel(kernel);
    if (program)
      clReleaseProgram(program);
    if (queue)
      clReleaseCommandQueue(queue);
    if (context)
      clReleaseContext(context);
  }
#endif

  void Run(unsigned opcode, unsigned token, unsigned reset, unsigned mode) {
#ifdef USE_HLS_SIM
    decode(weights->packed.data(), weights->parameters.data(), output.data(),
           opcode, token, reset, mode, words);
#elif !defined(USE_CPU_ONLY)
    unsigned arguments[] = {opcode, token, reset, mode, words};
    for (int index = 0; index < 5; ++index) {
      CheckOpenCL(clSetKernelArg(kernel, 3 + index, sizeof(unsigned),
                                 &arguments[index]),
                  "set scalar argument");
    }
    struct Event {
      cl_event value = nullptr;
      ~Event() {
        if (value)
          clReleaseEvent(value);
      }
    } event;
    CheckOpenCL(clEnqueueTask(queue, kernel, 0, nullptr, &event.value),
                "decode");
    const size_t count = mode == 2 ? kOutputWords : (mode == 1 ? 513 : 1);
    CheckOpenCL(
        clEnqueueReadBuffer(queue, result, CL_TRUE, 0, count * sizeof(float),
                            output.data(), 0, nullptr, nullptr),
        "read result");
    cl_ulong begin = 0, end = 0;
    CheckOpenCL(clGetEventProfilingInfo(event.value, CL_PROFILING_COMMAND_START,
                                        sizeof(begin), &begin, nullptr),
                "profile start");
    CheckOpenCL(clGetEventProfilingInfo(event.value, CL_PROFILING_COMMAND_END,
                                        sizeof(end), &end, nullptr),
                "profile end");
    if (opcode == 1)
      profile_ns += end - begin;
#else
    (void)opcode;
    (void)token;
    (void)reset;
    (void)mode;
    throw std::runtime_error("this executable was built for CPU only");
#endif
  }
};

Device::Device(const Weights& weights, const std::string& xclbin)
    : impl_(std::make_unique<Impl>()) {
  impl_->words = weights.packed.size();
#ifdef USE_HLS_SIM
  (void)xclbin;
  impl_->weights = &weights;
#elif !defined(USE_CPU_ONLY)
  std::ifstream file(xclbin, std::ios::binary);
  std::vector<unsigned char> binary((std::istreambuf_iterator<char>(file)), {});
  if (binary.empty())
    throw std::runtime_error("cannot read xclbin: " + xclbin);
  cl_uint count = 0;
  CheckOpenCL(clGetPlatformIDs(0, nullptr, &count), "enumerate platforms");
  std::vector<cl_platform_id> platforms(count);
  CheckOpenCL(clGetPlatformIDs(count, platforms.data(), nullptr), "platforms");
  cl_device_id device = nullptr;
  for (cl_platform_id platform : platforms) {
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device,
                       nullptr) == CL_SUCCESS)
      break;
    device = nullptr;
  }
  if (!device)
    throw std::runtime_error("no FPGA device; load the overlay first");
  cl_int status = 0, binary_status = 0;
  impl_->context =
      clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  CheckOpenCL(status, "create context");
  impl_->queue = clCreateCommandQueue(impl_->context, device,
                                      CL_QUEUE_PROFILING_ENABLE, &status);
  CheckOpenCL(status, "create queue");
  const unsigned char* data = binary.data();
  const size_t size = binary.size();
  impl_->program = clCreateProgramWithBinary(impl_->context, 1, &device, &size,
                                             &data, &binary_status, &status);
  CheckOpenCL(status, "create program");
  CheckOpenCL(binary_status, "binary status");
  CheckOpenCL(
      clBuildProgram(impl_->program, 1, &device, nullptr, nullptr, nullptr),
      "build program");
  impl_->kernel = clCreateKernel(impl_->program, "decode", &status);
  CheckOpenCL(status, "create decode kernel");
  impl_->packed = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY,
                                 weights.packed.size() * sizeof(uint64_t),
                                 nullptr, &status);
  CheckOpenCL(status, "packed buffer");
  impl_->parameters = clCreateBuffer(impl_->context, CL_MEM_READ_ONLY,
                                     weights.parameters.size() * sizeof(float),
                                     nullptr, &status);
  CheckOpenCL(status, "parameter buffer");
  impl_->result = clCreateBuffer(impl_->context, CL_MEM_READ_WRITE,
                                 sizeof(impl_->output), nullptr, &status);
  CheckOpenCL(status, "result buffer");
  CheckOpenCL(clEnqueueWriteBuffer(impl_->queue, impl_->packed, CL_TRUE, 0,
                                   weights.packed.size() * sizeof(uint64_t),
                                   weights.packed.data(), 0, nullptr, nullptr),
              "load packed");
  CheckOpenCL(
      clEnqueueWriteBuffer(impl_->queue, impl_->parameters, CL_TRUE, 0,
                           weights.parameters.size() * sizeof(float),
                           weights.parameters.data(), 0, nullptr, nullptr),
      "load parameters");
  CheckOpenCL(clSetKernelArg(impl_->kernel, 0, sizeof(cl_mem), &impl_->packed),
              "packed argument");
  CheckOpenCL(
      clSetKernelArg(impl_->kernel, 1, sizeof(cl_mem), &impl_->parameters),
      "parameter argument");
  CheckOpenCL(clSetKernelArg(impl_->kernel, 2, sizeof(cl_mem), &impl_->result),
              "result argument");
#else
  (void)xclbin;
  throw std::runtime_error("this executable was built for CPU only");
#endif
  impl_->Run(0, 0, 0, 0);
  if (impl_->output[0] != 0)
    throw std::runtime_error("kernel rejected model LOAD");
}

Device::~Device() = default;

int Device::DecodeToken(int token, Context& context, bool logits, bool state) {
  const unsigned mode = state ? 2 : (logits ? 1 : 0);
  const uint64_t before = impl_->profile_ns;
  impl_->Run(1, token, context.reset_pending, mode);
  const int next_token = static_cast<int>(impl_->output[0]);
  if (next_token < 0 || next_token >= kVocabSize) {
    throw std::runtime_error("decode returned an invalid token/status");
  }
  if (logits)
    std::copy_n(impl_->output.data() + 1, kVocabSize, context.logits.begin());
  if (state)
    std::copy_n(impl_->output.data() + 513, kNumLayers * kDim,
                context.state.begin());
  ++context.kernel_calls;
  context.kernel_profile_ns += impl_->profile_ns - before;
  return next_token;
}

int Decode(int token, const Weights& weights, Context& context, Device* device,
           bool logits, bool state) {
  if (token < 0 || token >= kVocabSize)
    throw std::out_of_range("token ID");
  const int next_token =
      device ? device->DecodeToken(token, context, logits, state)
             : decode_cpu(token, weights, context);
  context.reset_pending = false;
  ++context.forward_calls;
  return next_token;
}

}  // namespace matmulfree

#endif  // BUILD_DECODE_KERNEL
