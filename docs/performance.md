# Performance and measurement

This page keeps three kinds of evidence separate: checks of the current public
source, estimates from HLS synthesis, and historical measurements on KV260.
Only the last category is a board throughput/energy measurement. Preparing a
source tree, passing cosimulation or copying an older xclbin does not establish
new board results.

## Result sets and provenance

| Result set | Scope | Board measurement |
|---|---|---|
| Current public-source preparation | Clean CPU build, native HLS/reference regression, ARM build and refreshed HLS synthesis | Pending |
| 2026-10-24 reorganized implementation | Six text cases, chat, tensor/state/reset checks, HLS C simulation and short RTL cosimulation | Not rerun after reorganization |
| 2026-10-23 whole-token implementation | Routed platform, correctness checks and paired CPU/PL SOM-energy measurements | Historical, measured |

The machine-readable evidence is in [results/](results/):

- [Public-source validation](results/public-validation.json) records the checks
  actually performed for the prepared source and its source hashes.
- [Model manifest](../model/manifest.json) pins the checkpoint and tokenizer.
- [Historical summary](results/historical-2026-10-23.json) preserves per-backend
  summary metrics, hardware/report hashes and the corresponding historical
  source hashes.
- [Historical retained runs](results/historical-runs-2026-10-23.json) contains
  the per-run measurements used by that summary, without local machine paths.

The historical source names in the data, such as `src/mega.cpp`, identify the
earlier implementation. They are not files required to build this repository.
The original source, raw telemetry, detailed tool reports and board bundles are
retained locally; those large archives are not published by this source tree.
The compact records make the stated summary traceable but are not a substitute
for the complete historical experiment archive.

## Current-source correctness and implementation status

The model computation has not intentionally changed during repository packaging.
The public CMake build is independent of local scaffolding. The local reference
suite compares the C++ CPU implementation and native execution of the HLS branch
against the pinned, unmodified C implementation.

Recorded checks for the reorganized source include:

- Six 256-step generate cases: three seeded greedy prompts, one seeded sampled
  prompt, an empty prompt and a Unicode prompt.
- A 128-step chat comparison.
- A fixed stream of 512 tokens, comparing all 512 logits and all 640 state
  values per step, followed by a reset and replay of the same 512-token stream.
- Equal binary dumps for C, C++ CPU and native HLS execution. The recorded
  dump SHA256 is
  `340834932482fe63bac2fde9aefa1b2757d2e9e6e4d9fd5fb00135e6ff42c0f6`.
- Vitis HLS C simulation of 512 complete forwards, with maximum absolute
  difference zero on the checked tensors.
- RTL cosimulation of LOAD followed by two complete forwards with logits/state
  output, passing C post-check with maximum absolute difference zero.

The current preparation's refreshed checks and compiler outcomes are listed in
`public-validation.json`. Its board-validation field remains false. The earlier
finite tests do not establish exact equivalence for every possible input or
constitute a board run of the newly prepared source.

For the 2026-10-24 HLS synthesis, the kernel estimate was:

| Resource | HLS estimate |
|---|---:|
| LUT | 43,553 |
| FF | 32,896 |
| BRAM_18K | 177 |
| URAM | 18 |
| DSP | 217 |

The target period was 6.667 ns and the HLS estimate was 6.727 ns. The top-level
maximum latency was `?` because the top includes LOAD and variable loop bounds.
Its reported minimum of 71 cycles is not the full-model decode latency.
These are pre-route kernel estimates, not the complete platform's placed/routed
resources or a timing-closure result. A refreshed estimate is retained in the
public validation record when available.

## Historical board throughput and energy

The following results were measured on **2026-10-23**, before the module and
repository reorganization. The target was KV260/K26, PL clock 150 MHz, batch 1,
with one Cortex-A53 running the CPU reference at a verified 1.333333 GHz. The
benchmark was pinned to CPU2 and the power sampler to CPU0. Both backends used
the same model, input stream and loaded overlay.

Eight alternating CPU/PL pairs were collected. Pair 0 was discarded; entries
below are medians across the remaining seven runs. Each run warmed up for 16
tokens, reset state, then executed 8,192 forwards. The CPU path in this historical
comparison used the original C forward implementation, not a newly measured
version of the current C++ CPU runtime.

| Metric | Cortex-A53 CPU | PS + PL whole-token decode |
|---|---:|---:|
| Forward tokens/s | 768.98 | 1,303.83 |
| Mean wall latency, ms/forward | 1.3004 | 0.7670 |
| SOM active input power, W | 3.5027 | 3.3744 |
| SOM total energy, mJ/forward | 4.5501 | 2.5871 |
| SOM idle-subtracted energy, mJ/forward | 0.2848 | 0.0708 |
| Minimum run throughput, tokens/s | 767.90 | 1,303.64 |
| Maximum run throughput, tokens/s | 771.40 | 1,304.78 |

This is about 1.696x throughput and 43.1% lower total SOM energy per forward than
the paired CPU baseline. It is not a new performance claim for the current
source. Per-run medians of individual token latency were:

| Backend | p50, ms | p95, ms | p99, ms |
|---|---:|---:|---:|
| CPU | 1.2941 | 1.3432 | 1.3828 |
| PS + PL | 0.7635 | 0.7792 | 0.8227 |

The main wall-time region includes embedding, every layer and nonlinear
operation, classifier, argmax, XRT task submission and result readback. The
accelerator returns a 4-byte selected token ID. Individual-token timers are
enabled for both backends and their overhead is included. The region excludes
checkpoint loading/packing, device initialization, tokenization, printing and
correctness checks.

Consequently, these are full-forward throughput numbers, not measurements of
an entire text-generation service or time to first token for a supplied prompt.
Initialization was recorded separately: historical medians were approximately
2.29 ms for CPU and 245.21 ms for PL initialization. They are not TTFT.

The historical OpenCL event interval was 0.6913 ms/forward. It does not isolate
the exact compute, driver and AXI-transfer components. Process CPU-time/wall-time
ratios were approximately 0.999 for CPU and 0.180 for the PL host, including XRT
threads; this is not a measurement of total PS utilization.

A supplemental 512-forward BOS-started greedy feedback run produced 769.57
CPU and 1,295.21 PL tokens/s, with 4.5402 and 2.6116 mJ/forward respectively.
It had one run per backend and no EOS-based early exit or text IO. It is not
pooled with the paired fixed-input comparison.

## Historical physical implementation

The recorded complete routed platform, including interfaces, used:

| Resource | Routed usage |
|---|---:|
| LUT | 35,836 |
| Registers | 39,925 |
| Block RAM tiles | 80 |
| RAMB36/FIFO instances | 76 |
| RAMB18 instances | 8 |
| URAM | 18 |
| DSP | 181 |

At 150 MHz, setup WNS was +0.704 ns with TNS zero; hold WHS was +0.010 ns with
THS zero. These belong to that routed build, not an inferred maximum clock from
HLS and not a timing result for a new link of the current source.

The corresponding HLS AXI-model RTL transactions were:

| Transaction | Cycles | Time at 150 MHz, ms |
|---|---:|---:|
| LOAD | 350,481 | 2.336540 |
| Decode including reset | 97,682 | 0.651213 |
| Decode retaining state | 97,059 | 0.647060 |

These decode transactions returned full logits/state. They exclude Linux/XRT
and have a different output mode from the greedy board benchmark. They should
not be subtracted from wall latency and labeled as an exact host-overhead
measurement.

Vivado's historical vectorless device-power estimate was 3.784 W: 3.475 W
dynamic and 0.309 W device static, including a default PS8 contribution of
2.560 W. There was no measured SAIF activity input. This is a modeled device
boundary with tool assumptions, not an INA260 measurement or isolated PL power.

## Measurement protocol

Use the following protocol for a new build and record new results separately.

1. Validate the exact model, CPU/PL tensor outputs, reset/replay and generated
   text before collecting performance. Save source, host and xclbin hashes.
2. Use the same loaded overlay for the paired CPU and PL runs. Pin the benchmark
   to CPU2 and the sampler to CPU0. Record actual CPU frequency/governor, PL
   clock, board identity, kernel/XRT versions and sensor names.
3. Generate a 512-token fixed stream with unsigned 32-bit LCG state initialized
   to 20261023. Its first token is BOS (1). For later entries, update
   `seed = 1664525 * seed + 1013904223` modulo 2^32 and use `3 + seed % 509`.
4. Warm up 16 forwards, then reset state. Execute 8,192 forwards by repeating
   that input stream, without resetting every 512 tokens. Both backends use the
   same inputs regardless of their predictions. This controls workload rather
   than measuring a particular generated story.
5. Collect eight pairs, ordering CPU then PL for even pairs and PL then CPU
   for odd pairs. Exclude pair 0 from the fixed-stream summary. Keep every run.
6. Sample INA260 `power1_input` about every 50 ms using the Linux monotonic time
   base. The historical unit was micro-watts, converted to watts. Verify the
   rail and units on the actual image; an arbitrary hwmon device is not enough.
7. Capture idle windows before and after each run. Integrate power over the
   benchmark's own monotonic start/end timestamps, with linearly interpolated
   endpoints and trapezoidal integration. Retain raw samples and timestamps.
8. Report per-run values and median/range across the retained runs. For latency
   percentiles, state whether a median of per-run percentiles or pooled samples
   was used. The historical table uses the former.

The optional benchmark source below implements the forward stream, warmup,
reset, timing, counters and diagnostic dumps. Power collection remains a
separate sampler so sysfs reads do not occur inside each timed forward.
For the historical experiment, adjacent idle acquisition included settling
delays; preserve the chosen windows and any discarded margins with a new run.

## Power and energy boundaries

The historical INA260 VCC_SOM input measurement includes PS, PL, DDR and module
power-conversion losses. It excludes the carrier and wall adapter. It is neither
wall power nor the PL-only rail.

For `N` forwards in a measured interval:

```text
elapsed = end_monotonic - start_monotonic
throughput = N / elapsed
energy = integral_start^end P_SOM(t) dt
active_power = energy / elapsed
energy_per_forward = energy / N
idle_power = (idle_before_mean + idle_after_mean) / 2
incremental_energy_per_forward = (energy - idle_power * elapsed) / N
```

Idle-subtracted energy includes changes in PS, interfaces, memory and PL; it
does not isolate the accelerator's switching energy. Idle power itself can
change with overlay resources, clocks and software. Preserve total energy as
the primary result alongside the incremental estimate. For short runs, increase
the number of forwards so sensor sampling is sufficient.

For comparison with FeFET/CIM, match the model, precision, batch, included
operations, output selection and residency/loading policy. Compare an array
read/MVM energy against the corresponding FPGA operator boundary, or complete
model token energy against a complete system boundary including necessary
peripherals and host work. FPGA SOM energy and a FeFET array-only estimate cannot
be divided and called a like-for-like device-efficiency improvement. Include
ADC/DAC, buffering, digital nonlinearities and data transfer where applicable.

Optional operation-equivalent reporting uses 915,456 matrix entries per forward,
including zero weights. Under a convention of one MAC-equivalent per entry and
two operations per MAC, historical PL throughput corresponds to approximately
1.1936 GMAC-equivalent/s or 2.3872 GOPS. These are workload conventions; ternary
hardware does not execute that many physical multiplications. Token/s and
energy/token are the primary language-model metrics.

## Bottlenecks and interpretation

The current GEMV has 128 output lanes and an II=1 input loop. The sequential
FP32 sum inside RMSNorm has achieved II=7 in the recorded synthesis because of
the accumulator dependency. A forward includes 47 normalizations over 6,736
elements; multiplying that element count by II suggests roughly 47k cycles of
reduction work, not a measured exclusive time attribution. Reductions, scaling,
nonlinearities, streams and output serialization all contribute to total
latency. The steady-state model is not limited by streaming full weights from
DDR, because this 1M implementation preloads them on chip.

The exact exponential is pipelined: its recorded 76-cycle latency is not a
76-cycle per-element initiation interval. Replacing it with a LUT approximation
changes the numerical contract and must be validated separately. Increasing
GEMV lanes, changing language or reducing bit widths cannot by themselves
establish a throughput/energy improvement. Use a new routed build and the
same measurement boundary for an optimization comparison.

## Standalone benchmark source

To reproduce timing or request diagnostic tensor dumps without unpublished
maintainer files, save this driver under the ignored output directory. It uses
only the public forward API and is the current C++ driver; its output must not
be mislabeled as a rerun of the historical C benchmark.

```bash
mkdir -p outputs/validation outputs/reference
cat > outputs/benchmark.cpp <<'CPP'
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "decode.hpp"

namespace {

double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct Window {
  double start, end, p50, p95, p99, maximum, process_seconds;
  uint64_t kernel_calls, profile_ns;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 5) {
      throw std::runtime_error(
          "usage: matmulfree_bench MODEL [steps] [repeats] [fixed|greedy]");
    }
    const int steps = argc > 2 ? std::stoi(argv[2]) : 1024;
    const int repeats = argc > 3 ? std::stoi(argv[3]) : 1;
    const std::string protocol = argc > 4 ? argv[4] : "fixed";
    if (steps < 1 || steps > 100000 || repeats < 1 || repeats > 101 ||
        (protocol != "fixed" && protocol != "greedy"))
      throw std::runtime_error("invalid benchmark arguments");
    const char* backend_environment = std::getenv("MATMULFREE_BACKEND");
#ifdef USE_HLS_SIM
    const std::string backend =
        backend_environment ? backend_environment : "sim";
#elif defined(USE_CPU_ONLY)
    const std::string backend =
        backend_environment ? backend_environment : "cpu";
#else
    const std::string backend =
        backend_environment ? backend_environment : "fpga";
#endif
    if (backend != "cpu" && backend != "fpga" && backend != "sim")
      throw std::runtime_error("invalid backend");
#ifdef USE_HLS_SIM
    if (backend == "fpga")
      throw std::runtime_error("simulation build has no XRT backend");
#elif defined(USE_CPU_ONLY)
    if (backend != "cpu")
      throw std::runtime_error("CPU build has no device backend");
#else
    if (backend == "sim")
      throw std::runtime_error("XRT build has no simulation backend");
#endif
    const char* path = std::getenv("MATMULFREE_DUMP");
    std::unique_ptr<FILE, int (*)(FILE*)> dump(
        path ? std::fopen(path, "wb") : nullptr, std::fclose);
    if (path && !dump)
      throw std::runtime_error("cannot open tensor dump");
    const double load_start = Now();
    matmulfree::Weights weights;
    matmulfree::LoadWeights(weights, argv[1]);
    matmulfree::Context context;
    std::unique_ptr<matmulfree::Device> device;
    if (backend != "cpu") {
      const char* binary = std::getenv("MATMULFREE_XCLBIN");
      device = std::make_unique<matmulfree::Device>(
          weights, binary ? binary : "decode.xclbin");
    }
    const double load_seconds = Now() - load_start;
    std::array<int, 512> tokens;
    tokens[0] = 1;
    unsigned seed = 20261023;
    for (int index = 1; index < 512; ++index) {
      seed = 1664525u * seed + 1013904223u;
      tokens[index] = 3 + seed % 509;
    }
    for (int index = 0; index < 16; ++index)
      matmulfree::Decode(tokens[index], weights, context, device.get());
    std::vector<Window> windows;
    std::vector<double> elapsed;
    std::vector<double> latency(steps);
    volatile float sink = 0;
    uint64_t total_calls = 0, total_profile = 0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      matmulfree::ResetContext(context);
      const uint64_t calls_before = context.kernel_calls;
      const uint64_t profile_before = context.kernel_profile_ns;
      const clock_t process_start = std::clock();
      Window window{};
      window.start = Now();
      int next = 1;
      for (int index = 0; index < steps; ++index) {
        const double token_start = Now();
        const int token = protocol == "greedy" ? (index == 0 ? 1 : next)
                                               : tokens[index % 512];
        next = matmulfree::Decode(token, weights, context, device.get(),
                                  dump != nullptr, dump != nullptr);
        sink = context.logits[0];
        latency[index] = (Now() - token_start) * 1000;
        if (dump) {
          std::fwrite(&token, 4, 1, dump.get());
          std::fwrite(&next, 4, 1, dump.get());
          std::fwrite(context.logits.data(), 4, matmulfree::kVocabSize,
                      dump.get());
          std::fwrite(context.state.data(), 4,
                      matmulfree::kNumLayers * matmulfree::kDim, dump.get());
        }
      }
      window.end = Now();
      window.process_seconds =
          static_cast<double>(std::clock() - process_start) / CLOCKS_PER_SEC;
      window.kernel_calls = context.kernel_calls - calls_before;
      window.profile_ns = context.kernel_profile_ns - profile_before;
      if (device && window.kernel_calls != static_cast<uint64_t>(steps))
        throw std::runtime_error("kernel call count mismatch");
      total_calls += window.kernel_calls;
      total_profile += window.profile_ns;
      std::sort(latency.begin(), latency.end());
      window.p50 = latency[steps / 2];
      window.p95 = latency[(steps - 1) * 95 / 100];
      window.p99 = latency[(steps - 1) * 99 / 100];
      window.maximum = latency.back();
      windows.push_back(window);
      elapsed.push_back((window.end - window.start) / steps);
    }
    std::printf("{\"windows\":[");
    for (size_t index = 0; index < windows.size(); ++index) {
      const Window& w = windows[index];
      std::printf(
          "%s{\"repeat\":%zu,\"start_monotonic_s\":%.9f,\"end_monotonic_s\":%."
          "9f,\"forward_calls\":%d,\"elapsed_s\":%.9f,\"decode_wall_s\":%.9f,"
          "\"xrt_event_profile_ns\":%llu,\"latency_p50_ms\":%.6f,\"latency_p95_"
          "ms\":%.6f,\"latency_p99_ms\":%.6f,\"latency_max_ms\":%.6f,\"process_"
          "cpu_s\":%.6f}",
          index ? "," : "", index, w.start, w.end, steps, w.end - w.start,
          w.end - w.start, static_cast<unsigned long long>(w.profile_ns), w.p50,
          w.p95, w.p99, w.maximum, w.process_seconds);
    }
    std::sort(elapsed.begin(), elapsed.end());
    const double median =
        (elapsed[(repeats - 1) / 2] + elapsed[repeats / 2]) * 0.5;
    std::printf(
        "],\"median_ms_per_token\":%.9f,\"median_tokens_per_second\":%.6f,"
        "\"load_seconds\":%.6f,\"warmup_steps\":16,\"input_protocol\":\"%s\","
        "\"sink\":%.9g}\n",
        median * 1000, 1.0 / median, load_seconds, protocol.c_str(),
        static_cast<double>(sink));
    std::fprintf(stderr,
                 "MATMULFREE_STATS "
                 "{\"calls\":%llu,\"kernel_profile_ns\":%llu,\"whole_token\":"
                 "true,\"backend\":\"%s\"}\n",
                 static_cast<unsigned long long>(total_calls),
                 static_cast<unsigned long long>(total_profile),
                 backend.c_str());
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark: " << error.what() << '\n';
    return 1;
  }
}
CPP
```


Build a CPU benchmark:

```bash
c++ -O3 -std=c++17 -ffp-contract=off -DUSE_CPU_ONLY -Isrc \
  src/context.cpp src/decode.cpp src/vocab.cpp src/weight.cpp \
  outputs/benchmark.cpp -lm -lpthread -o outputs/cpu/matmulfree_bench
MATMULFREE_BACKEND=cpu outputs/cpu/matmulfree_bench model/stories1M.bin 8192 1 fixed
```

For the ARM/XRT benchmark, use the host cross-compilation command in
`deploy.md`, substituting `outputs/benchmark.cpp` for `src/main.cpp` and
`outputs/host/matmulfree_bench` for the output. Do not define `USE_CPU_ONLY` for
that binary. It runs either backend based on `MATMULFREE_BACKEND`:

```bash
# On KV260, with the matching overlay already loaded:
MATMULFREE_BACKEND=cpu taskset -c 2 ./matmulfree_bench model/stories1M.bin 8192 1 fixed
MATMULFREE_BACKEND=fpga MATMULFREE_XCLBIN="$PWD/decode.xclbin" \
  taskset -c 2 ./matmulfree_bench model/stories1M.bin 8192 1 fixed
```

Capture stdout JSON and stderr stats for each run. The driver's clocks use the
monotonic time base used by Linux `time.monotonic()`; correlate the reported
window timestamps with sensor samples rather than the surrounding shell-command
duration. An XRT run asserts that the kernel-call count equals the forward
count. `fixed` is the primary protocol; `greedy` feeds back predictions and is
a separate supplementary workload.

For diagnostics, `MATMULFREE_DUMP=/absolute/path/dump.bin` requests logits and
state, and writes one record per forward: INT32 input token, INT32 argmax,
512 FP32 logits, 640 FP32 state values. The dump format is local little-endian
on the validated x86/AArch64 targets. A dump-enabled run is not a performance
measurement: it returns more data and performs file IO in the forward loop.

## Reference tensor dump

Download the pinned C reference as described in `deploy.md`. Its `TESTING`
definition disables the original CLI main, allowing the following small
standalone reference program:

```bash
mkdir -p outputs/validation outputs/reference
cat > outputs/reference_dump.c <<'CPP'
#define TESTING
#include "run.c"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  Model model;
  build_model(&model, argv[1]);
  FILE* output = fopen(argv[2], "wb");
  if (!output) return 2;
  int tokens[512];
  unsigned seed = 20261023;
  tokens[0] = 1;
  for (int index = 1; index < 512; ++index) {
    seed = 1664525u * seed + 1013904223u;
    tokens[index] = 3 + seed % 509;
  }
  for (int repeat = 0; repeat < 2; ++repeat) {
    reset_state(&model);
    for (int index = 0; index < 512; ++index) {
      int token = tokens[index];
      float* logits = forward(&model, token);
      int next = sample_argmax(logits, 512);
      fwrite(&token, 4, 1, output);
      fwrite(&next, 4, 1, output);
      fwrite(logits, 4, 512, output);
      fwrite(model.state.h, 4, 640, output);
    }
  }
  fclose(output);
  free_model(&model);
  return 0;
}
CPP
```


Compile, generate two reset/replay copies, and compare with the public CPU
benchmark's diagnostic mode:

```bash
cc -O3 -ffp-contract=off -Ioutputs/reference outputs/reference_dump.c -lm \
  -o outputs/reference/reference_dump
outputs/reference/reference_dump model/stories1M.bin outputs/validation/reference.bin
MATMULFREE_DUMP="$PWD/outputs/validation/cpu.bin" MATMULFREE_BACKEND=cpu \
  outputs/cpu/matmulfree_bench model/stories1M.bin 512 2 fixed > outputs/validation/cpu.json
cmp outputs/validation/reference.bin outputs/validation/cpu.bin
```

The expected file length is `2 * 512 * (8 + 1152 * 4) = 4,726,784` bytes.
The two halves must match after reset/replay. Use the same diagnostic invocation
with `MATMULFREE_BACKEND=fpga` and the AArch64/XRT benchmark on KV260, then compare
the files. A native HLS benchmark can be compiled by linking the public host
sources with `USE_CPU_ONLY`/`USE_HLS_SIM` and an object of `decode.cpp` compiled
with `BUILD_DECODE_KERNEL`/`USE_NATIVE_MATH`, using Vitis headers.

For a new result set, retain the model hash, all source/build artifact hashes,
command lines, raw tensor checks and power samples alongside the summary.
Publish numbers only for the exact validated host/bitstream pair.
