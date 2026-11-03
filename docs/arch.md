# Architecture

This implementation runs the published stories1M HGRN-Bit checkpoint from
[matmul-free-lm.c](https://github.com/ruhai-lin/matmul-free-lm.c). The checkpoint
and tokenizer are pinned in [model/manifest.json](../model/manifest.json).
The hardware is specialized for this model; it does not interpret arbitrary
Hugging Face model configurations.

## Model and execution boundary

| Property | Value |
|---|---:|
| Hidden dimension | 128 |
| FFN intermediate dimension | 272 |
| Layers | 5 |
| Vocabulary | 512 |
| CLI forward-step limit | 512 |
| Training parameters | 988,368 |
| Exported FP32 values, including 31 projection scales | 72,943 |
| Exported ternary coefficients | 915,456 |

The 31 additional exported scales explain why the counts of stored FP32 values
and ternary coefficients sum to 988,399 rather than the training parameter count.

HGRN-Bit replaces attention with an elementwise recurrence. The persistent state
is five vectors of 128 floats: 2,560 bytes in total. There is no attention KV
cache, position-dependent attention scan or RoPE operation. The hardware accepts
a token ID and the previous recurrent state; it has no position argument.
The CLI's 512-step limit is a software/model setting, not a growing state-cache
allocation. The benchmark can call the forward API for longer fixed workloads.

The PS runs checkpoint loading, parameter packing, BPE tokenization, sampling,
CLI control and text output. After initialization, PL performs embedding lookup,
every layer, final normalization, classifier and argmax. Each forward token
requires one XRT task submission and one result readback. Prompt tokens take the
same complete forward path as generated tokens.

## Source organization

| File | Responsibility |
|---|---|
| `src/main.cpp` | CLI, prompt processing, generation and chat loops |
| `src/decode.cpp` | HLS top and private primitives; CPU and XRT implementations |
| `src/decode.hpp` | Public forward API and device lifetime |
| `src/context.cpp/.hpp` | Host tensors, recurrent state, reset and counters |
| `src/weight.cpp/.hpp` | Fixed-shape validation, matrix metadata and packing |
| `src/vocab.cpp/.hpp` | Tokenizer, printable pieces and seeded sampler |
| `src/tensor.hpp` | Fixed dimensions, offsets, tensor types and kernel declaration |

The namespace is `matmulfree`. The HLS top `extern "C" decode` is in global
scope, matching Vitis' generated cosimulation stubs. Small hardware primitives
are private helpers inside `decode.cpp`, rather than separately packaged kernels.

The compile-time paths are:

| Definition | Effect |
|---|---|
| `BUILD_DECODE_KERNEL` | Compile the synthesizable HLS branch |
| `USE_CPU_ONLY` | Build the CPU host without OpenCL/XRT |
| `USE_CPU_ONLY` and `USE_HLS_SIM` | Call HLS C++ from the native host |
| `USE_NATIVE_MATH` | Use native math while compiling the HLS branch with g++ |
| None of the above | Build the ARM/XRT host and CPU fallback selected by CLI |

The default CMake build needs only published sources. `MATMULFREE_HLS_SIM=ON`
adds a native HLS executable and needs Vitis headers. The separate
`MATMULFREE_LOCAL_TOOLS` option is off by default: it enables private benchmark
and reference files when those are present in a local checkout. The public build
does not depend on these files.

## Forward computation

For a BitLinear matrix `W`, its input normalization weights `gamma`, and exported
weight scale `sw`, the reference computation is:

```text
r = 1 / sqrt(ordered_sum(x[j] * x[j]) / n + 1e-6)
xn[j] = gamma[j] * (r * x[j])
amax = max(1e-5, max(abs(xn[j])))
sx = 127 / amax
q[j] = clip(roundf(xn[j] * sx), -128, 127)
acc[row] = sum_j ternary_W[row,j] * q[j]     # INT32
y[row] = float(acc[row]) * (sw / sx)
```

Each matrix has its own normalization weights and scale. Although the i/f/g
projections share the same input vector, their fully normalized and quantized
vectors cannot simply be shared. The current source computes each BitLinear
separately.

One layer follows this order:

```text
u = mixer_norm(x)
i = BitLinear_i(u)
f = BitLinear_f(u)
g = BitLinear_g(u)
forget = sigmoid(f)
if layer > 0:
    forget = lower_bound + (1 - lower_bound) * forget
state = forget * state + (1 - forget) * silu(i)
gated_state = gate_norm(g) * silu(state)
x = x + BitLinear_o(gated_state)

u = ffn_norm(x)
[a, b] = BitLinear_w13(u)
h = silu(a) * b
x = x + BitLinear_w2(h)
```

After five layers, final RMSNorm and the classifier produce 512 logits. Greedy
selection scans them in increasing token-ID order and updates the winner only
on a strictly larger value, preserving the reference's tie behavior.

The matrices, in `[output, input]` order, are:

| Role | Count | Shape |
|---|---:|---|
| i/f/g/o | 20 | `[128, 128]` |
| Fused FFN gate/up (`w13`) | 5 | `[544, 128]` |
| FFN down (`w2`) | 5 | `[128, 272]` |
| Classifier | 1 | `[512, 128]` |

## Scheduler and shared datapath

`decode` executes a flat loop over 31 projections: six per layer, followed by
the classifier. Before and after each projection, its stage selects the relevant
normalization, gates, state updates or residual addition. One syntactic
`bitlinear` call site encourages reuse across all matrix roles.

Inside BitLinear, a DATAFLOW region connects three processes:

```text
quantize -- int8 stream --> ternary_gemv_engine -- int32 stream --> dequantize
         -- scale stream ------------------------------------->
```

The GEMV engine has four weight banks and 32 output lanes per bank, for 128 lanes
in total. A lane adds, subtracts or holds an INT8 activation according to its
2-bit coefficient. The inner input-column loop achieves II=1 in the recorded
HLS synthesis. The engine first reads a complete quantized input vector, then
walks output tiles and streams valid output rows. DATAFLOW does not make all
normalization work overlap with matrix computation: the quantizer must first
finish its reduction and absmax pass, and the GEMV buffers its complete input.

The source requests function sharing and explicit RAM bindings. The recorded
hierarchy contains one BitLinear dataflow module and one ternary GEMV engine.
Other helpers may be specialized by HLS; source directives alone do not establish
a physical instance count. The numerical FP32/FP64 portions still use floating
point operators and DSP resources even though ternary dot products use no
multipliers.

## Checkpoint and packed weights

The checkpoint is 1,207,260 bytes, little-endian:

1. Eight INT32 header fields: magic `0x4c464d4d`, version 1, dim, hidden_dim,
   layers, vocabulary, sequence limit and FP32-value count.
2. 72,943 FP32 values.
3. 915,456 signed INT8 ternary values, stored row-major by projection.

The host checks the exact size and expected header and rejects coefficients
outside {-1, 0, +1}. The checkpoint itself is not rewritten.

The FP32 section has the following offsets, measured in floats:

| First offset | Values | Contents |
|---:|---:|---|
| 0 | 65,536 | Token embedding |
| 65,536 | 640 | Forget-gate lower bounds |
| 66,176 | 640 | Mixer normalization weights |
| 66,816 | 640 | Gate normalization weights |
| 67,456 | 640 | FFN normalization weights |
| 68,096 | 128 | Final normalization weights |
| 68,224 | 4,719 | Per-projection input norms and scalar weight scales |

The host packs ternary coefficients as:

```text
[projection][128-output-row tile][input column][bank][32 lanes]
```

Within each uint64, lane 0 occupies bits `[1:0]`. Codes are `00` for zero,
`01` for +1 and `10` for -1. Packing does not generate `11`. The 544-output FFN
projection has five output tiles, with the final 96 rows padded with zeros.
Invalid rows do not become output values.

There are 30,528 uint64 words, or 244,224 bytes including padding, compared with
228,864 bytes for a padding-free 2-bit representation. LOAD distributes word `j`
to `cache[j % 4][j / 4]`. Each bank has capacity for 10,000 words; this model uses
7,632 words per bank. FP32 parameters use a separate 291,772-byte array bound to
URAM. Physical RAM consumption also includes banking, rounding, interfaces,
FIFOs and scratch arrays; payload bytes are not a resource-utilization report.

## Kernel interface and lifetime

```cpp
extern "C" void decode(const uint64_t* packed, const float* parameters,
                       float* result, unsigned opcode, unsigned token,
                       unsigned reset, unsigned output_mode, unsigned words);
```

`packed` and `parameters` share the `model` AXI master bundle. `result` uses
`output_mem`; all pointer/scalar registers and control use AXI-Lite `control`.
The deployment maps both masters to HP0. DDR is read during LOAD and the result
is written during each decode; weights and recurrent state are not streamed
from DDR during steady-state computation.

| Operation | Behavior |
|---|---|
| `opcode=0` | Copy packed/FP32 parameters, initialize 31 descriptors, clear state |
| `opcode=1` | Decode one token using cached parameters |
| `reset!=0` during decode | Clear the recurrent state before processing this token |

The expected LOAD word count is 30,528. The declared AXI depth for packed weights
is 40,000 words; a cosimulation testbench must allocate that declared depth even
when the unused tail is not consumed by hardware. The result buffer's maximum
size is 1,153 floats (4,612 bytes):

| Mode | Result words | Meaning |
|---:|---:|---|
| 0 | 1 | Float-encoded greedy token ID |
| 1 | 513 | Token ID followed by 512 logits |
| 2 | 1,153 | Token ID, logits, then 640 state floats |

LOAD returns status zero in `result[0]`. Negative status values indicate invalid
LOAD size/layout or an invalid decode request. The host checks status/token
range. At initialization it writes the parameter buffers and submits LOAD once.
For each forward call it sets scalar arguments, enqueues one task and performs a
blocking result read. It also records the OpenCL event's profiling interval.

Greedy generation requests mode 0; nonzero temperature requests mode 1 and uses
the host sampler. Mode 2 exists for validation. XRT timing, AXI transactions,
RTL cycle counts and application wall time have different boundaries; see
[performance.md](performance.md).

The accelerator has one persistent state context. A `Device` must not interleave
independent host `Context` objects as if they had separate PL states.
`ResetContext` clears the host state and marks the next device forward for
reset. There is no multi-session state virtualization or concurrent request
scheduler in this version.

## Numerical contract

The intended reference is the exported checkpoint interpreted by the pinned C
implementation, not arbitrary BF16 execution of the upstream Hugging Face model.

- RMSNorm epsilon is `1e-6f`; squared terms are accumulated in index order.
- `gamma * (r * x)` retains the reference's multiplication order.
- Activation absmax has a `1e-5f` floor; `roundf` rounds ties away from zero.
- Quantized activations are clipped to [-128,127]; dot products use INT32.
- Gate/state/residual operations and visible tensors use FP32.
- Compiler FP contraction is disabled; unsafe/fast-math reassociation is not used.
- `arm_exp` uses a 32-entry table and an FP64 polynomial with explicit FMA,
  then converts to FP32 for sigmoid/SiLU.

The FP64 exponential intermediates are intentional. An earlier implementation
using the default HLS exponential diverged at an A8 rounding boundary on a
recorded input stream. A small intermediate floating-point error can change an
INT8 activation and then affect subsequent recurrent state. A replacement
nonlinearity, reassociated reduction or narrower floating-point format requires
its own accuracy validation. Finite exact-match tests do not prove equivalence
for every possible floating-point input.

## Sources, adaptations and license

- Model/export/reference: [matmul-free-lm.c](https://github.com/ruhai-lin/matmul-free-lm.c),
  reference revision `f1cf9331b5a1760d5dbc2aa8899f6917a672d1db`.
  The model repository revision is separately pinned in the model manifest.
  The tokenizer, sampler, generate/chat conventions originate in Karpathy's
  llama2.c through that project. This repository ports them to C++ ownership and
  containers while preserving the tested behavior.
- Module organization and CPU/HLS split:
  [llama2.hls](https://github.com/ruhai-lin/llama2.hls), reference revision
  `728f19ce0f0dffd8ae229e2056cd20fb0ec3f648`, together with the MixerLoop
  whole-token/shared-engine approach. Their model computations are not substituted
  for HGRN-Bit.
- FPGA baseline reference:
  [matmulfree_KRIA](https://github.com/aebarthyi/matmulfree_KRIA), revision
  `0b41aa044303aeb31872c65b90b169e76e7b51a5`. The ternary add/subtract concept,
  PS/PL separation and validation workflow informed this design; its Chisel
  implementation and driver are not included.
- Exponential adaptation:
  [Arm optimized-routines math](https://github.com/ARM-software/optimized-routines/tree/0b5e3775c6c04761a2ecda8218eb45958c28be02/math),
  revision `0b5e3775c6c04761a2ecda8218eb45958c28be02`, specifically `expf.c`
  and `exp2f_data.c`. The table/range reduction/polynomial were adapted to HLS
  with explicit intermediate operations and FMA calls; this is not the original
  C implementation verbatim.

The root [LICENSE](../LICENSE) preserves MIT notices for Ruhai Lin, Andrej
Karpathy and Arm. Public source/build documentation does not depend on a local
copy of the reference repositories.

C++ formatting uses the checked-in Google-derived `.clang-format` (2-space
indentation, 80 columns). Public helpers and types follow the existing source
conventions; formatting or documentation changes must not silently change the
numerical or kernel ABI contract.
