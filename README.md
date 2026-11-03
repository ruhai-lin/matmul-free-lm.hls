# matmul-free-lm.hls

Run a small [MatMul-free language model](https://arxiv.org/abs/2406.02528) on an
AMD KV260 with one Vitis HLS `decode` kernel. This is the FPGA inference companion
to [matmul-free-lm.c](https://github.com/ruhai-lin/matmul-free-lm.c), using its
published stories1M checkpoint and tokenizer.

Every linear layer uses int8 activations and ternary weights in {-1, 0, +1}.
The FPGA computes their matrix-vector products with integer additions and
subtractions. Tokens are mixed by an elementwise gated recurrence, so the model
keeps one state vector per layer and has no attention KV cache.

One kernel invocation runs the complete forward pass: all five layers,
normalization, recurrent-state updates, FFN, classifier and greedy output
selection. Weights and state stay on chip after initialization. The ARM host
handles tokenization, optional temperature sampling and text output.

The current implementation targets **stories1M on KV260**, with Vitis/Vivado
2025.2. Model dimensions are fixed in the source; other checkpoints require
implementation changes. The same C++ sources also provide a CPU path for trying
the model without a board.

## quick start

From the repository root, download the checkpoint and its 512-token vocabulary:

```bash
mkdir -p model
MODEL_REV=ac7a1b8f9c3018bc2ed0df08d4d438fdd0e0cff3
MODEL_URL="https://huggingface.co/ruhai-lin/matmul-free-lm.c/resolve/$MODEL_REV/stories1M"
curl -fL "$MODEL_URL/stories1M.bin" -o model/stories1M.bin
curl -fL "$MODEL_URL/tok512.bin" -o model/tok512.bin
```

Build the CPU executable with a C++17 compiler and CMake:

```bash
cmake -S . -B outputs/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/cpu --parallel
outputs/cpu/matmulfree --weights model/stories1M.bin \
  --tokenizer model/tok512.bin --prompt "Once upon a time" \
  --max_seq 128 --temp 0 --seed 42
```

This path runs on the CPU and does not need Xilinx software. The generated text
starts with:

> Once upon a time, there was a little girl named Lily. She loved to play outside
> in the sunshine. One day, she found a big, shiny rock. She picked it up and put
> it in her box.

For temperature sampling, use `--temp 1 --top_p 0.9`. The executable accepts the
short options from the C project as well, for example `-i`, `-n`, `-t`, `-p`,
`-s` and `-z`. Use `--help` for the full CLI. `--max_seq` counts forward steps,
including prompt processing, and is capped at 512 in this implementation.

The model files are downloaded separately and are checked against
[model/manifest.json](model/manifest.json); the deployment guide includes the
checksum commands.

## run on KV260

Follow [deploy.md](docs/deploy.md) to synthesize `decode`, link the official
KV260 platform, cross-compile the ARM host and install the overlay. The guide
contains the actual tool commands and does not require private build scripts.
Once the matching host, model and bitstream are installed, generation is:

```bash
./matmulfree_host --weights model/stories1M.bin --tokenizer model/tok512.bin \
  --backend fpga --xclbin decode.xclbin --prompt "Once upon a time" \
  --max_seq 128 --temp 0 --seed 42
```

Greedy decoding returns one token ID per invocation. Temperature sampling
returns logits to the host, where the sampler chooses the next token. There is
one device launch per forward token in both cases.

## source and documentation

`src/decode.cpp` contains the HLS top and its private compute primitives, together
with the CPU/XRT host implementation selected at compile time. `main`, `weight`,
`context` and `vocab` handle the CLI, checkpoint loading, runtime tensors and
tokenizer/sampler respectively. Build products belong in `outputs/`.

- [Deployment](docs/deploy.md): model files, CPU/HLS/ARM builds, installation,
  correctness checks and restoring the starter application.
- [Architecture](docs/arch.md): model equations, numerical behavior, scheduling,
  packing, on-chip storage and the kernel interface.
- [Performance](docs/performance.md): benchmark protocol, throughput, latency,
  energy, implementation resources and the provenance of each result.

The performance document distinguishes historical board measurements from
validation of the current source. A rebuilt bitstream needs its own board
validation before it can be associated with new hardware measurements.

## acknowledgements

The model and reference behavior come from
[matmul-free-lm.c](https://github.com/ruhai-lin/matmul-free-lm.c), which follows
[llama2.c](https://github.com/karpathy/llama2.c) and the HGRN-Bit architecture of
[MatMul-free LM](https://github.com/ridgerchu/matmulfreellm).
[llama2.hls](https://github.com/ruhai-lin/llama2.hls) and MixerLoop informed the
module organization and whole-token scheduling;
[matmulfree_KRIA](https://github.com/aebarthyi/matmulfree_KRIA) informed the FPGA
baseline. The exponential primitive adapts
[Arm optimized-routines](https://github.com/ARM-software/optimized-routines).
Pinned sources and adaptation details are recorded in [arch.md](docs/arch.md).

## license

MIT. The [license](LICENSE) includes the notices for adapted Karpathy and Arm
code. Model provenance is recorded separately in `model/manifest.json`.
