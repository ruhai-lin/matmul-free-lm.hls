# Build and deploy

This guide builds the published sources directly. It does not assume local
`scripts/`, `references/` or `tests/` directories, or a downloadable prebuilt
bitstream. All commands use Bash on Linux or WSL. Run them from the repository
root unless a step explicitly changes directory.

The target is stories1M on KV260/K26 (`xck26-sfvc784-2LV-c`), with the official
`xilinx_kv260_base_202520_1` platform and a 150 MHz kernel clock. CPU generation
needs no FPGA tools. The documented FPGA toolchain is Vitis/Vivado 2025.2; the
ARM host was cross-compiled using the 2022.2 ZynqMP Linux SDK recipe. The board
must have a compatible Ubuntu/XRT installation and `xmutil`.

The current reorganized source has local CPU/HLS/RTL validation. Historical
board results belong to an earlier source snapshot. Follow the correctness
steps after loading a new build; [performance.md](performance.md) records the
evidence and remaining board-validation boundary.

## 1. Workspace and model

Use the same shell for the build sections so the variables remain available:

```bash
set -euo pipefail
REPO_ROOT=$(pwd)
mkdir -p outputs/{cpu,host,hls,link,logs,reports,validation,reference}
mkdir -p model
MODEL_REV=ac7a1b8f9c3018bc2ed0df08d4d438fdd0e0cff3
MODEL_URL="https://huggingface.co/ruhai-lin/matmul-free-lm.c/resolve/$MODEL_REV/stories1M"
curl -fL "$MODEL_URL/stories1M.bin" -o model/stories1M.bin
curl -fL "$MODEL_URL/tok512.bin" -o model/tok512.bin
```

Check both downloaded files against the committed manifest:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
manifest = json.loads(Path('model/manifest.json').read_text())
for item in manifest['files']:
    path = Path(item['destination'])
    data = path.read_bytes()
    assert len(data) == item['bytes'], f'wrong size: {path}'
    assert hashlib.sha256(data).hexdigest() == item['sha256'], f'wrong hash: {path}'
    print('PASS', path)
PY
```

The host validates the fixed checkpoint header and ternary range as well.
`tok512.bin` is required; a 32K tokenizer is not interchangeable with this model.

`model/manifest.json` is versioned. Downloaded model files and build artifacts
are ignored. The raw checkpoint remains the canonical C-project export; the
FPGA's 2-bit representation is produced in memory by the host.

## 2. CPU generation

Install a C++17 compiler, CMake 3.16 or newer, and Python 3 for the verification
examples. Configure a fresh build directory:

```bash
cmake -S . -B outputs/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build outputs/cpu --parallel
outputs/cpu/matmulfree --weights model/stories1M.bin \
  --tokenizer model/tok512.bin --prompt "Once upon a time" \
  --max_seq 128 --temp 0 --seed 42
```

The default CMake build compiles only public `src/` files. It defines
`USE_CPU_ONLY`, disables floating-point contraction in the model implementation,
and has no OpenCL/XRT dependency. Use a fresh build directory after moving a
checkout: CMake caches contain absolute paths.

The positional checkpoint and original short flags are accepted too:

```bash
outputs/cpu/matmulfree model/stories1M.bin -z model/tok512.bin \
  -i "A small dragon" -n 256 -t 0.8 -p 0.9 -s 42
```

Temperature zero selects argmax. Nonzero temperature uses the seeded host
sampler; the default temperature is 1 and top-p is 0.9. A zero/omitted seed uses
the current time. `--max_seq` includes prompt processing, is capped at 512, and
does not mean that many additional output tokens. The generate loop preserves
the C reference's BOS-based stopping convention. The model is a TinyStories
base model, although the host also exposes its inherited `--mode chat` loop.

## 3. Independent C reference

Download the reference into the ignored build tree rather than relying on a
vendored directory:

```bash
REFERENCE_REV=f1cf9331b5a1760d5dbc2aa8899f6917a672d1db
curl -fL "https://raw.githubusercontent.com/ruhai-lin/matmul-free-lm.c/$REFERENCE_REV/run.c" \
  -o outputs/reference/run.c
printf '%s  %s\n' \
  214e9b0b6595cae6b04d89f7bf5101e7eb1be14bccc1f6f85dc0f55e74fab186 \
  outputs/reference/run.c | sha256sum -c -
cc -O3 -ffp-contract=off outputs/reference/run.c -lm \
  -o outputs/reference/cpu_reference
```

The hash is for the LF upstream file. Its MIT notices remain in that file; the
adapted public C++ code also carries the required notices through the repository
license. The reference download is only needed for independent validation.

Compare deterministic text from CPU C++ against C:

```bash
python3 - <<'PY'
from pathlib import Path
import subprocess
root = Path.cwd()
cases = [('Once upon a time', 0, 42), ('The little girl', 0, 123),
         ('A small dragon', 0, 7), ('Once upon a time', .8, 42),
         ('', 0, 42), ('Hello, 世界!', .8, 123)]
for prompt, temp, seed in cases:
    args = [str(root/'model/stories1M.bin'), '-z', str(root/'model/tok512.bin'),
            '-i', prompt, '-n', '256', '-t', str(temp), '-s', str(seed)]
    def run(exe):
        return subprocess.run([str(exe), *args], check=True,
                              capture_output=True).stdout
    reference = run(root/'outputs/reference/cpu_reference')
    actual = run(root/'outputs/cpu/matmulfree')
    assert actual == reference, (prompt, temp, seed)
    print('PASS', repr(prompt), temp, seed)
PY
```

These text tests complement tensor/state checks; matching a short generated
story alone does not establish the full numerical contract.

## 4. Optional native HLS execution

Vitis headers allow g++ to execute the actual HLS C++ branch through the same
host interface. This remains native C++ execution, not RTL simulation:

```bash
VITIS_ROOT=${VITIS_ROOT:-/opt/xilinx/2025.2/Vitis}
cmake -S . -B outputs/sim -DCMAKE_BUILD_TYPE=Release \
  -DMATMULFREE_HLS_SIM=ON -DVITIS_ROOT="$VITIS_ROOT"
cmake --build outputs/sim --parallel
outputs/sim/matmulfree_sim --weights model/stories1M.bin \
  --tokenizer model/tok512.bin --prompt "Once upon a time" \
  --max_seq 128 --temp 0 --seed 42
```

The Vitis stream library can append a line beginning `INFO [HLS SIM]:` to stdout
at process exit. When comparing generated text, remove only that diagnostic
line. Preserve the remaining bytes, including whitespace and final newline.
The standalone benchmark and tensor-dump source in
[performance.md](performance.md#standalone-benchmark-source) can be linked against
this native HLS executable path as well.

`MATMULFREE_LOCAL_TOOLS` is an optional maintainer convenience and stays OFF for
all public instructions. Do not enable it on a clean public checkout: its local
reference/benchmark sources are deliberately outside the public file set.

## 5. HLS synthesis

Set paths for your installation. Vitis reads source/config paths through several
tools, so keep the hardware work directory on the Linux filesystem without
spaces, even if the checkout is on a Windows drive.

```bash
VITIS_ROOT=${VITIS_ROOT:-/opt/xilinx/2025.2/Vitis}
PLATFORM_DIR="$VITIS_ROOT/base_platforms/xilinx_kv260_base_202520_1"
PLATFORM="$PLATFORM_DIR/xilinx_kv260_base_202520_1.xpfm"
BUILD_WORK=${MATMULFREE_WORK:-$HOME/.cache/matmul-free-lm-hls}
[[ $BUILD_WORK = /* && $BUILD_WORK != *' '* ]]
mkdir -p "$BUILD_WORK"/{src,hls,link}
cp src/decode.cpp src/tensor.hpp "$BUILD_WORK/src/"
cat > "$BUILD_WORK/hls/hls_config.cfg" <<EOF
part=xck26-sfvc784-2LV-c
[hls]
flow_target=vitis
package.output.format=xo
package.output.syn=1
syn.top=decode
syn.file=$BUILD_WORK/src/decode.cpp
syn.cflags=-DBUILD_DECODE_KERNEL -I$BUILD_WORK/src -ffp-contract=off
syn.compile.unsafe_math_optimizations=0
clock=150MHz
EOF
(
  cd "$BUILD_WORK/hls"
  set +u; source "$VITIS_ROOT/settings64.sh"; set -u
  platforminfo --platform "$PLATFORM" > "$REPO_ROOT/outputs/logs/platform.txt"
  v++ --version > "$REPO_ROOT/outputs/logs/vitis-version.txt"
  v++ -c --mode hls --config "$BUILD_WORK/hls/hls_config.cfg" \
    --work_dir "$BUILD_WORK/hls/work" \
    2>&1 | tee "$REPO_ROOT/outputs/logs/hls.log"
)
cp "$BUILD_WORK/hls/work/decode.xo" outputs/hls/decode.xo
```

This builds `decode.cpp` with `BUILD_DECODE_KERNEL`; the other host sources are
not HLS translation units. No private testbench is needed for synthesis.
Reports are under `$BUILD_WORK/hls/work/hls/syn/report/`. Check the GEMV inner
loop's achieved II and instance hierarchy as well as overall resources. A
successful HLS compile does not establish routed timing closure.

For C simulation or RTL cosimulation, first create the public diagnostic driver
and reference tensor dump described in the performance document, then use the
testbench appendix at the end of this guide. Native simulation and RTL
cosimulation have different purposes and runtimes.

## 6. Link the KV260 platform

Keep v++'s input, output and intermediate work paths free of spaces:

```bash
cp outputs/hls/decode.xo "$BUILD_WORK/link/decode.xo"
(
  cd "$BUILD_WORK/link"
  set +u; source "$VITIS_ROOT/settings64.sh"; set -u
  v++ -l -t hw --platform "$PLATFORM" "$BUILD_WORK/link/decode.xo" \
    -o "$BUILD_WORK/link/decode.xclbin" --clock.default_freqhz 150000000 \
    --connectivity.sp decode_1.packed:HP0 \
    --connectivity.sp decode_1.parameters:HP0 \
    --connectivity.sp decode_1.result:HP0 \
    --vivado.synth.jobs 6 --vivado.impl.jobs 6 --save-temps \
    --temp_dir "$BUILD_WORK/link/_x" --log_dir "$BUILD_WORK/link/logs" \
    --report_dir "$BUILD_WORK/link/reports" \
    2>&1 | tee "$REPO_ROOT/outputs/logs/link.log"
)
cp "$BUILD_WORK/link/decode.xclbin" outputs/link/decode.xclbin
```

The kernel pointer arguments are `packed`, `parameters` and `result`; the two
AXI bundles are `model` and `output_mem`. These connectivity settings target
the official KV260 platform's HP0 path. If using a different platform, resolve
its memory/connectivity names and boot artifacts explicitly.

Use the final routed timing and utilization reports, not the HLS clock estimate,
to report physical implementation results. The routed DCP can also be inspected
without relying on a local Tcl script:

```bash
DCP=$(find "$BUILD_WORK/link/_x/link/vivado" -type f -name '*_routed.dcp' -print -quit)
[[ -n $DCP && -f $DCP ]]
cat > "$BUILD_WORK/report.tcl" <<'TCL'
set checkpoint [lindex $argv 0]
set output [lindex $argv 1]
file mkdir $output
open_checkpoint $checkpoint
report_timing_summary -delay_type min_max -report_unconstrained -file $output/timing.rpt
report_utilization -file $output/utilization.rpt
report_utilization -hierarchical -hierarchical_depth 4 -file $output/hierarchy.rpt
report_clocks -file $output/clocks.rpt
report_power -file $output/power_vectorless.rpt
close_design
TCL
(
  cd "$BUILD_WORK"
  set +u; source "${VIVADO_ROOT:-/opt/xilinx/2025.2/Vivado}/settings64.sh"; set -u
  vivado -mode batch -nojournal -log "$REPO_ROOT/outputs/logs/reports.log" \
    -source "$BUILD_WORK/report.tcl" -tclargs "$DCP" "$REPO_ROOT/outputs/reports"
)
```

The power command above is vectorless. Its assumptions and device boundary
must remain visible when reporting it alongside measured SOM input power.

## 7. Cross-compile the ARM host

Set `LINUX_SDK` to the **environment-setup script**, not its containing directory.
The example name is from the 2022.2 `xilinx-zynqmp-common` SDK; substitute the
location of your installed SDK. It must provide AArch64 compilers, headers and
the XRT/OpenCL target libraries.

```bash
LINUX_SDK=/path/to/xilinx-zynqmp-common-v2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
(
  unset LD_LIBRARY_PATH
  set +u; source "$LINUX_SDK"; set -u
  HOST_SOURCES=("$REPO_ROOT/src/context.cpp" "$REPO_ROOT/src/decode.cpp"
                "$REPO_ROOT/src/vocab.cpp" "$REPO_ROOT/src/weight.cpp")
  aarch64-xilinx-linux-g++ --sysroot="$SDKTARGETSYSROOT" \
    -O3 -std=c++17 -ffp-contract=off -Wno-unknown-pragmas \
    -I"$REPO_ROOT/src" -I"$SDKTARGETSYSROOT/usr/include/xrt" \
    "${HOST_SOURCES[@]}" "$REPO_ROOT/src/main.cpp" \
    -L"$SDKTARGETSYSROOT/usr/lib" \
    -lxilinxopencl -lxrt_coreutil -lpthread -lrt -ldl -lm \
    -o "$REPO_ROOT/outputs/host/matmulfree_host"
  if [[ -f "$REPO_ROOT/outputs/reference/run.c" ]]; then
    aarch64-xilinx-linux-gcc --sysroot="$SDKTARGETSYSROOT" -O3 -ffp-contract=off \
      "$REPO_ROOT/outputs/reference/run.c" -lm \
      -o "$REPO_ROOT/outputs/host/cpu_reference"
  fi
)
file outputs/host/matmulfree_host
```

The ELF should be AArch64. The XRT host supports both `--backend fpga` and
`--backend cpu`; it does not silently fall back to CPU when FPGA initialization
fails. The CPU-only native executable rejects FPGA selection. Verify target
library compatibility on the board before interpreting a runtime error as a
kernel failure.

## 8. Prepare a board directory

Use a new directory for each candidate build. The commands intentionally fail
if `outputs/bundle` already exists, avoiding a mixture of old and new artifacts.

```bash
BUNDLE="$REPO_ROOT/outputs/bundle"
mkdir "$BUNDLE"
mkdir "$BUNDLE/model"
cp outputs/host/matmulfree_host "$BUNDLE/"
if [[ -f outputs/host/cpu_reference ]]; then cp outputs/host/cpu_reference "$BUNDLE/"; fi
cp outputs/link/decode.xclbin "$BUNDLE/"
cp outputs/link/decode.xclbin "$BUNDLE/binary_container_1.bin"
cp "$PLATFORM_DIR/sw/boot/pl.dtbo" "$BUNDLE/"
cp model/stories1M.bin model/tok512.bin "$BUNDLE/model/"
cp model/manifest.json "$BUNDLE/model/manifest.json"
cp LICENSE "$BUNDLE/"
printf '%s\n' '{"shell_type":"XRT_FLAT","num_slots":"1"}' > "$BUNDLE/shell.json"
python3 - "$REPO_ROOT" "$BUNDLE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
root, bundle = map(Path, sys.argv[1:])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'board_validated': False, 'kernel': 'decode', 'target_clock_hz': 150000000,
            'model_manifest': json.loads((root/'model/manifest.json').read_text()),
            'source_sha256': {p.relative_to(root).as_posix(): digest(p)
                              for p in sorted((root/'src').glob('*pp'))},
            'xclbin_sha256': digest(bundle/'decode.xclbin')}
(bundle/'build.json').write_text(json.dumps(manifest, indent=2)+'\n')
(bundle/'SHA256SUMS').write_text(''.join(
    digest(p)+'  '+p.relative_to(bundle).as_posix()+'\n'
    for p in sorted(bundle.rglob('*')) if p.is_file()))
PY
```

The `.dtbo` must belong to the platform used for linking. The firmware name
`binary_container_1.bin` deliberately contains the same xclbin bytes as
`decode.xclbin`, following the tested XRT_FLAT deployment layout. Keep the source
files unchanged from synthesis through manifest creation, or the source hashes
would not describe the hardware build.

Transfer the directory using your board's actual host name/address:

```bash
BOARD_HOST=your-kv260-hostname
scp -r outputs/bundle "ubuntu@$BOARD_HOST:~/matmul-free-lm.hls"
```

## 9. Load and run on the board

Run the following **on KV260**, inside the transferred directory. Loading the
application replaces the currently active PL application.

```bash
cd ~/matmul-free-lm.hls
sha256sum -c SHA256SUMS
chmod +x matmulfree_host
sudo install -d /lib/firmware/xilinx/matmulfree
sudo install -m 644 binary_container_1.bin pl.dtbo shell.json \
  /lib/firmware/xilinx/matmulfree/
sudo xmutil unloadapp
sudo xmutil loadapp matmulfree
sudo xmutil listapps
./matmulfree_host --weights model/stories1M.bin --tokenizer model/tok512.bin \
  --backend fpga --xclbin decode.xclbin --prompt "Once upon a time" \
  --max_seq 128 --temp 0 --seed 42
```

For a deterministic comparison, run the same command with `--backend cpu` and
compare stdout. If the independent C reference was included, compare it too:

```bash
chmod +x cpu_reference
./cpu_reference model/stories1M.bin -z model/tok512.bin \
  -i "Once upon a time" -n 256 -t 0 -s 42 > reference.txt
./matmulfree_host --weights model/stories1M.bin --tokenizer model/tok512.bin \
  --backend cpu --prompt "Once upon a time" --max_seq 256 --temp 0 --seed 42 > cpu.txt
./matmulfree_host --weights model/stories1M.bin --tokenizer model/tok512.bin \
  --backend fpga --xclbin decode.xclbin --prompt "Once upon a time" \
  --max_seq 256 --temp 0 --seed 42 > fpga.txt
cmp reference.txt cpu.txt
cmp reference.txt fpga.txt
```

Repeat seeded sampling with `-t 0.8 -s 42`, and prompts such as `The little girl`
and `A small dragon`. For stronger validation, use mode-2 tensor dumps and
reset/replay as described in the performance document. Record the actual
bitstream/host hashes with the outcome. `build.json` starts with
`board_validated:false`; merely packaging or loading the overlay does not change
that evidence.

## 10. Restore the starter application

After completing validation and measurements:

```bash
sudo xmutil unloadapp
sudo xmutil loadapp k26-starter-kits
sudo xmutil listapps
```

Check that the default application loaded successfully. To shut down the board
afterward, run `sudo systemctl poweroff`. No build command in this guide connects
to, loads or powers off the board automatically.

## Optional HLS C/RTL testbench

The following self-contained testbench is generated under `outputs/`, not read
from an unpublished tests directory. It packs the checkpoint independently,
submits LOAD and compares every logit/state value against the reference dump.
Use the performance document's reference-dump appendix to create
`outputs/validation/reference.bin` first.

```bash
mkdir -p outputs/validation outputs/reference
cat > outputs/validation/decode_tb.cpp <<'CPP'
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "tensor.hpp"

using namespace matmulfree;

int main() {
  const char* path = std::getenv("MATMULFREE_MODEL");
  if (!path)
    return 2;
  FILE* file = fopen(path, "rb");
  if (!file)
    return 2;
  int32_t header[8];
  std::vector<float> parameters(kFloatParameters);
  std::vector<int8_t> ternary(kTernaryParameters);
  if (fread(header, sizeof(int32_t), 8, file) != 8 ||
      fread(parameters.data(), sizeof(float), parameters.size(), file) !=
          parameters.size() ||
      fread(ternary.data(), sizeof(int8_t), ternary.size(), file) !=
          ternary.size())
    return 2;
  fclose(file);
  const float* fp = parameters.data();
  const int8_t* src = ternary.data();
  std::vector<uint64_t> packed;
  unsigned pos = 0;
  for (unsigned k = 0; k < kNumProjections; ++k) {
    const MatrixShape shape = ProjectionShape(k);
    const unsigned n = shape.columns;
    const unsigned d = shape.rows;
    for (unsigned tile = 0; tile < (d + 127) / 128; ++tile)
      for (unsigned j = 0; j < n; ++j)
        for (unsigned bank = 0; bank < 4; ++bank) {
          uint64_t word = 0;
          for (unsigned lane = 0; lane < 32; ++lane) {
            unsigned row = tile * 128 + bank * 32 + lane;
            int v = row < d ? src[pos + row * n + j] : 0;
            word |= static_cast<uint64_t>(v == 1 ? 1 : (v == -1 ? 2 : 0))
                    << (2 * lane);
          }
          packed.push_back(word);
        }
    pos += n * d;
  }
  // The cosimulation AXI transactor snapshots the declared depth, including
  // the unused tail beyond the LOAD word count.
  unsigned words = packed.size();
  packed.resize(40000, 0);
  float result[kOutputWords];
  decode(packed.data(), fp, result, 0, 0, 0, 0, words);
  if (result[0] != 0)
    return 3;
  const char* expected = std::getenv("MATMULFREE_EXPECTED");
  FILE* ref = expected ? fopen(expected, "rb") : nullptr;
  if (!ref)
    return 2;
  unsigned seed = 20261023;
  const char* length = std::getenv("MATMULFREE_TEST_STEPS");
  int steps = length ? std::atoi(length) : 2;
  if (steps < 1 || steps > 512)
    return 2;
  for (int t = 0; t < steps; ++t) {
    unsigned token = 1;
    if (t) {
      seed = 1664525u * seed + 1013904223u;
      token = 3 + seed % 509;
    }
    decode(packed.data(), fp, result, 1, token, t == 0, 2, words);
    int ti, nt;
    float data[1152];
    if (fread(&ti, 4, 1, ref) != 1 || fread(&nt, 4, 1, ref) != 1 ||
        fread(data, 4, 1152, ref) != 1152)
      return 4;
    if (static_cast<int>(token) != ti || static_cast<int>(result[0]) != nt)
      return 5;
    float maximum = 0;
    for (int j = 0; j < 1152; ++j) {
      float err = std::fabs(result[j + 1] - data[j]);
      if (err > maximum)
        maximum = err;
      if (!std::isfinite(result[j + 1]) || result[j + 1] != data[j]) {
        printf("FAIL token=%d tensor_index=%d expected=%g actual=%g\n", t, j,
               data[j], result[j + 1]);
        return 6;
      }
    }
    printf("PASS full decode token=%d next=%d max_abs=%g\n", t, nt, maximum);
  }
  fclose(ref);
  return 0;
}
CPP
```


Add the testbench to the HLS configuration, then run C simulation:

```bash
cp outputs/validation/decode_tb.cpp "$BUILD_WORK/decode_tb.cpp"
cat >> "$BUILD_WORK/hls/hls_config.cfg" <<EOF
tb.file=$BUILD_WORK/decode_tb.cpp
tb.cflags=-I$BUILD_WORK/src
EOF
(
  cd "$BUILD_WORK/hls"
  set +u; source "$VITIS_ROOT/settings64.sh"; set -u
  MATMULFREE_MODEL="$REPO_ROOT/model/stories1M.bin" \
  MATMULFREE_EXPECTED="$REPO_ROOT/outputs/validation/reference.bin" \
  MATMULFREE_TEST_STEPS=512 \
    vitis-run --mode hls --csim --config "$BUILD_WORK/hls/hls_config.cfg" \
    --work_dir "$BUILD_WORK/hls/work" \
    2>&1 | tee "$REPO_ROOT/outputs/logs/csim.log"
)
```

Append these testbench settings once per newly generated configuration. For RTL
cosimulation, synthesize with that configuration and then replace `--csim` with
`--cosim`, using `MATMULFREE_TEST_STEPS=2` for a short LOAD-plus-two-token check.
The default testbench count is two if the variable is absent. Cosimulation of
the floating-point implementation can take several minutes even for this small
transaction count. Check the final `C/RTL co-simulation finished: PASS` and the
C post-check results; C testbench success before RTL execution is not enough.

Do not reuse a work directory containing an older synthesized DUT after changing
the public kernel source. Repeat synthesis before cosimulation. Neither
cosimulation nor a positive HLS timing estimate replaces route and board tests.
