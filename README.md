# qwen3.8-27b-img

Prebuilt **llama.cpp `llama-server`** compiled for **sm_120** (RTX 50-series / Blackwell) to serve `unsloth/Qwen3.8-27B-GGUF` (Q4_0, 16 GB) with MTP speculative decoding.

## Why prebuilt?
Official llama.cpp Linux binaries are **CPU-only** (CUDA builds ship only for Windows);
sm_120 Blackwell kernels are only compiled when CUDAToolkit >= 12.8. So a GPU build
must be compiled from source with `-DCMAKE_CUDA_ARCHITECTURES=120`. This repo hosts the
already-built binary so a fresh box skips the ~9 min compile.

## Arch / compat contract
- **GPU**: sm_120 (any RTX 50-series). Built with `-DCMAKE_CUDA_ARCHITECTURES=120`.
- **CPU**: generic x86-64 (`-DGGML_NATIVE=OFF`) — host-independent.
- **CUDA userland** (`libcudart.so.13` / `libcublas.so.13` / `libnccl.so.2`): from the
  `vastai/pytorch:cuda-13.0.3-auto` template's `/usr/local/cuda-13.0` (bundle does not ship them).
- **Driver** (`libcuda.so.1`): from the host (any driver present on a 5090 box).

## Usage (fresh box)
```bash
# 1. download bundle
curl -sL -o /workspace/llama.tar.gz \
  https://github.com/chiaqf/qwen3.8-27b-img/releases/download/v1.0.0/llama-server-portable.tar.gz
# 2. extract
cd /workspace && tar xzf llama.tar.gz     # -> llama-server + libggml*/libllama* .so, all in ./bundle? no: extracted to cwd
# 3. download model
export HF_TOKEN=...   # otherwise 16GB is slow
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_0.gguf --local-dir /workspace/models/Qwen3.8-27B-GGUF
# 4. serve (MTP)
LD_LIBRARY_PATH=/workspace:$LD_LIBRARY_PATH nohup /workspace/llama-server --model \
  /workspace/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 --n-gpu-layers 999 --parallel 1 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 3 --jinja \
  > /workspace/llama-server.log 2>&1 &
```

Note: the tarball was built with `tar czf -C bundle .` so files extract to the *current* directory.
Place them where they can resolve via `LD_LIBRARY_PATH` (self-referential rpath is NOT set).

## Verified (2026-08-15, RTX 5090, 595.71 driver)
- Prefill ~125 tok/s, decode ~127.5 tok/s
- MTP draft acceptance 0.72, mean draft len 3.17
- Round trip returned correct "PONG"

## Build recipe (if you need to rebuild)
```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
cmake -S /workspace/llama.cpp -B /workspace/llama.cpp/build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF
cmake --build /workspace/llama.cpp/build --config Release -j$(nproc) --target llama-server
mkdir -p bundle
cp -P /workspace/llama.cpp/build/bin/llama-server bundle/
cp -Pr /workspace/llama.cpp/build/bin/libggml*.so* /workspace/llama.cpp/build/bin/libllama*.so* \
  /workspace/llama.cpp/build/bin/libmtmd.so* bundle/
tar czf llama-server-portable.tar.gz -C bundle .
```
