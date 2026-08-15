#!/bin/bash
# Qwen 27B fast-path setup for a fresh vast.ai RTX 5090 (pytorch-cuda-13 template).
# Run ONCE via:  ssh root@HOST 'bash -s' < qwen-fastpath-setup.sh
# It is IDEMPOTENT and self-contained: bundle + hf CLI + model download + serve.
# Always run this in the BACKGROUND of the ssh session (redirect to a log) and
# check the log for DONE, never poll a live tty that may die.
set -u
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
HF_TOKEN="${HF_TOKEN:-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}"
URL="https://github.com/chiaqf/qwen3.8-27b-img/releases/download/v1.0.0/llama-server-portable.tar.gz"
GGUF="unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf"
LOG=/root/qwen-setup.log
exec > "$LOG" 2>&1   # swallow session-cutoff: everything persists in the log
echo "=== $(date -u +%H:%M:%S) setup start ==="

# 1. prebuilt bundle (replaces 9-min build)
mkdir -p /root/qwen-bundle /root/models
if [ ! -x /root/qwen-bundle/llama-server ]; then
  curl -sL -o /root/qwen-bundle/llama.tar.gz "$URL"
  tar xzf /root/qwen-bundle/llama.tar.gz -C /root/qwen-bundle && rm -f /root/qwen-bundle/llama.tar.gz
  echo "bundle: $(ls -la /root/qwen-bundle/llama-server | awk '{print $5}') bytes"
fi

# 2. hf CLI (write script to disk then run, NOT curl|bash — cutoff-safe)
if [ ! -x /root/.local/bin/hf ]; then
  curl -LsSf https://hf.co/cli/install.sh -o /root/hf-install.sh
  bash /root/hf-install.sh
  echo "hf bin: $(ls /root/.local/bin/hf 2>/dev/null)"
fi

# 3. model download (16GB, ~2min with token). No nohup needed: we're already detached.
if [ ! -f /root/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf ]; then
  HF_TOKEN="$HF_TOKEN" /root/.local/bin/hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_0.gguf --local-dir /root/models/Qwen3.8-27B-GGUF
fi
echo "model: $(stat -c %s /root/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf) bytes"

# 4. serve (256k ctx, MTP). Use pkill -x (NOT -f: -f matches this script's own argv).
pkill -x llama-server 2>/dev/null; sleep 1
LD_LIBRARY_PATH=/root/qwen-bundle nohup /root/qwen-bundle/llama-server \
  --model /root/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf \
  --host "$HOST" --port "$PORT" --ctx-size 262144 --n-gpu-layers 999 --parallel 1 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3 --jinja > /root/llama-server.log 2>&1 &

# 5. wait for health + MTP
for i in $(seq 1 60); do
  sleep 3
  H=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/health" 2>/dev/null)
  M=$(grep -c "creating MTP draft" /root/llama-server.log 2>/dev/null)
  echo "poll $i: health=$H mtp=$M"
  [ "$H" = "200" ] && [ "$M" -ge 1 ] && break
done
echo "=== $(date -u +%H:%M:%S) SETUP_DONE health=$H mtp=$M ==="
