#!/usr/bin/env bash
# Runtime hook: link the RunPod model-cached LTX-2.3 checkpoint into ComfyUI's
# checkpoints folder, then hand off to the normal worker-comfyui startup.
# RunPod's model cache lands at /runpod-volume/huggingface-cache/hub/ at boot,
# in standard Hugging Face cache layout. We resolve the snapshot dir (its name
# contains a hash that is unknown until runtime) and symlink the *.safetensors
# files into /comfyui/models/checkpoints so ComfyUI can see them by name.
set -u

CKPT_DIR=/comfyui/models/checkpoints
CACHE=/runpod-volume/huggingface-cache/hub
REPO_DIR="$CACHE/models--Lightricks--LTX-2.3-fp8"

mkdir -p "$CKPT_DIR"

echo "[prestart] looking for cached checkpoint under: $REPO_DIR"
if [ -d "$REPO_DIR/snapshots" ]; then
    SNAP="$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | head -n1)"
    if [ -n "${SNAP:-}" ]; then
        echo "[prestart] using snapshot: $SNAP"
        for f in "$SNAP"*.safetensors; do
            [ -e "$f" ] || continue
            ln -sf "$f" "$CKPT_DIR/$(basename "$f")"
            echo "[prestart]   linked $(basename "$f")"
        done
    else
        echo "[prestart] WARNING: no snapshot directory found inside $REPO_DIR"
    fi
else
    echo "[prestart] WARNING: model cache not found."
    echo "[prestart] -> On the endpoint, add  Lightricks/LTX-2.3-fp8  in the Model section."
fi

echo "[prestart] checkpoints folder now contains:"
ls -la "$CKPT_DIR" 2>/dev/null || true

# Hand off to the original worker-comfyui entrypoint.
exec /start.sh
