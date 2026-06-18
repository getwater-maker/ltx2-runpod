# LTX-2.3 image-to-video worker for RunPod Serverless (official ltx-pipelines, no ComfyUI)
# Checkpoint comes from RunPod model caching (Model field = Lightricks/LTX-2.3-fp8).
# Gemma + upsampler are baked in. Python pinned to 3.12 (3.14 breaks multiprocessing).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ---- Baked models FIRST, independent of the app venv (downloaded via ephemeral uvx)
# so that later venv/handler changes do NOT invalidate these big cached layers. ----
RUN mkdir -p /models \
 && wget -q -O /models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
      "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# Gemma text encoder (~24GB) — each ~5GB shard in its own layer (parallel pull + resumable)
RUN uvx --from "huggingface_hub[hf_transfer]" huggingface-cli download unsloth/gemma-3-12b-it --include "*.json" "*.model" "*.jinja" --local-dir /models/gemma
RUN uvx --from "huggingface_hub[hf_transfer]" huggingface-cli download unsloth/gemma-3-12b-it --include "model-00001-of-00005.safetensors" --local-dir /models/gemma
RUN uvx --from "huggingface_hub[hf_transfer]" huggingface-cli download unsloth/gemma-3-12b-it --include "model-00002-of-00005.safetensors" --local-dir /models/gemma
RUN uvx --from "huggingface_hub[hf_transfer]" huggingface-cli download unsloth/gemma-3-12b-it --include "model-00003-of-00005.safetensors" --local-dir /models/gemma
RUN uvx --from "huggingface_hub[hf_transfer]" huggingface-cli download unsloth/gemma-3-12b-it --include "model-00004-of-00005.safetensors" --local-dir /models/gemma
RUN uvx --from "huggingface_hub[hf_transfer]" huggingface-cli download unsloth/gemma-3-12b-it --include "model-00005-of-00005.safetensors" --local-dir /models/gemma

# ---- Official LTX-2 package, pinned to Python 3.12 (3.14's forkserver breaks loading) ----
ENV UV_PYTHON=3.12
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2
WORKDIR /app/LTX-2
RUN uv python install 3.12 \
 && uv venv --python 3.12 \
 && uv sync --frozen \
 && uv pip install runpod \
 && /app/LTX-2/.venv/bin/python --version

COPY handler.py /app/handler.py

ENV CKPT_FILE=ltx-2.3-22b-distilled-fp8.safetensors \
    GEMMA_DIR=/models/gemma \
    UPSAMPLER_PATH=/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    QUANT=fp8-scaled-mm \
    OFFLOAD=cpu

CMD ["/app/LTX-2/.venv/bin/python", "/app/handler.py"]
