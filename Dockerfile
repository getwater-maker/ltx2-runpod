# LTX-2.3 image-to-video worker for RunPod Serverless (길 B: official ltx-pipelines, no ComfyUI)
# Heavy models come from RunPod model caching (HF cache). Only the small spatial
# upsampler is baked in. The distilled pipeline is self-contained (no separate LoRA).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# uv (manages its own Python 3.12 per the repo's requires-python + uv.lock)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Official LTX-2 inference package (ltx-core + ltx-pipelines)
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2
WORKDIR /app/LTX-2
RUN uv sync --frozen
# RunPod serverless SDK into the same venv
RUN uv pip install runpod

# Bake the small spatial upsampler (~1 GB, public, no token)
RUN mkdir -p /models \
 && wget -q -O /models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
      "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

# Bake the Gemma text encoder directory (~24 GB, public, no token). RunPod's Model
# caching field only takes ONE model (the checkpoint), so we bake Gemma here.
RUN uv pip install "huggingface_hub[hf_transfer]"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN /app/LTX-2/.venv/bin/huggingface-cli download unsloth/gemma-3-12b-it \
      --local-dir /models/gemma --exclude "*.gguf" "original/*" \
 && ls -lh /models/gemma

COPY handler.py /app/handler.py

# Model selection (overridable as endpoint env vars without rebuilding)
ENV CKPT_REPO=Lightricks/LTX-2.3-fp8 \
    CKPT_FILE=ltx-2.3-22b-distilled-fp8.safetensors \
    GEMMA_DIR=/models/gemma \
    UPSAMPLER_PATH=/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    QUANT=fp8-scaled-mm \
    OFFLOAD=cpu

CMD ["/app/LTX-2/.venv/bin/python", "/app/handler.py"]
