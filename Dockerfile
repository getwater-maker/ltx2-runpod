# LTX-2.3 image-to-video worker for RunPod Serverless
# Method #4 (Hugging Face model caching): the heavy checkpoint is NOT baked in.
# It is pulled by RunPod's model cache at boot (set Model = Lightricks/LTX-2.3-fp8
# on the endpoint). Only the small text encoder is baked in for reliability.
FROM runpod/worker-comfyui:5.8.6-base

# Tools we rely on at build time
RUN apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# LTX-2.3 nodes + example workflows (LTX-2.3 is also in ComfyUI core; this adds the
# official node pack so the workflow loads identically to the docs).
RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo \
        /comfyui/custom_nodes/ComfyUI-LTXVideo \
 && if [ -f /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt ]; then \
        (uv pip install -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt \
         || pip install -r /comfyui/custom_nodes/ComfyUI-LTXVideo/requirements.txt); \
    fi

# Bake the Gemma text encoder (~9.5 GB) under the exact filename the workflow
# expects. Public repo -> no Hugging Face token needed.
RUN mkdir -p /comfyui/models/text_encoders \
 && wget -q -O /comfyui/models/text_encoders/comfy_gemma_3_12B_it.safetensors \
      "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors" \
 && ls -lh /comfyui/models/text_encoders/

# Bake the distilled LoRA (~7.6 GB) at the path the workflow references
# (loras/ltxv/ltx2/...). Public repo -> no token needed.
RUN mkdir -p /comfyui/models/loras/ltxv/ltx2 \
 && wget -q -O "/comfyui/models/loras/ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1.safetensors" \
      "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384-1.1.safetensors" \
 && ls -lh /comfyui/models/loras/ltxv/ltx2/

# Runtime: symlink the model-cached checkpoint into ComfyUI, then start the worker.
COPY prestart.sh /prestart.sh
RUN chmod +x /prestart.sh
CMD ["/prestart.sh"]
