# LTX-2.3 I2V worker for RunPod Serverless (Hugging Face model caching)

Custom [worker-comfyui](https://github.com/runpod-workers/worker-comfyui) image that runs
**LTX-2.3** image-to-video on RunPod Serverless using the **model caching** ("Hugging
Face cache") deployment method.

## How models are provided
- **Checkpoint** (`Lightricks/LTX-2.3-fp8`, ~27 GB) → **NOT** baked in. RunPod's model
  cache downloads it at boot. Add `Lightricks/LTX-2.3-fp8` in the endpoint's **Model**
  section. `prestart.sh` symlinks it into ComfyUI at runtime.
- **Text encoder** (`gemma_3_12B_it_fp4_mixed.safetensors`) → baked into the image as
  `comfy_gemma_3_12B_it.safetensors` (small, public, deterministic).
- No Hugging Face token required — all repos are public.

## Deploy
1. Push this folder to a GitHub repo.
2. RunPod → Serverless → New Endpoint → **Import Git Repository** → select this repo.
3. **Model** section: add `Lightricks/LTX-2.3-fp8`.
4. GPU: 48 GB (e.g. L40S / A6000 / A40).
5. Build & deploy.

The companion client tools (`send_to_runpod.py`, `영상만들기.bat`) live in the parent
project folder and call this endpoint.
