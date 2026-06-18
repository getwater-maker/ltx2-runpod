# LTX-2.3 image-to-video worker for RunPod Serverless (official ltx-pipelines)

Runs **LTX-2.3** image-to-video on RunPod Serverless using Lightricks' official
[LTX-2 inference package](https://github.com/Lightricks/LTX-2) — **no ComfyUI**.
The distilled two-stage pipeline is loaded once per worker and reused.

## Models (via RunPod model caching / HF cache)
Add these in the endpoint's **Model** section:
- `Lightricks/LTX-2.3-fp8` — checkpoint (public)
- `google/gemma-3-12b-it-qat-q4_0-unquantized` — text encoder (**gated**: accept the
  license on its HF page + set an HF token on the endpoint)

The spatial upsampler (~1 GB) is baked into the image.

## Request format
```json
{ "input": {
    "image": "<base64 image>",
    "prompt": "a gentle camera push-in, natural motion",
    "height": 768, "width": 1280,
    "num_frames": 121, "frame_rate": 25, "seed": 42
} }
```
Returns `{ "video_base64": "...", "filename": "output.mp4" }`.

## GPU
Use an **fp8-capable 48 GB GPU (Ada/Hopper, e.g. L40S)**. On Ampere (A40/A6000) fp8
matmul is unsupported — set `QUANT=fp8-cast` or use a larger GPU instead.

Env vars `CKPT_REPO`, `CKPT_FILE`, `GEMMA_REPO`, `QUANT`, `UPSAMPLER_PATH` override
defaults without rebuilding.
