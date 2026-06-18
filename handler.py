"""RunPod Serverless handler for LTX-2.3 image-to-video (official ltx-pipelines).

LAZY loading: only `import runpod` happens at module import, so the worker
reaches serverless.start() in seconds and becomes ready immediately (no risk of
being killed during a long import-time load). The heavy 53GB pipeline is loaded
on the FIRST job, inside try/except, and cached in a global. Any load/runtime
error is returned as the job result (visible via the /status API) instead of
crash-looping invisibly.

Models:
  - checkpoint: RunPod model cache (Model field = Lightricks/LTX-2.3-fp8)
  - gemma: baked into image at /models/gemma
  - upsampler: baked into image
"""
import os
import glob
import base64
import tempfile
import traceback

import runpod

CACHE = "/runpod-volume/huggingface-cache/hub"

PIPE = None
TILING = None


def _find_checkpoint(fname):
    """Find the checkpoint file ANYWHERE under the model cache, regardless of how
    RunPod named the repo dir (case / commit-hash variations)."""
    roots = [CACHE, "/runpod-volume/huggingface-cache", "/runpod-volume"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", fname), recursive=True)
        if hits:
            return hits[0]
        alt = [p for p in glob.glob(os.path.join(root, "**", "*.safetensors"), recursive=True)
               if "distilled" in os.path.basename(p).lower() and "fp8" in os.path.basename(p).lower()]
        if alt:
            return alt[0]
    listing = {r: os.listdir(r)[:30] for r in roots if os.path.isdir(r)}
    raise FileNotFoundError(
        f"checkpoint '{fname}' not in model cache. "
        f"Set endpoint Model field = Lightricks/LTX-2.3-fp8. Cache contents: {listing}")


def _ensure_loaded():
    """Load the pipeline once (on first job)."""
    global PIPE, TILING
    if PIPE is not None:
        return
    import torch
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode
    from ltx_core.model.video_vae import TilingConfig

    CKPT_FILE = os.environ.get("CKPT_FILE", "ltx-2.3-22b-distilled-fp8.safetensors")
    GEMMA_DIR = os.environ.get("GEMMA_DIR", "/models/gemma")
    UPSAMPLER_PATH = os.environ.get("UPSAMPLER_PATH", "/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    QUANT = os.environ.get("QUANT", "fp8-scaled-mm").strip()
    OFFLOAD = os.environ.get("OFFLOAD", "cpu").strip().lower()

    print(f"[init] cuda={torch.cuda.is_available()} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
    ckpt = _find_checkpoint(CKPT_FILE)
    if not os.path.isdir(GEMMA_DIR):
        raise FileNotFoundError(f"gemma dir not found: {GEMMA_DIR}")
    if not os.path.isfile(UPSAMPLER_PATH):
        raise FileNotFoundError(f"upsampler not found: {UPSAMPLER_PATH}")
    quant_policy = QuantizationKind(QUANT).to_policy(ckpt) if QUANT else None
    try:
        offload_mode = OffloadMode(OFFLOAD)
    except ValueError:
        offload_mode = OffloadMode.NONE
    print(f"[init] ckpt={ckpt} quant={QUANT} offload={offload_mode} loading...", flush=True)
    pipe = DistilledPipeline(
        distilled_checkpoint_path=ckpt,
        gemma_root=GEMMA_DIR,
        spatial_upsampler_path=UPSAMPLER_PATH,
        loras=[],
        quantization=quant_policy,
        offload_mode=offload_mode,
    )
    PIPE = pipe
    TILING = TilingConfig.default()
    print("[init] pipeline ready", flush=True)


def handler(job):
    try:
        _ensure_loaded()
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_core.model.video_vae import get_video_chunks_number

        inp = job.get("input", {}) or {}
        prompt = inp.get("prompt", "")
        height = int(inp.get("height", 768))
        width = int(inp.get("width", 1280))
        num_frames = int(inp.get("num_frames", 121))
        frame_rate = float(inp.get("frame_rate", 25))
        seed = int(inp.get("seed", 42))
        img_b64 = inp.get("image")
        if not img_b64:
            return {"error": "no input image (field 'image' base64 required)"}

        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "input.png")
            with open(img_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            out_path = os.path.join(td, "output.mp4")
            images = [ImageConditioningInput(path=img_path, frame_idx=0, strength=1.0)]
            video, audio = PIPE(
                prompt=prompt, seed=seed, height=height, width=width,
                num_frames=num_frames, frame_rate=frame_rate, images=images,
                tiling_config=TILING, enhance_prompt=False,
            )
            encode_video(video=video, fps=frame_rate, audio=audio, output_path=out_path,
                         video_chunks_number=get_video_chunks_number(num_frames, TILING))
            with open(out_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
        return {"video_base64": data, "filename": "output.mp4"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
