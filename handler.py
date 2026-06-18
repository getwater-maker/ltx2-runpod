"""RunPod Serverless handler for LTX-2.3 image-to-video (official ltx-pipelines).

Robust startup: the heavy pipeline is loaded once at import, but inside a
try/except. If loading fails, the worker does NOT crash-loop — it still starts
the serverless loop and returns the captured error/traceback as the job result,
so the failure is visible via the /status API instead of an invisible restart.

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
INIT_ERROR = None
INIT_LOG = []


def _log(msg):
    print(msg, flush=True)
    INIT_LOG.append(msg)


def _find_checkpoint(fname):
    """Find the checkpoint file ANYWHERE under the model cache, regardless of how
    RunPod named the repo dir (case / commit-hash variations). Falls back to any
    distilled-fp8 .safetensors. Raises with a full cache listing on failure."""
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
    listing = {}
    for root in roots:
        if os.path.isdir(root):
            listing[root] = os.listdir(root)[:30]
    raise FileNotFoundError(
        f"checkpoint '{fname}' not found in model cache. "
        f"Make sure the endpoint Model field = Lightricks/LTX-2.3-fp8. Cache contents: {listing}")


# ---- one-time load at import, errors captured (no crash loop) -------------
try:
    _log("[init] importing torch / ltx-pipelines ...")
    import torch
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.quantization_factory import QuantizationKind
    from ltx_pipelines.utils.types import OffloadMode
    from ltx_core.model.video_vae import TilingConfig

    CKPT_REPO = os.environ.get("CKPT_REPO", "Lightricks/LTX-2.3-fp8")
    CKPT_FILE = os.environ.get("CKPT_FILE", "ltx-2.3-22b-distilled-fp8.safetensors")
    GEMMA_DIR = os.environ.get("GEMMA_DIR", "/models/gemma")
    UPSAMPLER_PATH = os.environ.get("UPSAMPLER_PATH", "/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    QUANT = os.environ.get("QUANT", "fp8-scaled-mm").strip()
    OFFLOAD = os.environ.get("OFFLOAD", "cpu").strip().lower()

    _log(f"[init] cuda available={torch.cuda.is_available()} "
         f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    CKPT = _find_checkpoint(CKPT_FILE)
    if not os.path.isdir(GEMMA_DIR):
        raise FileNotFoundError(f"gemma dir not found: {GEMMA_DIR}")
    if not os.path.isfile(UPSAMPLER_PATH):
        raise FileNotFoundError(f"upsampler not found: {UPSAMPLER_PATH}")
    quant_policy = QuantizationKind(QUANT).to_policy(CKPT) if QUANT else None
    try:
        offload_mode = OffloadMode(OFFLOAD)
    except ValueError:
        offload_mode = OffloadMode.NONE
    _log(f"[init] ckpt={CKPT}")
    _log(f"[init] gemma={GEMMA_DIR} upsampler={UPSAMPLER_PATH} quant={QUANT} offload={offload_mode}")
    _log("[init] loading DistilledPipeline (may take minutes)...")
    PIPE = DistilledPipeline(
        distilled_checkpoint_path=CKPT,
        gemma_root=GEMMA_DIR,
        spatial_upsampler_path=UPSAMPLER_PATH,
        loras=[],
        quantization=quant_policy,
        offload_mode=offload_mode,
    )
    TILING = TilingConfig.default()
    _log("[init] pipeline ready")
except Exception:
    INIT_ERROR = traceback.format_exc()
    print("[init] FAILED:\n" + INIT_ERROR, flush=True)


def handler(job):
    if PIPE is None:
        return {"error": "pipeline failed to load at startup", "trace": INIT_ERROR, "init_log": INIT_LOG}
    try:
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
