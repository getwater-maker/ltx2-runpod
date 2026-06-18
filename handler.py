"""RunPod Serverless handler for LTX-2.3 image-to-video using the official
ltx-pipelines DistilledPipeline. The model is loaded ONCE at worker start and
reused for every request (good for batch jobs).

Models are resolved from RunPod's model cache (HF cache) at:
  /runpod-volume/huggingface-cache/hub/models--<org>--<repo>/snapshots/<hash>/
Add these in the endpoint's Model section:
  - Lightricks/LTX-2.3-fp8                       (checkpoint, public)
  - google/gemma-3-12b-it-qat-q4_0-unquantized   (text encoder, GATED -> needs HF token)
"""
import os
import glob
import base64
import tempfile
import traceback

import runpod

CACHE = "/runpod-volume/huggingface-cache/hub"


def _snapshot_dir(repo_id: str) -> str:
    base = os.path.join(CACHE, "models--" + repo_id.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(base, "*")))
    if not snaps:
        raise FileNotFoundError(
            f"Model cache missing for '{repo_id}'. Add it in the endpoint's Model section "
            f"(gated models also need an HF token)."
        )
    return snaps[-1]


def _find_file(root: str, name: str) -> str:
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"'{name}' not found under {root}")
    return hits[0]


# ---- one-time model load at worker startup -------------------------------
CKPT_REPO = os.environ.get("CKPT_REPO", "Lightricks/LTX-2.3-fp8")
CKPT_FILE = os.environ.get("CKPT_FILE", "ltx-2.3-22b-distilled-fp8.safetensors")
GEMMA_REPO = os.environ.get("GEMMA_REPO", "google/gemma-3-12b-it-qat-q4_0-unquantized")
UPSAMPLER_PATH = os.environ.get("UPSAMPLER_PATH", "/models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
QUANT = os.environ.get("QUANT", "fp8-scaled-mm").strip()

print("[init] importing ltx-pipelines...", flush=True)
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

print("[init] resolving model paths from cache...", flush=True)
CKPT = _find_file(_snapshot_dir(CKPT_REPO), CKPT_FILE)
GEMMA_DIR = _snapshot_dir(GEMMA_REPO)
QUANT_POLICY = QuantizationKind(QUANT).to_policy(CKPT) if QUANT else None
print(f"[init] ckpt      = {CKPT}", flush=True)
print(f"[init] gemma     = {GEMMA_DIR}", flush=True)
print(f"[init] upsampler = {UPSAMPLER_PATH}", flush=True)
print(f"[init] quant     = {QUANT or 'none'}", flush=True)

print("[init] loading DistilledPipeline (one-time, may take a minute)...", flush=True)
PIPE = DistilledPipeline(
    distilled_checkpoint_path=CKPT,
    gemma_root=GEMMA_DIR,
    spatial_upsampler_path=UPSAMPLER_PATH,
    loras=[],
    quantization=QUANT_POLICY,
)
TILING = TilingConfig.default()
print("[init] pipeline ready.", flush=True)


def handler(job):
    try:
        inp = job.get("input", {}) or {}
        prompt = inp.get("prompt", "")
        height = int(inp.get("height", 768))
        width = int(inp.get("width", 1280))
        num_frames = int(inp.get("num_frames", 121))   # (8*K)+1
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
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
                tiling_config=TILING,
                enhance_prompt=False,
            )
            encode_video(
                video=video,
                fps=frame_rate,
                audio=audio,
                output_path=out_path,
                video_chunks_number=get_video_chunks_number(num_frames, TILING),
            )
            with open(out_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
        return {"video_base64": data, "filename": "output.mp4"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
