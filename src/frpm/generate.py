"""Optional image-generation module: produces a new, realistic image of the
identity captured in a FRPM reference pack, using a user-supplied scene
description.

This is intentionally separate from the core reference-pack pipeline (see
``pyproject.toml``'s ``generate`` extra) since it pulls in a much heavier,
independent dependency stack (torch, diffusers, transformers) that most
FRPM users - who only want the curated reference pack itself, to hand off to
an external image-generation service - do not need.

Model choice & licensing
-------------------------
Uses Stable Diffusion 1.5 (``stable-diffusion-v1-5/stable-diffusion-v1-5``,
CreativeML OpenRAIL-M - permissive for personal/commercial use, standard
behavioral-use restrictions only) together with IP-Adapter Plus Face
(``h94/IP-Adapter``, Apache-2.0), which conditions generation on a cropped
face image via a CLIP vision encoder.

This deliberately avoids InstantID / PuLID / IP-Adapter-FaceID, even though
they generally produce a tighter identity lock: all three depend on
InsightFace's face-recognition backbone (antelopev2/buffalo_l), whose
pretrained weights are licensed "non-commercial research purposes only" -
the exact class of restrictive license FRPM's face-embedding model choice
(OpenCV SFace) was chosen to avoid. IP-Adapter Plus Face uses a plain CLIP
image encoder instead, with no such restriction, at the cost of a looser
identity match than InstantID/PuLID would give.

Runs on CPU only (no GPU was detected on this machine); this is
substantially slower than GPU inference (potentially many minutes per
image) but requires no extra hardware.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("frpm.generate")

SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter-plus-face_sd15.bin"

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, anime, 3d render, cgi, "
    "deformed, disfigured, mutated, extra limbs, extra fingers, bad anatomy, "
    "blurry, low quality, worst quality, jpeg artifacts, watermark, text, logo"
)


def generate_identity_image(
    face_reference_path: str | Path,
    prompt: str,
    output_path: str | Path,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    width: int = 512,
    height: int = 768,
    steps: int = 28,
    guidance_scale: float = 7.0,
    ip_adapter_scale: float = 0.7,
    seed: Optional[int] = None,
) -> Path:
    """Generates a new image of the person shown in ``face_reference_path``,
    matching ``prompt``, using SD1.5 + IP-Adapter Plus Face. Runs entirely
    locally on CPU. Returns the path the image was saved to.

    Requires the ``generate`` extra (``pip install -e ".[generate]"``).
    """
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    from PIL import Image

    torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))

    face_reference_path = Path(face_reference_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    face_image = Image.open(face_reference_path).convert("RGB")

    logger.info("Loading Stable Diffusion 1.5 (%s) - first run downloads ~2GB, cached afterwards...", SD15_MODEL_ID)
    pipe = StableDiffusionPipeline.from_pretrained(
        SD15_MODEL_ID, torch_dtype=torch.float32, safety_checker=None, variant="fp16"
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    logger.info("Loading IP-Adapter Plus Face (%s) for identity conditioning...", IP_ADAPTER_WEIGHT)
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_WEIGHT)
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None

    logger.info(
        "Generating %dx%d image, %d steps on CPU - this will likely take several minutes...",
        width, height, steps,
    )
    start = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        ip_adapter_image=face_image,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    elapsed = time.time() - start
    logger.info("Generation finished in %.1fs.", elapsed)

    result.images[0].save(output_path)
    logger.info("Saved generated image to %s", output_path)
    return output_path


def _cli() -> None:
    import argparse

    from .utils import setup_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", required=True, help="Path to a face reference crop (e.g. from selected/faces/)")
    parser.add_argument("--prompt", required=True, help="Scene/pose/clothing description")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    setup_logging("INFO")
    generate_identity_image(
        face_reference_path=args.face,
        prompt=args.prompt,
        output_path=args.output,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        ip_adapter_scale=args.ip_adapter_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
