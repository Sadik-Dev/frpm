"""Synthesize missing perspectives of the identified person.

Upgrades pack *construction* itself: after the normal real-frame pipeline
selects the best real candidates, this optional stage recreates a highly
realistic image for each pose/expression category in
``selector.TARGET_CATEGORIES``. Two modes control which categories are
processed:

* ``missing`` (default) - only categories the source footage never
  captured well (e.g. no clean profile shot, no laughing frame).
* ``full`` - every target category, regardless of whether real footage
  already covers it, for a complete AI-reconstructed multi-perspective
  identity set (front/three-quarter/profile x neutral/smile/serious, plus
  laughing/talking/focused/confident etc.) seeded entirely from the
  person's own real photos.

Realism strategy: rather than generating from pure noise (a text-to-image
prompt alone), this uses **img2img** seeded from the person's own best-fit
*real* photo, combined with IP-Adapter Plus Face for identity conditioning
on that same photo. Starting from a real frame keeps skin tone, hair, and
general lighting continuity intact, and the (moderate, tunable) denoising
strength pushes the pose/expression toward the target category without
discarding the real photo entirely - this reads as far more photorealistic
and identity-consistent than pure txt2img.

Every synthesized image is written to its own ``synthesized/`` directory,
recorded in a manifest list (``synthesized_images``) completely separate
from real ``images``/``reference_pack``/``lora_dataset``, and tagged
``is_synthetic: true`` - real and AI-recreated content must never be
conflated.

Licensing: reuses ``generate.py``'s model choices (SD1.5 - CreativeML
OpenRAIL-M; IP-Adapter Plus Face - Apache-2.0), plus an optional GFPGAN
(Apache-2.0, TencentARC) face-restoration pass for extra sharpness/realism
(also 2x-upscales the result, since GFPGAN's restoration network operates
best when given upscaling headroom). CodeFormer was considered for the
same restoration step but is licensed "non-commercial purposes only"
(S-Lab License 1.0) and is therefore not used, consistent with FRPM's
existing licensing stance.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from . import captions as captions_mod
from .config import AppConfig
from .models import Candidate, SyntheticImageRecord
from .selector import MIN_CATEGORY_QUALITY, TARGET_CATEGORIES
from .utils import ensure_dir

logger = logging.getLogger("frpm.synthesis")

# Every category name the diversity selector targets, in the same order -
# used by "full" reconstruction mode to synthesize the complete set
# regardless of real coverage.
ALL_CATEGORY_NAMES: list[str] = [name for name, _ in TARGET_CATEGORIES]


# Explicit (pose_bucket_or_None, expression_tags) spec per selector.py
# category name - mirrors TARGET_CATEGORIES' own matcher definitions so the
# prompts we build stay consistent with what "having this category" means
# elsewhere in the pipeline.
CATEGORY_SPEC: dict[str, tuple[Optional[str], list[str]]] = {
    "front_neutral": ("front", ["neutral"]),
    "front_slight_smile": ("front", ["slight_smile"]),
    "front_serious": ("front", ["serious"]),
    "left_three_quarter_neutral": ("three_quarter_left", ["neutral"]),
    "left_three_quarter_smile": ("three_quarter_left", ["slight_smile"]),
    "left_three_quarter_serious": ("three_quarter_left", ["serious"]),
    "right_three_quarter_neutral": ("three_quarter_right", ["neutral"]),
    "right_three_quarter_smile": ("three_quarter_right", ["slight_smile"]),
    "right_three_quarter_serious": ("three_quarter_right", ["serious"]),
    "left_profile": ("profile_left", []),
    "right_profile": ("profile_right", []),
    "looking_down": ("looking_down", []),
    "looking_up": ("looking_up", []),
    "laughing": (None, ["laughing"]),
    "talking": (None, ["talking"]),
    "focused": (None, ["focused"]),
    "confident": (None, ["confident_smile"]),
}

POSE_PHRASES: dict[str, str] = {
    "front": "facing directly forward, front view",
    "three_quarter_left": "three-quarter view, face turned to the left",
    "three_quarter_right": "three-quarter view, face turned to the right",
    "profile_left": "side profile view, face turned fully to the left",
    "profile_right": "side profile view, face turned fully to the right",
    "looking_up": "looking upward, chin raised slightly",
    "looking_down": "looking downward, chin lowered slightly",
}

# Fallback search order of REAL poses to use as the img2img base image for
# each target pose, ordered by visual closeness (best first).
POSE_ADJACENCY: dict[str, list[str]] = {
    "front": ["front", "three_quarter_left", "three_quarter_right"],
    "three_quarter_left": ["three_quarter_left", "front", "profile_left"],
    "three_quarter_right": ["three_quarter_right", "front", "profile_right"],
    "profile_left": ["profile_left", "three_quarter_left", "front"],
    "profile_right": ["profile_right", "three_quarter_right", "front"],
    "looking_down": ["looking_down", "front"],
    "looking_up": ["looking_up", "front"],
}
# Expression-only categories (pose_key is None): prefer a clean, front-ish base.
_EXPRESSION_ONLY_ADJACENCY = ["front", "three_quarter_left", "three_quarter_right"]

# Per-category strength adjustment relative to cfg.synthesis_strength.
# Empirically, a pure text-prompt push (no ControlNet/pose conditioning) is
# too soft to reliably rotate a front-ish base image all the way to a full
# profile at typical strengths (~0.6) - verified by inspecting a real
# left_profile synthesis, which stayed noticeably front-of-profile. Larger
# target pose changes need a stronger push; expression-only changes (where
# preserving pose/identity fidelity matters most) need less.
#
# IMPORTANT: pushing strength too high (~0.8+) was *also* verified to cause
# the base SD1.5 checkpoint's own training-set biases to dominate over both
# the img2img base image and the IP-Adapter identity conditioning - in one
# verification run this produced an unrelated body-focused image instead of
# a face portrait. Boosts are therefore kept modest and capped well below
# that threshold; see ``REALISM_SUFFIX``/``DEFAULT_NEGATIVE_PROMPT`` below
# for the accompanying prompt-level guardrails against the same drift.
_STRENGTH_ADJUSTMENT: dict[str, float] = {
    "profile_left": 0.10,
    "profile_right": 0.10,
    "three_quarter_left": 0.05,
    "three_quarter_right": 0.05,
    "looking_up": 0.03,
    "looking_down": 0.03,
    "front": 0.0,
}
MAX_SYNTHESIS_STRENGTH = 0.75


def effective_synthesis_strength(category: str, base_strength: float) -> float:
    """Adjusts ``base_strength`` (from ``cfg.synthesis_strength``) based on
    how large a pose change ``category`` represents (see
    ``_STRENGTH_ADJUSTMENT`` above). Pure function - unit-testable."""
    pose_key, _expr_tags = CATEGORY_SPEC.get(category, (None, []))
    adjustment = _STRENGTH_ADJUSTMENT.get(pose_key, 0.0) if pose_key else 0.0
    return float(min(MAX_SYNTHESIS_STRENGTH, max(0.05, base_strength + adjustment)))


DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, anime, 3d render, cgi, "
    "deformed, disfigured, mutated, extra limbs, extra fingers, bad anatomy, "
    "blurry, low quality, worst quality, jpeg artifacts, watermark, text, logo, "
    "full body, wide shot, nsfw, nude, lingerie, swimsuit, revealing clothing, "
    "suggestive pose, multiple people, different person, duplicate face, "
    "cloned face, two faces, extra face, disembodied face"
)

# Explicitly anchors framing (head-and-shoulders portrait) on every prompt -
# without this, high-strength img2img runs were observed to drift toward
# generic full-body imagery unrelated to the source photo (see note above).
REALISM_SUFFIX = (
    "head and shoulders portrait photo, close-up, face clearly visible, "
    "photorealistic, natural lighting, sharp focus, realistic skin texture"
)

# SD1.5's UNet was trained at 512x512 and reliably produces a single
# coherent subject only near that resolution. Running img2img directly at
# FRPM's 1024x1024 face-crop resolution was verified to produce a visible
# "duplicate/ghost face" artifact at the frame edge (a well-known SD1.5
# failure mode above its native training resolution) *and* roughly 4x
# slower CPU inference for no quality benefit. Base images are therefore
# always downscaled to this working resolution before synthesis; the
# output is saved at this size rather than upscaled back (which would just
# add blur) - use --face-restore (GFPGAN) if a larger output is wanted, via
# its own upscale capability.
SYNTHESIS_WORKING_RESOLUTION = 512


def resize_for_synthesis(image, target: int = SYNTHESIS_WORKING_RESOLUTION):
    """Aspect-preserving resize of a PIL image so its longer side is
    ``target`` pixels - keeps the diffusion pipeline operating near SD1.5's
    native training resolution (see ``SYNTHESIS_WORKING_RESOLUTION`` above).
    Pure PIL operation - unit-testable without any ML model."""
    from PIL import Image as PILImage

    w, h = image.size
    if max(w, h) <= target:
        return image
    scale = target / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, PILImage.LANCZOS)



def _quality_of(c: Candidate) -> float:
    return c.quality.quality if c.quality else 0.0


def identify_missing_categories(
    candidates: list[Candidate], min_quality: float = MIN_CATEGORY_QUALITY
) -> list[str]:
    """Returns the ``TARGET_CATEGORIES`` names that have no real candidate
    at or above ``min_quality`` - i.e. gaps a synthesis pass should try to
    fill. Pure function, no ML - trivially unit-testable."""
    missing = []
    for name, matcher in TARGET_CATEGORIES:
        has_real = any(matcher(c) and _quality_of(c) >= min_quality for c in candidates)
        if not has_real:
            missing.append(name)
    return missing


def select_categories_to_synthesize(
    candidates: list[Candidate], mode: str = "missing", min_quality: float = MIN_CATEGORY_QUALITY
) -> list[str]:
    """Decides which categories a synthesis run should process.

    ``mode="missing"`` (default): only gaps the real footage didn't cover -
    see ``identify_missing_categories``.
    ``mode="full"``: every category in ``ALL_CATEGORY_NAMES``, regardless
    of existing real coverage - for a complete AI-reconstructed
    multi-perspective identity set. Pure function, no ML - unit-testable.
    """
    if mode == "full":
        return list(ALL_CATEGORY_NAMES)
    return identify_missing_categories(candidates, min_quality=min_quality)


def resolve_face_restore(explicit: Optional[bool], mode: str) -> bool:
    """Decides the effective face-restoration setting.

    An explicit ``--face-restore``/``--no-face-restore`` always wins. With
    neither given (``explicit=None``), face restoration is enabled
    automatically for ``mode="full"`` (a complete reconstruction pack
    benefits from the extra sharpness/realism by default) and left off for
    ``mode="missing"`` (matching prior default behavior). Pure function -
    unit-testable.
    """
    if explicit is not None:
        return explicit
    return mode == "full"


def pick_base_candidate(candidates: list[Candidate], category: str) -> Optional[Candidate]:
    """Picks the best real candidate to use as the img2img/identity base
    image for recreating ``category``, preferring poses visually closest to
    the target (see ``POSE_ADJACENCY``). Falls back to the single
    best-quality candidate overall if nothing in the preferred poses exists.
    Pure function, no ML - unit-testable with synthetic candidates."""
    scored = [c for c in candidates if c.quality is not None and c.pose is not None]
    if not scored:
        return None

    pose_key, _expr_tags = CATEGORY_SPEC.get(category, (None, []))
    adjacency = POSE_ADJACENCY.get(pose_key, _EXPRESSION_ONLY_ADJACENCY) if pose_key else _EXPRESSION_ONLY_ADJACENCY

    for pose_name in adjacency:
        matches = [c for c in scored if c.pose.pose.value == pose_name]
        if matches:
            return max(matches, key=_quality_of)

    return max(scored, key=_quality_of)


def build_synthesis_prompt(category: str) -> str:
    """Builds a text prompt describing ``category`` for img2img guidance.
    Pure function, no ML - unit-testable."""
    pose_key, expr_tags = CATEGORY_SPEC.get(category, (None, []))
    parts: list[str] = []
    if pose_key:
        parts.append(POSE_PHRASES[pose_key])
    for tag in expr_tags:
        phrase = captions_mod.EXPRESSION_PHRASES.get(tag)
        if phrase:
            parts.append(phrase)
    parts.append(REALISM_SUFFIX)
    return ", ".join(parts)


# --------------------------------------------------------------------------
# Heavy ML: actual image synthesis (requires the `generate` extra).
# --------------------------------------------------------------------------

def _patch_torchvision_functional_tensor() -> None:
    """Compatibility shim for GFPGAN's dependency chain.

    ``basicsr`` (required by ``gfpgan``) imports
    ``torchvision.transforms.functional_tensor``, which was removed in
    torchvision >= 0.17 (its functions moved to
    ``torchvision.transforms.functional``). Rather than pin FRPM to an old
    torchvision just for this optional face-restoration feature, register a
    shim module pointing at the new location. Inert if the old module still
    exists (older torchvision) or if torchvision itself is unavailable.
    """
    import sys

    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return  # still present natively - nothing to patch
    except ModuleNotFoundError:
        pass
    try:
        import torchvision.transforms.functional as _F

        sys.modules["torchvision.transforms.functional_tensor"] = _F
    except ImportError:
        pass


class _SynthesisPipeline:
    """Lazily-loaded, reused-across-calls SD1.5 img2img + IP-Adapter Plus
    Face pipeline, so synthesizing several missing categories in one run
    only pays the (~seconds-to-minutes) model-loading cost once."""

    # GFPGAN's restoration network was trained to also upscale by 2x, and
    # produces sharper output when allowed to do so rather than being
    # forced back down to the original (already-downscaled-to-512, see
    # SYNTHESIS_WORKING_RESOLUTION) size. Background pixels are upscaled
    # with a plain resize (bg_upsampler=None keeps that cheap); only the
    # detected face region gets GFPGAN's actual restoration network.
    FACE_RESTORE_UPSCALE = 2

    def __init__(self, ip_adapter_scale: float = 0.7):
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

        from .generate import IP_ADAPTER_REPO, IP_ADAPTER_WEIGHT, SD15_MODEL_ID

        torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))

        logger.info("Loading Stable Diffusion 1.5 img2img (%s)...", SD15_MODEL_ID)
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD15_MODEL_ID, torch_dtype=torch.float32, safety_checker=None
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        logger.info("Loading IP-Adapter Plus Face for identity conditioning...")
        self.pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_WEIGHT)
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)
        self._torch = torch
        self._face_restorer = None

    def _get_face_restorer(self):
        if self._face_restorer is not None:
            return self._face_restorer
        try:
            _patch_torchvision_functional_tensor()
            from gfpgan import GFPGANer

            # Apache-2.0 licensed weights, official TencentARC release URL;
            # GFPGANer downloads+caches this itself on first use.
            model_url = (
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
            )
            self._face_restorer = GFPGANer(
                model_path=model_url, upscale=self.FACE_RESTORE_UPSCALE, arch="clean",
                channel_multiplier=2, bg_upsampler=None,
            )
        except Exception as exc:
            logger.warning("Could not load GFPGAN face-restoration model (%s) - skipping restoration pass.", exc)
            self._face_restorer = False
        return self._face_restorer

    def restore_face(self, pil_image):
        """Runs an optional GFPGAN sharpening/realism pass. Returns the
        input image unchanged if GFPGAN isn't available/fails - this is a
        best-effort enhancement, never a hard requirement."""
        import cv2
        import numpy as np

        restorer = self._get_face_restorer()
        if not restorer:
            return pil_image
        try:
            bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            _, _, restored_bgr = restorer.enhance(bgr, has_aligned=False, only_center_face=True, paste_back=True)
            from PIL import Image

            return Image.fromarray(cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB))
        except Exception as exc:
            logger.warning("GFPGAN restoration pass failed (%s) - using un-restored image.", exc)
            return pil_image

    def synthesize(
        self,
        base_image,
        prompt: str,
        negative_prompt: str,
        strength: float,
        steps: int,
        guidance_scale: float,
        seed: Optional[int],
        apply_face_restore: bool,
    ):
        generator = self._torch.Generator("cpu").manual_seed(seed) if seed is not None else None
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            ip_adapter_image=base_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        if apply_face_restore:
            image = self.restore_face(image)
        return image


def run_synthesis_stage(
    candidates: list[Candidate],
    cfg: AppConfig,
    out_dir: Path,
    ensure_base_image_path: Callable[[Candidate], Optional[str]],
) -> list[SyntheticImageRecord]:
    """Selects categories per ``cfg.synthesis_mode`` (gaps only, or the
    full target set), recreates up to ``cfg.synthesis_count`` images for
    each, and writes them under ``out_dir/synthesized/``.

    ``ensure_base_image_path`` is a callback (provided by the pipeline)
    that returns the on-disk path to a candidate's face crop, generating it
    on demand if the candidate wasn't already part of the real selected
    pack - this stage never re-implements crop generation itself.
    """
    from PIL import Image

    categories = select_categories_to_synthesize(candidates, cfg.synthesis_mode)
    if not categories:
        logger.info("No missing pose/expression categories detected - nothing to synthesize.")
        return []

    if cfg.synthesis_mode == "full":
        logger.info(
            "Full reconstruction mode: recreating all %d target perspectives "
            "(front/three-quarter/profile x expressions), regardless of existing real coverage.",
            len(categories),
        )
    else:
        logger.info("%d categories missing from real footage - synthesizing to fill the gaps.", len(categories))

    if len(categories) > cfg.synthesis_max_categories:
        logger.info(
            "%d categories selected; capping synthesis to the first %d (--synthesis-max-categories) "
            "to bound CPU time. Skipped: %s",
            len(categories), cfg.synthesis_max_categories, ", ".join(categories[cfg.synthesis_max_categories:]),
        )
        categories = categories[: cfg.synthesis_max_categories]
    logger.info("Synthesizing %d categories: %s", len(categories), ", ".join(categories))

    synth_dir = ensure_dir(out_dir / "synthesized")
    pipeline = _SynthesisPipeline(ip_adapter_scale=cfg.synthesis_ip_adapter_scale)

    records: list[SyntheticImageRecord] = []
    for category in categories:
        base = pick_base_candidate(candidates, category)
        if base is None:
            logger.warning("No usable real candidate found to base '%s' on - skipping.", category)
            continue
        base_path = ensure_base_image_path(base)
        if not base_path or not Path(base_path).exists():
            logger.warning("Base image for '%s' not found on disk - skipping.", category)
            continue

        prompt = build_synthesis_prompt(category)
        strength = effective_synthesis_strength(category, cfg.synthesis_strength)
        base_image = resize_for_synthesis(Image.open(base_path).convert("RGB"))
        cat_dir = ensure_dir(synth_dir / category)

        for i in range(max(1, cfg.synthesis_count)):
            out_path = cat_dir / f"synth_{i:02d}.jpg"
            logger.info(
                "Synthesizing '%s' (%d/%d, strength=%.2f, steps=%d)...",
                category, i + 1, cfg.synthesis_count, strength, cfg.synthesis_steps,
            )
            start = time.time()
            try:
                image = pipeline.synthesize(
                    base_image=base_image,
                    prompt=prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    strength=strength,
                    steps=cfg.synthesis_steps,
                    guidance_scale=7.0,
                    seed=None,
                    apply_face_restore=cfg.face_restore,
                )
            except Exception as exc:
                logger.warning("Synthesis failed for '%s' (%d/%d): %s", category, i + 1, cfg.synthesis_count, exc)
                continue
            image.save(out_path, quality=95)
            logger.info("Saved %s in %.1fs.", out_path, time.time() - start)

            records.append(SyntheticImageRecord(
                file=str(out_path.relative_to(out_dir)).replace("\\", "/"),
                category=category,
                source_real_image=str(Path(base_path).relative_to(out_dir)).replace("\\", "/")
                if Path(base_path).is_relative_to(out_dir) else str(base_path),
                prompt=prompt,
                method="sd1.5-img2img+ip-adapter-plus-face" + ("+gfpgan" if cfg.face_restore else ""),
            ))

    return records
