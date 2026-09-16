"""Command-line interface."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from . import __version__
from .config import (
    DEFAULT_FPS,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_VIDEO_HEIGHT,
    DEFAULT_PACK_SIZE,
    DEFAULT_REFERENCE_PACK_SIZE,
    DEFAULT_TRIGGER_TOKEN,
    MAX_PACK_SIZE,
    MIN_PACK_SIZE,
    AppConfig,
)
from .errors import FRPMError
from .hardware import detect_hardware
from .models import IdentityClusterInfo
from .pipeline import Pipeline
from .utils import setup_logging

console = Console()
logger = logging.getLogger("frpm.cli")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frpm",
        description="Turn a YouTube video into a curated facial identity/expression reference pack "
        "for AI image-generation workflows (FLUX/PuLID/InstantID/LoRA/character sheets).",
    )
    p.add_argument("url", help="YouTube video URL")
    p.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help=f"Sampling rate in frames/sec (default: {DEFAULT_FPS})")
    p.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help=f"Hard cap on sampled frames, however long the video is (default: {DEFAULT_MAX_FRAMES})",
    )
    p.add_argument("--max-duration", type=float, default=None, help="Only process this many seconds of the video")
    p.add_argument("--start", type=float, default=0.0, help="Start timestamp in seconds (default: 0)")
    p.add_argument("--end", type=float, default=None, help="End timestamp in seconds")
    p.add_argument("--person", type=int, default=None, help="Pick identity N (1-based) instead of prompting interactively")
    p.add_argument(
        "--reference-image", default=None,
        help="Path to a photo of the target person's face; automatically selects the identity "
        "cluster that best matches it instead of prompting or auto-picking the dominant person",
    )
    p.add_argument(
        "--pack-size", type=int, default=DEFAULT_PACK_SIZE,
        help=f"Final pack size, {MIN_PACK_SIZE}-{MAX_PACK_SIZE} (default: {DEFAULT_PACK_SIZE})",
    )
    p.add_argument("--reference-pack-size", type=int, default=DEFAULT_REFERENCE_PACK_SIZE, help="Premium reference pack size, 12-20 (default: 16)")
    p.add_argument("--keep-video", action="store_true", help="Keep the downloaded source video instead of deleting it")
    p.add_argument("--keep-candidates", action="store_true", help="Keep all sampled candidate frames instead of pruning them")
    p.add_argument("--lora-mode", action="store_true", help="Also generate a lora_dataset/ with matching .txt caption files")
    p.add_argument("--trigger-token", default=DEFAULT_TRIGGER_TOKEN, help=f"Unique LoRA trigger token (default: {DEFAULT_TRIGGER_TOKEN})")

    p.add_argument(
        "--synthesize-missing", action="store_true",
        help="Recreate realistic images for pose/expression categories the source video didn't capture "
        "(img2img + IP-Adapter Plus Face, seeded from the person's own best real photo). "
        "Requires the 'generate' extra: pip install -e \".[generate]\"",
    )
    p.add_argument("--synthesis-count", type=int, default=1, help="How many images to synthesize per missing category (default: 1)")
    p.add_argument("--synthesis-strength", type=float, default=0.6, help="img2img denoising strength, 0-1 (default: 0.6; higher = more change from the base photo)")
    p.add_argument("--synthesis-max-categories", type=int, default=6, help="Cap on how many missing categories to synthesize per run, to bound CPU time (default: 6)")
    p.add_argument("--face-restore", action="store_true", help="Apply a GFPGAN face-restoration pass to synthesized images for extra sharpness/realism")

    report_group = p.add_mutually_exclusive_group()
    report_group.add_argument("--generate-report", dest="generate_report", action="store_true", default=True, help="Generate report.html (default: on)")
    report_group.add_argument("--no-report", dest="generate_report", action="store_false", help="Skip report.html generation")

    p.add_argument("--max-video-height", type=int, default=DEFAULT_MAX_VIDEO_HEIGHT, help=f"Preferred max video resolution in pixels (default: {DEFAULT_MAX_VIDEO_HEIGHT})")
    p.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Hardware preference (informational - the pipeline runs on CPU regardless; see README)")

    resume_group = p.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=True, help="Reuse valid cached intermediate results (default: on)")
    resume_group.add_argument("--force-restart", dest="resume", action="store_false", help="Ignore any cached intermediate results and start over")

    p.add_argument("--yes", "-y", action="store_true", help="Never prompt interactively; auto-pick the largest identity if ambiguous")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity (default: INFO)")
    p.add_argument("--version", action="version", version=f"frpm {__version__}")
    return p


def _prompt_identity(infos: list[IdentityClusterInfo]) -> int:
    table = Table(title=f"Found {len(infos)} people")
    table.add_column("#")
    table.add_column("Identity")
    table.add_column("Appearances", justify="right")
    for i, info in enumerate(infos, start=1):
        table.add_row(str(i), f"Person {info.cluster_id}", str(info.count))
    console.print(table)
    console.print(
        "[dim]Contact sheets for each identity were saved under "
        "<output>/<video_id>/contact_sheets/identity_candidate_*.jpg[/dim]"
    )
    while True:
        raw = console.input("Select target identity (number): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(infos):
            return infos[int(raw) - 1].cluster_id
        console.print(f"[red]Please enter a number between 1 and {len(infos)}.[/red]")


def _make_prompt_fn(assume_yes: bool):
    if assume_yes or not sys.stdin.isatty():
        return None
    return _prompt_identity


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.log_level)

    cfg = AppConfig(
        url=args.url,
        output_dir=Path(args.output),
        fps=args.fps,
        max_frames=args.max_frames,
        max_duration=args.max_duration,
        start=args.start,
        end=args.end,
        person=args.person,
        reference_image=args.reference_image,
        pack_size=args.pack_size,
        reference_pack_size=args.reference_pack_size,
        keep_video=args.keep_video,
        keep_candidates=args.keep_candidates,
        lora_mode=args.lora_mode,
        trigger_token=args.trigger_token,
        synthesize_missing=args.synthesize_missing,
        synthesis_count=args.synthesis_count,
        synthesis_strength=args.synthesis_strength,
        synthesis_max_categories=args.synthesis_max_categories,
        face_restore=args.face_restore,
        generate_report=args.generate_report,
        device=args.device,
        resume=args.resume,
        log_level=args.log_level,
        max_video_height=args.max_video_height,
        assume_yes=args.yes,
    )

    detect_hardware()

    try:
        pipeline = Pipeline(cfg, prompt_fn=_make_prompt_fn(args.yes))
        manifest, report_path = pipeline.run()
    except FRPMError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress up to the last completed stage was cached; rerun the same command to resume.[/yellow]")
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level safety net for a CLI tool
        logger.exception("Unexpected error")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        return 1

    out_dir = cfg.video_output_dir(manifest.video.video_id)
    console.print()
    console.print(f"[bold green]Done![/bold green] Identity pack created at: {out_dir}")
    console.print(f"  Final images: {len(manifest.images)}")
    console.print(f"  Reference pack: {len(manifest.reference_pack)}")
    if manifest.lora_dataset:
        console.print(f"  LoRA dataset: {len(manifest.lora_dataset)} images")
    if manifest.synthesized_images:
        console.print(f"  Synthesized (AI-recreated) images: {len(manifest.synthesized_images)}")
    console.print(f"  Manifest: {out_dir / 'manifest.json'}")
    if report_path:
        console.print(f"  Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
