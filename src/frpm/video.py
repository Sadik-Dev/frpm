"""Intelligent frame sampling.

Rather than decoding every frame, we ``grab()`` (cheap, no decode) through
the video and only ``retrieve()`` (decode) frames that land on the sampling
stride - this keeps long videos practical. The stride adapts automatically
so a video never yields more than ``cfg.max_frames`` sampled frames.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import AppConfig
from .errors import FRPMError
from .utils import ensure_dir, imwrite_unicode

logger = logging.getLogger("frpm.video")


@dataclass
class SampledFrame:
    frame_number: int
    timestamp: float
    path: str


@dataclass
class VideoInfo:
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int


def probe_video(video_path: Path) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FRPMError(f"Could not open video for reading: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        return VideoInfo(fps=fps, total_frames=total_frames, duration=duration, width=width, height=height)
    finally:
        cap.release()


def sample_frames(
    video_path: Path,
    cfg: AppConfig,
    out_dir: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[list[SampledFrame], VideoInfo]:
    import cv2

    ensure_dir(out_dir)
    info = probe_video(video_path)
    src_fps = info.fps

    start_frame = max(0, int(round(cfg.start * src_fps)))
    end_time = cfg.end if cfg.end is not None else info.duration
    if cfg.max_duration is not None:
        end_time = min(end_time, cfg.start + cfg.max_duration)
    end_frame = int(round(end_time * src_fps)) if end_time else info.total_frames
    end_frame = min(end_frame, info.total_frames) if info.total_frames else end_frame

    stride = max(1, round(src_fps / cfg.fps)) if cfg.fps > 0 else 1
    span = max(1, end_frame - start_frame)
    estimated = span // stride
    if estimated > cfg.max_frames:
        new_stride = max(stride, -(-span // cfg.max_frames))  # ceil division
        logger.warning(
            "Sampling at %.2f fps over this range would yield ~%d frames; "
            "increasing stride from every %d to every %d source frames to respect --max-frames=%d.",
            cfg.fps, estimated, stride, new_stride, cfg.max_frames,
        )
        stride = new_stride

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FRPMError(f"Could not open video for reading: {video_path}")

    results: list[SampledFrame] = []
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        while frame_idx < end_frame and len(results) < cfg.max_frames:
            ok = cap.grab()
            if not ok:
                break
            if (frame_idx - start_frame) % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    ts = frame_idx / src_fps if src_fps > 0 else float(frame_idx)
                    fname = out_dir / f"frame_{frame_idx:08d}.jpg"
                    imwrite_unicode(fname, frame, quality=95)
                    results.append(SampledFrame(frame_number=frame_idx, timestamp=ts, path=str(fname)))
                    if on_progress:
                        on_progress(len(results), min(cfg.max_frames, span // stride + 1))
            frame_idx += 1
    finally:
        cap.release()

    logger.info("Sampled %d frames (source fps=%.2f, stride=%d frames).", len(results), src_fps, stride)
    return results, info
