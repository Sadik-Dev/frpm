"""Central configuration: CLI-derived settings and tunable thresholds.

Keeping every tunable constant here (rather than scattered through the
codebase) makes the scoring/classification behaviour easy to audit and
adjust without touching pipeline logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---- CLI defaults -----------------------------------------------------
DEFAULT_FPS = 3.0
DEFAULT_MAX_FRAMES = 4000
DEFAULT_PACK_SIZE = 45
MIN_PACK_SIZE = 30
MAX_PACK_SIZE = 60
DEFAULT_REFERENCE_PACK_SIZE = 16
MIN_REFERENCE_PACK_SIZE = 12
MAX_REFERENCE_PACK_SIZE = 20
# Placeholder LoRA trigger word (a caption tag, not a credential/secret) -
# users override this via --trigger-token with their own unique token.
DEFAULT_TRIGGER_TOKEN = "person_xyz"  # noqa: S105 - not a password, see comment above
DEFAULT_MAX_VIDEO_HEIGHT = 1080
MIN_FACE_SIZE_PX = 40
DOMINANT_IDENTITY_RATIO = 0.70


@dataclass
class QualityWeights:
    sharpness: float = 0.30
    exposure: float = 0.20
    face_resolution: float = 0.20
    landmark_confidence: float = 0.20
    artifact: float = 0.05
    obstruction: float = 0.05


@dataclass
class PoseThresholds:
    front_yaw_max: float = 15.0
    three_quarter_yaw_max: float = 45.0
    pitch_up_deg: float = 12.0
    pitch_down_deg: float = 12.0
    # Sign calibration escape hatches: MediaPipe's facial transformation
    # matrix convention is derived analytically (see pose.py); flip these to
    # True if real-world footage ever shows angles/labels mirrored.
    invert_yaw: bool = False
    invert_pitch: bool = False


@dataclass
class ExpressionThresholds:
    smile_slight: float = 0.25
    smile_confident: float = 0.45
    smile_big: float = 0.70
    jaw_open_talking: float = 0.20
    jaw_open_big: float = 0.45
    eyes_closed: float = 0.55
    brow_raise: float = 0.35
    frown: float = 0.30
    surprise_combo: float = 0.40
    eyes_direction: float = 0.30
    focused_squint: float = 0.35


@dataclass
class DedupThresholds:
    phash_max_distance: int = 8
    embedding_min_similarity: float = 0.93
    blendshape_min_similarity: float = 0.97
    time_window_sec: float = 1.5
    max_per_window: int = 2


@dataclass
class AppConfig:
    url: str
    output_dir: Path
    fps: float = DEFAULT_FPS
    max_frames: int = DEFAULT_MAX_FRAMES
    max_duration: Optional[float] = None
    start: float = 0.0
    end: Optional[float] = None
    person: Optional[int] = None
    reference_image: Optional[Path] = None
    pack_size: int = DEFAULT_PACK_SIZE
    reference_pack_size: int = DEFAULT_REFERENCE_PACK_SIZE
    keep_video: bool = False
    keep_candidates: bool = False
    lora_mode: bool = False
    trigger_token: str = DEFAULT_TRIGGER_TOKEN
    generate_report: bool = True
    device: str = "auto"
    resume: bool = True
    log_level: str = "INFO"
    max_video_height: int = DEFAULT_MAX_VIDEO_HEIGHT
    assume_yes: bool = False
    min_quality: float = 0.0

    synthesize_missing: bool = False
    synthesis_mode: str = "missing"
    synthesis_count: int = 1
    synthesis_strength: float = 0.6
    synthesis_ip_adapter_scale: float = 0.7
    synthesis_max_categories: int = 6
    synthesis_steps: int = 30
    face_restore: bool = False

    quality_weights: QualityWeights = field(default_factory=QualityWeights)
    pose_thresholds: PoseThresholds = field(default_factory=PoseThresholds)
    expression_thresholds: ExpressionThresholds = field(default_factory=ExpressionThresholds)
    dedup_thresholds: DedupThresholds = field(default_factory=DedupThresholds)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.reference_image is not None:
            self.reference_image = Path(self.reference_image)
        if not (MIN_PACK_SIZE <= self.pack_size <= MAX_PACK_SIZE):
            self.pack_size = max(MIN_PACK_SIZE, min(MAX_PACK_SIZE, self.pack_size))
        if not (MIN_REFERENCE_PACK_SIZE <= self.reference_pack_size <= MAX_REFERENCE_PACK_SIZE):
            self.reference_pack_size = max(
                MIN_REFERENCE_PACK_SIZE, min(MAX_REFERENCE_PACK_SIZE, self.reference_pack_size)
            )
        if self.fps <= 0:
            self.fps = DEFAULT_FPS
        if self.max_frames <= 0:
            self.max_frames = DEFAULT_MAX_FRAMES

    @property
    def models_cache_dir(self) -> Path:
        d = Path.home() / ".cache" / "frpm" / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def video_output_dir(self, video_id: str) -> Path:
        return self.output_dir / video_id

    def cache_dir(self, video_id: str) -> Path:
        d = self.video_output_dir(video_id) / ".cache"
        d.mkdir(parents=True, exist_ok=True)
        return d
