"""Shared low-level helpers: logging, unicode-safe image IO, model
downloading/caching, JSON IO, and resilient per-item execution."""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import requests
from rich.logging import RichHandler

logger = logging.getLogger("frpm")

try:
    # On some managed/corporate Windows machines, HTTPS traffic is
    # intercepted with a custom root CA that's trusted by the OS but not by
    # Python's bundled `certifi` store, making `requests` fail SSL
    # verification even though the OS/browser trust the connection fine.
    # `truststore` makes Python's ssl module verify against the OS trust
    # store instead, which fixes this class of environment without
    # weakening verification. Safe/inert everywhere else.
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - best-effort, never fatal
    pass


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Third-party libraries are extremely chatty at INFO/DEBUG; keep them quiet.
    for noisy in ("absl", "mediapipe", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------
# Unicode/emoji-safe image IO (cv2.imread/imwrite mishandle non-ASCII paths
# on Windows because they go through fopen with the current codepage).
# --------------------------------------------------------------------------

def imread_unicode(path: str | Path):
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str | Path, image: np.ndarray, quality: int = 95) -> bool:
    import cv2

    path = Path(path)
    ext = path.suffix.lower() or ".jpg"
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


# --------------------------------------------------------------------------
# JSON caching helpers
# --------------------------------------------------------------------------

def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def read_json(path: str | Path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse cache file %s - ignoring it.", path)
        return None


# --------------------------------------------------------------------------
# Resilient per-item execution: never let one bad frame/face kill a run.
# --------------------------------------------------------------------------

@contextmanager
def tolerant(step_description: str) -> Iterator[None]:
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - intentional broad catch
        logger.warning("Skipping item during '%s': %s", step_description, exc)


# --------------------------------------------------------------------------
# Model downloading/caching
# --------------------------------------------------------------------------

MODEL_URLS = {
    "yunet": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    ),
    "sface": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
        "face_recognition_sface_2021dec.onnx"
    ),
    "face_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
        "float16/1/face_landmarker.task"
    ),
    "pose_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/"
        "float16/1/pose_landmarker_lite.task"
    ),
}

MODEL_FILENAMES = {
    "yunet": "face_detection_yunet_2023mar.onnx",
    "sface": "face_recognition_sface_2021dec.onnx",
    "face_landmarker": "face_landmarker.task",
    "pose_landmarker": "pose_landmarker_lite.task",
}


def ensure_model(name: str, cache_dir: Path) -> Path:
    """Download a model file into ``cache_dir`` if not already present.

    All models used by FRPM (OpenCV Zoo YuNet/SFace, MediaPipe Face/Pose
    Landmarker) are Apache-2.0 licensed and safe for personal/commercial use.
    """
    if name not in MODEL_URLS:
        raise KeyError(f"Unknown model '{name}'")
    dest = cache_dir / MODEL_FILENAMES[name]
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    ensure_dir(cache_dir)
    url = MODEL_URLS[name]
    logger.info("Downloading model '%s' (first run only) from %s", name, url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            written = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    written += len(chunk)
            if total and written != total:
                raise IOError(f"Incomplete download for {name}: {written}/{total} bytes")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return dest


def link_or_copy(src: Path, dst: Path) -> None:
    """Create ``dst`` pointing at ``src`` as cheaply as the OS allows.

    Tries a relative symlink first (fails without admin/dev-mode on some
    Windows setups), then a hardlink (same volume only), then falls back to
    a plain copy so the category folders always work reliably everywhere.
    """
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        return
    try:
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def sanitize_filename(name: str) -> str:
    keep = "-_.() "
    return "".join(c if c.isalnum() or c in keep else "_" for c in name).strip()[:150]
