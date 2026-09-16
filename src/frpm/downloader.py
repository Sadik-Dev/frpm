"""YouTube URL validation and video downloading via yt-dlp.

Kept local/offline-friendly: only yt-dlp talks to the network, and only to
fetch the video the user pointed us at. No telemetry, no other endpoints.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Optional

from .config import AppConfig
from .errors import DownloadFailedError, FRPMError, InvalidURLError, MissingFFmpegError, VideoUnavailableError
from .models import VideoMetadata
from .utils import ensure_dir, write_json

logger = logging.getLogger("frpm.downloader")

_YOUTUBE_HOST_RE = re.compile(
    r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com|youtube-nocookie\.com|youtu\.be)/",
    re.IGNORECASE,
)
_VIDEO_ID_PATTERNS = [
    re.compile(r"(?:v=|/videos/|embed/|shorts/|live/)([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
]


def is_valid_youtube_url(url: str) -> bool:
    """Fast, offline structural check - not a network call."""
    if not isinstance(url, str) or not url.strip():
        return False
    url = url.strip()
    if not _YOUTUBE_HOST_RE.match(url):
        return False
    return extract_video_id(url) is not None


def extract_video_id(url: str) -> Optional[str]:
    for pat in _VIDEO_ID_PATTERNS:
        m = pat.search(url)
        if m:
            return m.group(1)
    return None


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def download_video(cfg: AppConfig, url: str) -> tuple[Path, VideoMetadata]:
    """Download the video (<=cfg.max_video_height, no unnecessary audio) via
    yt-dlp. Returns the local file path and parsed metadata."""
    if not is_valid_youtube_url(url):
        raise InvalidURLError(
            f"'{url}' does not look like a valid YouTube URL. "
            "Expected something like https://www.youtube.com/watch?v=XXXXXXXXXXX "
            "or https://youtu.be/XXXXXXXXXXX."
        )

    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        raise MissingFFmpegError(
            "FFmpeg was not found on PATH and the 'imageio-ffmpeg' fallback is unavailable. "
            "Install FFmpeg (e.g. `winget install Gyan.FFmpeg` on Windows, or `apt install ffmpeg` "
            "on Linux, or `brew install ffmpeg` on macOS) and try again."
        )

    video_id_hint = extract_video_id(url) or "video"
    if cfg.keep_video:
        download_dir = cfg.output_dir / video_id_hint / "source"
    else:
        download_dir = cfg.output_dir / video_id_hint / ".cache" / "download"
    ensure_dir(download_dir)

    import yt_dlp

    height = cfg.max_video_height
    fmt = (
        f"bestvideo[height<={height}][ext=mp4]/bestvideo[height<={height}]/"
        f"best[height<={height}]/best"
    )
    ydl_opts = {
        "format": fmt,
        "outtmpl": str(download_dir / "%(id)s.%(ext)s"),
        "ffmpeg_location": ffmpeg_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "logger": _YtDlpLogger(),
        "retries": 3,
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise _translate_ytdlp_error(exc, url) from exc

    if info is None:
        raise DownloadFailedError(f"yt-dlp returned no information for '{url}'.")

    filepath = info.get("requested_downloads", [{}])[0].get("filepath") or ydl.prepare_filename(info)
    video_path = Path(filepath)
    if not video_path.exists():
        # merge_output_format may change the extension after post-processing
        candidate = video_path.with_suffix(".mp4")
        if candidate.exists():
            video_path = candidate
        else:
            raise DownloadFailedError(f"Downloaded file not found on disk (expected near {video_path}).")

    metadata = VideoMetadata(
        video_id=info.get("id", video_id_hint),
        title=info.get("title", ""),
        url=url,
        duration=float(info.get("duration") or 0.0),
        uploader=info.get("uploader") or "",
        upload_date=info.get("upload_date") or "",
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        fps=float(info.get("fps") or 0.0),
    )

    source_meta_dir = cfg.video_output_dir(metadata.video_id) / "source"
    ensure_dir(source_meta_dir)
    write_json(source_meta_dir / "metadata.json", metadata.model_dump())

    return video_path, metadata


class _YtDlpLogger:
    def debug(self, msg):
        if msg.startswith("[debug] "):
            return
        logger.debug(msg)

    def info(self, msg):
        logger.debug(msg)

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)


def _translate_ytdlp_error(exc: Exception, url: str) -> FRPMError:
    msg = str(exc).lower()
    if "private video" in msg:
        return VideoUnavailableError(f"This video is private and cannot be downloaded: {url}")
    if "sign in to confirm your age" in msg or "age" in msg and "restrict" in msg:
        return VideoUnavailableError(f"This video is age-restricted and could not be accessed anonymously: {url}")
    if "video unavailable" in msg or "has been removed" in msg:
        return VideoUnavailableError(f"This video is unavailable (removed or region-blocked): {url}")
    if "unsupported url" in msg:
        return InvalidURLError(f"'{url}' is not a supported/recognized video URL.")
    return DownloadFailedError(f"Failed to download '{url}': {exc}")
