"""User-facing error hierarchy. Messages here are shown directly to the CLI
user, so they should be short and actionable (no stack traces needed)."""
from __future__ import annotations


class FRPMError(Exception):
    """Base class for all expected/user-facing FRPM errors."""


class InvalidURLError(FRPMError):
    pass


class VideoUnavailableError(FRPMError):
    pass


class DownloadFailedError(FRPMError):
    pass


class MissingFFmpegError(FRPMError):
    pass


class NoFacesFoundError(FRPMError):
    pass


class NoIdentitySelectedError(FRPMError):
    pass
