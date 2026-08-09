"""Runtime configuration, read from environment variables.

The default view count is 4 rather than something larger, and that is a
measured decision rather than a cautious one. See MAX_VIEWS below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Service configuration.

    Attributes:
        max_views: Images accepted per job. The default of 4 is where an 8 GB
            card stops fitting the working set: measured peak was 7.32 GB at
            4 views and 8.25 GB at 8, and once allocation passes the physical
            limit the driver spills to host memory and per-view cost rises
            from 0.72 s to 1.16 s. Raise it if you have more VRAM.
        job_retention: Completed jobs kept in memory before the oldest are
            dropped. Results reference files on disk, so this bounds metadata
            rather than pixel data.
    """

    model: str = field(default_factory=lambda: _env_str("VGGT_MODEL", "facebook/VGGT-1B"))
    device: str | None = field(default_factory=lambda: os.environ.get("VGGT_DEVICE") or None)
    max_views: int = field(default_factory=lambda: _env_int("VGGT_MAX_VIEWS", 4))
    min_views: int = field(default_factory=lambda: _env_int("VGGT_MIN_VIEWS", 1))
    job_retention: int = field(default_factory=lambda: _env_int("VGGT_JOB_RETENTION", 100))
    artifact_dir: str = field(default_factory=lambda: _env_str("VGGT_ARTIFACT_DIR", "artifacts"))
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("VGGT_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
    )
    eager_load: bool = field(default_factory=lambda: _env_bool("VGGT_EAGER_LOAD", True))

    def __post_init__(self) -> None:
        if self.max_views < 1:
            raise ValueError("VGGT_MAX_VIEWS must be at least 1.")
        if self.min_views < 1 or self.min_views > self.max_views:
            raise ValueError("VGGT_MIN_VIEWS must be between 1 and VGGT_MAX_VIEWS.")
        if self.job_retention < 1:
            raise ValueError("VGGT_JOB_RETENTION must be at least 1.")
