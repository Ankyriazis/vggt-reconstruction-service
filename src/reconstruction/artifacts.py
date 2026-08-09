"""Persist reconstruction outputs to disk.

A reconstruction produces tens of millions of floats. Returning that inline
would make responses enormous and force clients to parse a wall of JSON, so
results are written once and served as files the client fetches on demand.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Points below this confidence are usually sky, motion blur, or surfaces seen
# at a grazing angle. Keeping them makes a point cloud look considerably worse.
DEFAULT_CONFIDENCE_PERCENTILE = 50.0


def write_results(result, directory: Path) -> dict[str, str]:
    """Write camera parameters, depth and a point cloud. Returns file names."""
    directory.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        directory / "cameras.npz",
        extrinsics=result.extrinsics,
        intrinsics=result.intrinsics,
    )
    np.savez_compressed(
        directory / "depth.npz",
        depth=result.depth,
        confidence=result.depth_confidence,
    )
    points_written = write_point_cloud(
        result.world_points, result.point_confidence, directory / "points.ply"
    )

    return {
        "cameras": "cameras.npz",
        "depth": "depth.npz",
        "point_cloud": "points.ply",
        "point_cloud_points": str(points_written),
    }


def write_point_cloud(
    world_points: np.ndarray,
    confidence: np.ndarray,
    path: Path,
    percentile: float = DEFAULT_CONFIDENCE_PERCENTILE,
) -> int:
    """Write a confidence-filtered point cloud as binary PLY.

    Binary rather than ASCII: a few million points as text is an order of
    magnitude larger and slow for any viewer to parse.
    """
    points = world_points.reshape(-1, 3)
    scores = confidence.reshape(-1)

    if scores.size == points.shape[0] and scores.size:
        threshold = float(np.percentile(scores, percentile))
        keep = scores >= threshold
        points = points[keep]

    finite = np.isfinite(points).all(axis=1)
    points = points[finite].astype(np.float32)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(points.tobytes())

    logger.info("wrote %s (%d points)", path.name, len(points))
    return len(points)
