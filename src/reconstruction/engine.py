"""3D reconstruction from unposed images, using VGGT.

No camera calibration is required: the model recovers extrinsics, intrinsics,
depth and a dense point map in a single feedforward pass.

Two properties of this model drive the whole service design. Loading it takes
roughly two and a half minutes, so it is loaded once at startup and never per
request. Inference takes seconds rather than milliseconds, which is why the
HTTP layer is a job queue and not a synchronous endpoint.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconstructionResult:
    """Outputs of one reconstruction, already on the CPU as plain arrays."""

    extrinsics: np.ndarray      # (V, 3, 4) camera-from-world
    intrinsics: np.ndarray      # (V, 3, 3)
    depth: np.ndarray           # (V, H, W, 1)
    depth_confidence: np.ndarray
    world_points: np.ndarray    # (V, H, W, 3)
    point_confidence: np.ndarray
    view_count: int
    inference_seconds: float
    peak_vram_gb: float | None
    free_vram_gb_at_start: float | None
    spilled: bool

    def summary(self) -> dict:
        payload = {
            "views": self.view_count,
            "inference_seconds": round(self.inference_seconds, 2),
            "peak_vram_gb": round(self.peak_vram_gb, 2) if self.peak_vram_gb else None,
            "free_vram_gb_at_start": (
                round(self.free_vram_gb_at_start, 2) if self.free_vram_gb_at_start else None
            ),
            "depth_shape": list(self.depth.shape),
            "points_total": int(self.world_points.shape[0] * self.world_points.shape[1]
                                * self.world_points.shape[2]),
        }
        if self.spilled:
            payload["warning"] = (
                "Working set exceeded free VRAM, so the driver paged to host memory. "
                "This can slow a job by more than 10x. Close other GPU applications, "
                "use a dedicated GPU, or lower VGGT_MAX_VIEWS."
            )
        return payload


class ReconstructionEngine:
    """Loads VGGT once and reconstructs scenes from image files."""

    def __init__(self, model: str = "facebook/VGGT-1B", device: str | None = None) -> None:
        from vggt.models.vggt import VGGT

        self.model_name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # bfloat16 needs compute capability 8.0+. Below that, float16 is the
        # best available and float32 on CPU, where autocast would only slow
        # things down.
        if self.device.startswith("cuda"):
            major = torch.cuda.get_device_capability()[0]
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.dtype = torch.float32

        # Capture the memory budget before loading. Afterwards PyTorch's
        # caching allocator has reserved most of the card, so mem_get_info
        # reports almost nothing free and is useless as a budget. What matters
        # is how much was available to this process at the start - the rest is
        # held by the desktop compositor, browser and editor.
        self.vram_total_gb: float | None = None
        self.vram_budget_gb: float | None = None
        if self.device.startswith("cuda"):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            self.vram_total_gb = total_bytes / 1024**3
            self.vram_budget_gb = free_bytes / 1024**3
            logger.info(
                "VRAM %.2f GB total, %.2f GB available to this process",
                self.vram_total_gb, self.vram_budget_gb,
            )

        logger.info("loading %s onto %s (%s)", model, self.device, self.dtype)
        started = time.perf_counter()
        self.model = VGGT.from_pretrained(model).to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        logger.info("loaded in %.1fs", self.load_seconds)

    @property
    def device_name(self) -> str:
        if self.device.startswith("cuda"):
            return torch.cuda.get_device_name(0)
        return "CPU"

    def vram_headroom_gb(self, view_count: int) -> float | None:
        """Budget minus the estimated working set. Negative means spilling.

        The budget is what was free when this process started, not the card's
        nominal capacity: on a workstation the desktop, browser and editor
        routinely hold 2-3 GB. When the working set exceeds the budget the
        driver pages to host memory rather than failing, so the symptom is a
        job that is inexplicably ten times slower, not an error.
        """
        if self.vram_budget_gb is None:
            return None
        return self.vram_budget_gb - self.estimate_vram_gb(view_count)

    @staticmethod
    def estimate_vram_gb(view_count: int) -> float:
        """Rough working-set estimate, fitted to measurements on this model.

        Measured peaks were 6.95 GB at 2 views and 7.31 GB at 4, giving a
        large fixed cost and a modest per-view one. Approximate by design: it
        exists to warn before a slow job, not to gate admission.
        """
        return 6.77 + 0.09 * view_count

    def reconstruct(self, image_paths: list[str | Path]) -> ReconstructionResult:
        """Reconstruct a scene from a list of image files.

        Args:
            image_paths: Two or more views of the same scene. One view works
                but yields only monocular depth, with no triangulated geometry.

        Raises:
            ValueError: if no images were supplied.
            torch.cuda.OutOfMemoryError: if the working set exceeds VRAM. The
                caller is expected to surface this as a job failure rather
                than let it kill the worker.
        """
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        if not image_paths:
            raise ValueError("At least one image is required.")

        paths = [str(p) for p in image_paths]
        images = load_and_preprocess_images(paths).to(self.device)

        on_cuda = self.device.startswith("cuda")
        budget_gb = self.vram_budget_gb if on_cuda else None
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
            headroom = self.vram_headroom_gb(len(paths))
            if headroom is not None and headroom < 0:
                # Do not refuse the job: the driver will complete it by paging
                # to host memory. Warn instead, because the symptom otherwise
                # is a job that is inexplicably ten times slower than the last.
                logger.warning(
                    "%d views need about %.2f GB but only %.2f GB was available "
                    "to this process; the driver will page to host memory and "
                    "this job will be slow",
                    len(paths), self.estimate_vram_gb(len(paths)), budget_gb,
                )

        started = time.perf_counter()
        try:
            with torch.no_grad():
                if on_cuda:
                    with torch.amp.autocast("cuda", dtype=self.dtype):
                        predictions = self.model(images)
                else:
                    predictions = self.model(images)

                # Camera parameters are predicted as a compact encoding; this
                # turns them into the extrinsic and intrinsic matrices that
                # downstream tools actually consume.
                extrinsics, intrinsics = pose_encoding_to_extri_intri(
                    predictions["pose_enc"], images.shape[-2:]
                )
            if on_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 1024**3 if on_cuda else None

            return ReconstructionResult(
                extrinsics=_to_numpy(extrinsics),
                intrinsics=_to_numpy(intrinsics),
                depth=_to_numpy(predictions["depth"]),
                depth_confidence=_to_numpy(predictions["depth_conf"]),
                world_points=_to_numpy(predictions["world_points"]),
                point_confidence=_to_numpy(predictions["world_points_conf"]),
                view_count=len(paths),
                inference_seconds=elapsed,
                peak_vram_gb=peak,
                free_vram_gb_at_start=budget_gb,
                # Explicit None checks: a budget of exactly 0.0 is meaningful
                # and a truthiness test would silently read it as "unknown".
                spilled=(peak is not None and budget_gb is not None and peak > budget_gb),
            )
        finally:
            del images
            if on_cuda:
                # Release the working set even on failure, so one oversized job
                # does not poison every job that follows it.
                torch.cuda.empty_cache()


def _to_numpy(tensor) -> np.ndarray:
    """Detach a tensor to a float32 numpy array, dropping the batch dimension."""
    if isinstance(tensor, np.ndarray):
        array = tensor
    else:
        array = tensor.detach().to(torch.float32).cpu().numpy()
    return array[0] if array.ndim >= 4 and array.shape[0] == 1 else array
