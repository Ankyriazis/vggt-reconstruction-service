"""Sweep view count against latency and peak VRAM.

    python tools/benchmark.py --images path/to/scene/*.png

This is the script that produced the table in README.md. It exists because the
default view count is a hardware claim, and a hardware claim should be
reproducible on the reader's own machine rather than taken on trust.

Watch for the point where peak VRAM approaches the card's capacity: past it
the driver spills to host memory and per-view cost rises sharply, which is the
real limit rather than the point where allocation finally fails.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reconstruction.engine import ReconstructionEngine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", required=True, help="glob for scene images")
    parser.add_argument("--model", default="facebook/VGGT-1B")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--views", default="2,4,8,16", help="comma-separated view counts to try"
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.images))
    if not files:
        print(f"error: no images matched {args.images!r}", file=sys.stderr)
        return 1

    engine = ReconstructionEngine(model=args.model, device=args.device)
    total = None
    if engine.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"device {engine.device_name}  {total:.1f} GB  dtype {engine.dtype}")
    else:
        print(f"device {engine.device_name}  dtype {engine.dtype}")
    print(f"model loaded in {engine.load_seconds:.1f}s")
    print(f"{len(files)} images available\n")

    print(f"{'views':>6} {'seconds':>9} {'peak VRAM':>11} {'per view':>10}")
    for count in [int(v) for v in args.views.split(",")]:
        if count > len(files):
            print(f"{count:>6}   (only {len(files)} images available, skipped)")
            continue
        try:
            result = engine.reconstruct(files[:count])
        except torch.cuda.OutOfMemoryError:
            print(f"{count:>6}   OUT OF MEMORY")
            torch.cuda.empty_cache()
            break

        vram = f"{result.peak_vram_gb:.2f} GB" if result.peak_vram_gb else "n/a"
        flag = ""
        if result.peak_vram_gb and total and result.peak_vram_gb > total:
            flag = "  <- exceeds VRAM, spilling to host memory"
        print(
            f"{count:>6} {result.inference_seconds:>9.2f} {vram:>11}"
            f" {result.inference_seconds / count:>9.2f}s{flag}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
