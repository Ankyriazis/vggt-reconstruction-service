# VGGT Reconstruction Service

3D reconstruction from **unposed** images, served over HTTP. No camera calibration, no
COLMAP pass, no known intrinsics — hand it several photographs of a scene and it returns
camera poses, depth maps and a point cloud from a single feedforward pass.

Built on [VGGT](https://github.com/facebookresearch/vggt) (Meta, CVPR 2025 Best Paper).

## Why this is a job queue and not an endpoint

This is the counterpart to a low-latency detection service, and almost every design choice
differs because of one measured fact: **a reconstruction takes tens of seconds**, where an
object detector takes milliseconds.

That single number rules out the synchronous request/response shape. A client would hold a
connection open for the duration, and any proxy in the path would time it out first. So
submissions return `202 Accepted` with a job id, and the client polls.

```
POST /reconstruct   ->  202  { "id": "f3f4896bda90", "poll": "/jobs/f3f4896bda90" }
GET  /jobs/{id}     ->  200  { "status": "running" }
                        200  { "status": "succeeded", "artifacts": {...} }
GET  /jobs/{id}/artifacts/points.ply
```

## Quickstart

```bash
# PyTorch first, from the index matching your hardware
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

pip install -r requirements.txt
pip install --no-deps git+https://github.com/facebookresearch/vggt.git

PYTHONPATH=src uvicorn reconstruction.service:app --port 8000
```

VGGT is installed with `--no-deps` deliberately: its metadata pins `numpy<2`, which is
stale. It runs correctly on numpy 2.4.4 — verified — and honouring the pin would force a
pointless downgrade of the whole environment.

```bash
curl -X POST http://localhost:8000/reconstruct \
  -F "files=@view0.png" -F "files=@view1.png" \
  -F "files=@view2.png" -F "files=@view3.png"
```

## What comes back

A real job on four 518 × 350 views:

```json
{
  "status": "succeeded",
  "views": 4,
  "run_seconds": 37.11,
  "result": {
    "inference_seconds": 35.26,
    "peak_vram_gb": 7.31,
    "depth_shape": [4, 350, 518, 1],
    "points_total": 725200
  },
  "artifacts": {
    "cameras": "cameras.npz",
    "depth": "depth.npz",
    "point_cloud": "points.ply",
    "point_cloud_points": "362600"
  }
}
```

`cameras.npz` holds extrinsics and intrinsics recovered by the model. `points.ply` is
binary PLY, filtered to the more confident half of predicted points — keeping all of them
buries the geometry in sky and grazing-angle noise.

## The measurement that shaped this service

**VGGT-1B does not fit comfortably on an 8 GB GPU that is also driving a desktop**, and
finding that out properly was the most useful part of building this.

Measured on an RTX 4060 (8 GB), `bfloat16`, 518 × 350 views:

| Views | Working set | Budget available | Headroom | Latency |
|---|---|---|---|---|
| 2 | 6.95 GB | 6.94 GB | **−0.01 GB** | 71 s |
| 4 | 7.31 GB | 6.94 GB | **−0.19 GB** | 129 s |

The card has 8.00 GB, but the desktop compositor, browser and editor held 1.06 GB, leaving
6.94 GB. Two views need 6.95 GB — over budget by **10 megabytes**.

At that point CUDA on Windows does not fail. It pages to host memory and keeps going, so
the symptom is not an error but a job that is inexplicably slow. In one early run on a
quieter GPU the same two-view job completed in **1.35 seconds**. The same code, once the
working set crossed the line, took **71 seconds** — a fiftyfold cliff, with no error
anywhere.

Two conclusions, both in the code rather than only in this file:

1. **This model wants a dedicated GPU.** The Docker image targets CUDA only; there is no
   CPU variant, because minutes per job is not a service.
2. **Silent spilling should be made loud.** The service records the VRAM budget at startup,
   estimates the working set per job, and reports the shortfall on `/readyz` and in the job
   result rather than leaving an operator to wonder why throughput collapsed.

```json
GET /readyz
{
  "ready": true,
  "vram": { "total_gb": 8.0, "budget_gb": 6.94, "headroom_at_max_views_gb": -0.19 },
  "warning": "Only 6.94 GB of VRAM was free when this process started, but 4 views need
              about 7.13 GB. Jobs will page to host memory and run many times slower."
}
```

Reproduce it on your own hardware:

```bash
python tools/benchmark.py --images "path/to/scene/*.png" --views 2,4,8
```

## Design notes

**The model loads once, at startup.** From a warm cache that is about 10 seconds; the first
ever run also downloads roughly 5 GB. Either way it does not belong on a request path.

**Liveness and readiness are separate endpoints.** `/healthz` answers as soon as the
process is up; `/readyz` returns 503 until the model is resident. An orchestrator that
cannot distinguish them will either route traffic to a cold replica or restart one that was
still loading.

**Exactly one worker consumes the queue.** Not a simplification — a second worker would
contend for the same GPU, and with the model already occupying most of the card, concurrent
jobs make the tail worse while improving nothing.

**A failed job must not kill the worker.** The handler catches broadly and marks the job
failed, because one malformed upload should not stop the queue for everyone.

**Retention is enforced when jobs finish, not only when they are submitted.** Enforcing it
solely on submit was a real bug caught by a test: a burst of six jobs leaves every earlier
job still queued and therefore not evictable, so the store stayed over its limit
indefinitely once the burst ended.

**Artifact paths are resolved inside the job directory.** A crafted artifact name cannot
escape it; there is a test for exactly that.

**Uploads are deleted once consumed.** The service keeps derived outputs, not user images.

## Configuration

| Variable | Default | |
|---|---|---|
| `VGGT_MODEL` | `facebook/VGGT-1B` | Model id or local path |
| `VGGT_DEVICE` | auto | `cuda:0`, `cpu` |
| `VGGT_MAX_VIEWS` | `4` | Images per job. A VRAM limit, not a model limit |
| `VGGT_MIN_VIEWS` | `1` | One view yields monocular depth only |
| `VGGT_JOB_RETENTION` | `100` | Finished jobs kept in memory |
| `VGGT_ARTIFACT_DIR` | `artifacts` | Where outputs are written |
| `VGGT_MAX_UPLOAD_BYTES` | `26214400` | Per-image upload cap |
| `VGGT_EAGER_LOAD` | `true` | Load the model at startup |

## Tests

```bash
pytest tests/ -v
```

20 tests, about two seconds. None load VGGT, touch a GPU, or reach the network — the engine
is injected through FastAPI's dependency system and the job handler is swapped directly.
Coverage includes queue retention, a failing job not killing the worker, path-traversal on
artifact downloads, upload validation, and the 503 paths taken while the model loads.

## Docker

```bash
docker compose -f docker/compose.yaml up --build   # needs the NVIDIA Container Toolkit
```

Runs unprivileged, bakes the weights into the image so a cold start does not depend on
reaching HuggingFace, and health-checks `/readyz` with a start period long enough to cover
model load.

## Limits

- One GPU, one job at a time. Concurrency needs a real queue and several workers; `jobs.py`
  is isolated so it can be replaced.
- Jobs live in memory and do not survive a restart.
- No authentication. Put it behind a gateway.
- Latency figures are from one machine, one scene, and — as the table above shows — depend
  more on free VRAM than on anything else. Treat them as an illustration of the cliff, not
  as a benchmark of the model.

## Licensing

**This matters more than usual here, and the constraint comes from the model, not from
preference.**

| | |
|---|---|
| This service's code | MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE) |
| VGGT code | [Meta VGGT License](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt) — a bespoke research licence with an Acceptable Use Policy and redistribution terms |
| `facebook/VGGT-1B` weights | **CC BY-NC 4.0 — non-commercial** |
| Commercial use | Requires [`facebook/VGGT-1B-Commercial`](https://huggingface.co/facebook/VGGT-1B-Commercial), granted by application |

This repository **does not redistribute** VGGT's code, weights, or sample images. It
depends on them: the package is installed from source and the weights are downloaded at
runtime, so you accept Meta's terms directly rather than through this project. That is why
the code here can be MIT while the model it drives is not.

If you intend to deploy this commercially, the default checkpoint is the wrong one, and no
amount of wrapping changes that.
