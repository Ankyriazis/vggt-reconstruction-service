"""HTTP service for 3D reconstruction from unposed images.

The contract is asynchronous by necessity. A reconstruction takes seconds to
tens of seconds and the model needs about two and a half minutes to load, so
submissions are accepted with 202 and a job id, and the client polls.

Liveness and readiness are separate for the same reason: a container that
answered "healthy" during a 150-second model load would be sent traffic it
cannot serve, and one whose healthcheck failed during that window would be
restarted forever.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse

from .artifacts import write_results
from .config import Settings
from .engine import ReconstructionEngine
from .jobs import JobQueue

logger = logging.getLogger(__name__)

_state: dict = {
    "engine": None,
    "queue": None,
    "settings": None,
    "started": 0.0,
    "artifact_root": None,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Uploads are staged in a folder of their own under the artifact directory,
# rather than the system temp directory, so that startup can find and delete
# whatever a previous process left there.
STAGING_DIR = ".uploads"
STAGING_PREFIX = "vggt-upload-"
JOB_ID = re.compile(r"[0-9a-f]{12}")


def get_queue() -> JobQueue:
    q = _state.get("queue")
    if q is None:
        raise HTTPException(status_code=503, detail="Service is still starting.")
    return q


def get_settings() -> Settings:
    return _state["settings"]


def require_engine() -> ReconstructionEngine:
    engine = _state.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="The model is still loading. This takes around 150 seconds "
                   "from cold; poll /readyz.",
        )
    return engine


def _job_dir(job_id: str) -> Path:
    return Path(_state["artifact_root"]) / job_id


def _staging_root() -> Path:
    return Path(_state["artifact_root"]) / STAGING_DIR


def _handle_job(job_id: str, image_paths: list[Path]):
    """Run one reconstruction. Executed on the worker thread.

    The uploads are deleted whatever the outcome. The service keeps derived
    outputs, not user images, and a job that failed has no more use for them
    than one that succeeded. A failed write leaves no partial directory either:
    a failed job serves no artifacts.
    """
    try:
        result = _state["engine"].reconstruct(image_paths)
        directory = _job_dir(job_id)
        try:
            artifacts = write_results(result, directory)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return result.summary(), artifacts
    finally:
        _discard_uploads(image_paths)


def _discard_uploads(image_paths: list[Path]) -> None:
    """Delete the staging directory a job's uploads were saved in."""
    for staging in {path.parent for path in image_paths}:
        if staging.parent == _staging_root():  # only directories the service made
            shutil.rmtree(staging, ignore_errors=True)


def _discard_job(job_id: str) -> None:
    """Delete an evicted job's outputs; the queue calls this as it evicts.

    On Windows a file that is being downloaded cannot be deleted. It is left
    for the next startup, which deletes it.
    """
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)


def _discard_leftovers(artifact_root: Path) -> None:
    """Delete the job directories and uploads a previous process left.

    Jobs live in memory, so once the process that ran them has gone nothing
    can reach their files, and the artifact directory is often a volume that
    outlives the container. Only names this service creates are touched, since
    the directory is configurable and may hold other things. That assumes one
    process per artifact directory, as one worker per GPU already does.
    """
    leftovers = [path for path in artifact_root.iterdir()
                 if path.is_dir() and JOB_ID.fullmatch(path.name)]
    staging = artifact_root / STAGING_DIR
    if staging.is_dir():
        leftovers += staging.glob(f"{STAGING_PREFIX}*")
    for path in leftovers:
        shutil.rmtree(path, ignore_errors=True)
    if leftovers:
        logger.info("deleted %d job directories and uploads left by a previous run",
                    len(leftovers))


def _read_chunks(handle, size: int = 1024 * 1024):
    with handle:
        while chunk := handle.read(size):
            yield chunk


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = _state.get("settings") or Settings()
    _state["settings"] = settings
    _state["started"] = time.time()

    artifact_root = Path(settings.artifact_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    _discard_leftovers(artifact_root)
    (artifact_root / STAGING_DIR).mkdir(exist_ok=True)
    _state["artifact_root"] = artifact_root

    job_queue = JobQueue(_handle_job, retention=settings.job_retention, on_evict=_discard_job)
    job_queue.start()
    _state["queue"] = job_queue

    if settings.eager_load:
        _state["engine"] = ReconstructionEngine(
            model=settings.model, device=settings.device
        )
    else:
        logger.warning("VGGT_EAGER_LOAD is off; no model will be loaded")

    yield

    job_queue.stop()
    _state["queue"] = None
    _state["engine"] = None


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is not None:
        _state["settings"] = settings

    app = FastAPI(
        title="VGGT Reconstruction Service",
        description="Feedforward 3D reconstruction from unposed images. "
                    "No camera calibration required.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/healthz", tags=["operations"])
    def healthz() -> dict:
        """Liveness. True as soon as the process is running."""
        return {
            "status": "alive",
            "uptime_seconds": round(time.time() - _state["started"], 1),
        }

    @app.get("/readyz", tags=["operations"])
    def readyz(response: Response) -> dict:
        """Readiness. False until the model has finished loading."""
        engine = _state.get("engine")
        if engine is None:
            response.status_code = 503
            return {"status": "loading", "ready": False}
        settings = _state["settings"]
        headroom = engine.vram_headroom_gb(settings.max_views)
        payload = {
            "status": "ready",
            "ready": True,
            "model": engine.model_name,
            "device": engine.device,
            "accelerator": engine.device_name,
            "dtype": str(engine.dtype).replace("torch.", ""),
            "load_seconds": round(engine.load_seconds, 1),
        }
        if engine.vram_budget_gb is not None:
            payload["vram"] = {
                "total_gb": round(engine.vram_total_gb, 2),
                "budget_gb": round(engine.vram_budget_gb, 2),
                "headroom_at_max_views_gb": (
                    round(headroom, 2) if headroom is not None else None
                ),
            }
            # Surface this on readiness so an operator sees it before the first
            # slow job, rather than diagnosing it afterwards.
            if headroom is not None and headroom < 0:
                payload["warning"] = (
                    f"Only {engine.vram_budget_gb:.2f} GB of VRAM was free when this "
                    f"process started, but {settings.max_views} views need about "
                    f"{engine.estimate_vram_gb(settings.max_views):.2f} GB. Jobs will "
                    "page to host memory and run many times slower. Use a dedicated "
                    "GPU or lower VGGT_MAX_VIEWS."
                )
        return payload

    @app.get("/metrics", tags=["operations"])
    def metrics(job_queue: JobQueue = Depends(get_queue)) -> dict:
        return {"jobs": job_queue.stats()}

    @app.post("/reconstruct", status_code=202, tags=["reconstruction"])
    async def reconstruct(
        response: Response,
        files: list[UploadFile] = File(..., description="Two or more views of one scene"),
        job_queue: JobQueue = Depends(get_queue),
        settings: Settings = Depends(get_settings),
        engine: ReconstructionEngine = Depends(require_engine),
    ) -> dict:
        """Accept images and queue a reconstruction.

        Returns 202 with a job id. Poll `/jobs/{id}` for the outcome.
        """
        if len(files) < settings.min_views:
            raise HTTPException(
                status_code=400,
                detail=f"At least {settings.min_views} image(s) required.",
            )
        if len(files) > settings.max_views:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"At most {settings.max_views} images per job. This limit is set "
                    f"by available VRAM; raise VGGT_MAX_VIEWS if your GPU has more."
                ),
            )

        staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=_staging_root()))
        saved: list[Path] = []
        try:
            for index, upload in enumerate(files):
                suffix = Path(upload.filename or "").suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{upload.filename!r} is not a supported image type "
                               f"({', '.join(sorted(IMAGE_SUFFIXES))}).",
                    )
                payload = await upload.read()
                if not payload:
                    raise HTTPException(status_code=400, detail=f"{upload.filename!r} is empty.")
                if len(payload) > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{upload.filename!r} exceeds "
                               f"{settings.max_upload_bytes // (1024 * 1024)} MB.",
                    )
                destination = staging / f"{index:03d}{suffix}"
                destination.write_bytes(payload)
                saved.append(destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

        job = job_queue.submit(saved)
        response.headers["Location"] = f"/jobs/{job.id}"
        return {
            "id": job.id,
            "status": job.status.value,
            "views": job.view_count,
            "poll": f"/jobs/{job.id}",
        }

    @app.get("/jobs", tags=["reconstruction"])
    def list_jobs(limit: int = 20, job_queue: JobQueue = Depends(get_queue)) -> dict:
        jobs = job_queue.list(limit=limit)
        return {"count": len(jobs), "jobs": [job.to_dict() for job in jobs]}

    @app.get("/jobs/{job_id}", tags=["reconstruction"])
    def get_job(job_id: str, job_queue: JobQueue = Depends(get_queue)) -> dict:
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No job {job_id!r}.")
        return job.to_dict()

    @app.get("/jobs/{job_id}/artifacts/{name}", tags=["reconstruction"])
    def get_artifact(job_id: str, name: str, job_queue: JobQueue = Depends(get_queue)):
        """Download one output file: cameras.npz, depth.npz or points.ply."""
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No job {job_id!r}.")
        if job.status.value != "succeeded":
            raise HTTPException(
                status_code=409,
                detail=f"Job {job_id} is {job.status.value}; no artifacts yet.",
            )

        # Resolve inside the job directory so a crafted name cannot escape it.
        root = _job_dir(job_id).resolve()
        candidate = (root / name).resolve()
        if not str(candidate).startswith(str(root)):
            raise HTTPException(status_code=404, detail=f"No artifact {name!r}.")

        # Open the file here rather than return a FileResponse, which opens it
        # only while sending, after the 200 has gone out. In between the job
        # can be evicted and its directory deleted. A file that is already
        # open stays readable when deleted.
        try:
            handle = candidate.open("rb")
        except OSError:  # missing, or a directory
            raise HTTPException(status_code=404, detail=f"No artifact {name!r}.") from None
        return StreamingResponse(
            _read_chunks(handle),
            media_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
            headers={
                "Content-Length": str(os.fstat(handle.fileno()).st_size),
                "Content-Disposition": f'attachment; filename="{job_id}-{name}"',
            },
        )

    return app


app = create_app()
