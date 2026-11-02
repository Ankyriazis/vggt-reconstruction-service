"""Tests for the reconstruction service.

None of these load VGGT, touch a GPU, or reach the network. The engine is
replaced through FastAPI's dependency system and the job handler is swapped
directly, which is the reason the queue and the HTTP layer were kept separate
from the model in the first place.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reconstruction import service as service_module  # noqa: E402
from reconstruction.artifacts import write_point_cloud  # noqa: E402
from reconstruction.config import Settings  # noqa: E402
from reconstruction.jobs import JobQueue, JobStatus  # noqa: E402
from reconstruction.service import create_app, require_engine  # noqa: E402

PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000100ffff0300000600"
    "05a4a8f1770000000049454e44ae426082"
)


class StubEngine:
    model_name = "stub"
    device = "cpu"
    dtype = "float32"
    load_seconds = 0.0
    device_name = "CPU"


def wait_for(job_queue: JobQueue, job_id: str, timeout: float = 5.0):
    """Poll until the job leaves a non-terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = job_queue.get(job_id)
        if job and job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def eventually(condition, timeout: float = 2.0) -> bool:
    """Poll until condition() holds, for effects that follow a job's status."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client with a stub engine and a job handler that does no real work."""
    def fake_handler(job_id, image_paths):
        directory = Path(service_module._state["artifact_root"]) / job_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "points.ply").write_bytes(b"ply\n")
        for path in image_paths:
            path.unlink(missing_ok=True)
        return {"views": len(image_paths), "inference_seconds": 0.01}, {
            "point_cloud": "points.ply"
        }

    monkeypatch.setattr(service_module, "_handle_job", fake_handler)
    app = create_app(
        Settings(eager_load=False, artifact_dir=str(tmp_path / "artifacts"), max_views=4)
    )
    app.dependency_overrides[require_engine] = lambda: StubEngine()
    with TestClient(app) as test_client:
        yield test_client


def upload(count: int, name="v.png"):
    return [("files", (f"{i}-{name}", PNG_1x1, "image/png")) for i in range(count)]


# ------------------------------------------------------------------ queue

def test_queue_runs_a_job_and_reports_success():
    job_queue = JobQueue(lambda job_id, paths: ({"ok": True}, {"a": "b"}))
    job_queue.start()
    try:
        job = job_queue.submit([Path("x.png")])
        finished = wait_for(job_queue, job.id)
        assert finished.status is JobStatus.SUCCEEDED
        assert finished.result == {"ok": True}
    finally:
        job_queue.stop()


def test_a_failing_job_does_not_kill_the_worker():
    """One bad job must not take the queue down with it."""
    calls = {"n": 0}

    def handler(job_id, paths):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return ({"ok": True}, {})

    job_queue = JobQueue(handler)
    job_queue.start()
    try:
        first = wait_for(job_queue, job_queue.submit([Path("a.png")]).id)
        assert first.status is JobStatus.FAILED
        assert "boom" in first.error

        second = wait_for(job_queue, job_queue.submit([Path("b.png")]).id)
        assert second.status is JobStatus.SUCCEEDED
    finally:
        job_queue.stop()


def test_retention_evicts_the_oldest_finished_jobs():
    """A burst larger than retention must not leave the store over its limit."""
    job_queue = JobQueue(lambda job_id, paths: ({}, {}), retention=3)
    job_queue.start()
    try:
        ids = [job_queue.submit([Path(f"{i}.png")]).id for i in range(6)]
        wait_for(job_queue, ids[-1])  # the newest is never the one evicted

        deadline = time.time() + 2.0
        while time.time() < deadline and job_queue.stats()["tracked"] > 3:
            time.sleep(0.02)

        assert job_queue.stats()["tracked"] <= 3
        assert job_queue.get(ids[0]) is None, "oldest finished job should be evicted"
        assert job_queue.get(ids[-1]) is not None, "newest job should be retained"
    finally:
        job_queue.stop()


def test_unknown_job_is_none():
    assert JobQueue(lambda job_id, paths: ({}, {})).get("nope") is None


def test_a_backlog_does_not_evict_finished_results():
    """Retention counts finished jobs, so waiting ones cannot push results out."""
    release = threading.Event()
    calls = []

    def handler(job_id, paths):
        calls.append(job_id)
        if len(calls) > 1:  # the first job finishes; the rest wait
            release.wait(timeout=5)
        return {}, {}

    job_queue = JobQueue(handler, retention=3)
    job_queue.start()
    try:
        first = wait_for(job_queue, job_queue.submit([]).id)
        waiting = [job_queue.submit([]).id for _ in range(3)]
        assert job_queue.get(first.id) is not None, "a backlog evicted a finished result"

        release.set()
        for job_id in waiting:
            wait_for(job_queue, job_id)
        assert eventually(lambda: job_queue.stats()["tracked"] == 3)
        assert job_queue.get(first.id) is None, "the oldest finished job goes first"
    finally:
        release.set()
        job_queue.stop()


def test_eviction_cleanup_runs_and_cannot_kill_the_worker():
    evicted = []

    def on_evict(job_id):
        evicted.append(job_id)
        raise OSError("disk went away")

    job_queue = JobQueue(lambda job_id, paths: ({}, {}), retention=1, on_evict=on_evict)
    job_queue.start()
    try:
        ids = [wait_for(job_queue, job_queue.submit([]).id).id for _ in range(3)]
        assert eventually(lambda: evicted == ids[:2])
    finally:
        job_queue.stop()


def test_a_stopped_queue_cannot_be_restarted():
    """stop() can return while a job still runs; a restart would add a second worker."""
    job_queue = JobQueue(lambda job_id, paths: ({}, {}))
    job_queue.start()
    job_queue.stop()
    with pytest.raises(RuntimeError, match="restarted"):
        job_queue.start()


# ------------------------------------------------------------------ artifacts

def test_point_cloud_is_confidence_filtered(tmp_path):
    points = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]], dtype=np.float32)
    confidence = np.array([[0.1, 0.2, 0.9, 0.95]], dtype=np.float32)
    written = write_point_cloud(points, confidence, tmp_path / "p.ply", percentile=50)
    assert written == 2, "expected the two lowest-confidence points to be dropped"


def test_point_cloud_drops_non_finite_points(tmp_path):
    points = np.array([[[0, 0, 0], [np.nan, 0, 0], [np.inf, 1, 1], [1, 1, 1]]], dtype=np.float32)
    confidence = np.ones((1, 4), dtype=np.float32)
    written = write_point_cloud(points, confidence, tmp_path / "p.ply", percentile=0)
    assert written == 2


def test_point_cloud_header_is_binary_ply(tmp_path):
    points = np.zeros((1, 3, 3), dtype=np.float32)
    write_point_cloud(points, np.ones((1, 3), np.float32), tmp_path / "p.ply", percentile=0)
    head = (tmp_path / "p.ply").read_bytes()[:64]
    assert head.startswith(b"ply\nformat binary_little_endian 1.0")


# ------------------------------------------------------------------ config

def test_max_views_must_be_positive():
    with pytest.raises(ValueError, match="MAX_VIEWS"):
        Settings(max_views=0)


def test_min_views_cannot_exceed_max_views():
    with pytest.raises(ValueError, match="MIN_VIEWS"):
        Settings(min_views=5, max_views=4)


# ------------------------------------------------------------------ http

def test_healthz(client):
    assert client.get("/healthz").json()["status"] == "alive"


def test_submit_returns_202_with_a_location_header(client):
    response = client.post("/reconstruct", files=upload(2))
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert response.headers["Location"] == f"/jobs/{body['id']}"


def test_job_reaches_succeeded_and_exposes_artifacts(client):
    job_id = client.post("/reconstruct", files=upload(2)).json()["id"]
    for _ in range(200):
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)
    assert body["status"] == "succeeded"
    assert body["artifacts"]["point_cloud"] == "points.ply"

    artifact = client.get(f"/jobs/{job_id}/artifacts/points.ply")
    assert artifact.status_code == 200
    assert artifact.content.startswith(b"ply")


def test_too_many_views_is_rejected_with_the_reason(client):
    response = client.post("/reconstruct", files=upload(5))
    assert response.status_code == 400
    assert "VRAM" in response.json()["detail"]


def test_non_image_upload_is_rejected(client):
    response = client.post(
        "/reconstruct", files=[("files", ("notes.txt", b"hello", "text/plain"))]
    )
    assert response.status_code == 400


def test_empty_upload_is_rejected(client):
    response = client.post("/reconstruct", files=[("files", ("a.png", b"", "image/png"))])
    assert response.status_code == 400


def test_unknown_job_returns_404(client):
    assert client.get("/jobs/deadbeef").status_code == 404


def test_artifact_of_unfinished_job_returns_409(client):
    """Asking for artifacts before a job finishes is a conflict, not a 404."""
    job_queue = service_module._state["queue"]
    # Stop the worker first, so the submitted job stays queued rather than
    # racing to completion before the assertion runs.
    job_queue.stop()
    job = job_queue.submit([])
    assert job.status is JobStatus.QUEUED

    response = client.get(f"/jobs/{job.id}/artifacts/points.ply")
    assert response.status_code == 409
    assert "queued" in response.json()["detail"]


def test_artifact_path_traversal_is_blocked(client):
    """A crafted artifact name must not escape the job directory."""
    job_id = client.post("/reconstruct", files=upload(2)).json()["id"]
    for _ in range(200):
        if client.get(f"/jobs/{job_id}").json()["status"] == "succeeded":
            break
        time.sleep(0.02)
    response = client.get(f"/jobs/{job_id}/artifacts/..%2F..%2Fconfig.py")
    assert response.status_code == 404


def test_readyz_is_503_without_a_model(tmp_path):
    """Readiness must not report ready during the long model load."""
    app = create_app(Settings(eager_load=False, artifact_dir=str(tmp_path)))
    with TestClient(app) as bare:
        assert bare.get("/healthz").status_code == 200
        assert bare.get("/readyz").status_code == 503
        assert bare.get("/readyz").json()["ready"] is False


def test_submitting_before_the_model_loads_is_503(tmp_path):
    app = create_app(Settings(eager_load=False, artifact_dir=str(tmp_path)))
    with TestClient(app) as bare:
        response = bare.post("/reconstruct", files=upload(2))
        assert response.status_code == 503
        assert "loading" in response.json()["detail"].lower()


# ------------------------------------------------------------------ files

class FakeResult:
    """Just enough of a ReconstructionResult for write_results and summary()."""

    extrinsics = np.zeros((1, 3, 4), np.float32)
    intrinsics = np.zeros((1, 3, 3), np.float32)
    depth = np.zeros((1, 2, 2, 1), np.float32)
    depth_confidence = np.ones((1, 2, 2), np.float32)
    world_points = np.zeros((1, 2, 2, 3), np.float32)
    point_confidence = np.ones((1, 2, 2), np.float32)

    def summary(self) -> dict:
        return {"views": 1}


class CheckingEngine(StubEngine):
    """Fails on anything that is not a PNG, the way the real loader does."""

    def reconstruct(self, image_paths):
        for path in image_paths:
            if not Path(path).read_bytes().startswith(PNG_1x1[:8]):
                raise ValueError(f"cannot identify image file {str(path)!r}")
        return FakeResult()


@contextmanager
def running_service(tmp_path, retention: int = 100):
    """The app with the real job handler, so the files it leaves are real too."""
    app = create_app(Settings(eager_load=False, artifact_dir=str(tmp_path / "artifacts"),
                              job_retention=retention))
    app.dependency_overrides[require_engine] = lambda: CheckingEngine()
    with TestClient(app) as test_client:
        service_module._state["engine"] = CheckingEngine()
        yield test_client


def run_job(client, files) -> tuple[str, str]:
    """Submit a job, wait for it to finish, and return its id and status."""
    job_id = client.post("/reconstruct", files=files).json()["id"]
    for _ in range(250):
        status = client.get(f"/jobs/{job_id}").json()["status"]
        if status in ("succeeded", "failed"):
            return job_id, status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


def test_uploads_are_deleted_whether_the_job_succeeds_or_fails(tmp_path):
    """A failed job used to keep the user's images, since only success deleted them."""
    with running_service(tmp_path) as client:
        not_an_image = [("files", ("a.png", b"not an image", "image/png"))]
        assert run_job(client, not_an_image)[1] == "failed"
        assert run_job(client, upload(2))[1] == "succeeded"
        assert list((tmp_path / "artifacts" / ".uploads").iterdir()) == []


def test_a_failed_write_leaves_no_partial_directory(tmp_path, monkeypatch):
    def write_partway(result, directory):
        directory.mkdir(parents=True)
        (directory / "cameras.npz").write_bytes(b"partial")
        raise OSError("No space left on device")

    monkeypatch.setattr(service_module, "write_results", write_partway)
    with running_service(tmp_path) as client:
        job_id, status = run_job(client, upload(2))
        assert status == "failed"
        assert not (tmp_path / "artifacts" / job_id).exists()


def test_an_evicted_job_takes_its_files_with_it(tmp_path):
    """Files of evicted jobs used to stay on disk, unreachable, indefinitely."""
    with running_service(tmp_path, retention=1) as client:
        first, _ = run_job(client, upload(2))
        second, _ = run_job(client, upload(2))
        assert eventually(lambda: client.get(f"/jobs/{first}").status_code == 404)
        assert eventually(lambda: not (tmp_path / "artifacts" / first).exists())
        assert client.get(f"/jobs/{second}/artifacts/points.ply").status_code == 200


def test_startup_deletes_what_a_previous_process_left(tmp_path):
    """Jobs do not survive a restart, so their files and uploads go too."""
    artifacts = tmp_path / "artifacts"
    old_job = artifacts / "0123456789ab"
    old_upload = artifacts / ".uploads" / "vggt-upload-x1y2z3"
    unrelated = artifacts / "notes"
    for directory in (old_job, old_upload, unrelated):
        directory.mkdir(parents=True)
        (directory / "file").write_bytes(b"x")

    with running_service(tmp_path):
        assert not old_job.exists()
        assert not old_upload.exists()
        assert (unrelated / "file").exists(), "only names the service creates are deleted"


def test_a_download_survives_its_job_being_evicted(tmp_path):
    """The handler opens the file itself, so deleting it mid-response is harmless.

    A FileResponse opened the file only while sending, after the 200 had gone
    out. On Windows the delete fails instead, and the file is left for the
    next startup, so there this test only checks that the download completes.
    """
    with running_service(tmp_path) as client:
        job_id, _ = run_job(client, upload(2))
        path = tmp_path / "artifacts" / job_id / "points.ply"
        expected = path.read_bytes()

        endpoint = next(route.endpoint for route in client.app.routes
                        if getattr(route, "path", "") == "/jobs/{job_id}/artifacts/{name}")
        response = endpoint(job_id, "points.ply", service_module._state["queue"])
        service_module._discard_job(job_id)  # what eviction does

        async def body():
            return b"".join([chunk async for chunk in response.body_iterator])

        assert asyncio.run(body()) == expected
        if sys.platform != "win32":
            assert not path.exists()
