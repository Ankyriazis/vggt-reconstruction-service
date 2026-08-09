"""Tests for the reconstruction service.

None of these load VGGT, touch a GPU, or reach the network. The engine is
replaced through FastAPI's dependency system and the job handler is swapped
directly, which is the reason the queue and the HTTP layer were kept separate
from the model in the first place.
"""

from __future__ import annotations

import sys
import time
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
