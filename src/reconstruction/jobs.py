"""Job queue for reconstruction work.

Reconstruction takes seconds to tens of seconds, which rules out answering
synchronously: a client would hold a connection open for the duration, and any
proxy in between would eventually time it out.

Exactly one worker consumes the queue. That is deliberate rather than a
simplification - a second worker would contend for the same GPU, and since the
model already occupies about 4.7 GB of an 8 GB card, concurrent jobs make the
tail worse while improving nothing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus
    view_count: int
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "id": self.id,
            "status": self.status.value,
            "views": self.view_count,
            "created_at": self.created_at,
            "queued_seconds": _elapsed(self.created_at, self.started_at),
            "run_seconds": _elapsed(self.started_at, self.finished_at),
        }
        if self.status is JobStatus.SUCCEEDED:
            payload["result"] = self.result
            payload["artifacts"] = self.artifacts
        if self.status is JobStatus.FAILED:
            payload["error"] = self.error
        return payload


def _elapsed(start: float | None, end: float | None) -> float | None:
    if start is None:
        return None
    return round((end if end is not None else time.time()) - start, 2)


class JobQueue:
    """Bounded, in-process job queue with a single worker thread.

    In-process on purpose: this is one GPU behind one service. Reaching for
    Redis or Celery here would add operational surface without removing the
    real constraint, which is that the card can only do one job at a time.
    The moment you need several GPU workers, or jobs to survive a restart,
    this is the piece to replace - and it is isolated so that it can be.
    """

    def __init__(self, handler, retention: int = 100) -> None:
        self._handler = handler
        self._retention = retention
        self._queue: queue.Queue[str] = queue.Queue()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._payloads: dict[str, list[Path]] = {}
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        if self._worker is not None:
            return
        self._stopping.clear()
        self._worker = threading.Thread(target=self._run, name="reconstruction", daemon=True)
        self._worker.start()
        logger.info("job worker started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        self._queue.put("")  # unblock the worker so it can notice the flag
        if self._worker is not None:
            self._worker.join(timeout=timeout)
            self._worker = None

    def submit(self, image_paths: list[Path]) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            status=JobStatus.QUEUED,
            view_count=len(image_paths),
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._payloads[job.id] = image_paths
            self._evict_locked()
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())[-limit:][::-1]

    def stats(self) -> dict:
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status.value] = counts.get(job.status.value, 0) + 1
            return {
                "tracked": len(self._jobs),
                "pending": self._queue.qsize(),
                "by_status": counts,
            }

    def _evict_locked(self) -> None:
        """Drop the oldest finished jobs once retention is exceeded.

        Only finished jobs are evicted; dropping a queued or running one would
        strand a client polling for a job id that quietly disappeared.
        """
        while len(self._jobs) > self._retention:
            for job_id, job in list(self._jobs.items()):
                if job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                    self._jobs.pop(job_id, None)
                    self._payloads.pop(job_id, None)
                    break
            else:
                return  # nothing evictable; let the queue exceed retention

    def _run(self) -> None:
        while not self._stopping.is_set():
            job_id = self._queue.get()
            if self._stopping.is_set() or not job_id:
                break

            with self._lock:
                job = self._jobs.get(job_id)
                paths = self._payloads.get(job_id, [])
            if job is None:
                continue

            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            try:
                result, artifacts = self._handler(job_id, paths)
                job.result = result
                job.artifacts = artifacts
                job.status = JobStatus.SUCCEEDED
            except Exception as error:  # noqa: BLE001 - a failed job must not kill the worker
                logger.exception("job %s failed", job_id)
                job.error = f"{type(error).__name__}: {error}"
                job.status = JobStatus.FAILED
            finally:
                job.finished_at = time.time()
                with self._lock:
                    self._payloads.pop(job_id, None)
                    # Also evict here, not only on submit. A burst of
                    # submissions leaves every earlier job still queued and so
                    # not yet evictable; without this the store would stay
                    # over its retention limit until the next submission.
                    self._evict_locked()
