# The job queue, checked with TLA+

A TLA+ model of the reconstruction job queue, model-checked with TLC. It
covers `JobQueue` (`src/reconstruction/jobs.py`) and the parts of the HTTP
layer that touch a job: `POST /reconstruct`, polling `GET /jobs/{id}`,
downloading `GET /jobs/{id}/artifacts/{name}`, the files a job leaves on disk,
and a process restart.

| Path | What it is |
|---|---|
| `spec/JobQueue.tla` | The queue, its worker threads, its clients and the files on disk. Each step cites the code it stands for. The `Code` constant switches the proposed fixes on one at a time |
| `spec/*.cfg` | One scenario each. The comment at the top says what it asks and what to expect; the runner reads the expectation from there |
| `scripts/model_check.py` | Runs TLC on every scenario; counterexamples go to `out/model-check/summary.md` |
| `scripts/tlc.py` | Runs TLC and parses its output |
| `tools/tla2tools.jar` | TLA+ tools (TLC 2.19; needs Java 11+). Not committed: copy it from another project or download the [latest release](https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar) |

## Model checking

    python scripts/model_check.py           # everything, about 80 seconds
    python scripts/model_check.py Fixed     # only configs whose name contains "Fixed"

It exits non-zero if any result differs from the one its config expects. A
counterexample in `summary.md` is a numbered list of steps; each says what
happened and what it changed.

## Properties

| Property | Kind | Meaning |
|---|---|---|
| `OneJobAtATime` | invariant | At most one job is on the GPU: "Exactly one worker consumes the queue" |
| `UnfinishedNeverDropped` | invariant | A queued or running job is always in the store, so its client can poll it |
| `SucceededIsDownloadable` | invariant | A job the store reports as succeeded has all its files |
| `StoreWithinRetentionWhenIdle` | invariant | Once nothing is queued or running, the store holds at most `VGGT_JOB_RETENTION` jobs |
| `LastResultsKept` | invariant | The `VGGT_JOB_RETENTION` jobs that finished last are in the store ("Finished jobs kept in memory") |
| `NoUserImagesKept` | invariant | Uploaded images are on disk only while their job is queued or running ("Uploads are deleted once consumed") |
| `FilesMatchStore` | invariant | Every job directory on disk belongs to a succeeded job in the store, to the running job, or to an evicted job whose delete is under way |
| `DownloadsComplete` | invariant | A download that starts sends the whole file |
| `EveryJobFinishes` | liveness | A submitted job finishes, unless its queue is stopped or the process restarts |
| `EvictedFilesDeleted` | liveness | An evicted job's directory is deleted |

## Results

| Config | Scenario | Result |
|---|---|---|
| `Service.cfg` | Today's code as deployed: 3 clients, jobs that fail in either step, retention 2, a restart (317,367 states) | all properties hold |
| `OldEviction.cfg` | The code before eviction ran as jobs finished | `StoreWithinRetentionWhenIdle` fails |
| `FailedJob.cfg` | A reconstruction fails | `NoUserImagesKept` fails |
| `Backlog.cfg` | A job finishes while another waits; retention 1 | `LastResultsKept` fails |
| `EvictedFiles.cfg` | A finished job is evicted | `FilesMatchStore` fails |
| `RestartFiles.cfg` | The process restarts after a job | `FilesMatchStore` fails |
| `QueueRestart.cfg` | `stop()` then `start()` on one queue | `OneJobAtATime` fails |
| `QueueRestartStalls.cfg` | `stop()` then `start()` on one queue | `EveryJobFinishes` fails |
| `Fixed.cfg` | The proposed fix: 3 clients, jobs that fail in either step, retention 1, a restart (430,577 states) | all properties hold |
| `FixedStop.cfg` | The proposed fix: 2 clients, jobs that fail in either step, retention 1, a `stop()` and a refused `start()`, a restart | all properties hold |
| `FixedWithoutCleanUploads.cfg` | The fix without `CleanUploads`, as in `FixedStop.cfg` | `NoUserImagesKept` fails |
| `FixedWithoutRetainFinished.cfg` | The fix without `RetainFinished` | `LastResultsKept` fails |
| `FixedWithoutDeleteFiles.cfg` | The fix without `DeleteFiles` | `FilesMatchStore` fails |
| `FixedWithoutOpenFirst.cfg` | The fix without `OpenFirst` | `DownloadsComplete` fails |
| `FixedWithoutSingleUse.cfg` | The fix without `SingleUse` | `OneJobAtATime` fails |
| `WitnessDeleteDuringDownload.cfg` | Does the fixed model delete a directory while its file is being sent? | yes: `NoDeleteDuringDownload` fails, as it should |
| `WitnessMissedResult.cfg` | Under the fix, can a client that polls slowly still miss its result? | yes: `NoMissedResult` fails, as it should |

## What holds today

`Service.cfg` checks every interleaving of three clients, the worker, jobs
that fail during inference or partway through writing, and a process restart.
The queue keeps its central promises: one job on the GPU at a time, a queued or
running job is never evicted, a failed job does not stop the worker, every job
finishes, and a job reported as succeeded always has its files (`write_results`
returns before the status is set, `jobs.py:168-171`).

Retention also comes back within its limit once the queue is idle. That is the
bug the test suite caught, and `OldEviction.cfg` shows the model would have
caught it too: with eviction on submit only, TLC finds two jobs submitted back
to back, both finish, and the store stays at two with retention 1.

## What TLC found

Most important first. Each one was then replayed against the real code (the
real `JobQueue`, `_handle_job` and FastAPI app, with a stub engine) and
reproduced.

1. **A failed job keeps the user's images, indefinitely** (`FailedJob.cfg`).
   `_handle_job` unlinks the uploads only after `write_results` returns
   (`service.py:71-72`). When `engine.reconstruct` raises, `_run` marks the job
   failed (`jobs.py:172-175`), and its `finally` drops the payload, which holds
   the only reference to the paths (`jobs.py:179`). Nothing deletes them after
   that. Upload validation checks only the file's suffix, that it is not empty
   and its size (`service.py:197-210`), so a file named `.png` that is not an
   image is enough to trigger this; so is running out of memory. The README says
   "Uploads are deleted once consumed. The service keeps derived outputs, not
   user images." Successful jobs also leave their staging directory behind,
   empty (`tempfile.mkdtemp`, `service.py:192`).

2. **Retention counts queued jobs, so a backlog evicts results as they
   finish** (`Backlog.cfg`). `_evict_locked` compares `len(self._jobs)` with
   the limit (`jobs.py:144`), and that count includes queued and running jobs.
   `VGGT_JOB_RETENTION` is documented as the number of finished jobs kept
   (README; `config.py:44`). With N jobs waiting, only R − N results are kept.
   Once N ≥ R, each result is evicted as soon as it finishes or the next job is
   submitted. Its client then gets 404 for a job that succeeded, which strands
   it in exactly the way the eviction docstring says eviction avoids
   (`jobs.py:141-142`). Replayed with retention 3: one finished job, three
   submitted after it, and the finished job is gone.

3. **Job directories outlive their jobs, on a volume that outlives the
   container** (`EvictedFiles.cfg`, `RestartFiles.cfg`). Eviction drops the
   job from memory only (`jobs.py:147-148`). Its directory stays, and the API
   can no longer reach it: the artifact endpoint answers 404 because the job is
   unknown. Jobs do not survive a restart, so every restart orphans every
   directory. In Docker the artifact directory is a named volume
   (`docker/compose.yaml`), so they survive the container as well. A write that
   fails partway leaves a partial directory for a failed job, whose artifacts
   answer 409. `config.py:45` says retention "bounds metadata rather than pixel
   data", but nothing bounds the pixel data: disk use grows with every job ever
   run. The four-view job in the main README writes a 4.4 MB point cloud on its
   own. Replayed with retention 1 and two jobs: the first answers 404, and its
   `cameras.npz`, `depth.npz` and `points.ply` are still on disk.

4. **`stop()` then `start()` runs two jobs at once** (`QueueRestart.cfg`).
   This one is latent: the service starts one queue per process and never
   restarts it, but the class's own `stop()`/`start()` allow it. `stop()` joins
   the worker for 5 seconds, then forgets it whether or not it has exited
   (`jobs.py:101-103`). A reconstruction takes tens of seconds, so the worker is
   usually still in its job. `start()` sees no worker, clears the flag and
   starts a second one (`jobs.py:91-95`). The first worker finds the flag clear
   when its job ends and keeps going, so two jobs share the GPU. Replayed: two
   jobs ran at once.

5. **`stop()` during a job leaves its stop item on the queue, and the next
   worker exits on it** (`QueueRestartStalls.cfg`). Also latent. A worker that
   is in a job when `stop()` is called exits on the flag once the job ends
   (`jobs.py:154`), without taking the `""` that `stop()` put on the queue
   (`jobs.py:100`). This needs no timeout: the join succeeds. The next
   `start()`'s worker takes the leftover item and exits. `_worker` still points
   at that dead thread, so later `start()` calls do nothing, and every job
   submitted after that stays queued forever. Replayed: a job submitted after
   `stop()` and `start()` was still queued 2 seconds later, with the worker
   dead.

## A fix that TLC checks

`Code` in `JobQueue.tla` switches each change on separately. With all of them
on, `Fixed.cfg` (430,577 states) and `FixedStop.cfg` find no violation of any
property. Each change is needed: with the other four on and that one off, its
property fails (`FixedWithout*.cfg`). To make sure the passes aren't vacuous,
the witness configs show that under the fix a directory does get deleted while
its file is being sent, and `DownloadsComplete` still holds.

1. **Delete the uploads whatever the outcome** (`CleanUploads`). In
   `_handle_job`, move the unlink into a `finally` and remove the staging
   directory with it. Also stage uploads in a directory the service owns rather
   than the system temp directory, and empty it at startup, because a process
   that dies with jobs queued leaves their uploads behind
   (`FixedWithoutCleanUploads.cfg` finds that case first).

2. **Count and evict finished jobs only, oldest first** (`RetainFinished`).
   Keep finished ids in a `deque`, append in `_run`'s `finally`, and evict from
   the left:

   ```python
   self._finished.append(job_id)
   while len(self._finished) > self._retention:
       self._jobs.pop(self._finished.popleft(), None)
   ```

   This replaces the scan in `_evict_locked`, and the call in `submit` goes away,
   since a submit adds no finished job. The store then holds every unfinished
   job plus the last R results, which is what the README says it does.

3. **Delete an evicted or failed job's directory** (`DeleteFiles`).
   `JobQueue` does not know where artifacts live, so give it an
   `on_evict(job_id)` callback that the service points at the job's directory.
   Collect the evicted ids under the lock and delete their directories after
   releasing it, so a poll never waits on the disk. In `_handle_job`, delete the
   job's directory if `write_results` raises. At startup, delete the job
   directories a previous process left. Match only names that look like job ids
   (12 hex characters), since `VGGT_ARTIFACT_DIR` is configurable and may hold
   other files.

4. **Open the file before answering** (`OpenFirst`). With change 3 alone, a
   download that races its job's eviction breaks (`FixedWithoutOpenFirst.cfg`).
   `get_artifact` checks that the file exists and returns a `FileResponse`.
   Starlette stats and opens the file only when it sends the response. If the
   file has gone before the stat, the client gets a 500; if it goes after the
   headers, a 200 cut off partway (checked against Starlette 1.10). Open the
   file in the handler, answer 404 if that fails, and stream from the open file.
   On Linux, which the Docker image runs, an open file stays readable after it
   is deleted. On Windows the delete fails instead, so retry it later rather
   than ignore the error.

5. **Make a `JobQueue` single-use** (`SingleUse`). `start()` after `stop()`
   raises. That is all the service needs, and it rules out findings 4 and 5. A
   queue that has to restart would instead need `stop()` to keep the worker
   until it has exited, and a stop item that a newer worker ignores.

What the fix leaves to the implementation:

- Retention is a count, not a time. A client that polls less often than R jobs
  finish can still miss its result (`WitnessMissedResult.cfg`). With R = 100
  and tens of seconds per job, that window is about an hour; a TTL would make it
  a fixed time.
- The deletes on Windows, above.

## What the model leaves out

- Time. A reconstruction is one inference step and one write step, and the
  5-second join "returns once the worker has exited, or while it is in a job".
- Upload validation, readiness, and the 503s while the model loads. Validation
  rejects a request before the job exists and removes its staging directory
  (`service.py:215-217`).
- `Job.to_dict` reads the job's fields without the lock while the worker writes
  them, and reads the status twice (`jobs.py:50, 56`). A poll can therefore
  return `"status": "running"` together with a `result`. This is harmless to a
  client that goes by the status, and is not modeled.
- The GIL. Each step is atomic, as the attribute writes it stands for are.
- A job's directory is treated as a single file.
- Bounds: at most 3 clients, 2 worker threads, one `stop()` and one restart.
  TLC checks every interleaving within those bounds.
