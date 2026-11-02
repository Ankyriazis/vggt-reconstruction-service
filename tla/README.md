# The job queue, checked with TLA+

A TLA+ model of the reconstruction job queue, model-checked with TLC. It
covers `JobQueue` (`src/reconstruction/jobs.py`) and the parts of the HTTP
layer that touch a job: `POST /reconstruct`, polling `GET /jobs/{id}`,
downloading `GET /jobs/{id}/artifacts/{name}`, the files a job leaves on disk,
and a process restart.

TLC found five problems in the queue as it was at commit 75e067f. They are
fixed, and the model checks the code both before and after the fix.

| Path | What it is |
|---|---|
| `spec/JobQueue.tla` | The queue, its worker threads, its clients and the files on disk. Each step cites the code it stands for, by its line in 75e067f. The `Code` constant switches the five fixes on one at a time |
| `spec/*.cfg` | One scenario each. The comment at the top says what it asks and what to expect; the runner reads the expectation from there |
| `scripts/model_check.py` | Runs TLC on every scenario; counterexamples go to `out/model-check/summary.md` |
| `scripts/tlc.py` | Runs TLC and parses its output |
| `tools/tla2tools.jar` | TLA+ tools (TLC 2.19; needs Java 11+). Not committed: copy it from another project or download the [latest release](https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar) |

## Model checking

    python scripts/model_check.py            # everything, about 50 seconds
    python scripts/model_check.py Without    # only configs whose name contains "Without"

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
| `LastResultsKept` | invariant | The `VGGT_JOB_RETENTION` jobs that finished last are in the store ("Finished jobs kept") |
| `NoUserImagesKept` | invariant | Uploaded images are on disk only while their job is queued or running ("Uploads are deleted once consumed") |
| `FilesMatchStore` | invariant | Every job directory on disk belongs to a succeeded job in the store, to the running job, or to an evicted job whose delete is under way |
| `DownloadsComplete` | invariant | A download that starts sends the whole file |
| `EveryJobFinishes` | liveness | A submitted job finishes, unless its queue is stopped or the process restarts |
| `EvictedFilesDeleted` | liveness | An evicted job's directory is deleted |

## Results

The code as it is:

| Config | Scenario | Result |
|---|---|---|
| `Service.cfg` | 3 clients, jobs that fail in either step, retention 1, a restart (425,393 states) | all properties hold |
| `ServiceStop.cfg` | 2 clients, jobs that fail in either step, retention 1, a `stop()` and a `start()` that raises, a restart | all properties hold |
| `WithoutCleanUploads.cfg` | Each fix undone in turn, in the setting of `ServiceStop.cfg` | `NoUserImagesKept` fails |
| `WithoutRetainFinished.cfg` | | `LastResultsKept` fails |
| `WithoutDeleteFiles.cfg` | | `FilesMatchStore` fails |
| `WithoutOpenFirst.cfg` | | `DownloadsComplete` fails |
| `WithoutSingleUse.cfg` | | `OneJobAtATime` fails |
| `WitnessDeleteDuringDownload.cfg` | Is a directory ever deleted while its file is being sent? | yes: `NoDeleteDuringDownload` fails, as it should |
| `WitnessMissedResult.cfg` | Can a client that polls slowly still miss its result? | yes: `NoMissedResult` fails, as it should |

The code before the fix (75e067f):

| Config | Scenario | Result |
|---|---|---|
| `Before.cfg` | 3 clients, jobs that fail in either step, retention 2, a restart (317,367 states) | the properties it lists hold |
| `OldEviction.cfg` | The code before eviction ran as jobs finished | `StoreWithinRetentionWhenIdle` fails |
| `FailedJob.cfg` | A reconstruction fails | `NoUserImagesKept` fails |
| `Backlog.cfg` | A job finishes while another waits; retention 1 | `LastResultsKept` fails |
| `EvictedFiles.cfg` | A finished job is evicted | `FilesMatchStore` fails |
| `RestartFiles.cfg` | The process restarts after a job | `FilesMatchStore` fails |
| `QueueRestart.cfg` | `stop()` then `start()` on one queue | `OneJobAtATime` fails |
| `QueueRestartStalls.cfg` | `stop()` then `start()` on one queue | `EveryJobFinishes` fails |

## What held before the fix

`Before.cfg` checks every interleaving of three clients, the worker, jobs that
fail during inference or partway through writing, and a process restart. The
queue kept its central promises: one job on the GPU at a time, a queued or
running job is never evicted, a failed job does not stop the worker, every job
finishes, and a job reported as succeeded always has its files (`write_results`
returns before the status is set).

Retention also came back within its limit once the queue was idle. That is the
bug the test suite caught earlier, and `OldEviction.cfg` shows the model would
have caught it too: with eviction on submit only, TLC finds two jobs submitted
back to back, both finish, and the store stays at two with retention 1.

## What TLC found

In 75e067f; the line numbers are from that commit. Each finding was replayed
against the real code (the real `JobQueue`, `_handle_job` and FastAPI app, with
a stub engine) and reproduced. `tests/test_service.py` now has a test for each
one, and each of those tests fails on 75e067f.

1. **A failed job kept the user's images, indefinitely** (`FailedJob.cfg`).
   `_handle_job` unlinked the uploads only after `write_results` returned
   (`service.py:71-72`). When `engine.reconstruct` raised, `_run` marked the job
   failed (`jobs.py:172-175`), and its `finally` dropped the payload, which held
   the only reference to the paths (`jobs.py:179`). Nothing deleted them after
   that. Upload validation checks only a file's suffix, that it is not empty,
   and its size (`service.py:197-210`), so a file named `.png` that is not an
   image was enough; so was running out of memory. Successful jobs left their
   staging directory behind, empty (`service.py:192`). The README promised
   "Uploads are deleted once consumed".

2. **Retention counted queued jobs, so a backlog evicted results as they
   finished** (`Backlog.cfg`). `_evict_locked` compared `len(self._jobs)` with
   the limit (`jobs.py:144`), and that count includes queued and running jobs.
   `VGGT_JOB_RETENTION` was documented as the number of finished jobs kept.
   With N jobs waiting, only R − N results were kept, and once N ≥ R each
   result was evicted as soon as it finished or the next job was submitted. Its
   client then got 404 for a job that had succeeded. That strands the client in
   exactly the way the eviction docstring said eviction avoids
   (`jobs.py:141-142`).

3. **Job directories outlived their jobs, on a volume that outlives the
   container** (`EvictedFiles.cfg`, `RestartFiles.cfg`). Eviction dropped a job
   from memory only (`jobs.py:147-148`). Its directory stayed, and the API could
   no longer reach it. Jobs do not survive a restart, so every restart orphaned
   every directory, and in Docker the artifact directory is a named volume. A
   write that failed partway left a partial directory behind. Disk use grew with
   every job ever run.

4. **`stop()` then `start()` could run two jobs at once** (`QueueRestart.cfg`).
   This was latent: the service makes one queue per process and never restarts
   it. `stop()` joins the worker for 5 seconds, then forgets it whether or not it
   has exited (`jobs.py:101-103`). `start()` then started a second worker
   (`jobs.py:91-95`), and the first, finding the stop flag clear when its job
   ended, kept going.

5. **`stop()` during a job left its stop item for the next worker**
   (`QueueRestartStalls.cfg`). Also latent. A worker that is in a job when
   `stop()` is called exits on the flag (`jobs.py:154`), without taking the `""`
   that `stop()` queued (`jobs.py:100`). This needs no timeout. The next
   `start()`'s worker took that item and exited. `_worker` still pointed at it,
   so later `start()` calls did nothing, and every job stayed queued forever.

## The fix

Each fix is one value of `Code` in `JobQueue.tla`, and `Service.cfg` checks the
code with all five. Each one is needed: the matching `Without*.cfg`, with only
that fix undone, fails.

1. **Uploads are deleted whatever the outcome** (`CleanUploads`).
   `_handle_job` deletes the job's staging directory in a `finally`. Uploads
   are staged in `.uploads` under the artifact directory rather than the system
   temp directory, and startup deletes the staging directories a previous
   process left (`_discard_leftovers`). `WithoutCleanUploads.cfg` finds that
   case first: a restart while a job is queued.

2. **Retention counts finished jobs, oldest finished first**
   (`RetainFinished`). `JobQueue` keeps finished ids in a deque and evicts from
   its left as each job finishes (`_finish_locked`). `submit` no longer evicts,
   since it adds no finished job. The store holds every unfinished job plus the
   last R results.

3. **An evicted or failed job's directory is deleted** (`DeleteFiles`).
   `JobQueue` takes an `on_evict` callback, which the service points at
   `_discard_job`. The worker calls it after releasing the lock, so a poll
   never waits on the disk, and an error in it is logged rather than stopping
   the worker. `_handle_job` deletes a directory that `write_results` left half
   written. Startup deletes the job directories a previous process left; it
   touches only names of 12 hex characters, since `VGGT_ARTIFACT_DIR` may hold
   other files.

4. **The artifact handler opens the file before answering** (`OpenFirst`).
   With fix 3 alone, a download that races its job's eviction breaks
   (`WithoutOpenFirst.cfg`). A `FileResponse` stats and opens the file only
   while sending: a 500 if the file had gone before the stat, a 200 cut off
   partway if after the headers (checked against Starlette 1.10). The handler
   now opens the file itself, answers 404 if it can't, and streams from the
   open file, which stays readable when deleted.

5. **A `JobQueue` is single-use** (`SingleUse`). `start()` after `stop()`
   raises, which rules out findings 4 and 5. That is all the service needs.

What the fix leaves open:

- Retention is a count, not a time. A client that polls less often than R jobs
  finish can still miss its result (`WitnessMissedResult.cfg`). With R = 100
  and tens of seconds per job, that is about an hour.
- On Windows a file that is being downloaded cannot be deleted. An eviction
  then leaves that directory for the next startup to delete. The model assumes
  POSIX, which the Docker image is.
- Startup deletes job directories in the artifact directory, so each artifact
  directory belongs to one service process. One worker per GPU already implies
  that.

## What the model leaves out

- Time. A reconstruction is one inference step and one write step, and the
  5-second join "returns once the worker has exited, or while it is in a job".
- Upload validation, readiness, and the 503s while the model loads. Validation
  rejects a request before the job exists and removes its staging directory.
- `Job.to_dict` reads the job's fields without the lock while the worker writes
  them, and reads the status twice. A poll can therefore return
  `"status": "running"` together with a `result`. That is harmless to a client
  that goes by the status, and is not modeled.
- The GIL. Each step is atomic, as the attribute writes it stands for are.
- A job's directory is treated as a single file.
- Bounds: at most 3 clients, 2 worker threads, one `stop()` and one restart.
  TLC checks every interleaving within those bounds.
