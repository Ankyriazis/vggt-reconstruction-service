------------------------------- MODULE JobQueue -------------------------------
(***************************************************************************)
(* The reconstruction job queue (src/reconstruction/jobs.py) and the parts *)
(* of the HTTP layer (src/reconstruction/service.py) that touch a job: the *)
(* POST /reconstruct that submits it, the GET /jobs/{id} that polls it, the *)
(* GET /jobs/{id}/artifacts/{name} that downloads it, and the files it      *)
(* leaves on disk. Each action cites the code it stands for, by its line   *)
(* in commit 75e067f, the code before the fixes this model led to.         *)
(*                                                                         *)
(* Code is the set of changes to 75e067f being checked. The code now has   *)
(* the last five; {} is the code as it was.                                *)
(*                                                                         *)
(*   "EvictOnSubmitOnly"  the code before eviction also ran as jobs        *)
(*                        finished: the retention bug the tests caught     *)
(*   "CleanUploads"       a job's uploads are deleted whatever its         *)
(*                        outcome, and startup deletes any a previous      *)
(*                        process left (service.py: _handle_job,           *)
(*                        _discard_leftovers)                              *)
(*   "RetainFinished"     retention counts finished jobs only, evicts them *)
(*                        in the order they finished, and only as a job    *)
(*                        finishes (jobs.py: JobQueue._finish_locked)      *)
(*   "DeleteFiles"        an evicted or failed job's directory is deleted, *)
(*                        and startup deletes job directories a previous   *)
(*                        process left (service.py: _discard_job via       *)
(*                        JobQueue's on_evict, _handle_job,                *)
(*                        _discard_leftovers)                              *)
(*   "OpenFirst"          the artifact handler opens the file before it    *)
(*                        answers, instead of leaving that to FileResponse *)
(*                        (service.py: get_artifact)                       *)
(*   "SingleUse"          start() after stop() raises (jobs.py:            *)
(*                        JobQueue.start)                                  *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    Jobs,           \* the jobs clients submit; each has its own client
    Retention,      \* VGGT_JOB_RETENTION
    Failures,       \* how a job may fail: SUBSET {"inference", "write"}
    Workers,        \* worker thread identities; each start() takes a new one
    QueueRestarts,  \* stop() calls on one JobQueue; the service makes none
    Redeploys,      \* process restarts: a deploy, a crash, `docker restart`
    Code            \* the changes to today's code being checked (above)

ASSUME Retention \in Nat \ {0}
ASSUME Failures \subseteq {"inference", "write"}
ASSUME QueueRestarts \in Nat /\ Redeploys \in Nat
ASSUME Cardinality(Workers) > QueueRestarts
ASSUME Code \subseteq {"EvictOnSubmitOnly", "CleanUploads", "RetainFinished",
                       "DeleteFiles", "OpenFirst", "SingleUse"}

Has(change) == change \in Code

StopItem    == "stop"   \* the "" that stop() puts on the queue (jobs.py:100)
NoJob       == "none"
NoWorker    == "none"
FirstWorker == CHOOSE w \in Workers : TRUE

Unfinished == {"queued", "running"}
Finished   == {"succeeded", "failed"}

VARIABLES
    status,     \* each Job's status: "new" until submitted, "lost" if a
                \* restart dropped it unfinished
    tracked,    \* the jobs in JobQueue._jobs
    order,      \* the order they were added to _jobs, in this process
    finished,   \* the order this process's jobs finished in
    queue,      \* JobQueue._queue: job ids and stop items
    stopping,   \* JobQueue._stopping
    worker,     \* JobQueue._worker
    wpc,        \* where each worker thread is in _run
    wjob,       \* the job each worker thread holds
    ctl,        \* the caller of stop() and start()
    stops,      \* stop() calls so far
    redeploys,  \* process restarts so far
    uploads,    \* jobs whose uploaded images are on disk
    disk,       \* each job's artifact directory: "none", "partial", "complete"
    doomed,     \* evicted jobs whose directory is still to be deleted
    cpc         \* where each job's client is

vars == <<status, tracked, order, finished, queue, stopping, worker, wpc, wjob,
          ctl, stops, redeploys, uploads, disk, doomed, cpc>>

-----------------------------------------------------------------------------
(* Client states:                                                          *)
(*   idle       not submitted yet                                          *)
(*   put        submit has added the job; its id is not on the queue yet   *)
(*   poll       has its 202 and polls GET /jobs/{id}                       *)
(*   download   saw "succeeded" and asks for an artifact                   *)
(*   respond    the handler returned a FileResponse; Starlette sends it    *)
(*   streaming  the handler opened the file and streams it ("OpenFirst")   *)
(*   done       has the outcome (and the file, if the job succeeded)       *)
(*   gone       a poll got 404: the job left the store before the client   *)
(*              saw how it ended                                           *)
(*   notfound   the artifact request got 404                               *)
(*   broken     the artifact request got a 500, or a 200 cut off partway   *)
(*   lost       the process restarted under it                             *)
CPC == {"idle", "put", "poll", "download", "respond", "streaming",
        "done", "gone", "notfound", "broken", "lost"}

WPC == {"unborn", "loop", "get", "claim", "infer", "write", "finally", "dead"}

TypeOK ==
    /\ status \in [Jobs -> {"new", "lost"} \cup Unfinished \cup Finished]
    /\ tracked \subseteq Jobs
    /\ order \in Seq(Jobs)
    /\ finished \in Seq(Jobs)
    /\ queue \in Seq(Jobs \cup {StopItem})
    /\ stopping \in BOOLEAN
    /\ worker \in Workers \cup {NoWorker}
    /\ wpc \in [Workers -> WPC]
    /\ wjob \in [Workers -> Jobs \cup {NoJob}]
    /\ ctl \in {"running", "putting", "joining", "stopped"}
    /\ stops \in 0..QueueRestarts
    /\ redeploys \in 0..Redeploys
    /\ uploads \subseteq Jobs
    /\ disk \in [Jobs -> {"none", "partial", "complete"}]
    /\ doomed \subseteq Jobs
    /\ cpc \in [Jobs -> CPC]

\* The service has just started: lifespan has created the queue and called
\* start() (service.py:86-87).
Init ==
    /\ status = [j \in Jobs |-> "new"]
    /\ tracked = {}
    /\ order = <<>>
    /\ finished = <<>>
    /\ queue = <<>>
    /\ stopping = FALSE
    /\ worker = FirstWorker
    /\ wpc = [w \in Workers |-> IF w = FirstWorker THEN "loop" ELSE "unborn"]
    /\ wjob = [w \in Workers |-> NoJob]
    /\ ctl = "running"
    /\ stops = 0
    /\ redeploys = 0
    /\ uploads = {}
    /\ disk = [j \in Jobs |-> "none"]
    /\ doomed = {}
    /\ cpc = [j \in Jobs |-> "idle"]

-----------------------------------------------------------------------------
(* JobQueue._evict_locked (jobs.py:138-151): while the store holds more    *)
(* than Retention jobs, drop the first finished one in insertion order;    *)
(* stop when none is finished. The jobs it drops from store t, given the   *)
(* insertion order ord and statuses st.                                    *)
(*                                                                         *)
(* With "RetainFinished" only finished jobs count against Retention, and   *)
(* they go in the order they finished.                                     *)
Evicted(t, ord, st) ==
    LET old    == IF Has("RetainFinished")
                  THEN SelectSeq(finished, LAMBDA j : j \in t)
                  ELSE SelectSeq(ord, LAMBDA j : j \in t /\ st[j] \in Finished)
        excess == IF Has("RetainFinished") THEN Len(old) - Retention
                                           ELSE Cardinality(t) - Retention
        n      == IF excess < 0 THEN 0
                  ELSE IF excess < Len(old) THEN excess ELSE Len(old)
    IN  {old[i] : i \in 1..n}

\* With "DeleteFiles", the evicting thread deletes the evicted jobs'
\* directories once it has released the lock (Purge).
Doom(e) == doomed' = IF Has("DeleteFiles") THEN doomed \cup e ELSE doomed

-----------------------------------------------------------------------------
(* Clients *)

(* POST /reconstruct writes the images to a staging directory              *)
(* (service.py:192-217), then JobQueue.submit adds the job and evicts      *)
(* under the lock (jobs.py:112-115). With "RetainFinished" submit does not *)
(* evict: it adds no finished job.                                         *)
Register(j) ==
    /\ cpc[j] = "idle"
    /\ LET t   == tracked \cup {j}
           ord == Append(order, j)
           st  == [status EXCEPT ![j] = "queued"]
           e   == IF Has("RetainFinished") THEN {} ELSE Evicted(t, ord, st)
       IN  /\ status' = st
           /\ order' = ord
           /\ tracked' = t \ e
           /\ Doom(e)
    /\ uploads' = uploads \cup {j}
    /\ cpc' = [cpc EXCEPT ![j] = "put"]
    /\ UNCHANGED <<finished, queue, stopping, worker, wpc, wjob, ctl, stops,
                   redeploys, disk>>

(* ...then puts the id on the queue once the lock is released             *)
(* (jobs.py:116), and the client gets its 202.                             *)
Put(j) ==
    /\ cpc[j] = "put"
    /\ queue' = Append(queue, j)
    /\ cpc' = [cpc EXCEPT ![j] = "poll"]
    /\ UNCHANGED <<status, tracked, order, finished, stopping, worker, wpc,
                   wjob, ctl, stops, redeploys, uploads, disk, doomed>>

(* GET /jobs/{id} (service.py:233-238). A poll that finds the job          *)
(* unfinished changes nothing, so only the poll that ends the wait is a    *)
(* step: it sees the outcome, or a 404 if the job has left the store.      *)
Poll(j) ==
    /\ cpc[j] = "poll"
    /\ \/ j \notin tracked
       \/ status[j] \in Finished
    /\ cpc' = [cpc EXCEPT ![j] = CASE j \notin tracked         -> "gone"
                                   [] status[j] = "succeeded" -> "download"
                                   [] OTHER                   -> "done"]
    /\ UNCHANGED <<status, tracked, order, finished, queue, stopping, worker,
                   wpc, wjob, ctl, stops, redeploys, uploads, disk, doomed>>

(* GET /jobs/{id}/artifacts/{name} (service.py:241-257) checks that the    *)
(* job succeeded and the file exists, then returns a FileResponse, which   *)
(* opens the file only when the response is sent. With "OpenFirst" the     *)
(* handler opens the file itself and streams from the open file, which     *)
(* deleting the directory does not affect (POSIX; the image is Linux).     *)
(* The handler checks the status before it opens the file; a job evicted   *)
(* in between either opens before the delete and streams, or answers 404,  *)
(* the two outcomes this step already has.                                 *)
Download(j) ==
    /\ cpc[j] = "download"
    /\ cpc' = [cpc EXCEPT ![j] =
                 IF j \in tracked /\ status[j] = "succeeded"
                    /\ disk[j] = "complete"
                 THEN IF Has("OpenFirst") THEN "streaming" ELSE "respond"
                 ELSE "notfound"]
    /\ UNCHANGED <<status, tracked, order, finished, queue, stopping, worker,
                   wpc, wjob, ctl, stops, redeploys, uploads, disk, doomed>>

(* Starlette sends the response: it stats the file, sends 200 and the      *)
(* headers, then opens the file. If the file has gone by then, the client  *)
(* gets a 500 or a 200 that stops partway.                                 *)
Respond(j) ==
    /\ cpc[j] \in {"respond", "streaming"}
    /\ cpc' = [cpc EXCEPT ![j] = IF cpc[j] = "streaming" \/ disk[j] = "complete"
                                 THEN "done" ELSE "broken"]
    /\ UNCHANGED <<status, tracked, order, finished, queue, stopping, worker,
                   wpc, wjob, ctl, stops, redeploys, uploads, disk, doomed>>

(* "DeleteFiles": the thread that evicted the job deletes its directory.   *)
Purge(j) ==
    /\ j \in doomed
    /\ doomed' = doomed \ {j}
    /\ disk' = [disk EXCEPT ![j] = "none"]
    /\ UNCHANGED <<status, tracked, order, finished, queue, stopping, worker,
                   wpc, wjob, ctl, stops, redeploys, uploads, cpc>>

-----------------------------------------------------------------------------
(* The worker: JobQueue._run (jobs.py:153-184), for worker thread w *)

\* while not self._stopping.is_set():
Loop(w) ==
    /\ wpc[w] = "loop"
    /\ wpc' = [wpc EXCEPT ![w] = IF stopping THEN "dead" ELSE "get"]
    /\ UNCHANGED <<status, tracked, order, finished, queue, stopping, worker,
                   wjob, ctl, stops, redeploys, uploads, disk, doomed, cpc>>

\* job_id = self._queue.get(), then break on a stop item or the flag.
Get(w) ==
    /\ wpc[w] = "get"
    /\ queue /= <<>>
    /\ queue' = Tail(queue)
    /\ IF Head(queue) = StopItem \/ stopping
       THEN /\ wpc' = [wpc EXCEPT ![w] = "dead"]
            /\ UNCHANGED wjob
       ELSE /\ wpc' = [wpc EXCEPT ![w] = "claim"]
            /\ wjob' = [wjob EXCEPT ![w] = Head(queue)]
    /\ UNCHANGED <<status, tracked, order, finished, stopping, worker, ctl,
                   stops, redeploys, uploads, disk, doomed, cpc>>

\* Look the job up under the lock, skip it if it has gone, mark it RUNNING.
Claim(w) ==
    /\ wpc[w] = "claim"
    /\ IF wjob[w] \in tracked
       THEN /\ status' = [status EXCEPT ![wjob[w]] = "running"]
            /\ wpc' = [wpc EXCEPT ![w] = "infer"]
            /\ UNCHANGED wjob
       ELSE /\ wpc' = [wpc EXCEPT ![w] = "loop"]
            /\ wjob' = [wjob EXCEPT ![w] = NoJob]
            /\ UNCHANGED status
    /\ UNCHANGED <<tracked, order, finished, queue, stopping, worker, ctl,
                   stops, redeploys, uploads, disk, doomed, cpc>>

(* The handler raised and _run marks the job FAILED (jobs.py:172-175).     *)
(* Only the success path unlinks the uploads (service.py:71-72), so today  *)
(* they stay, and so does whatever write_results had written (dir).        *)
Fail(j, dir) ==
    /\ status' = [status EXCEPT ![j] = "failed"]
    /\ finished' = Append(finished, j)
    /\ uploads' = IF Has("CleanUploads") THEN uploads \ {j} ELSE uploads
    /\ disk' = [disk EXCEPT ![j] = IF Has("DeleteFiles") THEN "none" ELSE dir]

\* engine.reconstruct (service.py:68): out of memory, or an upload that has
\* an image's name but is not an image.
Infer(w) ==
    /\ wpc[w] = "infer"
    /\ \/ /\ wpc' = [wpc EXCEPT ![w] = "write"]
          /\ UNCHANGED <<status, finished, uploads, disk>>
       \/ /\ "inference" \in Failures
          /\ wpc' = [wpc EXCEPT ![w] = "finally"]
          /\ Fail(wjob[w], "none")
    /\ UNCHANGED <<tracked, order, queue, stopping, worker, wjob, ctl, stops,
                   redeploys, doomed, cpc>>

(* write_results writes the job's directory (artifacts.py:22-45), the      *)
(* uploads are unlinked (service.py:69-74), and _run records the result    *)
(* and SUCCEEDED (jobs.py:168-171). Writing can fail partway: a full disk. *)
Write(w) ==
    /\ wpc[w] = "write"
    /\ wpc' = [wpc EXCEPT ![w] = "finally"]
    /\ \/ /\ disk' = [disk EXCEPT ![wjob[w]] = "complete"]
          /\ uploads' = uploads \ {wjob[w]}
          /\ status' = [status EXCEPT ![wjob[w]] = "succeeded"]
          /\ finished' = Append(finished, wjob[w])
       \/ /\ "write" \in Failures
          /\ Fail(wjob[w], "partial")
    /\ UNCHANGED <<tracked, order, queue, stopping, worker, wjob, ctl, stops,
                   redeploys, doomed, cpc>>

\* finally: under the lock, drop the payload and evict (jobs.py:176-184).
Finally(w) ==
    /\ wpc[w] = "finally"
    /\ LET e == IF Has("EvictOnSubmitOnly") THEN {}
                ELSE Evicted(tracked, order, status)
       IN  /\ tracked' = tracked \ e
           /\ Doom(e)
    /\ wpc' = [wpc EXCEPT ![w] = "loop"]
    /\ wjob' = [wjob EXCEPT ![w] = NoJob]
    /\ UNCHANGED <<status, order, finished, queue, stopping, worker, ctl, stops,
                   redeploys, uploads, disk, cpc>>

-----------------------------------------------------------------------------
(* stop() and start() on one JobQueue. The service never does this: it     *)
(* starts one queue and stops it on shutdown. Tests and any other caller   *)
(* of the class can.                                                       *)

\* JobQueue.stop() (jobs.py:98-103): set the flag...
StopFlag ==
    /\ ctl = "running"
    /\ stops < QueueRestarts
    /\ stops' = stops + 1
    /\ stopping' = TRUE
    /\ ctl' = "putting"
    /\ UNCHANGED <<status, tracked, order, finished, queue, worker, wpc, wjob,
                   redeploys, uploads, disk, doomed, cpc>>

\* ...put the stop item...
StopPut ==
    /\ ctl = "putting"
    /\ queue' = Append(queue, StopItem)
    /\ ctl' = "joining"
    /\ UNCHANGED <<status, tracked, order, finished, stopping, worker, wpc,
                   wjob, stops, redeploys, uploads, disk, doomed, cpc>>

(* ...join the worker for up to 5 seconds, and forget it. The join returns *)
(* when the worker exits, or times out while it is in a job: a             *)
(* reconstruction takes tens of seconds.                                   *)
StopJoin ==
    /\ ctl = "joining"
    /\ \/ worker = NoWorker
       \/ wpc[worker] \in {"dead", "infer", "write"}
    /\ worker' = NoWorker
    /\ ctl' = "stopped"
    /\ UNCHANGED <<status, tracked, order, finished, queue, stopping, wpc, wjob,
                   stops, redeploys, uploads, disk, doomed, cpc>>

(* JobQueue.start() (jobs.py:90-96): no worker is recorded after stop(),   *)
(* so it clears the flag and starts a new worker thread. With "SingleUse"  *)
(* it raises instead.                                                      *)
Start ==
    /\ ctl = "stopped"
    /\ ~Has("SingleUse")
    /\ LET w == CHOOSE x \in Workers : wpc[x] = "unborn"
       IN  /\ wpc' = [wpc EXCEPT ![w] = "loop"]
           /\ worker' = w
    /\ stopping' = FALSE
    /\ ctl' = "running"
    /\ UNCHANGED <<status, tracked, order, finished, queue, wjob, stops,
                   redeploys, uploads, disk, doomed, cpc>>

-----------------------------------------------------------------------------
(* The process restarts: a deploy, a crash, or `docker restart`            *)
(* (compose.yaml restarts it unless stopped). What was in memory is gone;  *)
(* files stay, and the artifact directory is a named volume, so it         *)
(* outlives the container too. The new process starts a new queue and     *)
(* worker (service.py:77-91). With "CleanUploads" and "DeleteFiles",       *)
(* startup deletes what the old process left, which nothing can reach.    *)
Redeploy ==
    /\ redeploys < Redeploys
    /\ redeploys' = redeploys + 1
    /\ status' = [j \in Jobs |-> IF status[j] \in Unfinished THEN "lost"
                                                          ELSE status[j]]
    /\ tracked' = {}
    /\ order' = <<>>
    /\ finished' = <<>>
    /\ queue' = <<>>
    /\ stopping' = FALSE
    /\ worker' = FirstWorker
    /\ wpc' = [w \in Workers |-> IF w = FirstWorker THEN "loop" ELSE "unborn"]
    /\ wjob' = [w \in Workers |-> NoJob]
    /\ ctl' = "running"
    /\ uploads' = IF Has("CleanUploads") THEN {} ELSE uploads
    /\ disk' = IF Has("DeleteFiles") THEN [j \in Jobs |-> "none"] ELSE disk
    /\ doomed' = {}
    /\ cpc' = [j \in Jobs |->
                 IF cpc[j] \in {"put", "poll", "download", "respond", "streaming"}
                 THEN "lost" ELSE cpc[j]]
    /\ UNCHANGED stops

-----------------------------------------------------------------------------
WorkerStep(w) == Loop(w) \/ Get(w) \/ Claim(w) \/ Infer(w) \/ Write(w) \/ Finally(w)
ClientStep(j) == Register(j) \/ Put(j) \/ Poll(j) \/ Download(j) \/ Respond(j)

Next ==
    \/ \E j \in Jobs : ClientStep(j) \/ Purge(j)
    \/ \E w \in Workers : WorkerStep(w)
    \/ StopFlag \/ StopPut \/ StopJoin \/ Start
    \/ Redeploy

(* Threads finish what they have started: the workers, a submit holding a  *)
(* job id, a stop() under way, a pending delete. Clients need not come     *)
(* back, and nothing has to stop, start or restart.                        *)
Fairness ==
    /\ \A w \in Workers : WF_vars(WorkerStep(w))
    /\ \A j \in Jobs : WF_vars(Put(j)) /\ WF_vars(Purge(j))
    /\ WF_vars(StopPut \/ StopJoin)

Spec == Init /\ [][Next]_vars /\ Fairness

-----------------------------------------------------------------------------
(* Properties: what the code and its README promise *)

\* One job on the GPU at a time: "Exactly one worker consumes the queue"
\* (jobs.py:7-10).
OneJobAtATime ==
    Cardinality({w \in Workers : wpc[w] \in {"infer", "write"}}) <= 1

\* "Only finished jobs are evicted; dropping a queued or running one would
\* strand a client polling for a job id that quietly disappeared"
\* (jobs.py:141-142).
UnfinishedNeverDropped ==
    \A j \in Jobs : status[j] \in Unfinished => j \in tracked

\* A job the store reports as succeeded has all its files.
SucceededIsDownloadable ==
    \A j \in tracked : status[j] = "succeeded" => disk[j] = "complete"

\* Once nothing is queued or running, the store is within Retention: the bug
\* the tests caught ("Retention is enforced when jobs finish", README).
Idle ==
    /\ \A j \in Jobs : status[j] \notin Unfinished
    /\ \A w \in Workers : wpc[w] /= "finally"

StoreWithinRetentionWhenIdle ==
    Idle => Cardinality(tracked) <= Retention

\* VGGT_JOB_RETENTION is the number of "Finished jobs kept in memory"
\* (README; config.py:44-46): the Retention jobs this process finished last
\* are still in the store.
LastResultsKept ==
    \A i \in 1..Len(finished) :
        i > Len(finished) - Retention => finished[i] \in tracked

\* "Uploads are deleted once consumed. The service keeps derived outputs, not
\* user images" (README): images are on disk only for unfinished jobs.
NoUserImagesKept ==
    \A j \in uploads : status[j] \in Unfinished

\* Every job directory on disk can be reached through the API, or is about to
\* be: it belongs to a succeeded job in the store, to the job being run, or to
\* an evicted job whose delete is under way.
FilesMatchStore ==
    \A j \in Jobs : disk[j] /= "none" =>
        \/ j \in tracked /\ status[j] = "succeeded"
        \/ status[j] = "running"
        \/ j \in doomed

\* A download that has started sends the whole file.
DownloadsComplete ==
    \A j \in Jobs : cpc[j] /= "broken"

\* A submitted job finishes, unless its queue is stopped or the process
\* restarts.
EveryJobFinishes ==
    \A j \in Jobs :
        status[j] = "queued" ~> (status[j] \in Finished \cup {"lost"} \/ ctl /= "running")

\* An evicted job's directory is deleted.
EvictedFilesDeleted ==
    \A j \in Jobs : j \in doomed ~> j \notin doomed

-----------------------------------------------------------------------------
(* Witnesses. Each should FAIL on the fixed code, showing that the model   *)
(* reaches the situation and that the passes are not vacuous.              *)

\* A job's directory is deleted while its file is being sent.
NoDeleteDuringDownload ==
    \A j \in Jobs : cpc[j] = "streaming" => disk[j] /= "none"

\* A client polls too late and finds its job evicted.
NoMissedResult ==
    \A j \in Jobs : cpc[j] /= "gone"

=============================================================================
