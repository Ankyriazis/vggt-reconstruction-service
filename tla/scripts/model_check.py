"""Model-check spec/JobQueue.tla under each of its configurations.

    python scripts/model_check.py            # all checks
    python scripts/model_check.py Fixed      # only configs whose name contains "Fixed"

Each .cfg says at the top what it asks and, on its "Expected:" line, either
"all properties hold" or which property should fail. Writes
out/model-check/summary.md and each raw TLC log next to it, and exits
non-zero if any result differs from what its config expects.
"""
import re
import sys

from tlc import ROOT, describe_trace, function_value, run_tlc

SPEC = ROOT / "spec" / "JobQueue.tla"
OUT = ROOT / "out" / "model-check"

# Today's code first, then the proposed fix, each of its changes left out in
# turn, and the witnesses that its passes are not vacuous.
ORDER = ["Service", "OldEviction", "FailedJob", "Backlog", "EvictedFiles",
         "RestartFiles", "QueueRestart", "QueueRestartStalls", "Fixed", "FixedStop",
         "FixedWithoutCleanUploads", "FixedWithoutRetainFinished", "FixedWithoutDeleteFiles",
         "FixedWithoutOpenFirst", "FixedWithoutSingleUse",
         "WitnessDeleteDuringDownload", "WitnessMissedResult"]

# Variables a counterexample lists; the rest are bookkeeping.
SHOW = {"status", "tracked", "finished", "queue", "stopping", "worker", "wpc",
        "ctl", "uploads", "disk", "doomed", "cpc"}

CFG_KEYWORDS = {"SPECIFICATION", "INIT", "NEXT", "CONSTANT", "CONSTANTS", "INVARIANT", "INVARIANTS",
                "PROPERTY", "PROPERTIES", "SYMMETRY", "VIEW", "ALIAS", "CONSTRAINT", "CONSTRAINTS",
                "ACTION_CONSTRAINT", "ACTION_CONSTRAINTS", "CHECK_DEADLOCK", "POSTCONDITION"}


def read_cfg(cfg):
    """The question a .cfg asks, what it expects, and the properties it lists."""
    text = cfg.read_text(encoding="utf-8")
    header = [line[2:].strip() for line in text.splitlines() if line.startswith("\\*")]
    expected = next(line for line in header if line.startswith("Expected:"))
    question = " ".join(header[:header.index(expected)])
    m = re.match(r"Expected: (\w+) fails", expected)
    words = [w for line in text.splitlines() for w in line.split("\\*")[0].split()]
    properties, section = [], None
    for word in words:
        if word in CFG_KEYWORDS:
            section = word
        elif section in ("PROPERTY", "PROPERTIES"):
            properties.append(word)
    if m is None:
        return question, "ok", "", properties
    broken = m.group(1)
    return question, "property" if broken in properties else "invariant", broken, properties


# ---------------------------------------------------------------- traces

def _set(value):
    return re.findall(r"\w+", value.strip("{}")) if value else []


def _seq(value):
    return [v.strip().strip('"') for v in value.strip("<>").split(",") if v.strip()]


def step_text(label, prev, cur):
    """What happened in one step, naming the job or worker it was about."""
    if prev is None:
        return "start: the service is up and its worker is waiting for a job"

    def fn(var, values):
        return {k: v.strip('"') for k, v in (function_value(values.get(var, "")) or {}).items()}

    def moved(var):
        before, after = fn(var, prev), fn(var, cur)
        return next((k for k in after if before.get(k) != after[k]), None)

    j, w = moved("cpc"), moved("wpc")
    after = fn("cpc", cur).get(j)
    held = status = None
    if w is not None:
        held = fn("wjob", cur).get(w)
        held = held if held and held != "none" else fn("wjob", prev).get(w)
        status = fn("status", cur).get(held)
    evicted = sorted(set(_set(prev["tracked"])) - set(_set(cur["tracked"])))
    evicting = f", evicting {', '.join(evicted)}" if evicted else ""

    if label == "Register":
        return (f"{j}'s client POSTs /reconstruct: its images are staged and "
                f"submit adds {j} under the lock{evicting}")
    if label == "Put":
        return f"submit puts {j} on the queue, and {j}'s client gets 202"
    if label == "Poll":
        return {"gone": f"{j}'s client polls and gets 404: {j} is no longer in the store",
                "download": f"{j}'s client polls: succeeded",
                "done": f"{j}'s client polls: failed"}[after]
    if label == "Download":
        return {"notfound": f"{j}'s client asks for points.ply: 404",
                "respond": f"{j}'s client asks for points.ply: the handler finds the "
                           f"file and returns a FileResponse",
                "streaming": f"{j}'s client asks for points.ply: the handler opens "
                             f"the file and starts streaming it"}[after]
    if label == "Respond":
        if after == "broken":
            return (f"Starlette sends {j}'s file, which has been deleted: a 500, "
                    f"or a 200 cut off partway")
        return f"{j}'s file is sent in full"
    if label == "Purge":
        return f"the thread that evicted {moved('disk')} deletes its directory"
    if label == "Loop":
        if fn("wpc", cur)[w] == "dead":
            return f"{w} sees the stopping flag set and exits"
        return f"{w} finds the stopping flag clear and waits on the queue"
    if label == "Get":
        item = _seq(prev["queue"])[0]
        if fn("wpc", cur)[w] != "dead":
            return f"{w} takes {item} off the queue"
        if item == "stop":
            return f"{w} takes the stop item and exits"
        return f"{w} takes {item}, sees the stopping flag set and exits, dropping {item}"
    if label == "Claim":
        if fn("wpc", cur)[w] == "loop":
            return f"{w} looks {held} up and finds it gone"
        return f"{w} looks {held} up under the lock and marks it RUNNING"
    if label == "Infer":
        if status == "failed":
            return f"{w}: engine.reconstruct raises on {held}, which is marked FAILED"
        return f"{w}: engine.reconstruct runs {held}"
    if label == "Write":
        if status == "failed":
            return f"{w}: write_results fails partway for {held}, which is marked FAILED"
        return f"{w} writes {held}'s artifacts, unlinks its uploads and marks it SUCCEEDED"
    if label == "Finally":
        return f"{w}'s finally evicts under the lock{evicting or ': nothing to evict'}"
    if label == "StopFlag":
        return "stop() sets the stopping flag"
    if label == "StopPut":
        return "stop() puts the stop item on the queue"
    if label == "StopJoin":
        joined = prev["worker"]
        if fn("wpc", prev).get(joined) == "dead":
            return f"stop() joins {joined}, which has exited, and forgets it"
        return (f"stop()'s 5-second join times out while {joined} is still in a job, "
                f"and it forgets {joined}")
    if label == "Start":
        return f"start() finds no worker recorded, clears the flag and starts {w}"
    if label == "Redeploy":
        return "the process restarts: what was in memory is gone, the files stay"
    return label


# ---------------------------------------------------------------- main

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    cfgs = sorted(SPEC.parent.glob("*.cfg"),
                  key=lambda p: ORDER.index(p.stem) if p.stem in ORDER else len(ORDER))
    OUT.mkdir(parents=True, exist_ok=True)
    report = ["# Model checking the reconstruction job queue", ""]
    all_as_expected = True
    for cfg in cfgs:
        if only not in cfg.name:
            continue
        question, expected, prop, listed = read_cfg(cfg)
        # Expected failures run on one worker, for the shortest counterexample.
        workers = "auto" if expected == "ok" else "1"
        r = run_tlc(SPEC, cfg, OUT / "states" / cfg.stem, workers)
        (OUT / f"{cfg.name}.out").write_text(r.output, encoding="utf-8")
        # TLC names a violated invariant, but not a violated temporal
        # property; that one is known only if the .cfg lists just one.
        broken = r.violated or (listed[0] if r.outcome == "property" and len(listed) == 1 else "")
        as_expected = r.outcome == expected and broken == prop
        all_as_expected &= as_expected
        if r.outcome == "ok":
            verdict = "all properties hold"
        elif r.outcome in ("invariant", "action", "property"):
            verdict = f"{broken or 'a temporal property'} fails"
        else:
            verdict = "TLC failed, see " + str(OUT / f"{cfg.name}.out")
        tag = "as expected" if as_expected else "UNEXPECTED"
        print(f"{cfg.name:<34} {verdict:<36} {r.distinct_states:>9,} states  ({tag})")
        report += [f"## {cfg.name}", "", question, "",
                   f"**Result:** {verdict} ({tag}; {r.distinct_states:,} distinct states).", ""]
        if r.trace:
            report += ["Counterexample:", "", "```"]
            report += describe_trace(r, step_text, show=SHOW)
            report += ["```", ""]
    (OUT / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nCounterexamples: {OUT / 'summary.md'}")
    sys.exit(0 if all_as_expected else 1)


if __name__ == "__main__":
    main()
