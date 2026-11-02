"""Run TLC and turn its output into something readable."""
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JAR = Path(os.environ.get("TLA2TOOLS_JAR", ROOT / "tools" / "tla2tools.jar"))
JAR_URL = "https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar"


@dataclass
class State:
    label: str     # the action that led to this state, e.g. "Register"
    values: dict   # variable -> value, as TLC prints it


@dataclass
class Result:
    outcome: str              # "ok", "invariant", "action", "property" or "error"
    violated: str = ""        # the violated invariant or action property, if TLC names it
    trace: list = field(default_factory=list)
    loops_to: int = 0         # liveness counterexamples loop back to this step
    stuttering: bool = False  # ...or end by doing nothing forever
    distinct_states: int = 0
    output: str = ""


def run_tlc(spec: Path, cfg: Path, metadir: Path, workers="1", timeout: int = 900) -> Result:
    """Model-check spec with cfg; both must be in the same directory.

    With one worker the search is breadth-first and deterministic, so
    counterexamples are the shortest ones and the same on every run.
    """
    if not JAR.exists():
        raise SystemExit(f"TLA+ tools not found at {JAR}.\n"
                         f"Download {JAR_URL} there, or set TLA2TOOLS_JAR.")
    cmd = ["java", "-XX:+UseParallelGC", "-cp", str(JAR), "tlc2.TLC",
           "-deadlock", "-workers", str(workers), "-metadir", str(metadir),
           "-config", cfg.name, spec.name]
    # TLC exits non-zero when it finds a violation, which parse() reports.
    proc = subprocess.run(cmd, cwd=spec.parent, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, check=False)
    return parse(proc.stdout + proc.stderr)


def parse(out: str) -> Result:
    r = Result(outcome="error", output=out)
    # The last count is the final one; earlier ones are progress reports.
    if counts := re.findall(r"([\d,]+) distinct states found", out):
        r.distinct_states = int(counts[-1].replace(",", ""))
    if "Model checking completed. No error has been found." in out:
        r.outcome = "ok"
    elif m := re.search(r"Error: Invariant (\w+) is violated", out):
        r.outcome, r.violated = "invariant", m.group(1)
    elif m := re.search(r"Error: Action property (\w+) is violated", out):
        r.outcome, r.violated = "action", m.group(1)
    elif "Error: Temporal properties were violated" in out:
        r.outcome = "property"
    r.trace, r.loops_to, r.stuttering = _parse_trace(out)
    return r


_STATE = re.compile(r"^State (\d+): (?:<(\w+)|(Stuttering))")
_BACK = re.compile(r"^Back to state (\d+)")


def _parse_trace(out):
    trace, loops_to, stuttering = [], 0, False
    current, var = None, None
    for line in out.splitlines():
        if m := _STATE.match(line):
            if m.group(3):
                stuttering, current = True, None
            else:
                current = State(m.group(2), {})
                trace.append(current)
            var = None
        elif m := _BACK.match(line):
            loops_to, current = int(m.group(1)), None
        elif current is not None and line.startswith("/\\ "):
            var, _, value = line[3:].partition(" = ")
            current.values[var] = value.strip()
        elif current is not None and var and line.strip():
            current.values[var] += " " + line.strip()   # value wrapped over lines
        elif not line.strip():
            current, var = None, None
    return trace, loops_to, stuttering


_ENTRY = re.compile(r'(\w+) :> ("[^"]*"|\w+)')


def function_value(value: str) -> dict | None:
    """A TLC function such as (j1 :> "none" @@ j2 :> "queued") as a dict."""
    if not (value.startswith("(") and ":>" in value):
        return None
    return dict(_ENTRY.findall(value))


def changes(prev: dict, cur: dict, show=None) -> list:
    """The variables that changed, one entry per changed key of a function."""
    listed = []
    for name, value in cur.items():
        if show is not None and name not in show or prev.get(name) == value:
            continue
        before, after = function_value(prev.get(name, "")), function_value(value)
        if before is not None and after is not None:
            listed += [f"{name}[{k}] = {v}" for k, v in after.items() if before.get(k) != v]
        else:
            listed.append(f"{name} = {value}")
    return listed


def describe_trace(result: Result, text, show=None) -> list:
    """One line per step: what happened, then what it changed.

    text(label, prev, cur) describes the step that led from prev to cur
    (prev is None for the initial state); show limits which variables are
    listed.
    """
    lines, prev = [], None
    for i, st in enumerate(result.trace, 1):
        listed = changes(prev, st.values, show) if prev else []
        lines.append(f"{i}. {text(st.label, prev, st.values)}"
                     + (f"   [{'; '.join(listed)}]" if listed else ""))
        prev = st.values
    if result.loops_to:
        lines.append(f"... then back to step {result.loops_to}, forever")
    elif result.stuttering:
        lines.append("... then nothing else ever happens")
    return lines
