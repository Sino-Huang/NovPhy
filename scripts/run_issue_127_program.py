"""Issue-127: the external-baselines program summary (device-time budget, decision log, program table and claim).

Modes (CPU only; none reads evaluation data except through the children's published summaries):
- ``--budget``: device time of every #127 scheduler run (``.local-artifacts/gpu-window/queue.json``; job ids ``128-*`` ..
  ``139-*``): per GPU the union of busy intervals, per child its own union per GPU, per-process seconds, each child's own
  ledger total and its projected remaining GPU-h. Program budget = device time (Decision P-budget, #127
  issuecomment-6388410336); remaining = 400 - program device-h. Writes ``budget.json``.
- ``--decisions``: every ``## Decision P-<rule> (<date>): <title>`` comment (and ``Addendum to Decision P-<rule>``) on
  #127 and its children (``gh api``); writes ``decision-log.json`` and ``decision-log.md``.
- ``--summary [--allow-partial]``: after every child has published, the program table, within-family contrasts, the
  intersection-union program claim (U and, secondary, C*), the Tier-C triggers, the E1 native-action table, the
  novelty tables, compute and the not-run list; writes ``summary.json`` (schema ``issue_127_program_summary_v1``) and
  ``findings.md``. Reads ``budget.json`` and ``decision-log.json`` as written by the two modes above (hash-pinned).
- ``--validate``: hash-checks every input ``summary.json`` recorded and re-derives it and ``findings.md`` byte for byte;
  writes ``validation.json``; exit 0 iff both match.

Every interval is DESCRIPTIVE; tokens are as published by the children (delta-rule wording only).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from scripts import external_baselines as eb

ROOT = eb.ROOT
ARTIFACTS = ROOT / ".local-artifacts"
OUTPUT = ARTIFACTS / "issue-127-program-v1"
QUEUE = ARTIFACTS / "gpu-window" / "queue.json"
REPO = "Sino-Huang/NovPhy"
PROGRAM_ISSUE = eb.PROGRAM_ISSUE
BUDGET_GPU_HOURS = 400.0
SCHEMA_BUDGET = "issue_127_program_budget_v1"
SCHEMA_DECISIONS = "issue_127_decision_log_v1"
SCHEMA_SUMMARY = "issue_127_program_summary_v1"
SCHEMA_VALIDATION = "issue_127_program_validation_v1"
IDENTITY = "issue-127-program-v1"
# child issue -> artefact slug and kind ("grid": an eb child summary; "native_action": the E1 table). Later #127
# children (Tier C, coverage, follow-up) are added here; JOB_ID admits job ids 128-* .. 149-*.
CHILDREN = {
    128: {"slug": "direct-predictors", "kind": "grid", "label": "A (A1, A2, A3, A4)"},
    129: {"slug": "slotformer", "kind": "grid", "label": "B1 SlotFormer-E99"},
    130: {"slug": "dino-wm", "kind": "grid", "label": "C1 DINO-WM"},
    131: {"slug": "thick", "kind": "grid", "label": "D1 THICK"},
    132: {"slug": "aibirds-agents", "kind": "native_action", "label": "E1 AIBIRDS agents"},
    133: {"slug": "dreamerv3", "kind": "grid", "label": "T1 DreamerV3 (coverage)"},
    134: {"slug": "gamma-models", "kind": "grid", "label": "T3 gamma-models (coverage)"},
    135: {"slug": "ddlp", "kind": "grid", "label": "T2 DDLP (Tier C, triggered)"},
    136: {"slug": "diffusion-forcing", "kind": "grid", "label": "T4 Diffusion Forcing (coverage)"},
    137: {"slug": "diamond", "kind": "grid", "label": "T5 DIAMOND (coverage)"},
    138: {"slug": "vjepa2-ac", "kind": "grid", "label": "T6 V-JEPA 2-AC (coverage)"},
    139: {"slug": "deepphy", "kind": "grid", "label": "T7 DeepPHY (coverage)"},
    140: {"slug": "u-native", "kind": "grid", "label": "U-native (budget-safeguard control)"},
}
JOB_ID = re.compile(r"^1(2[89]|3[0-9]|4[0-9])-")
WORLD_MODEL_ARMS = ("B1", "C1-dino", "C1-car", "D1")
REFERENCE_RANKERS = ("U", "C*", "P", "J", "J_dec")
ISSUE_102_ROWS = {"random": "random", "prior": "prior", "U (F-15-macro)": "F-15-macro", "C* (C-F-15)": "C-F-15",
                  "J (J-1e-04)": "J-1e-04", "J_dec (J-dec-1e-01)": "J-dec-1e-01", "oracle": "oracle"}
SOURCES = {"issue_123_compute": ARTIFACTS / "issue-123-hjepa-comparison-v1" / "compute.json",
           "issue_102_summary": ARTIFACTS / "issue-102-closed-loop-v1" / "summary.json"}
BANNED_WORDS = ("SOTA", "state of the art", "state-of-the-art", "beats", "outperforms")

read_json = eb.read_json
write_json = eb.write_json
json_text = eb.json_text
sha256_of = eb.sha256_of


def log(message):
    eb.log(PROGRAM_ISSUE, message)


def utc_now():
    return eb.utc_now()


def child_dir(artifacts, issue):
    return Path(artifacts) / f"issue-{issue}-{CHILDREN[issue]['slug']}-v1"


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


# ---------------------------------------------------------------------------
# --budget: device time from the scheduler's run records
# ---------------------------------------------------------------------------

def parse_time(text):
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def run_intervals(queue, now):
    """Every #127 run with a launch: (child, gpu, start, end, job id, tag, running). A run without an end time (no
    ended_at / window_killed_at / killed_at) is still running and counts up to now."""
    out = []
    for job in queue["jobs"]:
        if not JOB_ID.match(job["id"]):
            continue
        child = int(job["id"].split("-")[0])
        for run in job.get("runs", []):
            if not run.get("launched_at"):
                continue
            ended = run.get("ended_at") or run.get("window_killed_at") or run.get("killed_at")
            # a run records its own GPU when its job was moved to another GPU between runs
            out.append({"child": child, "gpu": run.get("gpu", job.get("gpu")), "job": job["id"], "tag": run.get("tag"),
                        "start": parse_time(run["launched_at"]), "end": parse_time(ended) if ended else now,
                        "running": ended is None})
    return out


def union_seconds(intervals):
    """Length in seconds of the union of [start, end) intervals."""
    total, current = 0.0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                total += (current[1] - current[0]).total_seconds()
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current is not None:
        total += (current[1] - current[0]).total_seconds()
    return total


def device_block(runs):
    gpu_runs = [r for r in runs if r["gpu"] is not None]
    per_gpu = {str(g): union_seconds([(r["start"], r["end"]) for r in gpu_runs if r["gpu"] == g])
               for g in sorted({r["gpu"] for r in gpu_runs})}
    device = sum(per_gpu.values())
    process = sum((r["end"] - r["start"]).total_seconds() for r in gpu_runs)
    return {"device_seconds": device, "device_gpu_hours": device / 3600, "per_gpu_device_seconds": per_gpu,
            "process_seconds": process, "process_gpu_hours": process / 3600, "runs": len(gpu_runs),
            "running": sorted(r["tag"] for r in gpu_runs if r["running"]),
            "cpu_runs": len(runs) - len(gpu_runs)}


def _projection_128(smoke):
    """#128 smoke.json projection.remaining_gpu_hours (everything left after the spent ledger)."""
    return smoke["projection"]["remaining_gpu_hours"]


def _projection_129(smoke):
    """#129 smoke.json projection: total_gpu_h minus the already-run selection_gpu_h (training + readouts + score +
    benchmarks)."""
    p = smoke["projection"]
    return p["total_gpu_h"] - p["selection_gpu_h"]


# explicit per-child mapping of the latest smoke.json projection to projected remaining GPU-h (missing -> null)
PROJECTIONS = {128: ("smoke.json", _projection_128), 129: ("smoke.json", _projection_129)}


def projected_remaining(artifacts, issue):
    entry = PROJECTIONS.get(issue)
    if entry is None:
        return None, "no projection mapping (no smoke.json projection published yet)"
    path = child_dir(artifacts, issue) / entry[0]
    if not path.is_file():
        return None, f"{rel(path)} missing"
    try:
        return float(entry[1](read_json(path))), f"{rel(path)}: {entry[1].__doc__.strip()}"
    except (KeyError, TypeError):
        return None, f"{rel(path)}: projection fields missing"


def child_ledger(artifacts, issue):
    """The child's own ledger total: its published compute.json if present (#123 A3 rule as applied by the child),
    else the raw ledger/*.json sums (eb.ledger_totals without report substitution; pre-publication)."""
    base = child_dir(artifacts, issue)
    compute = base / "compute.json"
    if compute.is_file():
        c = read_json(compute)
        gpu = c.get("gpu_seconds_total", (c.get("ledger") or {}).get("gpu_seconds"))
        if gpu is not None:
            return {"source": rel(compute), "gpu_seconds": gpu, "gpu_hours": gpu / 3600}
    if (base / "ledger").is_dir():
        _, gpu, wall = eb.ledger_totals(base, lambda phase: None)
        return {"source": f"{rel(base / 'ledger')} (raw ledger sums, pre-publication)", "gpu_seconds": gpu,
                "gpu_hours": gpu / 3600, "wall_seconds": wall}
    return None


def budget(queue_path=QUEUE, artifacts=ARTIFACTS, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    runs = run_intervals(read_json(queue_path), now)
    program = device_block(runs)
    children = {}
    for issue in sorted(set(CHILDREN) | {r["child"] for r in runs}):
        block = device_block([r for r in runs if r["child"] == issue])
        projected, source = projected_remaining(artifacts, issue) if issue in CHILDREN else (None, "unregistered")
        children[str(issue)] = {**block, "ledger": child_ledger(artifacts, issue) if issue in CHILDREN else None,
                                "projected_remaining_gpu_h": projected, "projection_source": source}
    child_sum = sum(c["device_gpu_hours"] for c in children.values())
    return {"schema": SCHEMA_BUDGET, "now": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "queue": rel(queue_path),
            "rule": ("program budget = device time: per GPU the union of the intervals in which at least one #127 run "
                     "was running, summed over GPUs (Decision P-budget, #127 issuecomment-6388410336); a still-running "
                     "run counts up to now"),
            "budget_gpu_hours": BUDGET_GPU_HOURS, "program": program,
            "remaining_gpu_hours": BUDGET_GPU_HOURS - program["device_gpu_hours"], "children": children,
            "sum_child_device_gpu_hours": child_sum,
            "note": ("children share GPUs at the same time, so per-child device times can sum to more than the program "
                     "device time; per-process GPU-h counts every run's own wall time"),
            "projected_remaining_gpu_h_total": sum(c["projected_remaining_gpu_h"] or 0.0 for c in children.values())}


def budget_lines(b):
    lines = ["| child | device GPU-h | process GPU-h | runs (running) | child ledger GPU-h | projected remaining GPU-h |",
             "| --- | --- | --- | --- | --- | --- |"]
    for issue, c in b["children"].items():
        ledger = c["ledger"]
        lines.append(f"| #{issue} | {c['device_gpu_hours']:.2f} | {c['process_gpu_hours']:.2f} | {c['runs']} "
                     f"({len(c['running'])}) | {'n/a' if ledger is None else format(ledger['gpu_hours'], '.2f')} | "
                     f"{'n/a' if c['projected_remaining_gpu_h'] is None else format(c['projected_remaining_gpu_h'], '.2f')} |")
    p = b["program"]
    lines.append(f"| program | {p['device_gpu_hours']:.2f} | {p['process_gpu_hours']:.2f} | {p['runs']} "
                 f"({len(p['running'])}) | | |")
    lines += ["", "Per-GPU device hours: " + ", ".join(f"GPU {g}: {s / 3600:.2f}"
                                                         for g, s in p["per_gpu_device_seconds"].items()),
              f"Sum of per-child device hours {b['sum_child_device_gpu_hours']:.2f} (> program device hours when "
              "children share a GPU at the same time).",
              f"Budget {b['budget_gpu_hours']:.0f} GPU-h; program device time {p['device_gpu_hours']:.2f} GPU-h; "
              f"remaining {b['remaining_gpu_hours']:.2f} GPU-h (as of {b['now']})."]
    return lines


def run_budget(output=OUTPUT):
    b = budget()
    write_json(Path(output) / "budget.json", b)
    print("\n".join(budget_lines(b)))
    return 0


# ---------------------------------------------------------------------------
# --decisions: the automatic decision log
# ---------------------------------------------------------------------------

HEADING = re.compile(r"^##\s+Decision\s+(?P<rule>P-[A-Za-z]+)\s*\((?P<date>\d{4}-\d{2}-\d{2})\)\s*:\s*(?P<title>.+?)\s*$")
ADDENDUM = re.compile(r"^(?:##\s+)?Addendum to Decision\s+(?P<rule>P-[A-Za-z]+)\s*\(?(?P<date>\d{4}-\d{2}-\d{2})\)?"
                      r"\s*(?:\((?P<ref>[^)]*)\))?\s*:\s*(?P<title>.+?)\s*$")


def parse_heading(body):
    """{rule, date, title, addendum, refers_to} of a decision comment's first line, or None."""
    first = (body or "").lstrip().split("\n", 1)[0].strip()
    match = HEADING.match(first)
    if match:
        title = match["title"]
        return {"rule": match["rule"], "date": match["date"], "title": title,
                "addendum": title.lower().startswith("addendum"), "refers_to": None}
    match = ADDENDUM.match(first)
    if match:
        return {"rule": match["rule"], "date": match["date"], "title": f"addendum: {match['title']}",
                "addendum": True, "refers_to": match["ref"]}
    return None


def fetch_comments(issue):
    """Every comment of an issue: [{html_url, created_at, body}] (gh api, paginated)."""
    out = subprocess.run(["gh", "api", "--paginate", f"repos/{REPO}/issues/{issue}/comments", "--jq",
                          ".[] | {html_url, created_at, body}"], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def decision_log(comments_by_issue):
    decisions = []
    for issue, comments in comments_by_issue.items():
        for c in comments:
            parsed = parse_heading(c["body"])
            if parsed is not None:
                decisions.append({"issue": int(issue), "url": c["html_url"], "created_at": c["created_at"], **parsed})
    decisions.sort(key=lambda d: (d["created_at"], d["url"]))
    return decisions


def _md_text(text):
    return text.replace("|", "/").replace("[", "\\[").replace("]", "\\]")


def decision_md(log_):
    lines = ["# #127 automatic decision log", "",
             f"{len(log_['decisions'])} decisions on #{', #'.join(str(i) for i in log_['issues'])} "
             f"(fetched {log_['fetched_at']}; order: created_at, url).", ""]
    for issue in log_["issues"]:
        rows = [d for d in log_["decisions"] if d["issue"] == issue]
        name = "program" if issue == PROGRAM_ISSUE else CHILDREN[issue]["label"]
        lines += [f"## #{issue} ({name}): {len(rows)}", ""]
        if rows:
            lines += ["| created (UTC) | rule | title |", "| --- | --- | --- |"]
            lines += [f"| {d['created_at']} | {d['rule']} | [{_md_text(d['title'])}]({d['url']}) |" for d in rows]
        lines.append("")
    return "\n".join(lines)


def run_decisions(output=OUTPUT):
    issues = [PROGRAM_ISSUE, *sorted(CHILDREN)]
    log_ = {"schema": SCHEMA_DECISIONS, "fetched_at": utc_now(), "repo": REPO, "issues": issues,
            "rule": "comments whose first line is '## Decision P-<rule> (<date>): <title>' or 'Addendum to Decision P-<rule>'",
            "decisions": decision_log({i: fetch_comments(i) for i in issues})}
    write_json(Path(output) / "decision-log.json", log_)
    (Path(output) / "decision-log.md").write_text(decision_md(log_) + "\n")
    counts = {i: sum(d["issue"] == i for d in log_["decisions"]) for i in issues}
    rules = {}
    for d in log_["decisions"]:
        rules[d["rule"]] = rules.get(d["rule"], 0) + 1
    print(json.dumps({"decisions": len(log_["decisions"]), "per_issue": counts, "per_rule": rules}, indent=2))
    return 0


# ---------------------------------------------------------------------------
# --summary: program table, claim, Tier-C triggers, native-action table, novelty, compute
# ---------------------------------------------------------------------------

def load_children(artifacts, allow_partial):
    """{issue: {"status", "summary", "compute", "inputs"}}; a missing grid child is refused unless allow_partial."""
    out = {}
    for issue, spec in sorted(CHILDREN.items()):
        base = child_dir(artifacts, issue)
        path = base / "summary.json"
        entry = {"status": "pending", "summary": None, "compute": None, "inputs": []}
        if path.is_file():
            if spec["kind"] == "grid":
                entry["summary"] = eb.load_child_summary(path)
                if entry["summary"]["issue"] != issue:
                    raise ValueError(f"{path} is the summary of #{entry['summary']['issue']}")
            else:
                entry["summary"] = read_json(path)
            entry["status"] = "published"
            for name in ("summary.json", "records-manifest.json", "compute.json"):
                p = base / name
                entry["inputs"].append({"name": f"issue_{issue}_{name}", "artifact": rel(p),
                                        "sha256": sha256_of(p) if p.is_file() else None})
            if (base / "compute.json").is_file():
                entry["compute"] = read_json(base / "compute.json")
        elif not allow_partial:
            raise ValueError(f"#{issue} has not published {rel(path)}; use --allow-partial for a partial summary")
        out[issue] = entry
    return out


def contrast_entry(c):
    return {"estimate": c["estimate"], "token": c["token"], "reverse_token": c["reverse_token"],
            "guards_pass": c["guards_pass"]}


def program_table(children):
    rows = []
    for issue, entry in children.items():
        if entry["summary"] is None or CHILDREN[issue]["kind"] != "grid":
            continue
        for view, block in entry["summary"]["arms"].items():
            compute = block["compute"]
            seconds = [float(s) for s in (compute.get("training_gpu_seconds") or {}).values() if s is not None]
            exclusive = compute.get("exclusive_training_gpu_seconds")
            rows.append({"child": issue, "view": view, "role": block["role"], "program_set": block["program_set"],
                         "guards_pass": block["guards_pass"], "arm_guards_pass": block["guards"]["pass"],
                         "global_guards_pass": block["global_guards_pass"], "dropped": block["dropped"],
                         "auc": block["ranker"]["auc"],
                         "contrasts": {k: contrast_entry(c) for k, c in block["contrasts"].items()},
                         "disposition": block["disposition"]["disposition"],
                         "budget_safeguard": block["budget_safeguard"],
                         "compute": {"trainable_parameters": compute.get("trainable_parameters"),
                                     "training_gpu_hours": sum(seconds) / 3600 if seconds else None,
                                     "exclusive_training_gpu_hours": (sum(float(s) for s in exclusive.values()) / 3600
                                                                      if isinstance(exclusive, dict) else None),
                                     "macs_per_candidate": compute.get("macs_per_candidate"),
                                     "wall_per_decision": compute.get("wall_per_decision")}})
    return rows


def blocks_by_view(children):
    return {view: block for e in children.values() if e["summary"] is not None and "arms" in e["summary"]
            for view, block in e["summary"]["arms"].items()}


def claim_entries(children, against="U"):
    """The program_claim entries of every published program-set arm; against C* the disposition is re-derived from
    the published X - C* contrast (eb.disposition: 2 iff C* - X supported and the arm's guards pass)."""
    entries = {}
    for e in children.values():
        s = e["summary"]
        if s is None or "program_claim_entries" not in s:
            continue
        for view, entry in s["program_claim_entries"].items():
            if against == "C*":
                block = s["arms"][view]
                entry = {**entry, "disposition": eb.disposition(block["contrasts"]["X-C*"],
                                                                block["guards_pass"])["disposition"]}
            entries[view] = entry
    return entries


def program_claims(children):
    u = eb.program_claim(claim_entries(children, "U"))
    c = eb.program_claim(claim_entries(children, "C*"))
    c["claim"] = "C* ranks launches better than every baseline in this set on the sealed NovPhy grid (secondary)"
    return {"U": u, "C*": c}


def child_extra(children, issue, key):
    s = children.get(issue, {}).get("summary")
    return None if s is None else (s.get("extra") or {}).get(key)


def tier_c(children):
    """The frozen #127 Tier-C triggers and P-follow-up, evaluated mechanically on the published blocks."""
    blocks = blocks_by_view(children)
    fired = sorted(v for v, b in blocks.items() if b["budget_safeguard"]["fires"])
    d1 = blocks.get("D1")
    df = None if d1 is None else bool(d1["contrasts"]["X-U"]["token"] == "supported"
                                      or d1["contrasts"]["X-J_dec"]["token"] == "supported")
    b1 = blocks.get("B1")
    readout = None if b1 is None or "G_readout" not in b1["guards"] else bool(b1["guards"]["G_readout"] is not True)
    positions = child_extra(children, 129, "positions") or {}
    b1_225 = ((positions.get("B1") or {}).get("pred_225") or {}).get("mean")
    u_225 = ((positions.get("U") or {}).get("pred_225") or {}).get("mean")
    position = None if b1_225 is None or u_225 is None else bool(b1_225 > u_225)
    ddlp = None if readout is None and position is None else bool(readout or position)
    wm_one = sorted(v for v in WORLD_MODEL_ARMS if v in blocks and blocks[v]["disposition"]["disposition"] == 1)
    any_one = sorted(v for v, b in blocks.items() if b["disposition"]["disposition"] == 1)
    return {"U-native": {"fires": bool(fired), "arms": fired,
                         "rule": "any arm's budget safeguard fires (X - U supported and X's training GPU-s > U's)"},
            "Diffusion Forcing / gamma-models": {"fires": df, "rule": "D1 - U or D1 - J_dec supported (published tokens)",
                                                 "evaluable": d1 is not None},
            "DDLP": {"fires": ddlp, "B1_fails_G_readout": readout, "B1_position_225": b1_225, "U_position_225": u_225,
                     "B1_position_exceeds_U": position,
                     "rule": ("B1 fails G-readout (where B1 publishes it) or B1's pooled mean position error at t = 225 "
                              "exceeds U's (#129 published position table, pred_225 means; B1 > U)")},
            "DreamerV3 / DIAMOND / V-JEPA 2-AC": {"fires": bool(wm_one), "arms": wm_one,
                                                  "rule": f"any world-model arm ({', '.join(WORLD_MODEL_ARMS)}) reaches disposition 1"},
            "P-follow-up": {"fires": bool(any_one), "arms": any_one, "rule": "any arm reaches disposition 1"}}


def within_family(children):
    out = {}
    for issue, e in children.items():
        s = e["summary"]
        if s is None or "within_family" not in s:
            continue
        for label, c in s["within_family"].items():
            out[f"#{issue} {label}"] = {"label": c.get("label", label), **contrast_entry(c)}
    return out


def native_action(children, issue_102):
    """E1 (as published by #132: summary.json "agents": {agent: {"pass1", "pass3"}}) beside the #102 N1 rows."""
    s = children.get(132, {}).get("summary")
    n1 = issue_102["sets"]["N1"]
    return {"label": ("native-action, DESCRIPTIVE: the agents choose continuous shots on the #102 N1 levels; not the "
                      "grid protocol; #102 rows as published"),
            "status": children.get(132, {}).get("status", "pending"),
            "agents": None if s is None else s.get("agents"),
            "engine_worker_hours": None if s is None else s.get("engine_worker_hours"),
            "issue_102_n1": {label: {k: n1[arm][k] for k in ("pass1", "pass3")} for label, arm in ISSUE_102_ROWS.items()
                             if arm in n1}}


def novelty_tables(children):
    out = {}
    for issue, e in children.items():
        section = child_extra(children, issue, "novelty")
        if section is None and e["compute"] is not None:
            section = e["compute"].get("novelty")
        out[str(issue)] = None if section is None else section.get("table")
    return out


def compute_section(children, budget_):
    arms = {f"#{r['child']} {r['view']}": r["compute"] for r in program_table(children)}
    engine = {str(i): (e["summary"].get("engine_seconds") or 0) / 3600 for i, e in children.items()
              if e["summary"] is not None and CHILDREN[i]["kind"] == "grid"}
    e1 = native_action_engine(children)
    return {"device": {"program_gpu_hours": budget_["program"]["device_gpu_hours"],
                       "per_gpu_device_seconds": budget_["program"]["per_gpu_device_seconds"],
                       "per_child_gpu_hours": {i: c["device_gpu_hours"] for i, c in budget_["children"].items()},
                       "budget_gpu_hours": budget_["budget_gpu_hours"], "remaining_gpu_hours": budget_["remaining_gpu_hours"],
                       "as_of": budget_["now"]},
            "process_gpu_hours": {"program": budget_["program"]["process_gpu_hours"],
                                  **{i: c["process_gpu_hours"] for i, c in budget_["children"].items()}},
            "arms": arms, "engine_worker_hours": {**engine, "132": e1}}


def native_action_engine(children):
    s = children.get(132, {}).get("summary")
    return None if s is None else s.get("engine_worker_hours")


def not_run(children):
    blocks = blocks_by_view(children)
    out = []
    for issue, e in children.items():
        if e["status"] == "pending":
            out.append({"child": issue, "arm": None, "status": "pending", "reason": "child summary not published"})
        pending = child_extra(children, issue, "pending_arms") or {}
        out += [{"child": issue, "arm": a, "status": "not_run", "reason": str(r)} for a, r in sorted(pending.items())]
    for view, block in sorted(blocks.items()):
        if block["dropped"]:
            out.append({"child": next(i for i, e in children.items() if e["summary"] and view in e["summary"].get("arms", {})),
                        "arm": view, "status": "dropped", "reason": block["dropped"]})
    listed = {r["arm"] for r in out}
    out += [{"child": None, "arm": a, "status": "not_run", "reason": "no published arm block"}
            for a in eb.PROGRAM_SET if a not in blocks and a not in listed]
    return out


def reference_rows(issue_123):
    rankers = issue_123["inventories"]["grid"]["rankers"]
    return {name: {"auc": rankers[name]["auc"], "top1_rate": rankers[name]["top1_rate"]} for name in REFERENCE_RANKERS}


def nn_rows(children):
    c = children.get(128, {}).get("compute")
    nn = None if c is None else c.get("no_model_references")
    return None if nn is None else {"label": nn.get("label"), "pooled_grid_auc": nn.get("pooled_grid_auc")}


def build(output=OUTPUT, artifacts=ARTIFACTS, allow_partial=False, sources=None):
    """(summary dict, findings text) from the children's published summaries and this output's budget.json /
    decision-log.json; deterministic (no timestamps of its own)."""
    output = Path(output)
    sources = dict(SOURCES if sources is None else sources)
    for name in ("budget.json", "decision-log.json"):
        if not (output / name).is_file():
            raise ValueError(f"{output / name} missing; run --budget and --decisions first")
    budget_, decisions = read_json(output / "budget.json"), read_json(output / "decision-log.json")
    children = load_children(artifacts, allow_partial)
    inputs = [{"name": n, "artifact": rel(p), "sha256": sha256_of(p)} for n, p in
              (("budget", output / "budget.json"), ("decision_log", output / "decision-log.json"),
               *sources.items())]
    inputs += [i for e in children.values() for i in e["inputs"]]
    issue_123, issue_102 = read_json(sources["issue_123_compute"]), read_json(sources["issue_102_summary"])
    summary = {
        "schema": SCHEMA_SUMMARY, "identity": IDENTITY, "partial": bool(allow_partial), "inputs": inputs,
        "children": {str(i): {"status": e["status"], "label": CHILDREN[i]["label"],
                              "identity": (e["summary"] or {}).get("identity"),
                              "frozen_at": (e["summary"] or {}).get("frozen_at"),
                              "global_guards_pass": (e["summary"] or {}).get("global_guards_pass")}
                     for i, e in children.items()},
        "program_set": list(eb.PROGRAM_SET), "program_table": program_table(children),
        "references": reference_rows(issue_123), "nn_references": nn_rows(children),
        "within_family": within_family(children), "program_claim": program_claims(children),
        "tier_c": tier_c(children), "native_action": native_action(children, issue_102),
        "novelty": novelty_tables(children), "compute": compute_section(children, budget_),
        "not_run": not_run(children),
        "decision_log": {"fetched_at": decisions["fetched_at"], "count": len(decisions["decisions"]),
                         "per_issue": {str(i): sum(d["issue"] == i for d in decisions["decisions"])
                                       for i in decisions["issues"]}}}
    return summary, findings_md(summary, children)


def fmt_auc(block):
    return eb.rc.fmt_auc(block)


def fmt_block(block):
    return eb.rc.fmt_block(block)


def findings_md(summary, children):
    lines = ["# #127 external-baselines program: findings", "",
             ("Every interval is a level-clustered percentile bootstrap (10 000 draws, PCG64 7201), DESCRIPTIVE; "
              f"delta = {eb.DELTA}. Tokens are as published by each child (a failed guard makes every token "
              f"{eb.STOP_TOKEN}). Wording follows the dispositions only."), ""]
    if summary["partial"]:
        pending = [i for i, c in summary["children"].items() if c["status"] == "pending"]
        lines += [f"**Partial summary** (--allow-partial): children not yet published: #{', #'.join(pending)}.", ""]
    claim = summary["program_claim"]["U"]
    lines += ["## Program-level claim (intersection-union over " + ", ".join(eb.PROGRAM_SET) + ")", ""]
    if claim["holds"]:
        lines.append(f"**Licensed:** \"{claim['claim']}\" (disposition 2 for every member; scoped to the sealed #109 "
                     "evaluation split, Gate-B mixed grid cells).")
    else:
        lines.append("**Not licensed.** Vetoes: " + "; ".join(f"{v['arm']} ({v['reason']})" for v in claim["vetoes"]) + ".")
    cstar = summary["program_claim"]["C*"]
    lines += ["", f"Secondary, C* in place of U: {'licensed' if cstar['holds'] else 'not licensed'} "
              f"({len(cstar['vetoes'])} vetoes).", ""]
    lines += ["## Dispositions per arm", ""]
    for row in summary["program_table"]:
        n = row["disposition"]
        text = eb.DISPOSITION_LABELS[n].replace("X", row["view"], 1) if n in (1, 3) else (
            f"U ranks launches better than {row['view']} (scoped)" if n == 2 else "per-contrast tokens only")
        lines.append(f"- #{row['child']} {row['view']} ({row['role']}{', program set' if row['program_set'] else ''}): "
                     f"disposition {n}: {text}" + (f"; dropped ({row['dropped']})" if row["dropped"] else ""))
    blocks = {f"#{i} {v}": {**b, "view": f"#{i} {v}"} for i, e in children.items() if e["summary"] is not None
              and "arms" in e["summary"] for v, b in e["summary"]["arms"].items()}
    lines += ["", "## Program table (grid; X − U)", "", *eb.format_arm_rows(blocks), "",
              "## Secondary contrasts", "", "| arm | X − C* [CI] | token | X − P [CI] | token | X − J_dec [CI] | token |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in summary["program_table"]:
        c = row["contrasts"]
        lines.append(f"| #{row['child']} {row['view']} | " + " | ".join(
            f"{fmt_block(c[k]['estimate'])} | {c[k]['token']}" for k in ("X-C*", "X-P", "X-J_dec")) + " |")
    lines += ["", "## References (Gate-B / #101 records, as published by #123)", "", "| ranker | grid AUC [CI] | top-1 |",
              "| --- | --- | --- |"]
    lines += [f"| {n} | {fmt_auc(r['auc'])} | {eb.rc.fmt_num(r['top1_rate'])} |" for n, r in summary["references"].items()]
    if summary["nn_references"]:
        lines += ["", f"No-model references (#128, {summary['nn_references']['label']}):", ""]
        lines += [f"- {n}: {fmt_auc(b)}" for n, b in (summary["nn_references"]["pooled_grid_auc"] or {}).items()]
    lines += ["", "## Within-family contrasts (as published)", "", "| contrast | estimate [CI] | token | reverse token |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {fmt_block(c['estimate'])} | {c['token']} | {c['reverse_token']} |"
              for k, c in summary["within_family"].items()]
    lines += ["", "## Tier-C triggers and P-follow-up (frozen #127 rules, evaluated mechanically)", ""]
    for name, t in summary["tier_c"].items():
        state = "not evaluable (inputs not published)" if t["fires"] is None else ("fires" if t["fires"] else "does not fire")
        arms = f" ({', '.join(t['arms'])})" if t.get("arms") else ""
        lines.append(f"- {name}: {state}{arms}. Rule: {t['rule']}.")
    na = summary["native_action"]
    lines += ["", "## E1 native-action table (DESCRIPTIVE, separate from the grid protocol)", "", na["label"] + ".", "",
              "| row | pass@1 | pass@3 |", "| --- | --- | --- |"]
    for name, row in (na["agents"] or {}).items():
        lines.append(f"| E1 {name} | {_pass(row.get('pass1'))} | {_pass(row.get('pass3'))} |")
    lines += [f"| #102 {name} | {_pass(r['pass1'])} | {_pass(r['pass3'])} |" for name, r in na["issue_102_n1"].items()]
    if na["agents"] is None:
        lines.append(f"\nE1: {na['status']}.")
    lines += ["", "## Novelty (P-novelty, DESCRIPTIVE; never in a disposition)", ""]
    for issue, table in summary["novelty"].items():
        lines += [f"### #{issue}", ""] + (eb.format_novelty_rows(table) if table else ["not published"]) + [""]
    comp = summary["compute"]
    lines += ["## Compute", "",
              f"Program device time {comp['device']['program_gpu_hours']:.2f} GPU-h of {comp['device']['budget_gpu_hours']:.0f} "
              f"(remaining {comp['device']['remaining_gpu_hours']:.2f}; as of {comp['device']['as_of']}); per-process "
              f"{comp['process_gpu_hours']['program']:.2f} GPU-h (shared-GPU runs counted per process).", "",
              "| arm | params | training GPU-h (recorded) | exclusive-equivalent GPU-h | MACs / candidate | wall / decision (s) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm, c in comp["arms"].items():
        lines.append(f"| {arm} | {_num(c['trainable_parameters'], ',')} | {_num(c['training_gpu_hours'], '.2f')} | "
                     f"{_num(c['exclusive_training_gpu_hours'], '.2f')} | {_num(c['macs_per_candidate'], '.3g')} | "
                     f"{_num(c['wall_per_decision'], '.4f')} |")
    lines += ["", "Engine worker-hours: " + ", ".join(f"#{i} {_num(h, '.2f')}" for i, h in comp["engine_worker_hours"].items()),
              "", "## Not run / dropped", ""]
    lines += [f"- #{r['child'] if r['child'] is not None else '?'} {r['arm'] or '(child)'}: {r['status']} ({r['reason']})"
              for r in summary["not_run"]] or ["- none"]
    d = summary["decision_log"]
    lines += ["", "## Decision log", "", f"{d['count']} automatic decisions (fetched {d['fetched_at']}); per issue: "
              + ", ".join(f"#{i} {n}" for i, n in d["per_issue"].items()) + ". See decision-log.md.", ""]
    return "\n".join(lines)


def _pass(block):
    if block is None:
        return "n/a"
    if isinstance(block, (int, float)):
        return f"{block:.4f}"
    return fmt_auc(block)


def _num(value, spec):
    return "n/a" if value is None else format(value, spec)


def run_summary(output=OUTPUT, artifacts=ARTIFACTS, allow_partial=False, sources=None):
    summary, findings = build(output, artifacts, allow_partial, sources)
    write_json(Path(output) / "summary.json", summary)
    (Path(output) / "findings.md").write_text(findings)
    log(f"summary: {len(summary['program_table'])} arm rows; claim holds {summary['program_claim']['U']['holds']}")
    return 0


def _current_sha(artifact):
    path = Path(artifact) if Path(artifact).is_absolute() else ROOT / artifact
    return sha256_of(path) if path.is_file() else None


def validate(output=OUTPUT, artifacts=ARTIFACTS, sources=None):
    """Re-derive summary.json / findings.md byte for byte after hash-checking every recorded input."""
    output = Path(output)
    recorded = read_json(output / "summary.json")
    drift = [i["artifact"] for i in recorded["inputs"] if _current_sha(i["artifact"]) != i["sha256"]]
    error = None
    summary_ok = findings_ok = False
    if drift:
        error = f"inputs changed since the summary: {drift}"
    else:
        summary, findings = build(output, artifacts, recorded["partial"], sources)
        summary_ok = json_text(summary) == (output / "summary.json").read_text()
        findings_ok = findings == (output / "findings.md").read_text()
        if not (summary_ok and findings_ok):
            error = "re-derived publication differs"
    ok = error is None
    write_json(output / "validation.json", {"schema": SCHEMA_VALIDATION, "identity": IDENTITY,
                                            "summary_sha256": sha256_of(output / "summary.json"),
                                            "summary_identical": summary_ok, "findings_identical": findings_ok,
                                            "input_drift": drift, "error": error, "validated_at": utc_now(), "pass": ok})
    log(f"validate: {'pass' if ok else 'FAIL: ' + error}")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("budget", "decisions", "summary", "validate"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--allow-partial", action="store_true", help="--summary before every child has published")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.budget:
        return run_budget()
    if args.decisions:
        return run_decisions()
    if args.summary:
        return run_summary(allow_partial=args.allow_partial)
    return validate()


if __name__ == "__main__":
    raise SystemExit(main())
