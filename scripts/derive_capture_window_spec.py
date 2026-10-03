"""Issue #113 work item 2: the macro capture-window spec for the #104/#110 campaign.

Zero engine seconds. Every number comes from retained native traces (every native
step, not only the 601 observed frames) of the single-shot captures that exist on
disk: the #109 smoke (N1 rolling type010103 / sliding type010105), the #77 N2
appearance captures (novelty level 1, type010101/02) and the #77 n2n normal side
(type010102, v1 admissible + v2 recovered).

What the scan measures per shot:
  * the native stop kind and step (the old window: engine rest, clear, fail, or the
    30 000-step cap);
  * which dynamic bodies still move at the end (engine stability thresholds);
  * when the non-bird bodies reach their final engine rest;
  * support-set changes among non-bird bodies at frame lags 1 / 5 / 15;
  * when the launched bird is destroyed, and a decay projection of the time a
    still-rolling bird would need to reach the engine's 0.01 u/s bird-death speed.

The frozen rule (``RULE``) keeps the old window for every shot that the old player
censored, and continues every shot that reached engine rest for ``REST_TAIL_FRAMES``
more observation frames. ``--validate`` rescans every trace and recomputes the spec.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = "issue-113-capture-window-v1"
SCHEMA = "issue_113_capture_window_spec_v1"
OUTPUT = ROOT / ".local-artifacts" / IDENTITY
VALIDATION_COMMAND = "python -u -m scripts.derive_capture_window_spec --validate"
CAMPAIGNS = ("issue-109-capture-smoke-v1", "issue-77-n2-appearance-v1", "issue-77-n2n-v1", "issue-77-n2n-v2")
SCAN_PROCESSES = 40

FIXED_DELTA_SECONDS = 0.0004
OBSERVATION_STRIDE = 50
ENGINE_SPEED = 0.01            # |v| threshold of the engine stable_entered candidate (sqrMagnitude 1e-4)
ENGINE_ANGULAR = 0.01          # deg/s threshold of the engine stable_entered candidate
BIRD_DEATH_SPEED = 0.01        # ABBird.CheckVelocityToDie (no magnet)
LAGS = (1, 5, 15)
PROJECTED_CAPS_SECONDS = (12, 16, 20, 24, 30, 40)

PRE_REST_CAP_STEPS = 30000
REST_TAIL_FRAMES = 50
REST_TAIL_STEPS = REST_TAIL_FRAMES * OBSERVATION_STRIDE
TAIL_TERMINAL = "rest_tail_complete"
RULE = {
    "name": "run-to-rest plus a fixed post-rest tail; the old 12 s cap is unchanged for shots not at rest",
    "rest_trigger": ("the engine stable_entered event, unchanged: every dynamic Rigidbody2D has |v|^2 <= 1e-4 and "
                     "|omega| <= 0.01 deg/s for 100 consecutive native steps, after the launch"),
    "pre_rest_cap_steps": PRE_REST_CAP_STEPS,
    "rest_tail_frames": REST_TAIL_FRAMES,
    "rest_tail_steps": REST_TAIL_STEPS,
    "hard_limit_steps": PRE_REST_CAP_STEPS + REST_TAIL_STEPS,
    "tail": ("stable_entered at step s <= pre_rest_cap_steps no longer finalizes; recording continues to s + "
             "rest_tail_steps and then finalizes with terminal reason rest_tail_complete"),
    "tail_cancel": ("stable_exited during a tail cancels it; recording continues and censors at pre_rest_cap_steps "
                    "if that step has passed, otherwise the next stable_entered starts a new tail"),
    "clear_and_fail": "level_clear and level_fail finalize immediately, as before, also inside a tail",
    "censoring": ("a shot not in a tail at pre_rest_cap_steps is censored there (native_time_window_limit), exactly "
                  "where the old player censored it"),
    "old_window_identity": ("for every shot the prefix up to the old stop step is the old trace: the change only "
                            "removes the stable_entered finalization and adds frames after it"),
    "terminal_reason": TAIL_TERMINAL,
    "stop_kind": f"{TAIL_TERMINAL} maps to stable_without_clear (scene at engine rest, no clear)",
    "engine_environment": {"NOVPHY_NATIVE_MAX_SHOT_STEPS": PRE_REST_CAP_STEPS,
                           "NOVPHY_NATIVE_REST_TAIL_STEPS": REST_TAIL_STEPS},
    "manifest_fields": {"maximum_shot_steps": PRE_REST_CAP_STEPS, "rest_tail_steps": REST_TAIL_STEPS},
}
RATIONALE = [
    "the macro events are absent because a stable shot stops on the first rest step (0 settled frames after it), "
    "not because the cap is short",
    "at the 12 s cap the structure (every non-bird body) is already at engine rest in almost every censored shot; "
    "the only moving body is the launched bird rolling on the ground (see per_level_type.censored_moving_at_end)",
    "a longer cap therefore adds frames of a lone rolling bird; reaching engine rest needs the bird to fall below "
    "0.01 u/s and die (projection table), at a capture and storage cost proportional to the added seconds",
    "structure changes stop well before the cap (last non-bird support change), so collapse yield is set by "
    "whether the shot reaches the structure (candidate inventory), not by the window",
    f"{REST_TAIL_FRAMES} tail frames (1.0 s) hold a settle debounce of up to 35 frames plus the Delta = 15 macro "
    "request horizon and a 15-frame temporal-head history inside the window, and cover the bird-death check that "
    "follows rest (storm onset on novelty level 8 needs the first bird destroyed)",
]


def log(message):
    print(f"[{IDENTITY}] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def relative(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ inputs

def shots():
    """Every complete retained single shot with its template level type."""
    rows = []
    for campaign in CAMPAIGNS:
        for result_path in sorted((ROOT / ".local-artifacts" / campaign / "results").glob("*.json")):
            result = json.loads(result_path.read_text())
            segments = result.get("segments") or []
            if result.get("failure") or not segments:
                continue
            segment = segments[0]
            template = unquote(str(segment.get("source_bindings", {}).get("scenario_template_id", "")))
            match = re.search(r"novelty_level_(\d+)/(type\d+)/", template)
            if match is None:
                raise ValueError(f"{result_path}: no template level type")
            native = Path(segment["native_root"])
            if not native.is_absolute():
                native = ROOT / native
            rows.append({"campaign": campaign, "branch": result.get("member_identity") or result_path.stem,
                         "novelty_level": int(match.group(1)), "family": match.group(2),
                         "native_root": relative(native)})
    return rows


# ------------------------------------------------------------------- scan

def _moving(body, speed, angular):
    return math.hypot(*body["velocity"]) > speed or abs(body["angular_velocity_degrees_per_second"]) > angular


def scan_shot(shot):
    """One pass over every native step of one retained trace."""
    root = ROOT / shot["native_root"]
    manifest = json.loads((root / "native-manifest.json").read_text())
    first = manifest["first_fixed_step"]
    last_moving_any = last_moving_nonbird = -1
    bird_speed = {}
    supports, events = [], []
    end_moving = set()
    count = 0
    for descriptor in manifest["chunks"]:
        with gzip.open(root / descriptor["path"], "rb") as stream:
            chunk = json.loads(stream.read())
        events.extend((e["fixed_step"] - first, e["event_type"], tuple(e["participants"])) for e in chunk["events"])
        for sample in chunk["fixed_step_samples"]:
            step = sample["fixed_step"] - first
            moving = set()
            for entity in sample["entities"]:
                body = entity.get("body")
                if (not entity.get("body_present") or not body or body.get("body_type") != "dynamic"
                        or not body.get("simulated")):
                    continue
                kind = entity["scenario_object_id"].split(":")[0]
                if _moving(body, ENGINE_SPEED, ENGINE_ANGULAR):
                    moving.add(kind)
                if entity["scenario_object_id"] == "bird:0000":
                    bird_speed[step] = math.hypot(*body["velocity"])
            if moving:
                last_moving_any = step
            if moving - {"bird"}:
                last_moving_nonbird = step
            if step % OBSERVATION_STRIDE == 0:
                supports.append(frozenset(
                    (s["supporter_entity_id"], s["supported_entity_id"]) for s in sample["supports"]
                    if "bird:" not in s["supporter_entity_id"] and "bird:" not in s["supported_entity_id"]))
            end_moving = moving
            count += 1
    steps = count - 1
    if manifest["status"] == "complete":
        terminal = manifest["terminal_evidence"]["reason"]
    elif manifest["failure"] == "native_time_window_limit":
        terminal = "censored"
    else:
        raise ValueError(f"{shot['native_root']}: failed trace in a complete branch")
    launch = next(step for step, kind, _ in events if kind == "bird_launched")
    bird_destroyed = next((step for step, kind, who in events
                           if kind == "entity_destroyed" and "runtime:bird:0000" in who), None)
    changes = {}
    last_change_frame = None
    for lag in LAGS:
        flags = [supports[t + lag] != supports[t] for t in range(len(supports) - lag)]
        changes[str(lag)] = {"positive_frames": sum(flags), "frames": len(flags)}
    for t in range(1, len(supports)):
        if supports[t] != supports[t - 1]:
            last_change_frame = t
    projection = None
    if terminal == "censored" and bird_destroyed is None and steps in bird_speed:
        end, before = bird_speed[steps], bird_speed.get(steps - 2500)
        if end <= BIRD_DEATH_SPEED:
            projection = 0.0
        elif before and 0 < end < before:
            rate = end / before                   # per physics second
            projection = math.log(BIRD_DEATH_SPEED / end) / math.log(rate)
    return {**shot, "terminal": terminal, "steps": steps, "frames": len(supports), "launch_step": launch,
            "rest_step_any": last_moving_any + 1 if last_moving_any + 1 <= steps else None,
            "rest_step_nonbird": last_moving_nonbird + 1 if last_moving_nonbird + 1 <= steps else None,
            "moving_at_end": sorted(end_moving), "bird_destroyed_step": bird_destroyed,
            "bird_end_speed": round(bird_speed[steps], 6) if steps in bird_speed else None,
            "projected_extra_seconds_to_bird_death": None if projection is None else round(projection, 4),
            "support_changes": changes, "last_support_change_frame": last_change_frame}


def scan(rows, processes=SCAN_PROCESSES):
    with ProcessPoolExecutor(processes) as pool:
        return list(pool.map(scan_shot, rows, chunksize=2))


# ---------------------------------------------------------------- summary

def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return round(values[low] + (values[high] - values[low]) * (position - low), 4)


def level_type(row):
    return f"novelty_level_{row['novelty_level']}/{row['family']}"


def tail_frames(row):
    """Frames the frozen rule adds to one retained shot (upper bound: clear/fail can end a tail early)."""
    return REST_TAIL_FRAMES if row["terminal"] == "stable_entered" else 0


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[level_type(row)].append(row)
    table = {}
    for name, items in sorted(groups.items()):
        censored = [r for r in items if r["terminal"] == "censored"]
        stable = [r for r in items if r["terminal"] == "stable_entered"]
        old_frames = sum(r["frames"] for r in items)
        added = sum(tail_frames(r) for r in items)
        projections = [r["projected_extra_seconds_to_bird_death"] for r in censored
                       if r["projected_extra_seconds_to_bird_death"] is not None]
        unprojected = len(censored) - len(projections)
        table[name] = {
            "shots": len(items),
            "stop_kinds_old_window": dict(sorted(Counter(r["terminal"] for r in items).items())),
            "settle": {
                "shots_reaching_engine_rest": len(stable),
                "share": round(len(stable) / len(items), 4),
                "rest_step_after_decision_p10_p50_p90": [percentile([r["steps"] for r in stable], q)
                                                         for q in (.1, .5, .9)],
                "settled_frames_old_window": len(stable),
                "settled_frames_new_window_upper_bound": len(stable) * (REST_TAIL_FRAMES + 1),
                "settled_frames_per_shot_new_window_upper_bound": round(len(stable) * (REST_TAIL_FRAMES + 1)
                                                                        / len(items), 3),
            },
            "censored_moving_at_end": dict(sorted(Counter("+".join(r["moving_at_end"]) or "none"
                                                          for r in censored).items())),
            "censored_structure_at_rest_at_cap": sum(r["rest_step_nonbird"] is not None for r in censored),
            "censored_launched_bird_end_speed_p50_p90": [percentile([r["bird_end_speed"] for r in censored
                                                                     if r["bird_end_speed"] is not None], q)
                                                         for q in (.5, .9)],
            "collapse": {
                "shots_with_nonbird_support_change": sum(r["last_support_change_frame"] is not None
                                                         for r in items),
                "share": round(sum(r["last_support_change_frame"] is not None for r in items) / len(items), 4),
                "last_change_frame_p50_p90_max": [percentile([r["last_support_change_frame"] for r in items
                                                              if r["last_support_change_frame"] is not None], q)
                                                  for q in (.5, .9, 1.0)],
                "positive_frames_per_lag": {str(lag): {
                    "positive": sum(r["support_changes"][str(lag)]["positive_frames"] for r in items),
                    "frames": sum(r["support_changes"][str(lag)]["frames"] for r in items)} for lag in LAGS},
            },
            "launched_bird_destroyed_in_window": sum(r["bird_destroyed_step"] is not None for r in items),
            "longer_cap_projection": {
                "label": "DESCRIPTIVE model projection (per-shot exponential decay of the launched bird's speed "
                         "over the last physics second, extrapolated to the 0.01 u/s death speed); not observed",
                "censored": len(censored), "censored_without_projection": unprojected,
                "bird_dead_by_cap_seconds": {str(cap): sum(1 for p in projections if 12 + p <= cap)
                                             for cap in PROJECTED_CAPS_SECONDS},
            },
            "frames": {"old_window": old_frames, "added_by_rule_upper_bound": added,
                       "relative_increase_upper_bound": round(added / old_frames, 4)},
        }
    every = rows
    old = sum(r["frames"] for r in every)
    added = sum(tail_frames(r) for r in every)
    censored = [r for r in every if r["terminal"] == "censored"]
    longer = {}
    for cap in PROJECTED_CAPS_SECONDS[1:]:
        extra = sum((cap - 12) / (FIXED_DELTA_SECONDS * OBSERVATION_STRIDE) for _ in censored)
        longer[str(cap)] = {"frames_added": int(extra), "relative_increase": round(extra / old, 4)}
    totals = {
        "shots": len(every), "old_window_frames": old, "rule_added_frames_upper_bound": added,
        "rule_relative_increase_upper_bound": round(added / old, 4),
        "censored": len(censored),
        "censored_structure_at_rest_at_cap": sum(r["rest_step_nonbird"] is not None for r in censored),
        "censored_only_launched_bird_moving": sum(r["moving_at_end"] == ["bird"] for r in censored),
        "longer_fixed_cap_cost": {"label": "every censored shot runs to the longer cap", "by_cap_seconds": longer},
    }
    return table, totals


def novelty_levels_without_evidence():
    return {"levels": [2, 3, 4, 5, 6, 7, 8],
            "statement": ("no retained capture exists for novelty levels 2-8 (or for level 0/1 families outside "
                          "type010101/02/03/05); their settle, collapse and storm-onset yields are measured by the "
                          "#104 rendered smoke under this rule and reported there, not estimated here")}


def derive(rows=None):
    rows = rows if rows is not None else scan(shots())
    table, totals = summarize(rows)
    sources = {campaign: sum(r["campaign"] == campaign for r in rows) for campaign in CAMPAIGNS}
    spec = {
        "schema": SCHEMA, "identity": IDENTITY, "issue": 113, "consumer_issues": [104, 110],
        "engine_seconds": 0, "validation_command": VALIDATION_COMMAND,
        "rule": RULE, "rationale": RATIONALE,
        "evidence": {"sources": sources, "per_level_type": table, "totals": totals,
                     "uncovered": novelty_levels_without_evidence(),
                     "thresholds": {"engine_speed": ENGINE_SPEED, "engine_angular_deg_s": ENGINE_ANGULAR,
                                    "bird_death_speed": BIRD_DEATH_SPEED, "lags_frames": list(LAGS)}},
        "not_decided_here": ("the #113 item-1 predicate definitions (settle debounce, per-Delta structure change); "
                             "the rule only guarantees the frames those definitions need"),
    }
    return spec, rows


def findings(spec):
    lines = [f"# #113 capture-window spec (`{IDENTITY}`)", "",
             f"Frozen {spec.get('frozen_at', '(unfrozen)')}; zero engine seconds; `{VALIDATION_COMMAND}`.", "",
             "## Rule", ""]
    lines += [f"- **{key}**: {value}" for key, value in spec["rule"].items() if key != "engine_environment"]
    lines += ["", "## Evidence per level type (retained traces, every native step)", "",
              "| level type | shots | stop kinds (old window) | at engine rest | settled frames old -> new (upper) "
              "| censored: structure at rest at cap | censored moving at end | collapse shots | last change frame "
              "p50/p90/max | bird destroyed in window | frames +% |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in spec["evidence"]["per_level_type"].items():
        settle, collapse = row["settle"], row["collapse"]
        lines.append(
            f"| {name} | {row['shots']} | {row['stop_kinds_old_window']} | {settle['shots_reaching_engine_rest']} "
            f"({settle['share']:.2f}) | {settle['settled_frames_old_window']} -> "
            f"{settle['settled_frames_new_window_upper_bound']} | {row['censored_structure_at_rest_at_cap']}/"
            f"{row['stop_kinds_old_window'].get('censored', 0)} | {row['censored_moving_at_end']} | "
            f"{collapse['shots_with_nonbird_support_change']} ({collapse['share']:.2f}) | "
            f"{collapse['last_change_frame_p50_p90_max']} | {row['launched_bird_destroyed_in_window']} | "
            f"{100 * row['frames']['relative_increase_upper_bound']:.1f} |")
    totals = spec["evidence"]["totals"]
    lines += ["", "## Longer fixed cap (rejected alternative)", "",
              f"- censored shots: {totals['censored']}; structure at rest at the cap: "
              f"{totals['censored_structure_at_rest_at_cap']}; only the launched bird moving: "
              f"{totals['censored_only_launched_bird_moving']}",
              "- projected launched-bird death by cap (DESCRIPTIVE projection, censored shots per level type):"]
    for name, row in spec["evidence"]["per_level_type"].items():
        projection = row["longer_cap_projection"]
        lines.append(f"  - {name}: {projection['bird_dead_by_cap_seconds']} of {projection['censored']} "
                     f"({projection['censored_without_projection']} without a projection)")
    lines.append("- frame cost of a longer cap for every censored shot: " + ", ".join(
        f"{cap} s +{100 * value['relative_increase']:.0f}%"
        for cap, value in totals["longer_fixed_cap_cost"]["by_cap_seconds"].items()))
    lines += [f"- the frozen rule adds at most {100 * totals['rule_relative_increase_upper_bound']:.1f}% frames",
              "", "## Rationale", ""] + [f"- {item}" for item in spec["rationale"]]
    lines += ["", f"## Not covered\n\n- {spec['evidence']['uncovered']['statement']}",
              f"- {spec['not_decided_here']}", ""]
    return "\n".join(lines)


def outputs(spec, rows):
    return {"spec.json": json_text(spec),
            "shots.json": json_text(sorted(rows, key=lambda r: (r["campaign"], r["branch"]))),
            "findings.md": findings(spec)}


def prepare(output=OUTPUT):
    output = Path(output)
    if (output / "spec.json").exists():
        raise ValueError("spec already frozen; use --validate")
    spec, rows = derive()
    spec["frozen_at"] = utc_now()
    output.mkdir(parents=True, exist_ok=True)
    for name, text in outputs(spec, rows).items():
        (output / name).write_text(text)
    log(f"frozen {spec['frozen_at']}: {len(rows)} shots")
    return spec


def validate(output=OUTPUT):
    output = Path(output)
    frozen = json.loads((output / "spec.json").read_text())
    spec, rows = derive()
    spec["frozen_at"] = frozen["frozen_at"]
    problems = [name for name, text in outputs(spec, rows).items() if (output / name).read_text() != text]
    report = {"identity": IDENTITY, "shots": len(rows), "problems": problems, "validated": not problems,
              "spec_sha256": "sha256:" + sha256((output / "spec.json").read_bytes()).hexdigest()}
    print(json_text(report), end="")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="list the retained shots; writes nothing")
    mode.add_argument("--prepare", action="store_true", help="scan and freeze the spec")
    mode.add_argument("--validate", action="store_true", help="rescan and compare every output")
    args = parser.parse_args()
    if args.dry_run:
        rows = shots()
        print(json_text({"shots": len(rows), "by_campaign": dict(Counter(r["campaign"] for r in rows)),
                         "by_level_type": dict(Counter(level_type(r) for r in rows))}), end="")
    elif args.prepare:
        prepare()
    elif not validate():
        sys.exit(1)


if __name__ == "__main__":
    main()
