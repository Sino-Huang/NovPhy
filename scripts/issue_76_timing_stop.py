"""Record the observed C2 native-time window error and stop unused C2 attempts."""
import argparse
import math
from pathlib import Path

from scripts import issue_76_expansion as files
from scripts import run_issue_76_canonical_smoke as smoke


def timing_evidence(output=smoke.OUTPUT):
    root = output / "attempts/issue-76-canonical-01/aligned"
    captures = list(root.glob("capture-v2:*"))
    if len(captures) != 1:
        raise ValueError("timing stop requires the retained single first-shot capture")
    paths = sorted(captures[0].glob("frame_*.json"))
    first, last = files.read(paths[0]), files.read(paths[-1])
    steps = last["fixed_step"] - first["fixed_step"]
    span = last["fixed_time_seconds"] - first["fixed_time_seconds"]
    measured = span / steps
    if not math.isclose(measured, .0004, abs_tol=1e-8) or span >= 1:
        raise ValueError("retained timing does not support the pre-release window diagnosis")
    return {"schema": "issue_76_native_time_window_stop_v1", "episode": "issue-76-canonical-01",
            "frame_count": len(paths), "first_fixed_step": first["fixed_step"], "last_fixed_step": last["fixed_step"],
            "recorded_seconds": span, "measured_seconds_per_fixed_step": measured,
            "native_fixed_delta_seconds": .0004, "declared_watchdog_steps": 600,
            "watchdog_window_seconds": .24, "original_drag_hold_seconds": 1.,
            "minimum_drag_steps_before_release_dispatch": 2500,
            "watchdog_overshoot_steps": max(0, steps - 600),
            "disposition": "engineering_window_invalid_before_bird_release",
            "unused_C2_episodes": [2, 3, 4], "failed_episode_may_be_retried": False,
            "new_renderer_frames": 0, "fresh_access_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", required=True)
    parser.parse_args()
    report = timing_evidence()
    target = smoke.OUTPUT / "native-time-window-stop.json"
    budget_path = smoke.OUTPUT / "budget.json"
    budget = files.read(budget_path)
    if target.exists():
        saved = files.read(target)
        if saved["evidence"] != report:
            raise ValueError("timing-stop evidence changed")
    else:
        files.write(target, {"evidence": report, "budget_before_stop": budget})
    smoke.capture_budget(budget_path, {**budget, "stopped": True,
                                     "stop_reason": "native_timestep_window_invalid_not_resource_exhaustion"})
    print(f"[canonical-timing] C2 stopped: {report['frame_count']} retained frames cover only {report['recorded_seconds']:.6f}s; no retries", flush=True)


if __name__ == "__main__":
    main()
