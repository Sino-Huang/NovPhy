"""Record a provenance-field correction without changing captures or criteria."""
import argparse
from pathlib import Path

from scripts import run_issue_76_action_replay as run
from scripts.run_issue_76_order_probe import immutable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        print("No-write: source-lineage lookup correction only; captures and criteria unchanged.")
        return
    plan = run.files.read(run.ROOT / "execution-plan.json")
    paths = ("scripts/run_issue_76_action_replay.py", "scripts/validate_issue_76_action_replay.py",
             "tests/test_issue_76_action_replay_validation.py")
    corrected = (*paths, "tests/test_issue_76_observation_lineage_binding.py")
    value = {"identity": "issue-76-replay-observation-lineage-correction-v1", "execution_plan_identity": plan["identity"],
        "original_source_text": {path: plan["source_text"][path] for path in paths},
        "corrected_source_text": {path: (run.files.ROOT / path).read_text() for path in corrected},
        "recording_source_text": Path(__file__).read_text(),
        "reason": "observation lineage is derived; original scenario lineage is in source_bindings.source_scenario_lineage_identity",
        "additional_check": "decision and captured observations must share the validated derived observation lineage",
        "captures_repeated": False, "comparison_limits_changed": False, "assignments_changed": False,
        "model_scoring_authorized": False, "fresh_access": False,
        "original_validation_failure": "ValueError: decision/capture observation bindings differ",
        "failed_validation_wall_seconds": None,
        "missing_cost_note": "prior failing validation path did not save a wall-time receipt; not represented as zero"}
    immutable(run.ROOT / "observation-lineage-correction.json", value)
    immutable(run.metadata.OUTPUT / "observation-lineage-correction.json", value)
    run.load_plan()
    print("Provenance correction frozen; original capture results and numerical criteria retained.")


if __name__ == "__main__":
    main()
