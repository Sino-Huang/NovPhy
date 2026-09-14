"""Isolated angular pilot using the unchanged fixed-action replay supervisor."""
import argparse
import shutil
import time

from scripts import prepare_issue_76_angular_replay as metadata
from scripts import validate_issue_76_action_replay as validation

supervisor = validation.run
SOURCES = (*validation.SOURCES, "scripts/run_issue_76_angular_replay.py",
           "tests/test_issue_76_angular_execution.py")


def load_plan(root=metadata.ROOT):
    plan = supervisor.files.read(root / "execution-plan.json")
    if (plan["identity"] != "issue-76-angular-replay-execution-v1"
            or plan["inventory"] != supervisor.files.read(root / "inventory.json")
            or not plan["capture_execution_authorized"] or plan["model_scoring_authorized"]
            or plan["new_optimizer_updates"] or plan["fresh_access"]):
        raise ValueError("angular pilot requires its exact capture-only execution freeze")
    for path, source in plan["source_text"].items():
        if (supervisor.files.ROOT / path).read_text() != source:
            raise ValueError(f"angular pilot source changed: {path}")
    return plan


def prepare(root=metadata.ROOT, output=metadata.OUTPUT):
    inventory = supervisor.files.read(root / "inventory.json")
    sources = dict(inventory["source_text"])
    for path, source in sources.items():
        if (supervisor.files.ROOT / path).read_text() != source:
            raise ValueError("angular inventory source changed")
    sources.update({path: (supervisor.files.ROOT / path).read_text() for path in SOURCES})
    plan = {"identity": "issue-76-angular-replay-execution-v1", "inventory": inventory,
            "source_text": sources, "comparability_limits": validation.comparison.LIMITS,
            "comparability_source_frozen": True, "capture_execution_authorized": True,
            "readiness_action": inventory["members"][0]["actions"][0],
            "model_scoring_authorized": False, "new_optimizer_updates": 0, "fresh_access": False}
    validation.immutable(root / "execution-plan.json", plan)
    validation.immutable(output / "execution-plan.json", plan)
    if not (root / "player").exists():
        started = time.monotonic()
        shutil.copytree(inventory["player_source"], root / "player")
        validation.immutable(root / "player-preparation.json",
                             {"source": inventory["player_source"], "wall_seconds": time.monotonic() - started,
                              "new_player_build": False})
    load_plan(root)
    print("Angular execution frozen; existing player copied; no capture started.")


def validate_smoke(root=metadata.ROOT, output=metadata.OUTPUT):
    started = time.monotonic()
    plan = load_plan(root)
    budget = supervisor.files.read(root / "capture-budget.json")
    if budget["running"] or budget["stopped"]:
        raise ValueError("angular smoke has not completed cleanly")
    members = supervisor.selected_members(plan, root, True)
    cases = [validation.read_case(root, plan, member) for member in members]
    compared = validation.comparison.compare(cases[0], cases[1], plan["comparability_limits"])
    report = {"execution_plan_identity": plan["identity"], "member_identities": [m["identity"] for m in members],
              "validated": compared["comparable"], "comparability": compared,
              "capture_budget": budget, "validation_wall_seconds": time.monotonic() - started,
              "decision_image_paths": [case["decision_image_path"] for case in cases],
              "gameplay_success_used_as_continuation_criterion": False,
              "model_scoring_authorized": False, "fresh_access": False, "advancement_authorized": False}
    validation.immutable(root / "smoke-validation.json", report)
    validation.immutable(output / "smoke-validation.json", report)
    print({"angular_smoke_validated": report["validated"], "failures": compared["failures"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for mode in ("prepare", "smoke", "validate-smoke", "run"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.validate_smoke:
        validate_smoke()
    elif args.smoke or args.run:
        supervisor.run(load_plan(), metadata.ROOT, smoke=args.smoke)
    else:
        print("No-write: angular capture requires its own execution freeze and validated smoke.")
