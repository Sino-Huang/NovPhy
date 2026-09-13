"""Record the specific JSON object-order correction without rewriting evidence."""
import argparse
from pathlib import Path

from scripts import run_issue_76_fixed_development as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        print("No-write: JSON policy-key order only; no predictions, selection or fresh access.")
        return
    plan = run.fit.files.read(run.ROOT / "execution-plan.json")
    driver = "scripts/run_issue_76_fixed_development.py"
    paths = (driver, "tests/test_issue_76_fixed_development_json.py")
    value = {"identity": "issue-76-fixed-development-json-order-correction-v1",
        "execution_plan_identity": plan["identity"], "original_execution_source": plan["source_text"][driver],
        "corrected_source_text": {path: (run.fit.files.ROOT / path).read_text() for path in paths},
        "recording_source_text": Path(__file__).read_text(),
        "diagnosis": "JSON writer sorts keys; object-key iteration order is not the frozen policy-pair order",
        "correction": "validate identical policy-key sets and reconstruct in-memory maps in the original model.pairs order",
        "score_values_changed": False, "selection_rule_changed": False, "predictions_rerun": False,
        "member_or_model_inventory_changed": False, "tolerances_changed": False, "fresh_access": False,
        "original_execution_plan_retained": True, "original_validation_plan_retained": True,
        "prior_validation_failure": "ValueError: fixed-policy inventory differs",
        "prior_validation_budget_retained": run.fit.require_finished_budget(run.ROOT, "independent-validation"),
        "regression": "sorted JSON round-trip preserves every value and reproduces the original calibration choice; missing policies still fail"}
    run.immutable(run.ROOT / "json-order-correction.json", value)
    run.immutable(run.OUTPUT / "json-order-correction.json", value)
    run.load_plan()
    print("Technical correction frozen; original scores, source freezes and spent validation budget retained.", flush=True)


if __name__ == "__main__":
    main()
