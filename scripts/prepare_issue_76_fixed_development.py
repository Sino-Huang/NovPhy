"""Freeze metadata-only development membership and the complete retained model pool."""
import argparse
from pathlib import Path

from scripts import run_issue_76_endpoint_anchor as source

ROOT = source.fit.files.ROOT / ".local-artifacts/issue-76-fixed-development-v1"
OUTPUT = source.fit.files.ROOT / "data/issue-76-fixed-development"
STAGES = ("repaired-representation", "matched-dynamics", "boundary-dynamics", "full-duration-dynamics",
          "lower-rate-dynamics", "boundary-focus", "local-anchor", "endpoint-anchor")
ENTRY_FIELDS = ("member_identity", "base_cluster", "exposure_role", "family", "path", "frames", "usable")


def make_inventory():
    parent = source.fit.files.read(source.ROOT / "plan.json")
    data_root = Path(parent["data_root"])
    index = source.fit.files.read(data_root / "data-index.json")
    assert index["plan_identity"] == parent["data_binding"]["identity"]
    entries = [{key: entry[key] for key in ENTRY_FIELDS} for entry in index["entries"]
               if entry["exposure_role"] in ("calibration", "model_selection")]
    assert all(sum(e["exposure_role"] == role for e in entries) == 50 for role in ("calibration", "model_selection"))
    models = []
    for stage in STAGES:
        root = source.fit.files.ROOT / ".local-artifacts" / f"issue-76-{stage}-v1"
        identity = source.fit.files.read(root / "plan.json")["identity"]
        for seed in parent["seeds"]:
            for pure in (False, True):
                checkpoint = source.fit.model_path(root, seed, pure, "predictor")
                assert checkpoint.is_file(), f"missing declared checkpoint: {checkpoint}"
                models.append(dict(recipe=stage, seed=seed, pure=pure, source_plan_identity=identity,
                                   checkpoint_paths=[str(checkpoint)], parameter_weights=[1.]))
    for seed in parent["seeds"]:
        for pure in (False, True):
            paths = [str(source.fit.model_path(source.fit.files.ROOT / ".local-artifacts" /
                         f"issue-76-{stage}-v1", seed, pure, "predictor")) for stage in ("local-anchor", "endpoint-anchor")]
            models.append(dict(recipe="fixed-midpoint", seed=seed, pure=pure,
                               source_plan_identities=["issue-76-local-anchor-v1", "issue-76-endpoint-anchor-v1"],
                               checkpoint_paths=paths, parameter_weights=[.5, .5]))
    return dict(identity=ROOT.name, data_root=str(data_root), data_plan_identity=index["plan_identity"],
        data_plan_path=str(data_root / "plan.json"), entries=entries, models=models, seeds=parent["seeds"],
        recipes=[*STAGES, "fixed-midpoint"], capacity=parent["capacity"],
        checkpoint_file_existence_checked=True, checkpoint_completion_validation_pending=True,
        score_execution_authorized=False, fresh_access=False, optimization_performed=False,
        failed_training_qualifications_preserved=True,
        source_text=Path(__file__).read_text(), protocol_text=(source.fit.files.ROOT /
            "docs/issue-76-fixed-development-protocol.md").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        print("No-write: metadata only; 100 assigned development members, 48 saved models and six fixed midpoints.")
        return
    inventory = make_inventory()
    source.immutable(ROOT / "inventory.json", inventory)
    source.immutable(OUTPUT / "inventory.json", inventory)
    print(dict(assigned_members=len(inventory["entries"]), model_instances=len(inventory["models"]),
               scoring_performed=False, fresh_access=False), flush=True)


if __name__ == "__main__":
    main()
