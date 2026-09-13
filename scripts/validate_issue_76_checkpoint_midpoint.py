"""Independently reconstruct the fixed midpoint and replay every reported curve."""
from pathlib import Path

import torch

from scripts import diagnose_issue_76_checkpoint_midpoint as probe


def main():
    torch.set_num_threads(1)
    run = probe.run
    report = run.fit.files.read(probe.OUTPUT)
    parent = run.fit.files.read(run.OUTPUT / "report.json")
    plan = parent["plan"]
    assert report["source_plan_identity"] == plan["identity"] and report["parent_plan_identity"] == plan["source_plan_identity"]
    assert report["weights"] == [.5, .5] and report["evaluation"] == plan["evaluation"]
    for path, text in report["source_text"].items():
        assert (run.fit.files.ROOT / path).read_text() == text, "probe source differs"
    rows = []
    with run.fit.fitting_budget(probe.ROOT, "validation", 180, "cpu") as budget:
        for seed in plan["seeds"]:
            for pure in (False, True):
                saved = [torch.load(run.fit.model_path(root, seed, pure, "predictor"), map_location="cpu", weights_only=False)
                         for root in (run.source.ROOT, run.ROOT)]
                models = [run.fit.new_predictor(plan, pure, "cpu").eval().requires_grad_(False) for _ in range(3)]
                for checkpoint, model, identity, counts in zip(saved, models,
                        (plan["source_plan_identity"], plan["identity"]), ((1800, 1785, 15), (900, 893, 7))):
                    assert checkpoint["complete"] and checkpoint["failure"] is None and checkpoint["plan_identity"] == identity
                    assert (checkpoint["updates_completed"], checkpoint["updates_applied"], checkpoint["updates_skipped"]) == counts
                    model.load_state_dict(checkpoint["model"])
                models[2].load_state_dict({key: .5 * saved[0]["model"][key] + .5 * saved[1]["model"][key]
                                          for key in saved[0]["model"]})
                for entry in plan["evaluation"]:
                    budget.check()
                    shard = run.fit.load_shard(Path(plan["data_root"]), entry, plan["data_binding"], fitting=True)
                    carrier = run.fit.load_carrier(Path(plan["data_root"]), seed, entry, plan["data_binding"])
                    reference = next(r for r in parent["rows"] if r["seed"] == seed and r["pure"] == pure
                                     and r["member_identity"] == entry["member_identity"])
                    rows.append({**reference, "midpoint": run.source.source.source.source.drift_curves(models[2], shard, carrier),
                                 "error_comparison": probe.compare_predictions(models[0], models[1], shard, carrier)})
        assert rows == report["rows"], "replayed midpoint or parent-error curves differ"
        summaries = probe.summarize(plan, rows)
        assert summaries == report["summaries"]
        assert report["midpoint_qualification_supported"] == all(s["qualification_supported"] for s in summaries)
        assert not any(report[k] for k in ("optimization_performed", "checkpoint_writes", "fresh_access", "advancement_passed"))
        assert report["budget"] == run.fit.require_finished_budget(probe.ROOT, "midpoint")
        assert report["replay_budget"] == run.fit.require_finished_budget(probe.ROOT.parent / "issue-76-trajectory-retention-v1", "replay")
        budget.check()
    value = dict(validated=True, source_plan_identity=plan["identity"], reconstructed_midpoints=6,
                 replayed_rows=len(rows), replayed_policy_curves=sum(len(r["midpoint"]) for r in rows),
                 midpoint_qualification_supported=report["midpoint_qualification_supported"],
                 optimization_performed=False, checkpoint_writes=False, fresh_access=False,
                 additional_validation_budget_path=str(probe.ROOT / "budgets/validation.json"))
    run.immutable(probe.OUTPUT.with_name("validation.json"), value)
    print(value, flush=True)


if __name__ == "__main__":
    main()
