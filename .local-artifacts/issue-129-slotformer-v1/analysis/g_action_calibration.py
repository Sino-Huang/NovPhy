"""#129 G-action calibration context (DESCRIPTIVE; fit split only)."""
import json, time
import torch
from scripts import external_baselines as eb
from scripts import run_decision_dynamics_retrain as rt
from scripts import run_gate_b as gb
from scripts import run_tau_ad_within_checkpoint as t96
from world_model.training.matched_dynamics import pairs_for
torch.set_num_threads(4)
OUT = eb.ROOT / ".local-artifacts/issue-129-slotformer-v1"
began = time.time()
items = eb.grid_items()
ordinals = [it["ordinal"] for it in items]
grid_actions = torch.stack([eb.launch5(it["action"]) for it in items])
anchors = eb.fit_anchors()["carriers"]
members = sorted(anchors)
shots = eb.FitShots()
branches = rt.complete_branches()
# (a) engine truth: the true E99 carrier at min(225, end) of each executed grid shot of each fit level
grid_shot = {}
for i, b in enumerate(branches):
    for e in b["candidate"]["inventory_entries"]:
        if e["inventory"] == "grid":
            grid_shot[(b["member"], e["ordinal"])] = i
complete = [m for m in members if all((m, o) in grid_shot for o in ordinals)]
idx = [grid_shot[(m, o)] for m in complete for o in ordinals]
pos = [min(225, int(shots.ends[i])) for i in idx]
truth = shots.carrier_rows(idx, pos).float().reshape(len(complete), len(ordinals), -1)
truth_launch = torch.stack([shots.launch5[i] for i in idx]).reshape(len(complete), len(ordinals), -1)
assert torch.equal(truth_launch, grid_actions[None].expand(len(complete), -1, -1)), "grid shot launches differ"
short = sum(1 for p in pos if p < 225)
# also the slot-only view (header dropped) for context
out = {"truth": {"G_action_carrier_236": eb.action_sensitivity(truth),
                 "G_action_slots_only_234": eb.action_sensitivity(truth[..., 2:]),
                 "levels": len(complete), "levels_missing_a_grid_shot": len(members) - len(complete),
                 "grid_shots_ending_before_225": short,
                 "rule": "true E99 fit carrier at min(225, end) of the level's 16 executed grid shots (fit store)"}}
# (b) U and C*: Gate-B checkpoints from the fit anchors, the request's fixed pair to frame 225 (eb._reference_rollout)
z0 = torch.stack([anchors[m] for m in members])
for label, name in (("U", "unit"), ("C*", "c_star")):
    per_seed = {}
    for seed in eb.SEEDS:
        model = eb._reference_predictor(name, seed)
        pair = pairs_for(model)[t96.arm_spec(gb.UNITS[name]["request"])["pair_index"]]
        with torch.no_grad():
            zz = z0.repeat_interleave(len(items), 0)
            aa = grid_actions.repeat(len(members), 1)
            z, position = zz.clone(), 0
            while position < 225:
                z = model.carrier(z, aa, pair)
                position += pair.delta
        e = z.float().reshape(len(members), len(items), -1)
        per_seed[str(seed)] = {"G_action_carrier_236": eb.action_sensitivity(e),
                               "G_action_slots_only_234": eb.action_sensitivity(e[..., 2:]),
                               "nonfinite": int((~torch.isfinite(e)).any(-1).sum())}
    out[label] = {"per_seed": per_seed, "rollout": f"request {gb.UNITS[name]['request']}, pair delta {pair.delta}"}
# (c) B1 / B1-m: the frozen readouts (same call, eb.action_sensitivity on fit_anchor_endpoints)
for arm in ("B1", "B1-m"):
    out[arm] = {"per_seed": {str(s): {"G_action_carrier_236": eb.read_json(
        OUT / f"readouts/{arm}/seed-{s}/readouts.json")["G_action"]} for s in eb.SEEDS},
        "source": f"readouts/{arm}/seed-*/readouts.json (frozen; readouts-manifest.json)"}
out["threshold"] = eb.GUARD["action_sensitivity_min"]
out["seconds"] = time.time() - began
print(json.dumps(out, indent=1))
(OUT / "analysis" / "g-action-calibration.raw.json").write_text(json.dumps(out, indent=1) + "\n")
