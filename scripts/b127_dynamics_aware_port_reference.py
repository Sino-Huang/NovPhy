"""#128 A4 G-port: the official DynamicsAware code (hand-crafted loss) vs the NovPhy adapter on fixed CPU float32 inputs.

Two steps, two envs:

    # 1. official side, env b127-dynaware, the pinned upstream clone imported unmodified (never the vendored copy)
    PHYRE_CACHE_DIR=data/external/dynamics_aware/phyre_cache CUDA_VISIBLE_DEVICES="" \
        /srv/home/shua0171/miniconda3/envs/b127-dynaware/bin/python scripts/b127_dynamics_aware_port_reference.py \
        reference .local-artifacts/issue-128-direct-predictors-v1/gport/a4
    # 2. our side, env novphy: world_model.baselines.dynamics_aware.novphy in the official configuration
    CUDA_VISIBLE_DEVICES="" python scripts/b127_dynamics_aware_port_reference.py \
        compare .local-artifacts/issue-128-direct-predictors-v1/gport/a4

Inputs (reference): PHYRE ball_within_template fold 0, the first two train tasks (sorted task ids) with >= 4 solving
and >= 4 non-solving actions among the 100k cache, 4 + 4 actions per task drawn with RandomState(SEED_INPUT) (the
official task-balanced layout: consecutive groups of n_samples_per_task = 8 per task), simulated with the official
ParallelPhyreSimulator settings (stride 60, featurized objects, MAX_LEN 17, ``standardize_features_shape``).
Model: ``neural_agent.build_model`` with the Table-1 configuration, ``torch.manual_seed(SEED_INIT)``, wrapped in
``nn.DataParallel`` as ``neural_agent.train`` does (a pass-through on CPU). Two updates replay the body of the
official training loop line by line (scheduler.step(); model.train(); forward with embeddings;
``neural_agent.sample_distance_loss``; zero_grad; ``model.module.ce_loss(...).mean()``; ce + weight x aux; backward;
Adam step) with Adam(3e-4) and CosineAnnealingLR(T_max=100 000): the first update runs at the schedule's step-0 LR
(the official loop steps the scheduler before the first update, so it is already lr0 (1 + cos(pi / T)) / 2 > 0),
the second at the next positive LR, on the same inputs. The compare side loads the official initial weights into
``A4Net(in_channels=7, action_size=3)`` and runs ``novphy.train_step`` twice with ``novphy.phyre_trajectories``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
CLONE = Path(os.environ.get("DYNAWARE_REF", "/mnt/array/sukaih/novphy-baselines-src/DynamicsAware@eb65576"))
COMMIT = "eb65576d8a9e9b1f74c64518b5dd68af9d0a7eff"
SEED_INIT, SEED_INPUT = 20261024, 7
N_PER_TASK, N_TASKS, MAX_LEN, UPDATES, LR = 8, 2, 17, 100_000, 3e-4
HP = {"regression_type": "binned", "repr_merging_method": "outer", "distance_cutoff": 1.414,
      "n_regression_bins": 20, "time_slice": 0, "embedding_dim": 256, "embeddor_type": "linear", "weight": 1.0}
TOLERANCE = {"loss_rel": 1e-4, "grad_norm_rel": 1e-4, "tensor_rel_l2": 1e-3, "param_atol": 1e-6, "param_rtol": 1e-5,
             "distance_atol": 1e-6, "lr_rel": 1e-12}
LR_PROBE_STEPS = (1, 2, 3, 1000, 50_000, 99_999, 100_000)


def sha256(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(out_dir):
    sys.path.insert(0, str(CLONE / "agents"))
    import numpy as np
    from torch import nn
    import phyre
    import nets
    import neural_agent
    import rollout_dist
    from psim import standardize_features_shape

    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids, _, _ = phyre.get_fold("ball_within_template", 0)
    cache = phyre.get_default_100k_cache("ball")
    rng = np.random.RandomState(SEED_INPUT)
    chosen, actions, solved = [], [], []
    for task_id in sorted(train_ids):
        statuses = cache.load_simulation_states(task_id)
        pos = np.flatnonzero(statuses == int(phyre.SimulationStatus.SOLVED))
        neg = np.flatnonzero(statuses == int(phyre.SimulationStatus.NOT_SOLVED))
        if len(pos) < N_PER_TASK // 2 or len(neg) < N_PER_TASK // 2:
            continue
        picks = np.concatenate([rng.choice(neg, N_PER_TASK // 2, replace=False),
                                rng.choice(pos, N_PER_TASK // 2, replace=False)])
        chosen.append(task_id)
        actions.append(cache.action_array[picks])
        solved.append(statuses[picks] == int(phyre.SimulationStatus.SOLVED))
        if len(chosen) == N_TASKS:
            break
    simulator = phyre.initialize_simulator(chosen, "ball")
    task_inds = np.repeat(np.arange(N_TASKS), N_PER_TASK)
    actions = np.concatenate(actions).astype(np.float32)
    is_solved = torch.from_numpy(np.concatenate(solved))
    feats = []
    for t, a in zip(task_inds, actions):
        result = simulator.simulate_action(int(t), a, stride=60, need_images=False, need_featurized_objects=True)
        assert int(result.status) == (1 if is_solved[len(feats)] else -1), "simulation disagrees with the cache"
        feats.append(result.featurized_objects.features)
    rollouts, masks = standardize_features_shape(feats, MAX_LEN)
    rollouts, masks = torch.from_numpy(rollouts), torch.from_numpy(masks)
    observations = torch.LongTensor(simulator.initial_scenes)[torch.from_numpy(task_inds)]
    actions_t = torch.from_numpy(actions)
    distances = [rollout_dist.measure_all_distances(rollouts[i:i + N_PER_TASK], masks[i:i + N_PER_TASK],
                                                    HP["distance_cutoff"], cross_task=False, time_slice=0)
                 for i in range(0, len(task_inds), N_PER_TASK)]

    torch.manual_seed(SEED_INIT)
    model = neural_agent.build_model("resnet18", action_space_dim=3, fusion_place="last", action_hidden_size=256,
                                     action_layers=1, embedding_dim=HP["embedding_dim"],
                                     embeddor_type=HP["embeddor_type"],
                                     repr_merging_method=HP["repr_merging_method"],
                                     n_regressor_outputs=HP["n_regression_bins"])
    init = {k: v.clone() for k, v in model.state_dict().items()}
    model = nn.DataParallel(model)
    model.train()
    model.to(nets.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    steps = []
    for _ in range(2):
        # --- the body of neural_agent.train's loop (use_sample_distance_aux_loss=1), verbatim order ---
        scheduler.step()
        model.train()
        logits, embeddings = model(observations, actions_t, get_embeddings=True)
        aux_loss = neural_agent.sample_distance_loss(model, embeddings, task_inds, rollouts, masks,
                                                     np.arange(N_TASKS), N_PER_TASK, HP)
        optimizer.zero_grad()
        classification_loss = model.module.ce_loss(logits, is_solved).mean()
        loss = classification_loss + aux_loss * HP["weight"]
        loss.backward()
        lr_used = optimizer.param_groups[0]["lr"]
        grads = {k: p.grad.detach().clone() for k, p in model.module.named_parameters()}
        optimizer.step()
        # ---
        state = optimizer.state_dict()["state"]
        steps.append({"lr": lr_used, "logits": logits.detach().clone(), "embeddings": embeddings.detach().clone(),
                      "ce": classification_loss.item(), "aux": aux_loss.item(), "loss": loss.item(), "grads": grads,
                      "after": {k: v.detach().clone() for k, v in model.module.state_dict().items()},
                      "adam": {i: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in s.items()}
                               for i, s in state.items()}})
    probe_opt = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=LR)
    probe = torch.optim.lr_scheduler.CosineAnnealingLR(probe_opt, T_max=UPDATES)
    lrs, done = {}, 0
    for s in LR_PROBE_STEPS:
        while done < s:
            probe_opt.step()
            probe.step()
            done += 1
        lrs[s] = probe_opt.param_groups[0]["lr"]
    import torchvision
    meta = {"commit": COMMIT, "clone": str(CLONE), "tasks": chosen, "seed_init": SEED_INIT,
            "seed_input": SEED_INPUT, "hp": HP, "updates": UPDATES, "lr": LR,
            "files": {f: sha256(CLONE / "agents" / f) for f in ("nets.py", "neural_agent.py", "rollout_dist.py",
                                                                 "psim.py")},
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "torchvision": torchvision.__version__, "numpy": np.__version__,
                         "phyre": str(Path(phyre.__file__).parent)}}
    torch.save({"meta": meta, "inputs": {"observations": observations, "actions": actions_t, "is_solved": is_solved,
                                         "task_inds": torch.from_numpy(task_inds), "rollouts": rollouts,
                                         "masks": masks},
                "distances": distances, "init": init, "steps": steps, "lr_probe": lrs}, out_dir / "reference.pt")
    (out_dir / "reference.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print(json.dumps({"tasks": chosen, "losses": [(s["ce"], s["aux"], s["loss"]) for s in steps],
                      "lrs": [s["lr"] for s in steps]}))
    return 0


def rel(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def compare(out_dir):
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.dynamics_aware import novphy as dn

    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    ref = torch.load(out_dir / "reference.pt", weights_only=False)
    x = ref["inputs"]
    traj = dn.phyre_trajectories(x["rollouts"], x["masks"])
    t = TOLERANCE
    rows = {}
    torch.manual_seed(SEED_INIT)
    model = dn.A4Net(in_channels=7, action_size=3, hp=HP)
    own_init = model.state_dict()
    rows["init_identical_from_seed"] = (set(own_init) == set(ref["init"])
                                        and all(torch.equal(own_init[k], ref["init"][k]) for k in ref["init"]))
    model.load_state_dict(ref["init"])
    dist_rows = []
    for g, d_ref in enumerate(ref["distances"]):
        d = dn.trajectory_distances(dn.select(traj, slice(g * N_PER_TASK, (g + 1) * N_PER_TASK)),
                                    HP["distance_cutoff"])
        dist_rows.append({"max_abs": float((d - d_ref).abs().max()),
                          "bins_equal": bool(torch.equal(torch.round(d * (HP["n_regression_bins"] - 1)),
                                                         torch.round(d_ref * (HP["n_regression_bins"] - 1))))})
    rows["distances"] = {"per_task": dist_rows,
                         "pass": all(r["max_abs"] <= t["distance_atol"] and r["bins_equal"] for r in dist_rows)}
    optimizer, scheduler = dn.make_optimizer(model, UPDATES, LR)
    steps = []
    for i, s in enumerate(ref["steps"]):
        lr_before = scheduler.get_last_lr()[0]
        out = dn.train_step(model, optimizer, scheduler, x["observations"], x["actions"], x["is_solved"], traj,
                            N_PER_TASK, HP)
        lr_used = optimizer.param_groups[0]["lr"]
        grads = {k: p.grad.detach() for k, p in model.named_parameters()}
        gnorm = torch.sqrt(sum(g.double().square().sum() for g in grads.values()))
        gnorm_ref = torch.sqrt(sum(g.double().square().sum() for g in s["grads"].values()))
        grad_rel = {k: rel(grads[k], s["grads"][k]) for k in s["grads"] if s["grads"][k].norm() > 0}
        zero_grads = [k for k in s["grads"] if s["grads"][k].norm() == 0]
        after = model.state_dict()
        param_bad = [k for k, v in s["after"].items()
                     if not torch.allclose(after[k].double(), v.double(), atol=t["param_atol"], rtol=t["param_rtol"])]
        param_max = max(float((after[k].double() - v.double()).abs().max()) for k, v in s["after"].items()
                        if v.is_floating_point())
        row = {"lr_before_step": lr_before, "lr": lr_used, "lr_ref": s["lr"],
               "lr_rel": abs(lr_used - s["lr"]) / s["lr"],
               "ce_rel": abs(out["ce"].item() - s["ce"]) / abs(s["ce"]),
               "aux_rel": abs(out["aux"].item() - s["aux"]) / abs(s["aux"]),
               "loss_rel": abs(out["loss"].item() - s["loss"]) / abs(s["loss"]),
               "logits_rel_l2": rel(out["logits"].detach(), s["logits"]),
               "embeddings_rel_l2": rel(out["embeddings"].detach(), s["embeddings"]),
               "grad_norm_rel": float(abs(gnorm - gnorm_ref) / gnorm_ref), "grad_tensors": len(grad_rel),
               "grad_rel_l2_max": max(grad_rel.values()), "grad_rel_l2_argmax": max(grad_rel, key=grad_rel.get),
               "zero_grad_tensors_ref": zero_grads,
               "zero_grads_match": all(float(grads[k].abs().max()) == 0.0 for k in zero_grads),
               "params_buffers_checked": len(s["after"]), "params_not_allclose": param_bad,
               "param_max_abs_diff": param_max,
               "losses": {"ce": out["ce"].item(), "aux": out["aux"].item(), "loss": out["loss"].item(),
                          "ref": [s["ce"], s["aux"], s["loss"]]}}
        row["pass"] = (row["lr_rel"] <= t["lr_rel"] and row["lr"] > 0
                       and max(row["ce_rel"], row["aux_rel"], row["loss_rel"]) <= t["loss_rel"]
                       and max(row["logits_rel_l2"], row["embeddings_rel_l2"], row["grad_rel_l2_max"])
                       <= t["tensor_rel_l2"]
                       and row["grad_norm_rel"] <= t["grad_norm_rel"] and row["zero_grads_match"] and not param_bad)
        steps.append(row)
        print(f"step {i}: lr {lr_used:.6e} loss_rel {row['loss_rel']:.2e} grad_max {row['grad_rel_l2_max']:.2e} "
              f"params_bad {len(param_bad)} pass {row['pass']}")
    probe_opt, probe = dn.make_optimizer(torch.nn.Linear(1, 1), UPDATES, LR)
    lr_rows, done = {}, 0
    for s_, lr_ref in ref["lr_probe"].items():
        while done < s_:
            probe_opt.step()
            probe.step()
            done += 1
        lr_rows[str(s_)] = {"ours": probe_opt.param_groups[0]["lr"], "ref": lr_ref,
                            "match": abs(probe_opt.param_groups[0]["lr"] - lr_ref) <= t["lr_rel"] * LR}
    record = {"schema": "issue_127_gport_v1", "issue": 128, "arm": "A4", "family": "dynamics_aware",
              "official": {"upstream": "https://github.com/facebookresearch/DynamicsAware", "commit": COMMIT,
                           "entry": "agents/neural_agent.py train() loop body, sample_distance_loss, "
                                    "rollout_dist.measure_all_distances, nets.ResNetFilmAction",
                           "env": "b127-dynaware", "meta": ref["meta"]},
              "ours": {"module": "world_model/baselines/dynamics_aware/novphy.py",
                       "sha256": sha256(ROOT / "world_model/baselines/dynamics_aware/novphy.py"),
                       "vendor_json_sha256": sha256(ROOT / "world_model/baselines/dynamics_aware/vendor/VENDOR.json"),
                       "configuration": "A4Net(in_channels=7, action_size=3) = the official configuration; "
                                        "novphy.phyre_trajectories; novphy.train_step; novphy.make_optimizer",
                       "env": "novphy", "torch": torch.__version__},
              "inputs": {"tasks": ref["meta"]["tasks"], "batch": int(x["actions"].shape[0]),
                         "n_samples_per_task": N_PER_TASK, "precision": "float32", "device": "cpu"},
              "tolerance": t, "init_identical_from_seed": rows["init_identical_from_seed"],
              "distances": rows["distances"], "steps": steps, "lr_schedule": lr_rows}
    record["pass"] = (rows["distances"]["pass"] and all(s["pass"] for s in steps)
                      and all(r["match"] for r in lr_rows.values()))
    (out_dir / "gport.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"G-port A4: {'pass' if record['pass'] else 'FAIL'} -> {out_dir / 'gport.json'}")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("reference", "compare"):
        raise SystemExit(__doc__)
    raise SystemExit(reference(sys.argv[2]) if sys.argv[1] == "reference" else compare(sys.argv[2]))
