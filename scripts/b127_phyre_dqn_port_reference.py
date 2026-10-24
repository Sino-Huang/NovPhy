"""#128 G-port for A1 DQN-px / A2 DQN-car: the official PHYRE DQN code vs the NovPhy adapter, CPU float32.

Two modes, two envs:

    # official side, conda env b127-phyre (phyre built from the pinned clone, torch 2.13 + torchvision 0.28)
    CUDA_VISIBLE_DEVICES="" python scripts/b127_phyre_dqn_port_reference.py reference [out_dir]
    # NovPhy side, conda env novphy: replays the recorded inputs through world_model.baselines.phyre_dqn.novphy
    CUDA_VISIBLE_DEVICES="" python scripts/b127_phyre_dqn_port_reference.py compare [out_dir]

``out_dir`` defaults to ``.local-artifacts/issue-128-direct-predictors-v1/gport``; ``compare`` writes ``gport.json``
(schema ``issue_127_gport_v1``) there.

**A1 (official configuration).** ``reference`` runs the unmodified official ``agents/neural_agent.train`` from the
clone (``PHYRE_REF``, default ``/mnt/array/sukaih/novphy-baselines-src/phyre@d0765dd``) on real PHYRE-B data: 8
training tasks of ball_within_template fold 0 (one per template, ``train[::200][:8]``), the first 1000 actions of the
shipped 100k simulation cache, ``ResNet18FilmAction`` (7-colour one-hot 256 x 256 input, 3-d action, hidden 256,
fusion "last"), batch 64, class-balanced, Adam lr 3e-4, cosine over ``updates=3`` (so the three updates run at
2.25e-4, 0.75e-4 and 0: the schedule's first LR, a later positive LR and the end of the cosine). Instrumentation
only, no official line changed: a wrapper around ``neural_agent.build_model`` adds a forward pre-hook (initial
state, batch inputs), a wrapper around ``nets.ResNet18FilmAction.ce_loss`` records logits / targets / loss in train
mode, global optimizer step hooks record gradients, LRs, parameters and Adam state, ``neural_agent.XE_EVAL_SIZE``
is set to 64 (the batch-0 eval-loss subset; eval mode, no effect on training) and ``Tensor.pin_memory`` is the
identity (CPU only). ``compare`` loads the initial state into ``A1Net("onehot7", action_size=3)`` (strict), rebuilds
the batches with the adapter's ``train_indices_sampler`` (RandomState(42)) and runs ``train_step`` three times.

**A2 (head path on A2 shapes).** ``reference`` builds the adapter's ``A2Net`` (matched width) with its FiLM module
replaced by the official ``nets.FilmActionNetwork`` (same weights) and runs the official
``ResNet18FilmAction.preprocess`` / ``_forward`` / ``ce_loss`` functions from the clone with the official optimizer,
scheduler and loop order (Adam, ``CosineAnnealingLR(T_max=3)``, ``scheduler.step()`` before each step) on fixed
carriers / launch4 vectors / labels; ``compare`` runs the adapter (``A2Net`` + ``train_step``) on the same batches.

**Schedule.** The official optimizer / scheduler pair is stepped through the native (100 000) and matched (9 000)
schedules on a dummy parameter; ``compare`` checks ``make_optimizer`` gives the same LR at every update.

Tolerances (#123 rule): loss and gradient-norm relative 1e-4; per-tensor gradient relative L2 1e-3 above
1e-6 x the global norm (absolute L2 below it); parameters after each step: >= 99.9 % within 1e-7 and every
element within 2 x LR + 1e-7; BatchNorm buffers and Adam moments allclose (rtol 1e-4, atol 1e-6 x max |.|);
LRs within 1e-12; identical batches.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "world_model/baselines/phyre_dqn"
OUT = ROOT / ".local-artifacts/issue-128-direct-predictors-v1/gport"
PHYRE_REF = Path(os.environ.get("PHYRE_REF", "/mnt/array/sukaih/novphy-baselines-src/phyre@d0765dd"))
PHYRE_CACHE = ROOT / "data/external/phyre_dqn/phyre_cache"
SEED_INIT, SEED_A2_INPUT, SAMPLER_SEED = 20261024, 7, 42
UPDATES, BATCH, LR, MAX_TRAIN_ACTIONS = 3, 64, 3e-4, 1000
A2_LEVELS, A2_EXAMPLES = 8, 256
SCHEDULES = {"native": 100_000, "matched": 9_000}
THREADS = 8
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
             "grad_floor_of_global_norm": 1e-6, "param_close_share": 0.999, "param_close_abs": 1e-7,
             "param_max_abs_over_lr": 2.0, "buffer_rtol": 1e-4, "buffer_atol_of_max": 1e-6, "lr_abs": 1e-12,
             "logit_max_abs": 1e-4}


def clone_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def sha256_of(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_adapter():
    """Import world_model/baselines/phyre_dqn as a standalone package (no world_model/__init__ side effects)."""
    if "phyre_dqn.novphy" in sys.modules:
        return sys.modules["phyre_dqn.novphy"]
    spec = importlib.util.spec_from_file_location("phyre_dqn", PACKAGE / "__init__.py",
                                                  submodule_search_locations=[str(PACKAGE)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["phyre_dqn"] = package
    spec.loader.exec_module(package)
    return importlib.import_module("phyre_dqn.novphy")


class StepRecorder:
    """Global optimizer step hooks: gradients, LR and global norm before, parameters / Adam state after."""

    def __init__(self, model):
        self.names = {id(p): name for name, p in model.named_parameters()}
        self.model, self.steps = model, []
        from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
        self.handles = [register_optimizer_step_pre_hook(self.pre), register_optimizer_step_post_hook(self.post)]

    def owns(self, optimizer):
        return all(id(p) in self.names for group in optimizer.param_groups for p in group["params"])

    def pre(self, optimizer, args, kwargs):
        if not self.owns(optimizer):
            return
        grads = {self.names[id(p)]: p.grad.detach().clone() for group in optimizer.param_groups
                 for p in group["params"] if p.grad is not None}
        norm = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads.values()])))
        self.steps.append({"grads": grads, "grad_norm": norm, "lr": optimizer.param_groups[0]["lr"]})

    def post(self, optimizer, args, kwargs):
        if not self.owns(optimizer):
            return
        self.steps[-1]["post_state"] = clone_state(self.model)
        self.steps[-1]["optimizer_state"] = {
            self.names[id(p)]: {k: (v.detach().clone() if torch.is_tensor(v) else v)
                                for k, v in optimizer.state[p].items()}
            for group in optimizer.param_groups for p in group["params"] if p in optimizer.state}

    def close(self):
        for handle in self.handles:
            handle.remove()


# ---------------------------------------------------------------------------------------------------------------
# official side (b127-phyre)
# ---------------------------------------------------------------------------------------------------------------

def reference_a1(out_dir):
    import nets
    import neural_agent
    import phyre

    cache = phyre.get_default_100k_cache("ball")
    train_ids = phyre.get_fold("ball_within_template", 0)[0]
    task_ids = tuple(train_ids[::200][:8])
    task_indices, is_solved, actions, _, observations = neural_agent.compact_simulation_data_to_trainset(
        "ball", **cache.get_sample(task_ids, MAX_TRAIN_ACTIONS))
    holder, batches, recorders = {}, [], []
    original_build, original_ce = neural_agent.build_model, nets.ResNet18FilmAction.ce_loss

    def pre_forward(module, args):
        if module.training:
            holder.setdefault("init_state", clone_state(module))
            batches.append({"observations": args[0].detach().clone(), "actions": args[1].detach().clone()})

    def build_model(**kwargs):
        model = original_build(**kwargs)
        model.register_forward_pre_hook(pre_forward)
        holder["model_kwargs"] = dict(kwargs)
        holder["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        recorders.append(StepRecorder(model))
        return model

    def ce_loss(self, decisions, targets):
        loss = original_ce(self, decisions, targets)
        if self.training:
            batches[-1].update(logits=decisions.detach().clone(), targets=targets.detach().clone(), loss=float(loss))
        return loss

    neural_agent.build_model, nets.ResNet18FilmAction.ce_loss = build_model, ce_loss
    neural_agent.XE_EVAL_SIZE = 64
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self
    torch.manual_seed(SEED_INIT)
    tmp = out_dir / "official-train-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    model = neural_agent.train(str(tmp), "ball", task_ids, cache=cache, train_batch_size=BATCH, learning_rate=LR,
                               max_train_actions=MAX_TRAIN_ACTIONS, updates=UPDATES, negative_sampling_prob=1.0,
                               save_checkpoints_every=-1, fusion_place="last", network_type="resnet18",
                               balance_classes=1, num_auccess_actions=0, eval_every=10_000, action_layers=1,
                               action_hidden_size=256, cosine_scheduler=1, dev_tasks_ids=None)
    tmp.rmdir()
    recorder = recorders[-1]
    recorder.close()
    neural_agent.build_model, nets.ResNet18FilmAction.ce_loss = original_build, original_ce
    if len(batches) != UPDATES or len(recorder.steps) != UPDATES:
        raise RuntimeError(f"expected {UPDATES} train-mode forwards and steps, got {len(batches)} / {len(recorder.steps)}")
    steps = [{**b, **s} for b, s in zip(batches, recorder.steps)]
    payload = {"arm": "A1", "task_ids": list(task_ids), "model_kwargs": holder["model_kwargs"],
               "trainable_parameters": holder["trainable_parameters"], "init_state": holder["init_state"],
               "data": {"task_indices": task_indices, "is_solved": is_solved, "actions": actions,
                        "observations": observations},
               "steps": steps, "final_state": clone_state(model), "num_colors": int(phyre.NUM_COLORS),
               "hyper": {"updates": UPDATES, "batch": BATCH, "lr": LR, "max_train_actions": MAX_TRAIN_ACTIONS,
                         "sampler_seed": SAMPLER_SEED, "seed_init": SEED_INIT}}
    torch.save(payload, out_dir / "reference-A1.pt")
    print(f"[phyre-port-reference] A1: losses {[round(s['loss'], 6) for s in steps]} at LRs "
          f"{[s['lr'] for s in steps]}, {holder['trainable_parameters']} params, "
          f"{int(is_solved.sum())}/{len(is_solved)} positive", flush=True)


def reference_a2(out_dir):
    import nets

    ours = load_adapter()
    official = nets.ResNet18FilmAction
    torch.manual_seed(SEED_INIT + 1)
    model = ours.A2Net(ours.A2_MATCHED_BASE_WIDTH)
    film = nets.FilmActionNetwork(4, model.stages[2][-1].bn2.num_features, hidden_size=256, num_layers=1)
    film.load_state_dict(model._action_network.state_dict())
    model._action_network = film
    model.action_networks = [None, None, None, film]
    init = clone_state(model)
    g = torch.Generator().manual_seed(SEED_A2_INPUT)
    carriers = torch.randn(A2_LEVELS, 236, generator=g)
    task_indices = torch.randint(0, A2_LEVELS, (A2_EXAMPLES,), generator=g)
    actions = torch.rand(A2_EXAMPLES, 4, generator=g) * torch.tensor([-0.35, 0.2, 1.0, 1.5]) + torch.tensor(
        [0.0, 0.0, 0.5, 0.0])
    is_solved = torch.rand(A2_EXAMPLES, generator=g) < 0.3
    rng = np.random.RandomState(SAMPLER_SEED)
    batch_indices = [rng.choice(A2_EXAMPLES, size=BATCH) for _ in range(UPDATES)]
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UPDATES)
    recorder = StepRecorder(model)
    batches = []
    for indices in batch_indices:
        scheduler.step()
        model.train()
        observations = carriers[task_indices[indices]]
        optimizer.zero_grad()
        preprocessed = official.preprocess(model, observations)
        logits = official._forward(model, actions[indices], **preprocessed)
        loss = official.ce_loss(model, logits, is_solved[indices])
        loss.backward()
        optimizer.step()
        batches.append({"logits": logits.detach().clone(), "loss": float(loss)})
    recorder.close()
    steps = [{**b, **s} for b, s in zip(batches, recorder.steps)]
    payload = {"arm": "A2", "base_width": ours.A2_MATCHED_BASE_WIDTH, "init_state": init,
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
               "data": {"carriers": carriers, "task_indices": task_indices, "actions": actions,
                        "is_solved": is_solved, "batch_indices": [torch.as_tensor(i) for i in batch_indices]},
               "steps": steps, "official_functions": ["nets.ResNet18FilmAction.preprocess",
                                                      "nets.ResNet18FilmAction._forward",
                                                      "nets.ResNet18FilmAction.ce_loss", "nets.FilmActionNetwork"],
               "hyper": {"updates": UPDATES, "batch": BATCH, "lr": LR, "seed_init": SEED_INIT + 1,
                         "seed_input": SEED_A2_INPUT}}
    torch.save(payload, out_dir / "reference-A2.pt")
    print(f"[phyre-port-reference] A2: losses {[round(s['loss'], 6) for s in steps]} at LRs "
          f"{[s['lr'] for s in steps]}, {payload['trainable_parameters']} params", flush=True)


def official_schedule(updates):
    """LR of update k = 1..updates under neural_agent.train's Adam + CosineAnnealingLR (scheduler stepped first)."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates)
    lrs = []
    for _ in range(updates):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    return torch.tensor(lrs, dtype=torch.float64)


def reference(out_dir):
    import warnings

    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
    os.environ.setdefault("PHYRE_CACHE_DIR", str(PHYRE_CACHE))
    sys.path.insert(0, str(PHYRE_REF / "agents"))
    import torchvision
    import phyre

    torch.set_num_threads(THREADS)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_a1(out_dir)
    reference_a2(out_dir)
    torch.save({name: official_schedule(t) for name, t in SCHEDULES.items()}, out_dir / "reference-schedule.pt")
    meta = {"torch": torch.__version__, "torchvision": torchvision.__version__, "phyre_file": phyre.__file__,
            "phyre_ref": str(PHYRE_REF), "agents_sha256": {name: sha256_of(PHYRE_REF / "agents" / name)
                                                           for name in ("nets.py", "neural_agent.py")},
            "python": sys.version.split()[0], "written_at": utc_now()}
    (out_dir / "reference-meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[phyre-port-reference] wrote {out_dir}", flush=True)


# ---------------------------------------------------------------------------------------------------------------
# NovPhy side (novphy)
# ---------------------------------------------------------------------------------------------------------------

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def compare_step(mine, official, lr_bound):
    tol = TOLERANCE
    floor = tol["grad_floor_of_global_norm"] * official["grad_norm"]
    above = [float((mine["grads"][k] - v).norm() / v.norm()) for k, v in official["grads"].items()
             if float(v.norm()) > floor]
    below = [float((mine["grads"][k] - v).norm()) for k, v in official["grads"].items() if float(v.norm()) <= floor]
    params = [k for k, v in official["post_state"].items() if k in mine["named_parameters"]]
    deltas = torch.cat([(mine["post_state"][k] - official["post_state"][k]).abs().flatten() for k in params])
    buffers = [k for k, v in official["post_state"].items() if k not in mine["named_parameters"]
               and v.is_floating_point()]
    buffer_ok = all(torch.allclose(mine["post_state"][k], official["post_state"][k], rtol=tol["buffer_rtol"],
                                   atol=tol["buffer_atol_of_max"] * max(1.0, float(official["post_state"][k].abs().max())))
                    for k in buffers)
    moments_ok, moment_max = True, 0.0
    for name, state in official["optimizer_state"].items():
        for key in ("exp_avg", "exp_avg_sq"):
            ref, got = state[key], mine["optimizer_state"][name][key]
            scale = max(float(ref.abs().max()), 1e-30)
            moment_max = max(moment_max, float((got - ref).abs().max()) / scale)
            moments_ok &= torch.allclose(got, ref, rtol=tol["buffer_rtol"], atol=tol["buffer_atol_of_max"] * scale)
        moments_ok &= float(mine["optimizer_state"][name]["step"]) == float(state["step"])
    out = {"loss": mine["loss"], "official_loss": official["loss"],
           "loss_relative_delta": abs(mine["loss"] - official["loss"]) / max(1.0, abs(official["loss"])),
           "logit_max_abs_delta": float((mine["logits"] - official["logits"]).abs().max()),
           "grad_norm": mine["grad_norm"], "official_grad_norm": official["grad_norm"],
           "grad_norm_relative_delta": abs(mine["grad_norm"] - official["grad_norm"]) / official["grad_norm"],
           "grad_tensor_max_relative_l2": max(above or [0.0]), "null_gradient_tensors": len(below),
           "null_gradient_max_abs_l2": max(below or [0.0]), "null_gradient_floor": floor,
           "lr": mine["lr"], "official_lr": official["lr"],
           "param_close_share": float((deltas <= tol["param_close_abs"]).float().mean()),
           "param_max_abs_delta": float(deltas.max()), "param_max_abs_bound": tol["param_max_abs_over_lr"] * lr_bound
           + tol["param_close_abs"], "buffers_checked": len(buffers), "buffers_allclose": bool(buffer_ok),
           "adam_moment_max_relative_delta": moment_max, "adam_state_allclose": bool(moments_ok)}
    out["pass"] = bool(out["loss_relative_delta"] <= tol["loss_relative"]
                       and out["logit_max_abs_delta"] <= tol["logit_max_abs"]
                       and out["grad_norm_relative_delta"] <= tol["grad_norm_relative"]
                       and out["grad_tensor_max_relative_l2"] <= tol["grad_tensor_relative_l2"]
                       and out["null_gradient_max_abs_l2"] <= floor
                       and abs(out["lr"] - out["official_lr"]) <= tol["lr_abs"]
                       and out["param_close_share"] >= tol["param_close_share"]
                       and out["param_max_abs_delta"] <= out["param_max_abs_bound"]
                       and buffer_ok and moments_ok)
    return out


def run_adapter_steps(nv, model, data, batch_indices, updates):
    """The adapter's training steps with the same step hooks as the reference; logits via a forward hook."""
    optimizer, scheduler = nv.make_optimizer(model, learning_rate=LR, updates=updates)
    recorder = StepRecorder(model)
    logits = []
    handle = model.register_forward_hook(lambda m, a, out: logits.append(out.detach().clone()) if m.training else None)
    losses = [float(nv.train_step(model, optimizer, scheduler, data, idx)) for idx in batch_indices]
    handle.remove()
    recorder.close()
    names = {name for name, _ in model.named_parameters()}
    return [{**s, "loss": loss, "logits": lg, "named_parameters": names}
            for s, loss, lg in zip(recorder.steps, losses, logits)]


def compare_arm(official_steps, mine_steps, init_state):
    lr_bound = max(s["lr"] for s in official_steps)
    steps = [compare_step(m, o, max(lr_bound, 1e-12)) for m, o in zip(mine_steps, official_steps)]
    previous = init_state
    for step, official in zip(steps, official_steps):  # how far the official step moved the parameters
        step["official_param_update_max_abs"] = max(
            float((official["post_state"][k] - previous[k]).abs().max()) for k in mine_steps[0]["named_parameters"])
        previous = official["post_state"]
    return steps, all(s["pass"] for s in steps) and len(steps) == len(official_steps) == UPDATES


def compare_a1(nv, ref):
    model = nv.A1Net("onehot7", action_size=ref["model_kwargs"]["action_space_dim"])
    missing = model.load_state_dict(ref["init_state"], strict=True)
    d = ref["data"]
    data = nv.TrainSet(task_indices=d["task_indices"], is_solved=d["is_solved"], actions=d["actions"],
                       observations=d["observations"], levels=tuple(ref["task_ids"]))
    sampler = nv.train_indices_sampler(data.is_solved, BATCH, np.random.RandomState(SAMPLER_SEED))
    batch_indices = [next(sampler) for _ in range(UPDATES)]
    same_batches = all(
        torch.equal(data.actions[idx], o["actions"]) and torch.equal(data.is_solved[idx], o["targets"])
        and torch.equal(data.observations[data.task_indices[idx]], o["observations"])
        for idx, o in zip(batch_indices, ref["steps"]))
    balanced = [int(data.is_solved[idx].sum()) for idx in batch_indices]
    mine = run_adapter_steps(nv, model, data, batch_indices, UPDATES)
    steps, ok = compare_arm(ref["steps"], mine, ref["init_state"])
    params = nv.trainable_parameters(model)
    out = {"configuration": "official: ResNet18FilmAction, 7-colour one-hot 256x256, action 3-d (PHYRE-B), hidden 256, "
                            "fusion last, batch 64 class-balanced, Adam 3e-4, cosine T_max=3",
           "official_model_kwargs": ref["model_kwargs"], "task_ids": ref["task_ids"],
           "examples": len(data.is_solved), "positives": int(data.is_solved.sum()),
           "state_dict_strict_load": not (missing.missing_keys or missing.unexpected_keys),
           "trainable_parameters": params, "official_trainable_parameters": ref["trainable_parameters"],
           "same_batches": bool(same_batches), "positives_per_batch": balanced,
           "num_colors": ref["num_colors"], "num_colors_equal": ref["num_colors"] == nv.nets.phyre.NUM_COLORS,
           "steps": steps}
    out["pass"] = bool(ok and same_batches and out["state_dict_strict_load"] and out["num_colors_equal"]
                       and params == ref["trainable_parameters"])
    return out


def compare_a2(nv, ref):
    model = nv.A2Net(ref["base_width"])
    missing = model.load_state_dict(ref["init_state"], strict=True)
    d = ref["data"]
    data = nv.TrainSet(task_indices=d["task_indices"], is_solved=d["is_solved"], actions=d["actions"],
                       observations=d["carriers"], levels=tuple(range(A2_LEVELS)))
    batch_indices = [i.numpy() for i in d["batch_indices"]]
    mine = run_adapter_steps(nv, model, data, batch_indices, UPDATES)
    steps, ok = compare_arm(ref["steps"], mine, ref["init_state"])
    params = nv.trainable_parameters(model)
    out = {"configuration": "A2 matched (base width %d) on 236-d carriers; official preprocess/_forward/ce_loss and "
                            "FilmActionNetwork vs the adapter's A2Net; Adam 3e-4, cosine T_max=3" % ref["base_width"],
           "official_functions": ref["official_functions"],
           "state_dict_strict_load": not (missing.missing_keys or missing.unexpected_keys),
           "trainable_parameters": params, "official_trainable_parameters": ref["trainable_parameters"],
           "u_parameters": nv.U_PARAMETERS, "parameter_ratio_to_u": params / nv.U_PARAMETERS,
           "within_matched_tolerance": abs(params / nv.U_PARAMETERS - 1) <= nv.MATCHED_TOLERANCE, "steps": steps}
    out["pass"] = bool(ok and out["state_dict_strict_load"] and params == ref["trainable_parameters"]
                       and out["within_matched_tolerance"])
    return out


def compare_schedule(nv, ref):
    out = {}
    for name, official in ref.items():
        updates = len(official)
        model = torch.nn.Linear(1, 1)
        optimizer, scheduler = nv.make_optimizer(model, learning_rate=LR, updates=updates)
        lrs = []
        for _ in range(updates):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
        delta = float((torch.tensor(lrs, dtype=torch.float64) - official).abs().max())
        out[name] = {"updates": updates, "lr_first": float(official[0]), "lr_last": float(official[-1]),
                     "max_abs_delta": delta, "pass": delta <= TOLERANCE["lr_abs"]}
    return out


def compare(out_dir, write=True):
    import warnings

    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.phyre_dqn import novphy as nv

    torch.set_num_threads(THREADS)
    refs = {name: out_dir / f"reference-{name}.pt" for name in ("A1", "A2", "schedule")}
    a1 = compare_a1(nv, torch.load(refs["A1"], map_location="cpu", weights_only=True))
    a2 = compare_a2(nv, torch.load(refs["A2"], map_location="cpu", weights_only=True))
    schedule = compare_schedule(nv, torch.load(refs["schedule"], map_location="cpu", weights_only=True))
    meta = json.loads((out_dir / "reference-meta.json").read_text())
    result = {"schema": "issue_127_gport_v1", "issue": 128, "checked_at": utc_now(), "tolerance": TOLERANCE,
              "official": {**meta, "commit": "d0765dd2f6d6cf41b933a3a57173c63d21fbfa6a",
                           "references": {k: sha256_of(v) for k, v in refs.items()}},
              "adapter": {"module": "world_model/baselines/phyre_dqn/novphy.py",
                          "module_sha256": sha256_of(PACKAGE / "novphy.py"),
                          "vendor_json_sha256": sha256_of(PACKAGE / "vendor/VENDOR.json"),
                          "torch": torch.__version__, "python": sys.version.split()[0]},
              "arms": {"A1": a1, "A2": a2}, "schedule": schedule}
    result["pass"] = bool(a1["pass"] and a2["pass"] and all(s["pass"] for s in schedule.values()))
    if write:
        (out_dir / "gport.json").write_text(json.dumps(result, indent=2) + "\n")
    for arm, r in result["arms"].items():
        worst = {k: max(s[k] for s in r["steps"]) for k in ("loss_relative_delta", "grad_norm_relative_delta",
                                                            "grad_tensor_max_relative_l2", "param_max_abs_delta",
                                                            "logit_max_abs_delta")}
        print(f"[phyre-gport] {arm}: pass={r['pass']} {worst}", flush=True)
    print(f"[phyre-gport] schedule: {schedule}; overall pass={result['pass']}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    if mode == "reference":
        reference(target)
    elif mode == "compare":
        sys.exit(compare(target))
    else:
        sys.exit("usage: b127_phyre_dqn_port_reference.py {reference|compare} [out_dir]")
