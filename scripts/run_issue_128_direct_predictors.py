"""Issue-128 runner: the direct success predictors of #127 Q1 (A1 DQN-px, A2 DQN-car, A2-native, A4 DynDQN, A3 Phy-Q).

Every arm predicts P(pig removal | level anchor, launch) directly; the primary score is the success logit (Q for A3)
and the ranking cost is its negative (``eb.score_rows``, key ``neg_score``). Fit split trains; policy split selects;
the evaluation split is read only by ``--score`` after ``--freeze-readouts``. Contract:
``novphy-spec-notes/round3/child-runner-contract.md``.

Data paths (where every model input and label comes from):
- training example = one per fit shot (13 919): observation = the shot's LEVEL ANCHOR, never the shot's own frames:
  frame 0 (pre-launch) of the level's anchor branch (``eb.fit_anchor_frames()``; the frame 0 of every other branch of
  the level equals it up to capture noise, <= 32 differing pixels and mean |diff| <= 0.01 grey levels,
  checked exhaustively on fit and policy by ``--smoke``: ``accounting.anchor_frame_pre_launch``), the anchor carrier
  (``eb.fit_anchors()["carriers"]``, A2) or the anchor's symbolic state (A3-sym, ``pq.load_symbolic_store("fit")``);
  action = ``eb.launch4`` of the shot's launch (``FitShots.launch4``); label = the engine verdict
  (``FitShots.verdicts``, pig removal). Gathered in ``Context.trainset`` -> the family's ``trainset``.
- policy readouts (selection score, G-rank): one observation per policy level, ``Context.anchors("policy", kind)``
  (``eb.policy_anchor_frames()`` / ``eb.policy_anchors()["carriers"]`` / the policy symbolic store), held FIXED across
  the cell's 16 grid candidates; only ``eb.launch4(candidate action)`` varies (``policy_rows``); labels enter only in
  ``eb.policy_grid_auc``. G-rank: the 16 grid launches from each of the 480 fit anchors (``fit_costs``).
- scoring: the Gate-B anchor of each evaluation level (``eb.read_evaluation_anchors``; A3-sym:
  ``symbolic_evaluation_anchors``), fixed across the cell's candidates (``score_arm_seed``).

Modes (``python -u -m scripts.run_issue_128_direct_predictors --<mode>``; GPU modes only via the night-window
scheduler, ``scripts/gpu_window_issue_128.json``):

  --selection --gpu g --workers k [--arms ...]   the pre-declared configurations, seed 20260908 (resumable); A4
                                                  (deferred selection) only after the freeze and its Tier-T G-fid
  --selection-readouts --gpu g --workers k        policy grid AUC, G-rank (480 fit anchors x 16 grid), MACs
  --smoke                                         data accounting, G-port, tests, measured throughput, selection,
                                                  projection vs the child cap -> smoke.json (CPU)
  --prepare                                       freeze plan.json (afterwards: re-verify only)
  --select-deferred                               post-freeze mechanical selection of A4 -> selection-deferred.json
  --train --gpu g                                 the frozen queue of GPU g (the selected seed-20260908 run is reused),
                                                  then A4's main seeds on their planned GPUs once A4 is selected
  --readouts --gpu g --workers k [--arms ...]     readouts.json per arm-seed (G-rank, policy AUC, MACs)
  --freeze-readouts [--arms group]                freeze one whole scoring group (default: the next): model / readouts
                                                  hashes, before any evaluation anchor is read
  --score [--arms group]                          one pass per frozen group on every mixed cell of every inventory
                                                  (GPU 0): 'primary' = A1, A2, A2-native, A4; 'a3' = A3-sym, A3-img later
  --score-novelty [--arms ...]                    after an arm's primary pass: its guard-passing models on the mixed
                                                  #110 novelty levels (P-novelty, DESCRIPTIVE) -> novelty/records/
  --publish / --validate                          records manifest, compute.json, summary.json, findings.md (re-run
                                                  after the A3 pass: an addendum, nothing re-scored)
  --throughput [--arms A4]                        measured seconds per native update (scratch) -> throughput/<arm>.json

Families: ``phyre_dqn`` (A1, A2, A2-native; world_model/baselines/phyre_dqn/novphy.py), ``dynamics_aware`` (A4;
world_model/baselines/dynamics_aware/novphy.py), ``phyq`` (A3-sym symbolic state, A3-img frame;
world_model/baselines/phyq/novphy.py). A family whose adapter module cannot be imported raises the typed
``adapter_missing``; its arms stay registered.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch

from scripts import external_baselines as eb

ISSUE = 128
IDENTITY = "issue-128-direct-predictors-v1"
OUTPUT = eb.ROOT / ".local-artifacts" / IDENTITY
RUNNER = "scripts/run_issue_128_direct_predictors.py"
TESTS = ("tests/test_issue_128_direct_predictors.py", "tests/test_issue_128_phyre_dqn.py",
         "tests/test_issue_128_dynamics_aware.py", "tests/test_issue_128_phyq.py")
VALIDATION_COMMAND = "python -u -m scripts.run_issue_128_direct_predictors --validate"
SCHEMA_PLAN = "issue_128_plan_v1"
SCHEMA_RECORD = "issue_128_record_v1"
SCHEMA_MODEL = "issue_128_model_v1"
SCHEMA_READOUTS = "issue_128_readouts_v1"
SCHEMA_SMOKE = "issue_128_smoke_v1"
SCHEMA_FREEZE = "issue_128_readouts_manifest_v1"
SCORE_KEY = "neg_score"
CHECKPOINT_EVERY = 2000
LOG_EVERY = 500
GPUS = 4
CAP_GPU_HOURS = 66.0  # the child share's upper end (#127 budget table, A: ~35-66 GPU-h incl. fidelity)
CAP_WALL_HOURS = 48.0  # on 4 GPUs (round2 CONVENTIONS)
SCORING_OVERHEAD_GPU_HOURS = 2.0  # readouts + scoring of every arm-seed (projection term until measured)
SEEDS = eb.SEEDS
SELECTION_SEED = eb.SELECTION_SEED
U_PARAMETERS = 1_837_690
FIT_EXAMPLES, FIT_LEVELS = 13_919, 480  # eb.FitShots(); checked by --smoke


def phyq_configurations():
    """A3's <= 3 native-recipe configurations (PhyqA3's decision): pq.recipe on the fit split; {} without adapter."""
    try:
        from world_model.baselines.phyq import novphy as pq
    except ImportError:
        return {}
    return {name: {k: v for k, v in pq.recipe(name, FIT_EXAMPLES, FIT_LEVELS).items()
                   if k in ("updates", "learning_rate", "resets", "configuration")}
            for name in pq.CONFIGURATIONS}


def length_axis(updates, learning_rate=3e-4):
    """Training-length configurations (native first): the native update count and 1/10, 1/100 of it, each with the
    official LR and its own cosine schedule over the run (Decision P-other, selection design)."""
    return {"native": {"updates": updates, "learning_rate": learning_rate},
            "tenth": {"updates": updates // 10, "learning_rate": learning_rate},
            "hundredth": {"updates": updates // 100, "learning_rate": learning_rate}}


def lr_axis(updates):
    """LR configurations at a fixed (matched) budget: the official 3e-4 first, then 1e-4 and 1e-3."""
    return {"lr3e-4": {"updates": updates, "learning_rate": 3e-4},
            "lr1e-4": {"updates": updates, "learning_rate": 1e-4},
            "lr1e-3": {"updates": updates, "learning_rate": 1e-3}}


ARMS = {
    "A1": {"family": "phyre_dqn", "model_arm": "A1", "variant": "primary", "role": "primary", "program_set": True,
           "input": "frame", "budget": "native", "batch": 64, "configurations": length_axis(100_000),
           "cap_gpu_hours_per_seed": 2.0, "score": "logit", "deferred_scoring": False, "deferred_selection": False},
    "A2": {"family": "phyre_dqn", "model_arm": "A2", "variant": "matched", "role": "primary", "program_set": True,
           "input": "carrier", "budget": "matched (U's B1 recipe: 9 000 updates x batch 64)", "batch": 64,
           "configurations": lr_axis(9_000), "cap_gpu_hours_per_seed": None, "score": "logit",
           "deferred_scoring": False, "deferred_selection": False},
    "A2-native": {"family": "phyre_dqn", "model_arm": "A2", "variant": "native_size", "role": "secondary",
                  "program_set": False, "input": "carrier", "budget": "native (PHYRE DQN schedule, A1-size MLP)",
                  "batch": 64, "configurations": length_axis(100_000), "cap_gpu_hours_per_seed": None,
                  "score": "logit", "deferred_scoring": False, "deferred_selection": False},
    "A4": {"family": "dynamics_aware", "model_arm": "A4", "variant": "primary", "role": "primary",
           "program_set": True, "input": "frame", "budget": "native (Table-1 recipe: 100 000 updates x batch 512)",
           "batch": 512, "configurations": length_axis(100_000), "cap_gpu_hours_per_seed": 3.0, "score": "logit",
           "deferred_scoring": False, "deferred_selection": True},
    "A3-sym": {"family": "phyq", "model_arm": "A3-sym", "variant": "primary", "role": "secondary",
               "program_set": False, "input": "symbolic", "budget": "native Phy-Q", "batch": 32,
               "configurations": phyq_configurations(), "cap_gpu_hours_per_seed": None, "score": "Q",
               "deferred_scoring": True, "deferred_selection": False},
    "A3-img": {"family": "phyq", "model_arm": "A3-img", "variant": "primary", "role": "secondary",
               "program_set": False, "input": "frame", "budget": "native Phy-Q", "batch": 32,
               "configurations": phyq_configurations(), "cap_gpu_hours_per_seed": None, "score": "Q",
               "deferred_scoring": True, "deferred_selection": False},
}
WITHIN_FAMILY = {"A2-A1": ("A2", "A1"), "A4-A1": ("A4", "A1")}
# One scoring pass per group (Main, 2026-10-24): the program-set arms and A2-native together after A4's seeds are
# trained; A3 later (after its P-engine fidelity outcome). A group is frozen and scored as a whole.
SCORING_GROUPS = {"primary": ("A1", "A2", "A2-native", "A4"), "a3": ("A3-sym", "A3-img")}
# A4's selection runs after the plan freeze, once its Tier-T G-fid passes (#127 ladder: "A4 (Tier-T G-fid first)");
# its configurations, rule and budget are frozen in the plan, its main seeds run on these GPUs.
DEFERRED_TRAINING_GPUS = {"A4": {20260909: 0, 20260910: 1}}
GFID_SOURCE = {"A1": "fidelity/gfid.json", "A2": "fidelity/gfid.json", "A2-native": "fidelity/gfid.json",
               "A4": "fidelity/a4/gfid.json", "A3-sym": "fidelity/a3/gfid.json", "A3-img": "fidelity/a3/gfid.json"}
GPORT_SOURCE = {"A1": ("gport/gport.json", "A1"), "A2": ("gport/gport.json", "A2"),
                "A2-native": ("gport/gport.json", "A2"), "A4": ("gport/a4/gport.json", None),
                "A3-sym": ("gport/a3/gport.json", None), "A3-img": ("gport/a3/gport.json", None)}


def log(message):
    eb.log(ISSUE, message)


class AdapterMissing(RuntimeError):
    """The arm's family has no NovPhy adapter yet (typed reason ``adapter_missing``)."""


# ---------------------------------------------------------------------------------------------------------------
# families: one runner-facing interface per adapter module
# ---------------------------------------------------------------------------------------------------------------

class PhyreDQN:
    """A1 / A2 / A2-native (world_model/baselines/phyre_dqn/novphy.py)."""

    def __init__(self):
        from world_model.baselines.phyre_dqn import novphy as nv
        self.nv = nv

    def build(self, spec):
        return self.nv.build_model(spec["model_arm"], spec["variant"])

    def optimizer(self, model, config):
        return self.nv.make_optimizer(model, learning_rate=config["learning_rate"], updates=config["updates"])

    def trainset(self, spec, fit, anchors):
        return self.nv.build_trainset(fit.members, fit.launch4, fit.verdicts, anchors)

    def to(self, data, device):
        return data._replace(observations=data.observations.to(device))

    def batch(self, data, spec, rng):
        return next(self.nv.train_indices_sampler(data.is_solved, spec["batch"], rng))

    def step(self, model, optimizer, scheduler, data, indices, device):
        return {"loss": self.nv.train_step(model, optimizer, scheduler, data, indices, device)}

    def score(self, model, observation, launches):
        return self.nv.score_actions(model, observation, launches)


class DynamicsAware:
    """A4 (world_model/baselines/dynamics_aware/novphy.py; runner_train_step steps the scheduler itself)."""

    def __init__(self):
        from world_model.baselines.dynamics_aware import novphy as da
        self.da = da

    def build(self, spec):
        return self.da.build_model()

    def optimizer(self, model, config):
        return self.da.make_optimizer(model, updates=config["updates"], learning_rate=config["learning_rate"])

    def trainset(self, spec, fit, anchors):
        return self.da.runner_trainset(fit, anchors)

    def to(self, data, device):
        return self.da.data_to(data, device)

    def batch(self, data, spec, rng):
        return self.da.runner_batch(data, rng)

    def step(self, model, optimizer, scheduler, data, indices, device):
        return self.da.runner_train_step(model, optimizer, scheduler, data, indices, device)

    def score(self, model, observation, launches):
        return self.da.score_actions(model, observation, launches)


class Phyq:
    """A3-sym / A3-img: no adapter yet (world_model/baselines/phyq/novphy.py) -> AdapterMissing."""

    def __init__(self):
        try:
            from world_model.baselines.phyq import novphy as pq
        except ImportError as error:
            raise AdapterMissing(f"world_model/baselines/phyq/novphy.py: {error}") from error
        self.pq = pq

    def build(self, spec):
        return self.pq.build_model(spec["model_arm"])

    def optimizer(self, model, config):
        """AdamW + OuterSteps (a fresh AdamW state at every outer step; runner_train_step steps it)."""
        return self.pq.make_optimizer(model, config["updates"], config["learning_rate"],
                                      resets=config.get("resets", ()))

    def trainset(self, spec, fit, anchors):
        return self.pq.runner_trainset(spec["model_arm"], fit, anchors)

    def to(self, data, device):
        return self.pq.data_to(data, device)

    def batch(self, data, spec, rng):
        return self.pq.runner_batch(data, rng)

    def step(self, model, optimizer, scheduler, data, indices, device):
        return self.pq.runner_train_step(model, optimizer, scheduler, data, indices, device)

    def score(self, model, observation, launches):
        # pq.score_actions takes CPU inputs (the official PIL / numpy image transform, a CPU action lookup table) and
        # moves the network input and the action indices to the model's device itself
        return self.pq.score_actions(model, observation.cpu(), launches.cpu())


FAMILIES = {"phyre_dqn": PhyreDQN, "dynamics_aware": DynamicsAware, "phyq": Phyq}
_FAMILY_CACHE = {}


def family_of(arm):
    name = ARMS[arm]["family"]
    if name not in _FAMILY_CACHE:
        _FAMILY_CACHE[name] = FAMILIES[name]()
    return _FAMILY_CACHE[name]


def available(arm):
    try:
        family_of(arm)
        return True
    except AdapterMissing as error:
        log(f"{arm}: adapter_missing ({error})")
        return False


def trainable_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def launch_matrix(items):
    """eb.launch4 of {"action": ...} items, float32 [K, 4]."""
    return torch.stack([eb.launch4(item["action"]) for item in items])


class Context:
    """Data read once per process: fit shots, anchors per split and input kind, trainsets on the device."""

    def __init__(self, device):
        self.device = device
        self._fit, self._anchors, self._data = None, {}, {}

    def fit(self):
        if self._fit is None:
            self._fit = eb.FitShots()
        return self._fit

    def anchors(self, split, kind):
        if (split, kind) not in self._anchors:
            if split == "evaluation":
                raise ValueError("evaluation anchors are read by --score only (eb.read_evaluation_anchors)")
            if kind == "symbolic":
                from world_model.baselines.phyq import novphy as pq
                value = pq.load_symbolic_store(split)["states"]
            elif split == "fit":
                value = eb.fit_anchor_frames() if kind == "frame" else eb.fit_anchors()["carriers"]
            elif split == "policy":
                value = eb.policy_anchor_frames() if kind == "frame" else eb.policy_anchors()["carriers"]
            else:
                raise ValueError(f"unknown split {split}")
            self._anchors[(split, kind)] = value
        return self._anchors[(split, kind)]

    def trainset(self, arm):
        spec = ARMS[arm]
        key = (spec["family"], spec["input"], spec["model_arm"] if spec["family"] == "phyq" else None)
        if key not in self._data:
            family = family_of(arm)
            data = family.trainset(spec, self.fit(), self.anchors("fit", spec["input"]))
            self._data[key] = family.to(data, self.device)
        return self._data[key]


# ---------------------------------------------------------------------------------------------------------------
# training (resumable; batch b drawn from eb.batch_rng(seed, b); nonfinite loss = typed terminal failure)
# ---------------------------------------------------------------------------------------------------------------

def selection_dir(output, arm, config):
    return Path(output) / "selection" / arm / config / f"seed-{SELECTION_SEED}"


def main_dir(output, arm, seed):
    return Path(output) / "models" / arm / f"seed-{seed}"


def deferred_selection_path(output):
    return Path(output) / "selection-deferred.json"


# Post-freeze amendment #128 A2 (Decision P-fid repair 1, issuecomment-6464616983): an arm whose port defect was
# repaired re-runs its frozen selection (same configurations, seed and rule) on the corrected inputs; the frozen
# selected_configuration (chosen on the defective inputs) is replaced by the mechanical re-selection written by
# --select-deferred; until then the arm has no configuration (no main seeds, readouts or freeze).
REPAIR_RESELECTION = {"A3-sym": "https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6464616983"}


def arm_selection(output, plan, arm):
    """(configuration name, config) of an arm under the frozen plan: frozen in plan.json, or (deferred-selection
    arms and repair re-selections) the post-freeze mechanical result in selection-deferred.json; (None, None)
    before it exists."""
    spec = plan["arms"][arm]
    if arm in REPAIR_RESELECTION:
        path = deferred_selection_path(output)
        entry = eb.read_json(path)["arms"].get(arm) if path.is_file() else None
        if entry and entry["selected"]:
            return entry["selected"], spec["configurations"][entry["selected"]]
        return None, None
    if spec.get("selected_configuration"):
        return spec["selected_configuration"], spec["config"]
    path = deferred_selection_path(output)
    if spec["status"] == "selection_after_gfid" and path.is_file():
        entry = eb.read_json(path)["arms"].get(arm)
        if entry and entry["selected"]:
            return entry["selected"], spec["configurations"][entry["selected"]]
    return None, None


def model_dir(output, plan, arm, seed):
    """The run of an arm-seed under the frozen plan: seed 20260908 is the selected configuration's selection run."""
    if seed == SELECTION_SEED:
        name, _ = arm_selection(output, plan, arm)
        if name is None:
            raise ValueError(f"{arm}: no selected configuration yet")
        return selection_dir(output, arm, name)
    return main_dir(output, arm, seed)


def deferred_gate(output, arm):
    """A deferred-selection arm trains only after the plan freeze and after its G-fid passes."""
    if not ARMS[arm]["deferred_selection"]:
        return True
    if not (Path(output) / "plan.json").is_file():
        log(f"{arm}: selection runs after the plan freeze (deferred); skipped")
        return False
    if gfid_pass(output, arm) is not True:
        log(f"{arm}: selection waits for its G-fid ({GFID_SOURCE[arm]}) to pass; skipped")
        return False
    return True


def train_run(arm, config, seed, out_dir, context, max_steps=None):
    """Train one arm-seed under configuration ``config`` ({"updates", "learning_rate"}); resumable from resume.pt.
    Returns report.json's content, or {"partial": True, ...} when max_steps ends the call early."""
    spec, family, device = ARMS[arm], family_of(arm), context.device
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final, resume = out_dir / "model.pt", out_dir / "resume.pt"
    if final.is_file():
        return eb.read_json(out_dir / "report.json")
    updates = int(config["updates"])
    if device != "cpu":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    model = family.build(spec).to(device)
    optimizer, scheduler = family.optimizer(model, config)
    start, elapsed = 0, 0.0
    state = torch.load(resume, map_location="cpu", weights_only=False) if resume.is_file() else None
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng"])
        start, elapsed = state["step"], state["elapsed"]
        log(f"{arm} seed {seed}: resumed at update {start}")
    data = context.trainset(arm)
    stop_at = updates if max_steps is None else min(updates, start + max_steps)
    began, failure, trace, step = time.monotonic(), None, [], start - 1
    with open(out_dir / "train-log.jsonl", "a") as log_file:
        for step in range(start, stop_at):
            indices = family.batch(data, spec, eb.batch_rng(seed, step))
            out = family.step(model, optimizer, scheduler, data, indices, device)
            loss = float(out["loss"])
            if not math.isfinite(loss):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "loss": repr(loss)}
                break
            if step % LOG_EVERY == 0 or step == updates - 1:
                entry = {"step": step, "lr": optimizer.param_groups[0]["lr"], "loss": loss,
                         **{k: float(v) for k, v in out.items() if k != "loss" and torch.is_tensor(v) and v.ndim == 0},
                         "seconds": elapsed + time.monotonic() - began}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            stop = bool(eb.STOP_REQUESTED) and step + 1 < updates
            if stop or ((step + 1) % CHECKPOINT_EVERY == 0 and step + 1 < updates) or (
                    max_steps is not None and step + 1 == stop_at and step + 1 < updates):
                eb.atomic_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                         "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                                         "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                                         "step": step + 1, "elapsed": elapsed + time.monotonic() - began})
            if stop:
                raise eb.TrainingPaused(f"{arm} seed {seed} paused at update {step + 1} of {updates}")
    seconds = elapsed + time.monotonic() - began
    if failure is None and step + 1 < updates:
        return {"partial": True, "steps_done": step + 1, "seconds": seconds}
    complete = failure is None
    payload = {"schema": SCHEMA_MODEL, "arm": arm, "config": dict(config), "seed": seed, "updates": updates,
               "batch": spec["batch"], "complete": complete, "typed_failure": failure,
               "state_dict": model.state_dict() if complete else None, "parameters": trainable_parameters(model)}
    eb.atomic_torch(final, payload)
    report = {"arm": arm, "config": dict(config), "seed": seed, "updates": updates, "batch": spec["batch"],
              "complete": complete, "typed_failure": failure, "wall_seconds": seconds, "gpu_seconds": seconds,
              "seconds_per_update": seconds / max(1, step + 1),
              "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
              "model_sha256": eb.sha256_of(final), "parameters": payload["parameters"],
              "final": trace[-1] if trace else None, "finished_at": eb.utc_now()}
    eb.write_json(out_dir / "report.json", report)
    if resume.is_file():
        resume.unlink()
    log(f"{arm} seed {seed} {config}: {'complete' if complete else failure['typed_failure']} after {seconds:.0f}s")
    return report


def load_model(arm, path, device):
    """(model, payload) of a model.pt, or (None, payload) for a typed failure."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA_MODEL or payload["arm"] != arm:
        raise ValueError(f"{path}: model binding differs")
    if not payload["complete"]:
        return None, payload
    model = family_of(arm).build(ARMS[arm])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


# ---------------------------------------------------------------------------------------------------------------
# readouts (fit / policy only; deterministic_scoring())
# ---------------------------------------------------------------------------------------------------------------

def scores_for(arm, model, observation, items, device):
    launches = launch_matrix(items).to(device)
    return family_of(arm).score(model, observation.to(device), launches).float().cpu()


def policy_rows(arm, model, context):
    anchors = context.anchors("policy", ARMS[arm]["input"])
    return {member: eb.score_rows(entry["inventory"], scores_for(arm, model, anchors[member], entry["inventory"],
                                                                 context.device))
            for member, entry in eb.policy_cells("grid").items()}


def fit_costs(arm, model, context):
    """Ranking costs [480, 16]: -score of the 16 grid launches from every fit anchor."""
    anchors = context.anchors("fit", ARMS[arm]["input"])
    grid = eb.grid_items()
    return np.stack([-scores_for(arm, model, anchors[m], grid, context.device).numpy() for m in sorted(anchors)])


def readout_run(arm, run, context):
    """readouts.json of one trained run: policy grid AUC (selection score), G-rank, MACs per candidate."""
    from world_model.planning import scoring_harness as harness
    run = Path(run)
    began = time.monotonic()
    model, payload = load_model(arm, run / "model.pt", context.device)
    out = {"schema": SCHEMA_READOUTS, "arm": arm, "config": payload["config"], "seed": payload["seed"],
           "model_sha256": eb.sha256_of(run / "model.pt"), "typed_failure": payload["typed_failure"],
           "parameters": payload["parameters"]}
    if model is None:
        out.update({"policy_grid_auc": None, "G_rank": None, "macs_per_candidate": None})
    else:
        with harness.deterministic_scoring() as policy:
            rows = policy_rows(arm, model, context)
            out["policy_grid_auc"] = eb.policy_grid_auc(rows, SCORE_KEY)
            out["G_rank"] = eb.g_rank(fit_costs(arm, model, context))
            anchors = context.anchors("fit", ARMS[arm]["input"])
            observation = anchors[sorted(anchors)[0]].to(context.device)
            launches = launch_matrix(eb.grid_items()).to(context.device)
            out["macs_per_candidate"] = eb.macs_per_candidate(family_of(arm).score, model, observation, launches,
                                                              candidates=len(launches))
            out["scoring_policy"] = policy
    out["seconds"] = time.monotonic() - began
    eb.write_json(run / "readouts.json", out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------------------------------------------

def parse_arms(text, default):
    arms = list(default) if not text else [a.strip() for a in text.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}")
    return arms


def selection_runs(arms):
    return [(arm, config) for arm in arms for config in ARMS[arm]["configurations"]]


def device_name():
    return "cuda" if torch.cuda.is_available() else "cpu"


def selection(output, gpu, workers, arms, context=None, max_steps=None):
    """Train this worker's selection runs (fit split only); resumable; exit 75 on a stop signal."""
    context = context or Context(device_name())
    for i, (arm, config) in enumerate(selection_runs(arms)):
        if i % workers != gpu or not available(arm) or not deferred_gate(output, arm):
            continue
        run = selection_dir(output, arm, config)
        if (run / "model.pt").is_file():
            continue
        if eb.STOP_REQUESTED:
            return eb.EXIT_PAUSED
        try:
            report = train_run(arm, ARMS[arm]["configurations"][config], SELECTION_SEED, run, context, max_steps)
        except eb.TrainingPaused as paused:
            log(str(paused))
            return eb.EXIT_PAUSED
        if not report.get("partial"):
            eb.mark(output, f"selection-train-{arm}-{config}", report["wall_seconds"], report["gpu_seconds"])
    return 0


def selection_readouts(output, gpu, workers, arms, context=None):
    context = context or Context(device_name())
    for i, (arm, config) in enumerate(selection_runs(arms)):
        if i % workers != gpu or not available(arm):
            continue
        run = selection_dir(output, arm, config)
        if (run / "readouts.json").is_file() or not (run / "model.pt").is_file():
            continue
        if eb.STOP_REQUESTED:
            return eb.EXIT_PAUSED
        t0 = time.monotonic()
        result = readout_run(arm, run, context)
        eb.mark(output, f"selection-readouts-{arm}-{config}", time.monotonic() - t0,
                time.monotonic() - t0 if context.device != "cpu" else 0.0)
        log(f"selection readouts {arm}/{config}: policy grid AUC {result['policy_grid_auc']}, "
            f"G-rank {None if result['G_rank'] is None else result['G_rank']['pass']}")
    return 0


def selection_result(output, arms):
    """{arm: eb.select_configuration(...)} over the selection readouts (None for an arm with a missing readout)."""
    out = {}
    for arm in arms:
        results, complete = {}, True
        for config in ARMS[arm]["configurations"]:
            path = selection_dir(output, arm, config) / "readouts.json"
            if not path.is_file():
                complete = False
                continue
            report = eb.read_json(path)
            results[config] = {"auc": report["policy_grid_auc"], "parameters": report["parameters"]}
        out[arm] = {**eb.select_configuration(results), "complete": complete and bool(results)}
        if not out[arm]["complete"]:
            out[arm]["selected"] = None
    return out


THROUGHPUT_STEPS = 200


def measure_throughput(output, arm, context=None, steps=THROUGHPUT_STEPS):
    """Seconds per update of a scratch run of the native configuration (nothing kept): the median of the timed
    updates after the first quarter; writes throughput/<arm>.json."""
    context = context or Context(device_name())
    spec, family, device = ARMS[arm], family_of(arm), context.device
    config = spec["configurations"]["native"] if "native" in spec["configurations"] else next(
        iter(spec["configurations"].values()))
    if device != "cpu":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    model = family.build(spec).to(device)
    optimizer, scheduler = family.optimizer(model, config)
    data = context.trainset(arm)
    times = []
    for b in range(steps):
        if eb.STOP_REQUESTED:
            return eb.EXIT_PAUSED
        t0 = time.monotonic()
        family.step(model, optimizer, scheduler, data, family.batch(data, spec, eb.batch_rng(0, b)), device)
        if device != "cpu":
            torch.cuda.synchronize()
        times.append(time.monotonic() - t0)
    warm = times[len(times) // 4:]
    per = float(np.median(warm))
    out = {"arm": arm, "config": config, "batch": spec["batch"], "steps_timed": len(warm), "seconds_per_update": per,
           "mean_seconds_per_update": float(np.mean(warm)),
           "projected_gpu_hours_native_per_seed": per * config["updates"] / 3600.0,
           "projected_gpu_hours_selection": per * sum(c["updates"] for c in spec["configurations"].values()) / 3600.0,
           "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if device != "cpu" else None,
           "device": torch.cuda.get_device_name(0) if device != "cpu" else "cpu", "measured_at": eb.utc_now()}
    eb.write_json(Path(output) / "throughput" / f"{arm}.json", out)
    eb.mark(output, f"throughput-{arm}", sum(times), sum(times) if device != "cpu" else 0.0)
    log(f"throughput {arm}: {per:.4f} s/update at batch {spec['batch']} -> "
        f"{out['projected_gpu_hours_native_per_seed']:.2f} GPU-h per native seed")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# smoke, projection, plan
# ---------------------------------------------------------------------------------------------------------------

def lpt_queues(jobs, gpus=GPUS):
    loads, queues = [0.0] * gpus, [[] for _ in range(gpus)]
    for name, hours in sorted(jobs, key=lambda j: (-j[1], j[0])):
        k = int(np.argmin(loads))
        queues[k].append(name)
        loads[k] += hours
    return queues, loads


def gpu_hours_per_seed(output, arm, config):
    """Measured: the selection run's seconds per update x the configuration's updates."""
    report = eb.read_json(selection_dir(output, arm, config) / "report.json")
    return report["seconds_per_update"] * ARMS[arm]["configurations"][config]["updates"] / 3600.0


def fidelity_gpu_hours(output):
    walls = Path(output) / "fidelity" / "runs" / "walls.json"
    a1 = sum(eb.read_json(walls).values()) / 3600.0 if walls.is_file() else None
    a4_ledger = Path(output) / "fidelity" / "a4" / "ledger.json"
    a4 = eb.read_json(a4_ledger).get("gpu_hours") if a4_ledger.is_file() else None
    return {"A1_tier_e": a1, "A4_tier_t": a4}


# Projection terms not yet measured (replaced by measurements when they exist; Decision P-cap on #128):
A4_NATIVE_GPU_HOURS_PER_SEED_ESTIMATE = 10.0  # DynA4's FLOP projection 5-10 GPU-h, upper end (until throughput/A4.json)
A4_TIER_T_GPU_HOURS_ESTIMATE = 36.0  # DynA4's Tier-T P-cap (issuecomment-6382423910), upper end (until its ledger)
# Rule P-cap step 1 (unallocated program budget): the GPU-h drawn above the child cap and the decision that drew them.
P_CAP = {"decision": "https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-6386804015",
         "drawn_gpu_hours": 20.0}


def deferred_hours(output, arm):
    """(per-update seconds source, selection GPU-h, native GPU-h per seed) of a deferred-selection arm: measured
    throughput/<arm>.json if present, else the declared estimate; the main seeds are projected at native cost."""
    path = Path(output) / "throughput" / f"{arm}.json"
    native = ARMS[arm]["configurations"]["native"]["updates"]
    if path.is_file():
        per_update = eb.read_json(path)["seconds_per_update"] / 3600.0
        source = "measured"
    else:
        per_update = A4_NATIVE_GPU_HOURS_PER_SEED_ESTIMATE / native
        source = "estimate"
    selection = per_update * sum(c["updates"] for c in ARMS[arm]["configurations"].values())
    return source, selection, per_update * native


def projection(output, arms, selected, deferred=()):
    """GPU-h of everything left (main seeds of the selected configurations; deferred arms: selection + 2 main seeds
    at native cost; readouts / scoring) plus what was spent, against the child cap; wall on 4 GPUs (LPT queues)."""
    jobs, per_seed, deferred_jobs = [], {}, {}
    for arm in arms:
        config = selected[arm]["selected"]
        hours = gpu_hours_per_seed(output, arm, config)
        per_seed[arm] = {"config": config, "gpu_hours_per_seed": hours, "source": "measured",
                         "cap_gpu_hours_per_seed": ARMS[arm]["cap_gpu_hours_per_seed"],
                         "over_per_seed_cap": bool(ARMS[arm]["cap_gpu_hours_per_seed"] is not None
                                                   and hours > ARMS[arm]["cap_gpu_hours_per_seed"])}
        jobs += [(f"{arm}/{seed}", hours) for seed in SEEDS if seed != SELECTION_SEED]
    for arm in deferred:
        source, selection, native = deferred_hours(output, arm)
        cap = ARMS[arm]["cap_gpu_hours_per_seed"]
        per_seed[arm] = {"config": None, "gpu_hours_per_seed": native, "source": source,
                         "selection_gpu_hours": selection, "cap_gpu_hours_per_seed": cap,
                         "over_per_seed_cap": bool(cap is not None and native > cap)}
        deferred_jobs[arm] = selection + native * (len(SEEDS) - 1)
    queues, loads = lpt_queues(jobs)
    _, spent_gpu, _ = eb.ledger_totals(output, lambda phase: None)
    fid = fidelity_gpu_hours(output)
    if fid["A4_tier_t"] is None:
        fid = {**fid, "A4_tier_t": A4_TIER_T_GPU_HOURS_ESTIMATE, "A4_tier_t_source": "estimate"}
    remaining = sum(h for _, h in jobs) + sum(deferred_jobs.values()) + SCORING_OVERHEAD_GPU_HOURS
    total = remaining + spent_gpu / 3600.0 + sum(v for v in fid.values() if isinstance(v, float))
    # deferred arms: selection on one GPU (3 configurations in sequence), then 2 seeds on 2 GPUs in parallel
    wall = max(loads) + max([per_seed[a]["selection_gpu_hours"] + per_seed[a]["gpu_hours_per_seed"]
                             for a in deferred] or [0.0]) + SCORING_OVERHEAD_GPU_HOURS / GPUS
    budget = CAP_GPU_HOURS + P_CAP["drawn_gpu_hours"]
    return {"per_seed": per_seed, "queues": queues, "queue_hours": loads, "deferred_gpu_hours": deferred_jobs,
            "remaining_gpu_hours": remaining, "spent_ledger_gpu_hours": spent_gpu / 3600.0,
            "fidelity_gpu_hours": fid, "total_gpu_hours": total, "projected_wall_hours": wall,
            "cap_gpu_hours": CAP_GPU_HOURS, "cap_wall_hours": CAP_WALL_HOURS, "p_cap": P_CAP,
            "over_child_cap_gpu_hours": max(0.0, total - CAP_GPU_HOURS),
            "within_cap": bool(total <= budget and wall <= CAP_WALL_HOURS),
            "p_cap_arms": [a for a, v in per_seed.items() if v["over_per_seed_cap"]]}


def gport_pass(output, arm):
    path, key = GPORT_SOURCE[arm]
    path = Path(output) / path
    if not path.is_file():
        return None
    result = eb.read_json(path)
    return bool(result["arms"][key]["pass"] if key else result["pass"])


def gfid_pass(output, arm):
    path = Path(output) / GFID_SOURCE[arm]
    if not path.is_file():
        return None
    result = eb.read_json(path)
    return bool(result.get("pass"))


def run_tests():
    proc = subprocess.run(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS], cwd=eb.ROOT,
                          capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-400:]
    return {"command": "CUDA_VISIBLE_DEVICES='' python -m pytest -q " + " ".join(TESTS), "exit_code": proc.returncode,
            "tail": tail}


def ready_arms():
    """Arms whose adapter exists and that have configurations (A3 stays registered: adapter_missing)."""
    return [arm for arm, spec in ARMS.items() if spec["configurations"] and available(arm)]


CAPTURE_NOISE = {"max_differing_pixels": 32, "max_mean_abs_diff": 0.01}  # Decision P-other 2026-10-25


def frame_noise(a, b):
    """(differing pixels: positions where any channel differs, mean |diff| in grey levels over all 230 400 uint8
    elements) of two [3, 240, 320] frames."""
    diff = (a.to(torch.int16) - b.to(torch.int16)).abs()
    return int((diff > 0).any(0).sum()), float(diff.double().mean())


def _frame_check(split, anchors, keys, branches):
    """branches: iterable of (member, branch key, frame 0)."""
    exact, worst_n, worst_mean, differing, over, levels_differing, levels_over, count = True, 0, 0.0, 0, 0, set(), set(), 0
    for member, key, frame in branches:
        count += 1
        if key == keys[member]:
            exact &= torch.equal(frame, anchors[member])
            continue
        n, mean = frame_noise(frame, anchors[member])
        worst_n, worst_mean = max(worst_n, n), max(worst_mean, mean)
        if n:
            differing += 1
            levels_differing.add(member)
        if n > CAPTURE_NOISE["max_differing_pixels"] or mean > CAPTURE_NOISE["max_mean_abs_diff"]:
            over += 1
            levels_over.add(member)
    return {"split": split, "levels": len(anchors), "branches": count, "anchor_exact": bool(exact),
            "branches_differing": differing, "levels_differing": len(levels_differing),
            "worst_differing_pixels": worst_n, "worst_mean_abs_diff": worst_mean,
            "branches_over_tolerance": over, "levels_over_tolerance": len(levels_over),
            "pass": bool(exact and over == 0)}


def anchor_frame_check(fit):
    """Exhaustive, fit and policy: each level's anchor frame equals frame 0 of its own anchor branch exactly, and
    frame 0 of every other branch of the level within CAPTURE_NOISE (pre-launch, not candidate-specific; the
    measured differences are capture nondeterminism at a fixed location, uncorrelated with the launch)."""
    keys = eb.fit_anchors()["keys"]
    fit_check = _frame_check("fit", eb.fit_anchor_frames(), keys,
                             ((m, k, fit.frame(i, 0)) for i, (m, k) in enumerate(zip(fit.members, fit.keys))))
    store = eb.rc.PixelStore(eb.HJ123, "policy")
    pkeys = eb.policy_anchors()["keys"]
    pol_check = _frame_check("policy", eb.policy_anchor_frames(), pkeys,
                             ((k.rsplit("-c", 1)[0], k, store.frame(i, 0)) for i, k in enumerate(store.keys)
                              if k.rsplit("-c", 1)[0] in pkeys))
    return {"tolerance": CAPTURE_NOISE, "fit": fit_check, "policy": pol_check,
            "pass": bool(fit_check["pass"] and pol_check["pass"])}


# ---------------------------------------------------------------------------------------------------------------
# no-model nearest-fit references NN-px / NN-car (DESCRIPTIVE, seed-free like P; Decision P-other 2026-10-25)
# ---------------------------------------------------------------------------------------------------------------

NN_REFERENCES = {
    "NN-px": "nearest fit anchor by mean absolute difference over the 320x240 RGB anchor frames (exact integer sums)",
    "NN-car": "nearest fit anchor by L2 distance between E99 anchor carriers (float64)",
}
NN_RULE = ("score of a candidate = the engine verdict (1 / 0) of the identical launch executed in the nearest fit "
           "level; cost = -score (eb.score_rows); distance ties -> lowest fit member id; if that fit level did not "
           "execute the launch, the next fit level in (distance, member) order that did")
NN_COVERAGE = ("NN-px / NN-car are defined on grid and angle only; the fit split never executed offset or power "
               "launches, so those inventory columns are empty by construction")
SCHEMA_NN_REFERENCE = "issue_128_nn_reference_v1"


def launch_key(action):
    return tuple(sorted(action.items()))


def fit_verdicts_by_launch(fit):
    return {(m, launch_key(a)): bool(v) for m, a, v in zip(fit.members, fit.actions, fit.verdicts.tolist())}


def pixel_orders(queries, refs):
    """{query: [(mean |diff| grey levels, ref member)]} sorted by (distance, member); exact integer sums."""
    members = sorted(refs)
    r = torch.stack([refs[m].reshape(-1) for m in members]).to(torch.int16)
    out = {}
    for q, frame in sorted(queries.items()):
        d = (r - frame.reshape(-1).to(torch.int16)).abs().sum(1, dtype=torch.int64)
        out[q] = [(dist / r.shape[1], m) for dist, m in sorted(zip(d.tolist(), members))]
    return out


def carrier_orders(queries, refs):
    members = sorted(refs)
    r = torch.stack([torch.as_tensor(refs[m], dtype=torch.float64) for m in members])
    return {q: sorted(zip((r - torch.as_tensor(c, dtype=torch.float64)).norm(dim=1).tolist(), members))
            for q, c in sorted(queries.items())}


def nn_rows(order, inventory, verdicts):
    scores = [next((float(verdicts[(m, launch_key(item["action"]))]) for _, m in order
                    if (m, launch_key(item["action"])) in verdicts), float("nan")) for item in inventory]
    return eb.score_rows(inventory, torch.tensor(scores, dtype=torch.float64))


def nn_record(name, cells_by_inventory, orders, verdicts):
    """cells_by_inventory: {inventory: {member: entry}}; orders: {member: [(distance, fit member)]}."""
    return {"schema": SCHEMA_NN_REFERENCE, "plan_identity": IDENTITY, "reference": name, "rule": NN_RULE,
            "distance": NN_REFERENCES[name], "typed_failure": None, "engine_seconds": 0,
            "nearest": {m: {"fit_member": order[0][1], "distance": order[0][0]} for m, order in sorted(orders.items())
                        if any(m in cells for cells in cells_by_inventory.values())},
            "inventories": {inv: {m: {"rows": nn_rows(orders[m], e["inventory"], verdicts)}
                                  for m, e in sorted(cells.items())} for inv, cells in cells_by_inventory.items()}}


def nn_orders(frames, carriers):
    """NN-px / NN-car orders of query anchors against the 480 fit anchors."""
    return {"NN-px": pixel_orders(frames, eb.fit_anchor_frames()),
            "NN-car": carrier_orders(carriers, eb.fit_anchors()["carriers"])}


def _quantiles(values):
    v = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.quantile(v, 0.5)), "p95": float(np.quantile(v, 0.95)), "max": float(v.max()),
            "min": float(v.min())}


def policy_fit_redundancy(fit):
    """DISCLOSURE (not a gate): how close each policy anchor is to its nearest fit anchor (and to its nearest other
    policy anchor), and the pooled policy AUC of the no-model NN-px / NN-car verdict copies."""
    frames, carriers = eb.policy_anchor_frames(), eb.policy_anchors()["carriers"]
    orders = nn_orders(frames, carriers)
    pp = pixel_orders(frames, frames)
    pc = carrier_orders(carriers, carriers)
    verdicts = fit_verdicts_by_launch(fit)
    cells = {inv: eb.policy_cells(inv) for inv in eb.INVENTORIES}
    executed = {k for _, k in verdicts}
    out = {"policy_levels": len(frames),
           "nearest_fit_px_mean_abs_diff": _quantiles([o[0][0] for o in orders["NN-px"].values()]),
           "nearest_other_policy_px_mean_abs_diff": _quantiles([o[1][0] for o in pp.values()]),
           "nearest_fit_car_l2": _quantiles([o[0][0] for o in orders["NN-car"].values()]),
           "nearest_other_policy_car_l2": _quantiles([o[1][0] for o in pc.values()]),
           "policy_levels_closer_to_a_fit_level_than_to_any_other_policy_level_px": int(sum(
               orders["NN-px"][m][0][0] < pp[m][1][0] for m in frames)),
           "launch_coverage": {inv: [sum(launch_key(i["action"]) in executed
                                         for e in c.values() for i in e["inventory"]),
                                     sum(len(e["inventory"]) for e in c.values())] for inv, c in cells.items()},
           "nn_policy_auc": {}, "rule": NN_RULE}
    for name, order in orders.items():
        record = nn_record(name, cells, order, verdicts)
        out["nn_policy_auc"][name] = {inv: (eb.rc.pooled_auc(c, {m: record["inventories"][inv][m]["rows"] for m in c},
                                                             SCORE_KEY) if c else None)
                                      for inv, c in cells.items()}
    return out


def smoke(output):
    output = Path(output)
    began = time.monotonic()
    arms = ready_arms()
    now = [a for a in arms if not ARMS[a]["deferred_selection"]]
    deferred = [a for a in arms if ARMS[a]["deferred_selection"]]
    fit = eb.FitShots()
    accounting = {"fit_shots": len(fit), "fit_levels": len(fit.levels), "fit_positives": int(fit.verdicts.sum()),
                  "fit_anchor_frames": len(eb.fit_anchor_frames()), "fit_anchor_carriers": len(eb.fit_anchors()["carriers"]),
                  "policy_anchor_frames": len(eb.policy_anchor_frames()),
                  "policy_grid_mixed_cells": len(eb.policy_cells("grid")),
                  "policy_fit_level_overlap": len(set(eb.policy_anchors()["keys"]) & set(fit.levels)),
                  "policy_levels": len(eb.policy_anchors()["keys"]),
                  "fit_keys_are_fit_stream": all(k.startswith("issue-109-sc-fit") for k in fit.keys),
                  "anchor_frame_pre_launch": anchor_frame_check(fit),
                  "fit_examples_declared": (len(fit), len(fit.levels)) == (FIT_EXAMPLES, FIT_LEVELS),
                  "u_parameters": U_PARAMETERS}
    redundancy = policy_fit_redundancy(fit)
    selected = selection_result(output, now)
    missing = [a for a in now if not selected[a]["complete"]]
    gports = {arm: gport_pass(output, arm) for arm in ARMS}
    tests = run_tests()
    project = projection(output, [a for a in now if a not in missing], selected, deferred)  # ok needs no missing
    ok = bool(not missing and all(gports[a] for a in arms) and tests["exit_code"] == 0
              and accounting["fit_keys_are_fit_stream"] and accounting["anchor_frame_pre_launch"]["pass"]
              and accounting["policy_fit_level_overlap"] == 0 and accounting["fit_examples_declared"]
              and project is not None and project["within_cap"])
    evidence = {"schema": SCHEMA_SMOKE, "ok": ok, "at": eb.utc_now(), "arms": arms, "deferred_selection": deferred,
                "adapter_missing": [a for a in ARMS if a not in arms], "accounting": accounting,
                "selection": selected, "selection_missing": missing, "gport": gports,
                "gfid": {arm: gfid_pass(output, arm) for arm in ARMS}, "tests": tests,
                "throughput": {f"{arm}/{config}": eb.read_json(selection_dir(output, arm, config) / "report.json")
                               ["seconds_per_update"] for arm in now for config in ARMS[arm]["configurations"]
                               if (selection_dir(output, arm, config) / "report.json").is_file()},
                "projection": project, "policy_fit_redundancy": redundancy, "wall_seconds": time.monotonic() - began}
    eb.write_json(output / "smoke.json", evidence)
    log(f"smoke ok={ok}: arms {arms} (deferred selection {deferred}), missing selections {missing}, projection "
        f"{None if project is None else round(project['total_gpu_hours'], 1)} GPU-h")
    return 0 if ok else 1


def frozen_inputs(output):
    output = Path(output)
    names = {"runner": eb.ROOT / RUNNER, "smoke": output / "smoke.json",
             "port_reference": eb.ROOT / "scripts/b127_phyre_dqn_port_reference.py",
             "fidelity_runner": eb.ROOT / "scripts/b127_phyre_dqn_fidelity.py",
             "gport": output / "gport/gport.json", "gfid_band": output / "fidelity/band.json",
             "gfid": output / "fidelity/gfid.json"}
    for test in TESTS:
        names[f"tests_{Path(test).stem}"] = eb.ROOT / test
    for family in ("phyre_dqn", "dynamics_aware", "phyq"):
        base = eb.ROOT / "world_model/baselines" / family
        for rel in ("__init__.py", "novphy.py"):
            if (base / rel).is_file():
                names[f"{family}_{rel}"] = base / rel
        vendor = base / "vendor" / "VENDOR.json"
        if vendor.is_file():
            names[f"{family}_vendor_manifest"] = vendor
            for rel in sorted(eb.read_json(vendor)["files"]):
                names[f"{family}_vendor_{rel}"] = base / "vendor" / rel
    for rel in ("gport/a4/gport.json", "fidelity/a4/band.json", "gport/a3/gport.json", "a3/data/symbolic-fit.pt",
                "a3/data/symbolic-policy.pt"):
        if (output / rel).is_file():
            names[rel] = output / rel
    weights = eb.ROOT / "data/external/phyq/resnet18-5c106cde.pth"  # A3-img ImageNet init (Phy-Q pretrained=True)
    if weights.is_file():
        names["phyq_imagenet_resnet18"] = weights
    own = [{"name": n, "artifact": str(Path(p).relative_to(eb.ROOT)), "sha256": eb.sha256_of(p)}
           for n, p in names.items()]
    return eb.shared_inputs() + own


def frozen_plan(output, frozen_at):
    output = Path(output)
    evidence = eb.read_json(output / "smoke.json")
    arms = {}
    for arm, spec in ARMS.items():
        entry = {k: v for k, v in spec.items() if k != "configurations"}
        entry["configurations"] = spec["configurations"]
        if arm in evidence["arms"] and not spec["deferred_selection"]:
            config = evidence["selection"][arm]["selected"]
            run = selection_dir(output, arm, config)
            entry.update({"status": "ready", "selected_configuration": config,
                          "config": spec["configurations"][config], "selection": evidence["selection"][arm],
                          "reused_run": str(run.relative_to(eb.ROOT)), "reused_model_sha256": eb.sha256_of(run / "model.pt"),
                          "parameters": eb.read_json(run / "report.json")["parameters"],
                          "projected_gpu_hours_per_seed": evidence["projection"]["per_seed"][arm]["gpu_hours_per_seed"]})
        elif arm in evidence["arms"]:
            model = family_of(arm).build(spec)
            entry.update({"status": "selection_after_gfid", "selected_configuration": None,
                          "selection_rule": (f"after the freeze, once {GFID_SOURCE[arm]} passes: the configurations "
                                             f"above at seed {SELECTION_SEED} (--selection --arms {arm}), "
                                             "--selection-readouts, then --select-deferred applies "
                                             "eb.select_configuration mechanically (policy split only) and writes "
                                             "selection-deferred.json; the selected run is main seed 20260908"),
                          "deferred_training_gpus": {str(s): g for s, g in DEFERRED_TRAINING_GPUS[arm].items()},
                          "parameters": trainable_parameters(model),
                          "projected_gpu_hours_per_seed": evidence["projection"]["per_seed"][arm]["gpu_hours_per_seed"]})
        else:
            entry.update({"status": "adapter_missing", "selected_configuration": None})
        arms[arm] = entry
    project = evidence["projection"]
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "issue": ISSUE, "program_issue": eb.PROGRAM_ISSUE,
        "frozen_at": frozen_at, "frozen_before_training_and_evaluation": True, "runner": RUNNER,
        "validation_command": VALIDATION_COMMAND, "engine_seconds": 0, "seeds": list(SEEDS),
        "question": ("Q1: does U rank launches better than direct success predictors trained on the same fit shots "
                     "(A1 pixels, A2 the same carrier, A4 + trajectory supervision; A3 Phy-Q secondary)?"),
        "arms": arms, "score": {"key": SCORE_KEY, "rule": "primary score = success logit (Q for A3); cost = -score "
                                "(eb.score_rows); every mixed cell of every inventory from its Gate-B anchor"},
        "selection_rule": eb.SELECTION_RULE,
        "training": {"queues": project["queues"], "queue_projected_hours": project["queue_hours"],
                     "checkpoint_every": CHECKPOINT_EVERY, "batches": "batch b drawn from eb.batch_rng(seed, b)",
                     "typed_failure": "a nonfinite loss ends the run (model.pt complete False); never retried",
                     "reuse": "the selected configuration's seed-20260908 selection run is main seed 20260908"},
        "guards": {"arm": ["G_rank", "G_nonfinite", "G_trained", "G_port", "G_fid"],
                   "G_rank": "480 fit anchors x 16 grid launches per seed (eb.g_rank)",
                   "G_fid": "A1 Tier E (fidelity/gfid.json); A2 and A2-native inherit A1's; A4 Tier T "
                            "(fidelity/a4/gfid.json, failing until it exists); A3 its P-engine fidelity outcome",
                   "G_port": "gport/gport.json (A1; A2 and A2-native: the A2 head path), gport/a4/gport.json (A4)",
                   "global": ["G_isolation", "G_chronology", "G_scoring_policy", "G_records"],
                   "thresholds": eb.GUARD},
        "contrasts": {"per_arm": list(eb.CONTRAST_REFERENCES), "within_family": WITHIN_FAMILY,
                      "both_directions": True, "delta": eb.DELTA},
        "dispositions": eb.DISPOSITION_LABELS, "program_set": list(eb.PROGRAM_SET),
        "caps": {"gpu_hours": CAP_GPU_HOURS, "wall_hours_on_4_gpus": CAP_WALL_HOURS,
                 "per_seed": {a: s["cap_gpu_hours_per_seed"] for a, s in ARMS.items()},
                 "p_cap_arms": project["p_cap_arms"]},
        "deferred_scoring": [a for a, s in ARMS.items() if s["deferred_scoring"]],
        "scoring_groups": {name: list(group) for name, group in SCORING_GROUPS.items()},
        "scoring_rule": ("one readout freeze and one scoring pass per group, in order; group 'primary' after every "
                         "seed of A1, A2, A2-native and A4 is trained and read out; group 'a3' after A3's P-engine "
                         "fidelity outcome; the publication is re-derived after each pass, existing records are "
                         "never re-scored"),
        "no_model_references": {"names": NN_REFERENCES, "rule": NN_RULE, "coverage": NN_COVERAGE, "seed_free": True,
                                "scored": "in the first scoring pass, on every mixed cell of every inventory from "
                                          "the Gate-B anchors (frames / carriers of anchors-evaluation.pt) against "
                                          "the 480 fit anchors -> references/NN-px.json, references/NN-car.json",
                                "use": "DESCRIPTIVE rows beside U / C* / P (pooled AUC; X - NN-px, U - NN-px, "
                                       "X - NN-car, U - NN-car estimates without tokens); never in a disposition"},
        "disclosures": [
            "A1 / A2 action MLP width 256 (official code and shipped checkpoints) where the paper text says 512",
            "batch b of every run is drawn from PCG64([seed, b]) with the official sampler rule (resumable)",
            "A2 / A2-native G-fid is inherited from A1's pipeline check",
            "A3 (Phy-Q) is secondary, never in the program set; scored after its P-engine fidelity outcome; while "
            "its adapter is missing it is registered with status adapter_missing",
            "the evaluation split was scored by earlier tickets for our models; it is pristine for these arms",
            "fit / policy layout redundancy (#109 cohort: two generator templates; near-duplicate fit anchors): "
            "measured in smoke.json policy_fit_redundancy; it inflates policy-split selection AUCs for every arm "
            "alike; the NN-px / NN-car references measure the same effect on the evaluation split",
            "frame 0 of a level's branches differs by capture noise (smoke.json accounting.anchor_frame_pre_launch); "
            "training and scoring use the anchor branch's frame 0 only"],
        "inputs": frozen_inputs(output),
        "smoke_evidence": {k: evidence[k] for k in ("ok", "at", "accounting", "gport", "tests", "projection",
                                                    "policy_fit_redundancy")},
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = eb.read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY or not plan["smoke_evidence"]["ok"]:
        raise ValueError("plan.json is not the issue-128 protocol with a passed smoke")
    eb.verify_pins(plan["inputs"])
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    if (output / "models").exists() or (output / "records").exists():
        raise ValueError("main-arm models or records exist before the freeze; refusing")
    evidence = eb.read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed or projects over the cap; STOP RULE: no freeze")
    plan = frozen_plan(output, eb.utc_now())
    (output / "plan.json").write_text(eb.json_text(plan))
    load_plan(output)
    log(f"frozen plan published: queues {plan['training']['queue_projected_hours']}")
    return 0


def train_names(output, plan, gpu):
    """This GPU's frozen queue, then (deferred-selection arms, once selection-deferred.json selected a
    configuration) the arm's main seeds assigned to this GPU in the plan."""
    names = list(plan["training"]["queues"][gpu])
    for arm, spec in plan["arms"].items():
        if spec["status"] == "selection_after_gfid" and arm_selection(output, plan, arm)[0] is not None:
            names += [f"{arm}/{seed}" for seed, g in spec["deferred_training_gpus"].items() if g == gpu]
    return names


def train(output, gpu):
    plan = load_plan(output)
    context = Context(device_name())
    for name in train_names(output, plan, gpu):
        if eb.STOP_REQUESTED:
            return eb.EXIT_PAUSED
        arm, seed = name.split("/")
        seed = int(seed)
        config = arm_selection(output, plan, arm)[1]
        if config is None:  # #128 A2: a repaired arm before its re-selection
            log(f"queue {gpu}: {name} skipped (no configuration selected yet)")
            continue
        run = main_dir(output, arm, seed)
        finished_before = (run / "model.pt").is_file()
        try:
            report = train_run(arm, config, seed, run, context)
        except eb.TrainingPaused as paused:
            log(f"queue {gpu}: {paused}")
            return eb.EXIT_PAUSED
        if not report.get("partial") and not finished_before:
            eb.mark(output, f"train-{arm}-seed{seed}", report["wall_seconds"], report["gpu_seconds"])
    return 0


def select_deferred(output):
    """Post-freeze, mechanical: eb.select_configuration over a deferred-selection (or repair re-selection) arm's
    selection readouts (policy split only), written once to selection-deferred.json."""
    import fcntl
    plan = load_plan(output)
    path = deferred_selection_path(output)
    lock = open(Path(output) / ".selection-deferred.lock", "w")  # #128 A2: the A4 and A3-sym jobs may overlap
    fcntl.flock(lock, fcntl.LOCK_EX)
    done = eb.read_json(path)["arms"] if path.is_file() else {}
    for arm, spec in plan["arms"].items():
        if (spec["status"] != "selection_after_gfid" and arm not in REPAIR_RESELECTION) or arm in done:
            continue
        result = selection_result(output, [arm])[arm]
        if not result["complete"]:
            log(f"{arm}: selection readouts incomplete; nothing selected")
            continue
        run = selection_dir(output, arm, result["selected"])
        done[arm] = {**result, "gfid_pass": gfid_pass(output, arm), "selected_at": eb.utc_now(),
                     "reused_model_sha256": eb.sha256_of(run / "model.pt")}
        if arm in REPAIR_RESELECTION:
            done[arm].update({"repair": REPAIR_RESELECTION[arm],
                              "frozen_selected_configuration": spec["selected_configuration"]})
        log(f"{arm}: selected {result['selected']}")
    eb.write_json(path, {"schema": "issue_128_selection_deferred_v1", "plan_identity": IDENTITY, "arms": done})
    lock.close()
    return 0


def ready_plan_arms(output, plan):
    """Arms whose configuration is fixed (frozen, or selected post-freeze)."""
    return [a for a, s in plan["arms"].items() if s["status"] != "adapter_missing"
            and arm_selection(output, plan, a)[0] is not None]


def readouts(output, gpu, workers, arms=None):
    plan = load_plan(output)
    context = Context(device_name())
    names = [(arm, seed) for arm in ready_plan_arms(output, plan) if arms is None or arm in arms for seed in SEEDS]
    for i, (arm, seed) in enumerate(names):
        if i % workers != gpu:
            continue
        run = model_dir(output, plan, arm, seed)
        if (run / "readouts.json").is_file() or not (run / "model.pt").is_file():
            continue
        if eb.STOP_REQUESTED:
            return eb.EXIT_PAUSED
        t0 = time.monotonic()
        readout_run(arm, run, context)
        eb.mark(output, f"readouts-{arm}-seed{seed}", time.monotonic() - t0, time.monotonic() - t0)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# readout freeze (groups) and scoring
# ---------------------------------------------------------------------------------------------------------------

def freeze_path(output):
    return Path(output) / "readouts-manifest.json"


def frozen_groups(output):
    path = freeze_path(output)
    return eb.read_json(path)["groups"] if path.is_file() else []


def frozen_entry(output, arm, seed):
    for group in frozen_groups(output):
        if f"{arm}/{seed}" in group["entries"]:
            return group, group["entries"][f"{arm}/{seed}"]
    return None, None


# Post-freeze amendment #128 A1 (Decision P-other 2026-10-27, issuecomment-6426445961): an arm of a scoring group
# dropped with a typed reason under a posted decision is recorded as dropped in its group's freeze entry (no models,
# readouts or records); the rest of the group freezes and scores as frozen. Markers are written by the orchestrator.
SCHEMA_DROPPED = "issue_128_dropped_v1"
DROP_REASONS = ("fidelity_not_reproduced", "nonfinite_loss")  # P-fid drop; the runner's typed terminal failure
DROP_DECISION_PREFIX = "https://github.com/Sino-Huang/NovPhy/issues/128#issuecomment-"


def drop_marker_path(output, arm):
    return Path(output) / "dropped" / f"{arm}.json"


def drop_marker(output, arm):
    """The arm's validated drop marker ({arm, typed_reason, decision, written_at}), or None if there is none."""
    path = drop_marker_path(output, arm)
    if not path.is_file():
        return None
    marker = eb.read_json(path)
    if (marker.get("schema") != SCHEMA_DROPPED or marker.get("arm") != arm
            or marker.get("typed_reason") not in DROP_REASONS
            or not str(marker.get("decision", "")).startswith(DROP_DECISION_PREFIX) or not marker.get("written_at")):
        raise ValueError(f"{path}: not a valid {SCHEMA_DROPPED} drop marker")
    return marker


def dropped_arms(output):
    """{arm: {typed_reason, decision, marker, marker_sha256}} recorded in the readout-freeze groups."""
    return {arm: info for group in frozen_groups(output) for arm, info in group.get("dropped", {}).items()}


def freeze_readouts(output, arms=None):
    """Append a freeze group: sha256 of every model.pt and readouts.json of one scoring group (default: the first
    group not yet frozen); a group is frozen as a whole, never twice, and only when every arm-seed has readouts or
    the arm has a valid drop marker (recorded as dropped: reason, decision, marker sha256; no models / readouts)."""
    plan = load_plan(output)
    done = {name.split("/")[0] for group in frozen_groups(output) for name in group["entries"]} | set(
        dropped_arms(output))
    groups = [list(g) for g in plan["scoring_groups"].values()]
    if arms is None:
        arms = next((g for g in groups if not set(g) & done), None)
    if arms is None or sorted(arms) not in [sorted(g) for g in groups] or set(arms) & done:
        raise ValueError(f"freeze one whole scoring group not yet frozen ({plan['scoring_groups']}); got {arms}")
    entries, dropped = {}, {}
    for arm in arms:
        marker = drop_marker(output, arm)
        if marker is not None:
            path = drop_marker_path(output, arm)
            dropped[arm] = {"typed_reason": marker["typed_reason"], "decision": marker["decision"],
                            "marker": str(path.relative_to(eb.ROOT)), "marker_sha256": eb.sha256_of(path)}
            continue
        for seed in SEEDS:
            run = model_dir(output, plan, arm, seed)
            report = eb.read_json(run / "readouts.json")
            if report["model_sha256"] != eb.sha256_of(run / "model.pt"):
                raise ValueError(f"{arm}/{seed}: readouts.json does not bind the current model.pt")
            entries[f"{arm}/{seed}"] = {"model": str((run / "model.pt").relative_to(eb.ROOT)),
                                        "model_sha256": eb.sha256_of(run / "model.pt"),
                                        "readouts": str((run / "readouts.json").relative_to(eb.ROOT)),
                                        "readouts_sha256": eb.sha256_of(run / "readouts.json"),
                                        "typed_failure": report["typed_failure"]}
    group = {"arms": arms, "frozen_at": eb.utc_now(), "entries": entries}
    if dropped:
        group["dropped"] = dropped
    groups = frozen_groups(output) + [group]
    eb.write_json(freeze_path(output), {"schema": SCHEMA_FREEZE, "plan_identity": IDENTITY, "groups": groups})
    log(f"readouts frozen: {arms}" + (f" (dropped: {sorted(dropped)})" if dropped else ""))
    return 0


def symbolic_path(output):
    return Path(output) / "data" / "anchors-evaluation-symbolic.pt"


def symbolic_evaluation_anchors(output, anchors):
    """A3-sym's evaluation input (--score only, after the freeze and eb.read_evaluation_anchors): the 12-channel
    symbolic state of each Gate-B anchor attempt (pq.symbolic_anchors), cached with its read_at."""
    path = symbolic_path(output)
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)
    from world_model.baselines.phyq import novphy as pq
    states = pq.symbolic_anchors({m: eb.rc.eval_attempt(k) for m, k in anchors["keys"].items()})
    payload = {"schema": "issue_128_eval_symbolic_anchors_v1", "states": states, "keys": dict(anchors["keys"]),
               "read_at": eb.utc_now()}
    eb.atomic_torch(path, payload)
    return payload


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


@torch.no_grad()
def score_arm_seed(output, plan, arm, seed, cells, anchors, device):
    """Every mixed cell of every inventory from its Gate-B anchor; wall per decision on the grid cells."""
    _, entry = frozen_entry(output, arm, seed)
    if entry is None:
        raise ValueError(f"{arm}/{seed} is not in a readout-freeze group")
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
            "config": plan["arms"][arm]["selected_configuration"], "engine_seconds": 0}
    if entry["typed_failure"] is not None:
        return {**base, "typed_failure": entry["typed_failure"], "inventories": {}, "wall_per_decision": None}
    path = eb.ROOT / entry["model"]
    if eb.sha256_of(path) != entry["model_sha256"]:
        raise ValueError(f"{arm}/{seed}: model differs from the frozen readout manifest")
    model, _ = load_model(arm, path, device)
    kind = ARMS[arm]["input"]
    observations = {"frame": anchors["frames"], "carrier": anchors["carriers"]}.get(kind) or anchors.get(kind)
    if observations is None:
        raise ValueError(f"{arm}: no evaluation anchors of kind {kind}")
    inventories = {}
    for inventory in eb.INVENTORIES:
        inventories[inventory] = {
            member: {"rows": eb.score_rows(cell["inventory"],
                                           scores_for(arm, model, observations[member], cell["inventory"], device))}
            for member, cell in sorted(cells[inventory]["mixed"].items())}
    grid = sorted(cells["grid"]["mixed"].items())
    wall = eb.wall_per_decision(lambda mc: scores_for(arm, model, observations[mc[0]], mc[1]["inventory"], device),
                                grid)
    return {**base, "typed_failure": None, "model_sha256": entry["model_sha256"], "inventories": inventories,
            "wall_per_decision": wall}


def nn_reference_path(output, name):
    return Path(output) / "references" / f"{name}.json"


def score_nn_references(output, cells, anchors):
    """NN-px / NN-car on every mixed cell of every inventory from the Gate-B anchors (frames / carriers), written
    once in the first scoring pass (existing files are never rewritten); no model, seed-free."""
    names = [n for n in NN_REFERENCES if not nn_reference_path(output, n).is_file()]
    if not names:
        return
    fit = eb.FitShots()
    orders = nn_orders(anchors["frames"], anchors["carriers"])
    by_inventory = {inv: cells[inv]["mixed"] for inv in eb.INVENTORIES}
    for name in names:
        record = nn_record(name, by_inventory, orders[name], fit_verdicts_by_launch(fit))
        record["scored_at"] = eb.utc_now()
        eb.write_json(nn_reference_path(output, name), record)
        log(f"no-model reference {name} scored")


def nn_views(output):
    """{name: ({seed: record}, key)} of the scored NN references (seed-free: one record for every seed)."""
    views = {}
    for name in NN_REFERENCES:
        path = nn_reference_path(output, name)
        if path.is_file():
            record = eb.read_json(path)
            if record.get("schema") != SCHEMA_NN_REFERENCE or record.get("plan_identity") != IDENTITY:
                raise ValueError(f"{path}: binding differs")
            views[name] = ({seed: record for seed in SEEDS}, SCORE_KEY)
    return views


def score(output, arms=None):
    """Score one frozen scoring group once (default: the first frozen group with a missing record; resumable per
    record) in a process that did not train."""
    from world_model.planning import scoring_harness as harness
    output = Path(output)
    plan = load_plan(output)
    groups = [g["arms"] for g in frozen_groups(output)]
    dropped = dropped_arms(output)
    if arms is None:
        arms = next((g for g in groups
                     if not all(record_path(output, a, s).is_file() for a in g if a not in dropped
                                for s in SEEDS)), None)
    if arms is None or sorted(arms) not in [sorted(g) for g in groups]:
        raise ValueError(f"score one whole frozen scoring group ({groups}); got {arms}")
    if dropped:
        log(f"not scored (dropped): {sorted(a for a in arms if a in dropped)}")
        arms = [a for a in arms if a not in dropped]
    device = device_name()
    cells = eb.evaluation_cells()
    with harness.deterministic_scoring() as policy:
        path = eb.evaluation_anchors_path(output)
        anchors = (torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
                   else eb.read_evaluation_anchors(output, policy))
        if any(ARMS[a]["input"] == "symbolic" for a in arms):
            anchors = {**anchors, "symbolic": symbolic_evaluation_anchors(output, anchors)["states"]}
        score_nn_references(output, cells, anchors)
        for arm in arms:
            for seed in SEEDS:
                out = record_path(output, arm, seed)
                if out.is_file():
                    continue
                t0 = time.monotonic()
                record = score_arm_seed(output, plan, arm, seed, cells, anchors, device)
                record.update({"scoring_policy": policy, "scored_at": eb.utc_now()})
                eb.write_json(out, record)
                eb.mark(output, f"score-{arm}-seed{seed}", time.monotonic() - t0,
                        time.monotonic() - t0 if device != "cpu" else 0.0)
                log(f"scored {arm} seed {seed} ({time.monotonic() - t0:.0f}s)")
    return 0


# ---------------------------------------------------------------------------------------------------------------
# publication (eb.arm_block / eb.child_summary / eb.format_arm_rows) and validation
# ---------------------------------------------------------------------------------------------------------------

RECORD_PATTERNS = ("records/*--seed*.json", "references/*.json", "novelty/records/*--seed*.json",
                   "novelty/references/*.json")


def arm_guards(output, plan, arm, arm_records):
    """(guards, extra guards, G-fid entry) of one scored arm: the computation the publication's arm block uses
    (eb.arm_guard_block over G-rank / G-nonfinite / G-trained; G-port and G-fid from gport / gfid files)."""
    per_seed = {}
    for seed in SEEDS:
        report = eb.read_json(model_dir(output, plan, arm, seed) / "readouts.json")
        per_seed[seed] = {"G_rank": report["G_rank"], "typed_failure": arm_records[seed]["typed_failure"]}
    nonfinite = eb.nonfinite_share(arm_records.values(), SCORE_KEY)
    guards = eb.arm_guard_block(per_seed, ["G_rank", "G_nonfinite", "G_trained"], nonfinite)
    gfid = {"pass": gfid_pass(output, arm) is True, "source": GFID_SOURCE[arm],
            "inherited_from": "A1" if arm in ("A2", "A2-native") else None}
    extra = {"G_port": gport_pass(output, arm) is True, "G_fid": gfid["pass"]}
    return guards, extra, gfid


def failed_guards(guards, extra):
    """The G_* entries that are not True (eb.arm_block's arm-pass rule: the arm passes iff this is empty)."""
    merged = {**guards, **extra}
    return [k for k, v in merged.items() if k.startswith("G_") and v is not True]


# ---------------------------------------------------------------------------------------------------------------
# P-novelty (DESCRIPTIVE; Decision P-novelty on #128, issuecomment-6386830623): after an arm's primary pass, its
# guard-passing models score the 16 grid launches of every mixed #110 novelty level from that level's anchor
# ---------------------------------------------------------------------------------------------------------------

SCHEMA_NOVELTY_RECORD = "issue_128_novelty_record_v1"


def novelty_dir(output):
    return Path(output) / "novelty"


def novelty_record_path(output, arm, seed):
    return novelty_dir(output) / "records" / f"{arm}--seed{seed}.json"


def novelty_observations(arm, anchors):
    """{level: observation} of the arm's input kind: frames / carriers of anchors-novelty.pt, or (A3-sym) the
    symbolic state of each anchor attempt's decision-physics-v1.json (anchors["physics_paths"])."""
    kind = ARMS[arm]["input"]
    if kind == "frame":
        return anchors["frames"]
    if kind == "carrier":
        return anchors["carriers"]
    from world_model.baselines.phyq import novphy as pq
    return {level: pq.symbolic_state(Path(path).parent) for level, path in sorted(anchors["physics_paths"].items())}


@torch.no_grad()
def score_novelty_arm_seed(output, arm, seed, cells, observations, device):
    """The 16 grid launches of every mixed novelty level from its anchor, exactly as score_arm_seed scores a Gate-B
    cell (the frozen model, eb.score_rows of scores_for)."""
    _, entry = frozen_entry(output, arm, seed)
    path = eb.ROOT / entry["model"]
    if eb.sha256_of(path) != entry["model_sha256"]:
        raise ValueError(f"{arm}/{seed}: model differs from the frozen readout manifest")
    model, _ = load_model(arm, path, device)
    grid = {}
    for level in cells["mixed"]:
        items = cells["levels"][level]["inventory"]
        grid[level] = {"rows": eb.score_rows(items, scores_for(arm, model, observations[level], items, device))}
    return {"schema": SCHEMA_NOVELTY_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed,
            "model_sha256": entry["model_sha256"], "inventories": {"grid": grid}, "engine_seconds": 0}


def score_novelty(output, arms=None):
    """--score-novelty (GPU 0, once, resumable per record): refuses before the readout freeze and before the arms'
    primary records exist; never re-scores a primary record. Default arms: every arm with complete primary records
    and no novelty pass yet (A3 joins after its own primary pass). Arms failing a guard are listed, not scored."""
    from world_model.planning import scoring_harness as harness
    output = Path(output)
    plan = load_plan(output)
    if not freeze_path(output).is_file():
        raise ValueError("readouts-manifest.json missing: --score-novelty runs after the model freeze")
    scored = scored_arms(output, plan)
    passes_path = novelty_dir(output) / "passes.json"
    passes = eb.read_json(passes_path)["passes"] if passes_path.is_file() else []
    done = {a for p in passes for a in p["arms"]}
    dropped = dropped_arms(output)
    if arms and set(arms) & set(dropped):
        log(f"novelty: not scored (dropped): {sorted(set(arms) & set(dropped))}")
        arms = [a for a in arms if a not in dropped]
    arms = arms or [a for a in scored if a not in done]
    missing = [a for a in arms if a not in scored]
    if missing or not arms:
        raise ValueError(f"--score-novelty needs the primary records of {missing or arms} (run --score first)")
    records = load_records(output, arms)
    eligible, skipped = [], {}
    for arm in arms:
        guards, extra, _ = arm_guards(output, plan, arm, records[arm])
        failed = failed_guards(guards, extra)
        if failed:
            skipped[arm] = {"reason": "guard_failed", "failed_guards": failed}
            log(f"novelty: {arm} not scored (failed guards {failed})")
        else:
            eligible.append(arm)
    if not any(set(p["arms"]) == set(arms) for p in passes):
        last = max(r["scored_at"] for a in arms for r in records[a].values())
        passes.append({"arms": arms, "eligible": eligible, "skipped": skipped, "primary_last_scored_at": last,
                       "started_at": eb.utc_now()})
        eb.write_json(passes_path, {"schema": "issue_128_novelty_passes_v1", "plan_identity": IDENTITY,
                                    "passes": passes})
    device = device_name()
    with harness.deterministic_scoring() as policy:
        cells = eb.novelty_cells()
        path = eb.novelty_anchors_path(output)
        anchors = (torch.load(path, map_location="cpu", weights_only=True) if path.is_file()
                   else eb.read_novelty_anchors(output, policy, cells))
        eb.novelty_reference_records(output, policy, cells, anchors["carriers"])
        for arm in eligible:
            observations = novelty_observations(arm, anchors)
            for seed in SEEDS:
                out = novelty_record_path(output, arm, seed)
                if out.is_file():
                    continue
                t0 = time.monotonic()
                record = score_novelty_arm_seed(output, arm, seed, cells, observations, device)
                record.update({"scoring_policy": policy, "scored_at": eb.utc_now()})
                eb.write_json(out, record)
                eb.mark(output, f"score-novelty-{arm}-seed{seed}", time.monotonic() - t0,
                        time.monotonic() - t0 if device != "cpu" else 0.0)
                log(f"novelty scored {arm} seed {seed} ({time.monotonic() - t0:.0f}s)")
    return 0


def novelty_section(output, plan, records):
    """DESCRIPTIVE novelty block (no tokens, never in a disposition) when novelty records exist: eb.novelty_table of
    every novelty-scored arm plus U / C* / P, the skipped arms, and the pass chronology."""
    output = Path(output)
    passes_path = novelty_dir(output) / "passes.json"
    if not passes_path.is_file():
        return None
    passes = eb.read_json(passes_path)["passes"]
    arms = [a for p in passes for a in p["eligible"]
            if all(novelty_record_path(output, a, s).is_file() for s in SEEDS)]
    if not arms:
        return None
    from world_model.planning import scoring_harness as harness
    cells = eb.novelty_cells()
    anchors = torch.load(eb.novelty_anchors_path(output), map_location="cpu", weights_only=True)
    refs = eb.novelty_reference_records(output, None, cells, anchors["carriers"])  # read back, never rewritten
    views = {}
    chronology = {"anchors_read_at": anchors["read_at"],
                  "first_pass_primary_last_scored_at": passes[0]["primary_last_scored_at"], "arms": {}}
    ok = anchors["read_at"] >= passes[0]["primary_last_scored_at"]
    for arm in arms:
        by_seed = {}
        for seed in SEEDS:
            record = eb.read_json(novelty_record_path(output, arm, seed))
            frozen = frozen_entry(output, arm, seed)[1]
            if (record.get("schema") != SCHEMA_NOVELTY_RECORD or record.get("plan_identity") != IDENTITY
                    or record["model_sha256"] != frozen["model_sha256"]):
                raise ValueError(f"novelty record {arm}/{seed} binding differs")
            by_seed[seed] = record
        views[arm] = (by_seed, SCORE_KEY)
        primary_last = max(r["scored_at"] for r in records[arm].values())
        first = min(r["scored_at"] for r in by_seed.values())
        arm_ok = bool(primary_last <= first and anchors["read_at"] <= first)
        chronology["arms"][arm] = {"primary_last_scored_at": primary_last, "novelty_first_scored_at": first,
                                   "pass": arm_ok}
        ok = ok and arm_ok
    chronology["pass"] = bool(ok)
    return {"label": "DESCRIPTIVE (P-novelty; no tokens, never in a disposition)",
            "table": eb.novelty_table({**views, **refs}, cells),
            "skipped": {a: s for p in passes for a, s in p["skipped"].items()},
            "scoring_policy_pass": all(r.get("scoring_policy") == harness.SCORING_POLICY
                                       for by_seed, _ in views.values() for r in by_seed.values()),
            "chronology": chronology}


PUBLISHED = ("compute.json", "summary.json", "findings.md")


def scored_arms(output, plan):
    return [a for a in plan["arms"] if all(record_path(output, a, s).is_file() for s in SEEDS)]


def load_records(output, arms):
    records = {}
    for arm in arms:
        records[arm] = {}
        for seed in SEEDS:
            record = eb.read_json(record_path(output, arm, seed))
            if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY:
                raise ValueError(f"record {arm}/{seed} binding differs")
            records[arm][seed] = record
    return records


def global_guards(output, plan, records):
    from world_model.planning import scoring_harness as harness
    roster = {level["identity"] for level in eb.gb.evaluation_levels()}
    flat = [r for by_seed in records.values() for r in by_seed.values()]
    scored = [r for r in flat if not r["typed_failure"]]
    isolation = (all(set(r["inventories"]["grid"]) <= roster for r in scored)
                 and all(k.startswith("issue-109-sc-fit") for k in eb.FitShots().keys))
    anchors = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)
    chronology, detail = True, {}
    groups = frozen_groups(output)
    first_group = min(g["frozen_at"] for g in groups)
    for arm, by_seed in records.items():
        group, _ = frozen_entry(output, arm, SEEDS[0])
        first = min(r["scored_at"] for r in by_seed.values())
        ok = plan["frozen_at"] <= group["frozen_at"] <= first and anchors["read_at"] <= first
        if ARMS[arm]["input"] == "symbolic":
            symbolic = torch.load(symbolic_path(output), map_location="cpu", weights_only=True)["read_at"]
            ok = ok and group["frozen_at"] <= symbolic <= first
        detail[arm] = {"readouts_frozen_at": group["frozen_at"], "first_record_scored_at": first, "pass": bool(ok)}
        chronology &= ok
    chronology &= first_group <= anchors["read_at"]
    return {"G_isolation": {"pass": bool(isolation)},
            "G_chronology": {"pass": bool(chronology), "plan_frozen_at": plan["frozen_at"],
                             "first_readouts_frozen_at": first_group, "anchors_read_at": anchors["read_at"],
                             "arms": detail},
            "G_scoring_policy": {"pass": all(r.get("scoring_policy") == harness.SCORING_POLICY for r in flat)},
            "G_records": eb.records_guard(output, IDENTITY)}


def arm_compute(output, plan, arm, records):
    seconds, macs = {}, []
    for seed in SEEDS:
        run = model_dir(output, plan, arm, seed)
        seconds[str(seed)] = eb.read_json(run / "report.json")["gpu_seconds"]
        report = eb.read_json(run / "readouts.json")
        if report.get("macs_per_candidate") is not None:
            macs.append(report["macs_per_candidate"])
    walls = [r["wall_per_decision"] for r in records.values() if r.get("wall_per_decision") is not None]
    return {"trainable_parameters": plan["arms"][arm]["parameters"], "training_gpu_seconds": seconds,
            "macs_per_candidate": float(np.mean(macs)) if macs else None,
            "wall_per_decision": float(np.median(walls)) if walls else None}


def compute_tables(output, plan):
    output = Path(output)
    arms = scored_arms(output, plan)
    records = load_records(output, arms)
    guards_global = global_guards(output, plan, records)
    global_ok = all(g["pass"] is True for g in guards_global.values())
    cells = eb.evaluation_cells()
    references = eb.references()
    nn = nn_views(output)
    arm_views = {**{arm: (records[arm], SCORE_KEY) for arm in arms}, **nn}
    rows = {inventory: eb.view_rows(cells, inventory, arm_views, references) for inventory in eb.INVENTORIES}
    dropped = dropped_arms(output)
    if dropped:  # #128 A1: dropped arms' stand-in records (typed reason, no rows) as a view of their own
        stand_in = {arm: dropped_records(arm, info) for arm, info in dropped.items()}
        drop_rows = eb.view_rows(cells, "grid", {**arm_views, **{a: (r, SCORE_KEY) for a, r in stand_in.items()}},
                                 references)
    blocks, gfid = {}, {}
    for arm in arms:
        guards, extra, gfid[arm] = arm_guards(output, plan, arm, records[arm])
        blocks[arm] = eb.arm_block(rows["grid"], arm, records[arm], SCORE_KEY, guards, global_ok,
                                   arm_compute(output, plan, arm, records[arm]), ARMS[arm]["role"],
                                   ARMS[arm]["program_set"], extra_guards=extra)
    for arm, info in dropped.items():
        guards, extra, gfid[arm] = dropped_guards(output, plan, arm, info, stand_in[arm])
        blocks[arm] = eb.arm_block(drop_rows, arm, stand_in[arm], SCORE_KEY, guards, global_ok,
                                   {"trainable_parameters": plan["arms"][arm].get("parameters"),
                                    "training_gpu_seconds": {}, "macs_per_candidate": None, "wall_per_decision": None},
                                   ARMS[arm]["role"], ARMS[arm]["program_set"], extra_guards=extra,
                                   dropped=info["typed_reason"])
    within = {}
    for label, (first, second) in WITHIN_FAMILY.items():
        if first in blocks and second in blocks:
            guarded = global_ok and blocks[first]["guards"]["pass"] and blocks[second]["guards"]["pass"]
            within[label] = eb.within_family(drop_rows if dropped else rows["grid"], first, second, guarded)
    descriptive = {inventory: eb.ranker_table(rows[inventory], ["U", "C*", "P", "J", "J_dec", *nn, *arms])
                   for inventory in eb.INVENTORIES}
    nn_block = {"label": "DESCRIPTIVE no-model references (never in a disposition; estimates without tokens)",
                "rule": NN_RULE, "pooled_grid_auc": {n: descriptive["grid"]["rankers"][n]["auc"] for n in nn},
                "estimates": {f"{x} - {n}": {k: v for k, v in eb.contrast(rows["grid"], x, n, False).items()
                                            if k not in ("token", "reverse_token", "guards_pass")}
                              for n in nn for x in (*arms, "U")}}
    entries, gpu, wall = eb.ledger_totals(output, lambda phase: ledger_report(output, plan, phase))
    pending = [a for a in plan["arms"] if a not in arms and a not in dropped]
    extra_summary = {"selection": {a: plan["arms"][a].get("selection") for a in arms},
                     "selection_deferred": (eb.read_json(deferred_selection_path(output))["arms"]
                                            if deferred_selection_path(output).is_file() else {}),
                     "pending_arms": {a: plan["arms"][a]["status"] for a in pending}}
    if dropped:
        extra_summary["dropped"] = dropped
    engine = engine_block(output)
    summary = eb.child_summary(ISSUE, IDENTITY, plan["frozen_at"], blocks, within, guards_global, gfid,
                               extra_summary, **({"engine_seconds": engine["engine_seconds"]} if engine else {}))
    compute = {"schema": "issue_128_compute_v1", "identity": IDENTITY, "frozen_at": plan["frozen_at"],
               "summary": summary, "inventories": descriptive,
               "ledger": {"phases": entries, "gpu_seconds": gpu, "wall_seconds": wall,
                          "fidelity_gpu_hours": fidelity_gpu_hours(output),
                          "rule": "each arm-seed's training counted once from its report.json (#123 A3)"},
               "primary_cost_key": SCORE_KEY, "no_model_references": nn_block}
    if engine is not None:
        compute["engine"] = engine
    novelty = novelty_section(output, plan, records)
    if novelty is not None:
        compute["novelty"] = novelty
    return compute


def dropped_records(arm, info):
    """A dropped arm's stand-in records: its typed reason on every seed (no rows; view_rows yields no estimate)."""
    return {seed: {"arm": arm, "seed": seed, "typed_failure": info["typed_reason"]} for seed in SEEDS}


def dropped_guards(output, plan, arm, info, stand_in):
    """A dropped arm's guards: every seed a typed failure (G-rank / G-nonfinite / G-trained fail), G_fid False; the
    drop marker must still be the bytes recorded at the readout freeze."""
    if eb.sha256_of(eb.ROOT / info["marker"]) != info["marker_sha256"]:
        raise ValueError(f"{arm}: drop marker differs from the one recorded at the readout freeze")
    guards = eb.arm_guard_block({seed: {"typed_failure": r["typed_failure"]} for seed, r in stand_in.items()},
                                ["G_rank", "G_nonfinite", "G_trained"], None)
    gfid = {"pass": False, "source": GFID_SOURCE[arm], "inherited_from": None, "dropped": info["typed_reason"],
            "decision": info["decision"]}
    return guards, {"G_port": gport_pass(output, arm) is True, "G_fid": False}, gfid


# Post-freeze amendment #128 A3: engine time (game-server worker-seconds; A3's native-task fidelity) from
# ledger/engine-*.json into summary.json engine_seconds and compute.json; absent files -> outputs as before.
ENGINE_ATTEMPT_ARMS = {"sym": "A3-sym", "img": "A3-img"}  # b127_phyq_fidelity attempt ids "<variant>-attempt-<n>"


def engine_block(output):
    """{"engine_seconds", "per_arm", "unattributed_engine_seconds", "files"} summed over ledger/engine-*.json
    (attempts attributed to an arm by their variant prefix), or None when no engine ledger file exists."""
    paths = sorted((Path(output) / "ledger").glob("engine-*.json"))
    if not paths:
        return None
    total, unattributed, per_arm, files = 0.0, 0.0, {}, []
    for path in paths:
        block = eb.read_json(path)
        seconds = float(block.get("engine_seconds", 0.0))
        total += seconds
        attributed = 0.0
        for attempt, entry in sorted(block.get("attempts", {}).items()):
            arm = ENGINE_ATTEMPT_ARMS.get(attempt.split("-", 1)[0])
            if arm is not None:
                per_arm[arm] = per_arm.get(arm, 0.0) + float(entry["engine_seconds"])
                attributed += float(entry["engine_seconds"])
        unattributed += seconds - attributed
        files.append({"artifact": str(path.relative_to(eb.ROOT)), "engine_seconds": seconds})
    return {"engine_seconds": total, "engine_hours": total / 3600.0, "per_arm": per_arm,
            "unattributed_engine_seconds": unattributed, "files": files,
            "rule": "engine worker-seconds (game-server lifetimes, CPU) of ledger/engine-*.json; not GPU time"}


def ledger_report(output, plan, phase):
    if phase.startswith("train-"):
        arm, seed = phase[len("train-"):].rsplit("-seed", 1)
        return main_dir(output, arm, int(seed)) / "report.json"
    return None


def findings_md(plan, compute):
    summary = compute["summary"]
    lines = ["# Issue #128 findings: direct success predictors (A1, A2, A2-native, A4; A3 when scored) vs U", "",
             f"Plan frozen {plan['frozen_at']}. Score = success logit (Q for A3); cost = -score. Intervals: "
             "member-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE. delta = 0.02.", "",
             "## Global guards", "", "| guard | pass |", "| --- | --- |"]
    lines += [f"| {name} | {g['pass']} |" for name, g in summary["global_guards"].items()]
    lines += ["", "## Arms (grid, 311 mixed cells x 3 seeds)", "", *eb.format_arm_rows(summary["arms"]), "",
              "## Within-family contrasts", "", "| contrast | estimate | token | reverse token | guards |",
              "| --- | --- | --- | --- | --- |"]
    for label, c in summary["within_family"].items():
        lines.append(f"| {c['label']} | {eb.rc.fmt_block(c['estimate'])} | {c['token']} | {c['reverse_token']} | "
                     f"{c['guards_pass']} |")
    lines += ["", "## Secondary contrasts per arm", "", "| arm | contrast | estimate | token | reverse token |",
              "| --- | --- | --- | --- | --- |"]
    for arm, block in summary["arms"].items():
        for label, c in block["contrasts"].items():
            lines.append(f"| {arm} | {label} | {eb.rc.fmt_block(c['estimate'])} | {c['token']} | {c['reverse_token']} |")
    lines += ["", "## Compute", "", "| arm | params | training GPU-s per seed | MACs / candidate | wall / decision (s) |",
              "| --- | --- | --- | --- | --- |"]
    for arm, block in summary["arms"].items():
        c = block["compute"]
        per = ", ".join(f"{s}: {v:.0f}" for s, v in c["training_gpu_seconds"].items())
        params = "n/a" if c["trainable_parameters"] is None else f"{c['trainable_parameters']:,}"
        lines.append(f"| {arm} | {params} | {per} | {eb.rc.fmt_num(c['macs_per_candidate'], 0)} | "
                     f"{eb.rc.fmt_num(c['wall_per_decision'], 4)} |")
    pending = summary["extra"]["pending_arms"]
    if pending:
        lines += ["", "Not yet scored: " + ", ".join(f"{a} ({s})" for a, s in pending.items())
                  + " (A3 is added by a later scoring pass; nothing above is re-scored)."]
    dropped = summary["extra"].get("dropped")
    if dropped:
        lines += ["", "Dropped (no records; every token the stop token; disposition 4; vetoes the program claim): "
                  + ", ".join(f"{a} ({d['typed_reason']}; {d['decision']})" for a, d in dropped.items()) + "."]
    engine = compute.get("engine")
    if engine is not None:
        lines += ["", f"Engine time (CPU game-server worker-hours; A3 native-task fidelity): {engine['engine_hours']:.2f} h"
                  + "".join(f"; {a} {v / 3600.0:.2f} h" for a, v in sorted(engine["per_arm"].items())) + "."]
    nn = compute["no_model_references"]
    if nn["pooled_grid_auc"]:
        lines += ["", "## No-model nearest-fit references (DESCRIPTIVE, never in a disposition, no tokens)", "",
                  f"Rule: {nn['rule']}. {NN_COVERAGE}.", "", "| reference | pooled grid AUC [CI] |", "| --- | --- |"]
        lines += [f"| {n} | {eb.rc.fmt_auc(a)} |" for n, a in nn["pooled_grid_auc"].items()]
        lines += ["", "| contrast | estimate |", "| --- | --- |"]
        lines += [f"| {label} | {eb.rc.fmt_block(c['estimate'])} |" for label, c in nn["estimates"].items()]
    novelty = compute.get("novelty")
    if novelty is not None:
        lines += ["", "## Novelty (P-novelty, DESCRIPTIVE: no tokens, never part of a disposition)", "",
                  f"#110 novelty levels of #107's in-scope cells, mixed levels only; chronology pass "
                  f"{novelty['chronology']['pass']}.", "", *eb.format_novelty_rows(novelty["table"])]
        if novelty["skipped"]:
            lines += ["", "Not novelty-scored (failed guards): " + ", ".join(
                f"{a} ({', '.join(s['failed_guards'])})" for a, s in novelty["skipped"].items())]
    return "\n".join(lines) + "\n"


def rendered(plan, compute):
    return {"compute.json": eb.json_text(compute), "summary.json": eb.json_text(compute["summary"]),
            "findings.md": findings_md(plan, compute)}


def write_publication(output, plan):
    from world_model.planning import scoring_harness as harness
    with harness.deterministic_scoring():
        compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        (Path(output) / name).write_text(text)
    return compute


def publish(output):
    """records-manifest.json and the tables; every token is the stop token until --validate binds G-records. Run
    again after the A3 scoring pass (the addendum): existing records are only re-read."""
    output = Path(output)
    plan = load_plan(output)
    eb.write_json(output / "records-manifest.json", eb.records_manifest(output, IDENTITY, RECORD_PATTERNS))
    compute = write_publication(output, plan)
    log(f"published {list(compute['summary']['arms'])}; G-records "
        f"{compute['summary']['global_guards']['G_records']['pass']}")
    return 0


def spot_check(output, plan):
    """Re-score the first complete record (A2, then A1, then any) on the grid cells; scores must replicate."""
    from world_model.planning import scoring_harness as harness
    cells = eb.evaluation_cells()
    anchors = torch.load(eb.evaluation_anchors_path(output), map_location="cpu", weights_only=True)
    arms = scored_arms(output, plan)
    for arm in [a for a in ("A2", "A1") if a in arms] + [a for a in arms if a not in ("A2", "A1")]:
        for seed in SEEDS:
            record = eb.read_json(record_path(output, arm, seed))
            if record["typed_failure"]:
                continue
            with harness.deterministic_scoring():
                fresh = score_arm_seed(output, plan, arm, seed, cells, anchors, device_name())
            worst, worst_rel = 0.0, 0.0  # relative: |delta| / max(1, |score|) (CPU vs GPU re-score round-off)
            for member, block in record["inventories"]["grid"].items():
                for old, new in zip(block["rows"], fresh["inventories"]["grid"][member]["rows"]):
                    if (old["score"] is None) != (new["score"] is None):
                        worst = worst_rel = math.inf
                    elif old["score"] is not None:
                        delta = abs(old["score"] - new["score"])
                        worst = max(worst, delta)
                        worst_rel = max(worst_rel, delta / max(1.0, abs(old["score"])))
            return {"arm": arm, "seed": seed, "cells": len(record["inventories"]["grid"]), "max_abs_delta": worst,
                    "max_relative_delta": worst_rel, "tolerance_relative": 1e-4, "pass": bool(worst_rel <= 1e-4)}
    raise ValueError("no complete record to re-score")


def validate(output):
    output = Path(output)
    plan = load_plan(output)
    spot = None
    try:
        manifest = eb.read_json(output / "records-manifest.json")
        for entry in manifest["records"]:
            if eb.sha256_of(eb.ROOT / entry["artifact"]) != entry["sha256"]:
                raise ValueError(f"record {entry['artifact']} differs from records-manifest.json")
        if manifest != eb.records_manifest(output, IDENTITY, RECORD_PATTERNS):
            raise ValueError("records-manifest.json does not list exactly the current records")
        from world_model.planning import scoring_harness as harness
        with harness.deterministic_scoring():
            compute = compute_tables(output, plan)
        for name, text in rendered(plan, compute).items():
            if (output / name).read_bytes() != text.encode():
                raise ValueError(f"published {name} differs from the recomputation")
        spot = spot_check(output, plan)
        if not spot["pass"]:
            raise ValueError(f"re-scoring does not replicate the retained record: {spot}")
    except Exception as error:
        for name in PUBLISHED:
            (output / name).unlink(missing_ok=True)
        eb.write_receipt(output, IDENTITY, spot, f"{type(error).__name__}: {error}")
        write_publication(output, plan)
        log(f"validation FAILED: publication re-derived with every token {eb.STOP_TOKEN}")
        raise
    eb.write_receipt(output, IDENTITY, spot)
    compute = write_publication(output, plan)
    if not compute["summary"]["global_guards"]["G_records"]["pass"]:
        raise ValueError("the validation receipt does not pass G-records")
    log(f"validation passed: {len(manifest['records'])} records, spot check {spot}")
    return 0


def dry_run(output):
    for arm, spec in ARMS.items():
        state = "ready" if spec["configurations"] and available(arm) else "adapter_missing"
        log(f"{arm}: {spec['family']} {spec['role']} {state}; configurations {spec['configurations']}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("selection", "selection-readouts", "smoke", "prepare", "train", "readouts", "freeze-readouts",
                 "score", "publish", "validate", "dry-run", "throughput", "select-deferred",
                 "score-novelty"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="queue / worker index (CUDA_VISIBLE_DEVICES set to it)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--arms", default=None, help="comma-separated arm subset")
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, eb.request_stop)
    if args.selection or args.selection_readouts:
        arms = parse_arms(args.arms, [a for a, s in ARMS.items() if s["configurations"]])
        fn = selection if args.selection else selection_readouts
        return fn(OUTPUT, args.gpu, args.workers, arms)
    if args.throughput:
        context = Context(device_name())
        for arm in parse_arms(args.arms, ["A4"]):
            if measure_throughput(OUTPUT, arm, context) == eb.EXIT_PAUSED:
                return eb.EXIT_PAUSED
        return 0
    if args.train:
        return train(OUTPUT, args.gpu)
    if args.readouts:
        return readouts(OUTPUT, args.gpu, args.workers, parse_arms(args.arms, []) or None)
    if args.freeze_readouts:
        return freeze_readouts(OUTPUT, parse_arms(args.arms, []) or None)
    if args.score:
        return score(OUTPUT, parse_arms(args.arms, []) or None)
    if args.score_novelty:
        return score_novelty(OUTPUT, parse_arms(args.arms, []) or None)
    fn = {"smoke": smoke, "prepare": prepare, "publish": publish, "validate": validate, "dry_run": dry_run,
          "select_deferred": select_deferred}
    for name, f in fn.items():
        if getattr(args, name):
            return f(OUTPUT)
    parser.error("no mode given")


if __name__ == "__main__":
    raise SystemExit(main())
