"""Issue-78 ADD-EXP: external temporal-adaptation baselines (TAWM, VLWM).

Ports the two published temporal-adaptation mechanisms onto the frozen
~1.8M-parameter issue-74/77 ContinuousDynamics carrier, retrains them on the
frozen issue-77 N1 fitting corpus under the identical equal-update budget
(9000 updates, batch 64, three paired seeds, identical minibatches per paired
seed), and scores them on the frozen issue-77 evaluation states (4 N1
held-out + 17 N2, admissible candidates of the frozen 13-candidate inventory,
endpoints {15,30,60,150,225,600} observed frames) under the #77 contract.

Discipline (declared before any model outcome was seen, 2026-09-21):
- The frozen plan (.local-artifacts/issue-78-external-temporal-baselines-v1/
  plan.json) pins membership, estimands, metrics, contrasts, compute
  accounting, stop rules and the pre-declared disposition rules BEFORE any
  outcome is computed; every number is real, no placeholders.
- Every scheduled cell is executed or retained as a typed terminal failure;
  no outcome-conditioned exclusion, retry, replacement, or re-freeze.
- Estimand: normalized ranking regret against the t=600 end-of-window replay
  cost (right-censored; NOT a settled cost), exactly as #77.
- Paired bootstrap intervals (10000 draws, seed 7201, states as units, seed
  differences averaged before resampling) are DESCRIPTIVE; no beats/survives
  language; zero-shot conditions only (both baselines are trained on the N1
  corpus and evaluated zero-shot on N2; nothing is pooled).
- Compute accounting reports per-state wall time AND active-parameter/step
  counts AND candidate counts per arm; candidate count alone is never compute
  matching; inference is NOT equalized (recorded).
- `continuous_h5` comparator with selection_optimism=true (#74 readiness) is
  disclosed in every contrast that uses it.
- Wall-time-measured GPU phases (--train, --run-evaluation) hold an exclusive
  fcntl flock on /tmp/novphy-addexp-gpu.lock for the whole phase; dry-run,
  smoke, publish and validate need no lock. Peak CUDA allocation stays <4 GiB.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import math
from collections import Counter
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from scripts import run_issue_71_hybrid_readiness as old
from scripts import run_issue_72_matched_grid as grid
from scripts import run_issue_74_matched_dynamics as base
from scripts import run_issue_77_n1_train as train
from scripts import run_issue_77_n1_diagnostic as diag
from world_model.model import Abstraction, PredictionPair
from world_model.training import external_temporal_baselines as eb
from world_model.training.matched_dynamics import (
    CONTINUOUS_PAIRS, ContinuousDynamics, MatchedController, active_capacity,
    continuous_loss, parameter_count, work,
)

ROOT = base.ROOT
DYNAMICS = train.OUTPUT
N1_DIAG = diag.OUTPUT
N2_EVAL = ROOT / ".local-artifacts/issue-77-n2-eval-v1"
READINESS_74 = ROOT / ".local-artifacts/issue-74-matched-dynamics-v1/readiness.json"
OUTPUT = ROOT / ".local-artifacts/issue-78-external-temporal-baselines-v1"
SCHEMA = "issue_78_external_temporal_baselines_v1"
IDENTITY = "issue-78-external-temporal-baselines-v1"
TIMES = diag.TIMES
TASK_TIME = diag.TASK_TIME
HORIZONS = (1, 5, 15)
SEEDS = base.SEEDS
ARMS = ("tawm", "vlwm")
ADAPTIVE = "tawm_adaptive"
SYSTEMS = {f"{arm}_h{h}": (arm, PredictionPair(h, Abstraction.CONTINUOUS))
           for arm in ARMS for h in HORIZONS}
SYSTEMS[ADAPTIVE] = ("tawm", None)
REFERENCE_SYSTEMS = {f"continuous_h{h}": PredictionPair(h, Abstraction.CONTINUOUS)
                     for h in HORIZONS}
N1_CORPUS = "issue-77-n1-diagnostic-v1"
N2_CORPUS = "issue-77-n2-eval-v1"
ALLOWANCE_GPU_SECONDS = 4 * 3600
ALLOWANCE_DERIVED_BYTES = 5 * 2 ** 30
CONTROLLER_SEED_OFFSET = 7800
TAWM_PIN = "ffb61f8e2bcdb0030cb4a7175e0b782cdad9af4c"
VLWM_NO_CODE_SEARCH = (
    {"way": "arxiv-abstract", "query": "arxiv.org/abs/2606.21775",
     "date": "2026-09-21", "result": "paper page; no code link in the abstract or comments"},
    {"way": "arxiv-full-html-grep", "query": "arxiv.org/html/2606.21775v1 grep for code links",
     "date": "2026-09-21",
     "result": "no code-availability statement and no project github link in the full HTML"},
    {"way": "github-org-scan", "query": "github.com/PKU-ML and github.com/PKU-ICML repositories",
     "date": "2026-09-21",
     "result": "PKU-ML: 48 repositories, none matching VLWM/variable-length latent world models; "
               "PKU-ICML: not found"},
    {"way": "github-search", "query": "github repository search: 2606.21775 / VLWM latent world model",
     "date": "2026-09-21", "result": "no repository for this paper among the results"},
    {"way": "general-web", "query": "web search: VLWM variable-length latent world model github code",
     "date": "2026-09-21",
     "result": "only paper mirrors (arxiv, alphaXiv, semantic scholar) and unrelated repositories"},
    {"way": "papers-with-code", "query": "paperswithcode.com paper page (service redirects)",
     "date": "2026-09-21", "result": "no paper page with code; service redirects to generic search"},
    {"way": "openreview", "query": "openreview.net search for the exact title",
     "date": "2026-09-21", "result": "no matching submission"},
)
FILES = ("scripts/run_external_temporal_baselines.py",
         "world_model/training/external_temporal_baselines.py",
         "world_model/training/matched_dynamics.py")
read, write = base.read, base.write


def log(message):
    print(f"[issue-78] {message}", flush=True)


def memory(device):
    return base.memory(device)


def dynamics_args(args):
    return argparse.Namespace(output=args.dynamics, issue71=train.old.ROOT,
                              parser=train.old.repair.ROOT, device=args.device)


# ---------------------------------------------------------------- membership

def training_pools(plan77):
    pools = plan77["n1_lineages"]
    if len(pools["predictor"]) + 2400 != plan77["training"]["fit_lineages"]:
        raise ValueError("frozen #77 predictor pool disagrees with its frozen fit_lineages count")
    if len(pools["controller"]) + 600 != plan77["training"]["controller_lineages"]:
        raise ValueError("frozen #77 controller pool disagrees with its controller_lineages count")
    return {"predictor": {"issue71_count": 2400, "n1_lineages": sorted(pools["predictor"])},
            "controller": {"issue71_count": 600, "n1_lineages": sorted(pools["controller"])}}


def eval_membership():
    """Frozen evaluation states from the #77 diagnostic and N2 evaluation plans."""
    n1 = read(N1_DIAG / "plan.json")
    if n1["identity"] != N1_CORPUS:
        raise ValueError("unexpected issue-77 N1 diagnostic plan identity")
    n2 = read(N2_EVAL / "plan.json")
    if n2["identity"] != N2_CORPUS:
        raise ValueError("unexpected issue-77 N2 evaluation plan identity")
    states = []
    for sample in n1["samples"]:
        states.append(_state_entry(sample, N1_CORPUS,
                                   N1_DIAG / "targets" / f"state-{sample['state']['ordinal']:03d}.json"))
    for side in n2["sides"]:
        for sample in side["samples"]:
            states.append(_state_entry(sample, N2_CORPUS,
                                       N2_EVAL / "targets" / f"state-{sample['state']['identity']}.json"))
    if not any(s["candidates"] for s in states):
        raise ValueError("no evaluable frozen states")
    return states


def _state_entry(sample, corpus, target_path):
    state = sample["state"]
    if sample["sources"] and len(sample["sources"]) > 13:
        raise ValueError("candidate inventory exceeds the frozen 13-candidate grid")
    return {"state": state["identity"], "state_ordinal": state.get("ordinal"),
            "family": state["generator_family"], "corpus": corpus,
            "unsupported": not bool(sample["sources"]),
            "dropped_candidates": list(sample["dropped_candidates"]),
            "target_path": str(target_path),
            "candidates": [{"identity": s["identity"], "ordinal": s["ordinal"], "action": s["action"]}
                           for s in sample["sources"]]}


def target_path(args, state):
    return Path(state["target_path"])


# ---------------------------------------------------------------- frozen plan

def make_plan(args):
    plan77 = train.load_plan(dynamics_args(args))
    if read(args.dynamics / "n1-data.json")["lineages"] != 12:
        raise ValueError("requires completed issue-77 N1 shards")
    readiness74 = read(READINESS_74)
    if readiness74["selected_continuous_policy"] != "continuous_h5" \
            or readiness74["selection_optimism"] is not True:
        raise ValueError("issue-74 readiness must select continuous_h5 with selection optimism")
    states = eval_membership()
    evaluable = [s for s in states if not s["unsupported"]]
    candidates = sum(len(s["candidates"]) for s in evaluable)
    pools = training_pools(plan77)
    tawm_counts = eb.tawm_horizon_counts(9000)
    vlwm_counts = eb.vlwm_horizon_counts(9000)
    controller_capacity = parameter_count(MatchedController(pure=True))
    return {
        "schema": SCHEMA, "identity": IDENTITY,
        "prior_artifacts": {"dynamics": str(args.dynamics.resolve()),
                            "dynamics_plan_identity": plan77["identity"],
                            "n1_diagnostic": str(N1_DIAG), "n2_eval": str(N2_EVAL),
                            "issue74_readiness": str(READINESS_74),
                            "selected_continuous_policy": "continuous_h5",
                            "selection_optimism": True,
                            "selection_optimism_disclosure": "continuous_h5 was selected on "
                                                            "development evidence (issue-74 "
                                                            "readiness); this optimism is "
                                                            "disclosed in every contrast that "
                                                            "uses it"},
        "contract": plan77["contract"], "capacity": base.capacity_contract(),
        "seeds": list(SEEDS), "arms": list(ARMS),
        "carrier": {"module": "world_model.training.matched_dynamics.ContinuousDynamics",
                    "width": plan77["capacity"]["continuous_width"],
                    "parameters": plan77["capacity"]["continuous_parameters"],
                    "new_learned_components": "none; both ports reuse the frozen carrier unchanged"},
        "training": {"steps": 9000, "batch_size": 64, "learning_rate": .0001, "weight_decay": .0001,
                     "grad_clip": 1., "unroll": 4,
                     "schedule": "#71/#77 three fitting lineages per nine steps over the 2409-entry "
                                 "frozen pool (2400 issue-71 + 9 N1 predictor lineages); identical "
                                 "random minibatches per paired seed across arms and identical to "
                                 "the #77 reference-arm minibatch sequence",
                     "recipe": {"tawm": "carrier continuous_loss (local MSE + recursive MSE + .01 "
                                        "carrier bound penalty) at the scheduled horizon",
                                "vlwm": "Eq. 3 direct squared error against the observed carrier at "
                                        "t+k from every valid in-window start + .01 carrier bound "
                                        "penalty; no recursive rollout term (paper-fidelity port)"},
                     "horizon_schedule": {
                         "tawm": {"distribution": "uniform over the frozen grid {1,5,15}",
                                  "realization": "deterministic round-robin DELTAS[step % 3]",
                                  "counts": {f"h{h}": tawm_counts[h] for h in HORIZONS}},
                         "vlwm": {"distribution": "stage-wise curriculum over the frozen grid, "
                                                  "VLWM Eqs. 5-6 with K_max=3, T=9000: "
                                                  "j(tau)=min(3, ceil(3*tau/9000)), p_tau(k)= "
                                                  "Uniform over the first j stages",
                                  "realization": "round-robin within the eligible stage set",
                                  "stage_updates": {"stage_1_k_in_1": 3000,
                                                    "stage_2_k_in_1_5": 3000,
                                                    "stage_3_k_in_1_5_15": 3000},
                                  "counts": {f"h{h}": vlwm_counts[h] for h in HORIZONS}}},
                     "selection": "last update, no best-validation checkpoint",
                     "fit_lineages": pools["predictor"]["issue71_count"]
                                     + len(pools["predictor"]["n1_lineages"]),
                     "controller_lineages": pools["controller"]["issue71_count"]
                                            + len(pools["controller"]["n1_lineages"]),
                     "derived_pools": pools,
                     "optimizer_examples": 9000 * 64},
        "controller": {"arms": ["tawm"], "steps_per_round": 1800, "batch_size": 128,
                       "learning_rate": .001, "rounds": 2, "compute_weight": .0001,
                       "seed_offset": CONTROLLER_SEED_OFFSET, "pool_size": 603,
                       "capacity": {"tawm": controller_capacity},
                       "pool": "603-entry frozen controller pool (600 issue-71 + 3 N1 lineages)",
                       "machinery": "existing repo MatchedController(pure=True) with DP duration-MSE "
                                    "labels (controller_labels); identical to the repo `continuous "
                                    "adaptive` arm; no new selector",
                       "vlwm": "no controller; fixed systems only (per the frozen ticket design)"},
        "systems": {"tawm": "single shared-parameter h-conditioned continuous predictor trained over "
                            "the uniform mixed-h distribution (TAWM port, declared deviations "
                            "below)",
                    "vlwm": "action-sequence-conditioned direct z_{t+k} prediction with stage-wise "
                            "curriculum over k on the frozen grid (VLWM port, Eqs. 3-6); "
                            "paper-fidelity port, no reference code consulted",
                    "inventory": {name: {"arm": arm,
                                         "horizon": None if pair is None else pair.delta,
                                         "adaptive": pair is None}
                                  for name, (arm, pair) in SYSTEMS.items()}},
        "deviations": {
            "tawm": ["Delta-t conditioning replaced by h-conditioning on the frozen grid {1,5,15} "
                     "(observed-frame units; 1 frame = 50 native steps)",
                     "TAWM's LogUniform(Delta-t) sampler replaced by the uniform distribution over "
                     "the frozen grid (targets exist only at grid offsets); realized as "
                     "deterministic round-robin DELTAS[step % 3] PER UPDATE for exact "
                     "3000/3000/3000 update counts (the paper samples its interval per training "
                     "example; the per-update granularity is a declared granularity deviation)",
                     "evaluated (a) fixed per-h and (b) with h chosen by the existing learned "
                     "horizon-controller machinery (repo `continuous adaptive` arm); no new "
                     "selector, no free threshold",
                     "the TAWM README documents no uncertainty-gating control experiments; none "
                     "are claimed",
                     "STRUCTURAL EQUIVALENCE DISCLOSURE: on this carrier the TAWM cell shares the "
                     "#77 continuous reference cell's recipe - same seed-initialized "
                     "ContinuousDynamics, same generator-seeded minibatch stream over the same "
                     "2409-lineage pool, same continuous_loss, same AdamW settings, same "
                     "3000/3000/3000 horizon counts; it differs from the reference arm ONLY in "
                     "the within-cycle horizon ordering (reference PAIRS[step%9] blocks "
                     "1,1,1,5,5,5,15,15,15 versus TAWM round-robin 1,5,15). The "
                     "external_vs_fixed and training_effect_analog contrasts for tawm_h{h} vs "
                     "continuous_h{h} therefore estimate horizon-schedule-ordering effects under "
                     "an identical budget and identical initialization, NOT an independent "
                     "re-implementation, and the q1/q3 numbers must be read as such; the vlwm "
                     "arm carries the independent mechanism contrast"],
            "vlwm": ["single-shot mapping: the paper's action sequence a_{t:t+k-1} is the executed "
                     "launch action followed by k-1 null actions; under the carrier's single "
                     "5-value action slot the sequence collapses to the launch action, and the "
                     "sequence length k enters through the carrier's horizon conditioning",
                     "L=1 context (Eq. 4 minimal instantiation {s_t}); the carrier interface holds "
                     "one observed carrier",
                     "the paper's stop-gradient target encoder is unnecessary: targets are "
                     "observed carriers, not encoder outputs",
                     "curriculum stages are the ordered frozen grid {1},{1,5},{1,5,15} instead of "
                     "integer horizons 1..K_max"]},
        "fidelity_gate": {
            "criterion": "the mechanism is reproducible from the paper method section on our "
                         "carrier with no new learned component (pre-execution; the only place "
                         "the system matrix may change)",
            "tawm": "PASS: the TAWM recipe (shared-parameter dynamics conditioned on the "
                    "transition length, trained over a mixed-length distribution; the reference "
                    "README states the architecture-agnostic recipe) is realized by the existing "
                    "HorizonConditioner FiLM conditioning plus the mixed-h training schedule; "
                    "zero new learned modules",
            "tawm_reference": {"repo": "github.com/anh-nn01/Time-Aware-World-Model",
                               "commit": TAWM_PIN,
                               "clone": "refs/tawm under the issue-78 artifact directory "
                                        "(reference-only consultation; never trained at its "
                                        "original scale)",
                               "paper": "arXiv 2506.08441 (ICML 2025)"},
            "vlwm": "PASS by construction: paper Eqs. 3-6 fully specify the port onto the carrier",
            "vlwm_reference": {"repo": "none exists",
                               "code": "paper-fidelity port, no reference code consulted",
                               "paper": "arXiv 2606.21775"},
            "vlwm_no_code_search": [dict(w) for w in VLWM_NO_CODE_SEARCH],
            "thick_fallback": "NOT triggered: both ports passed the gate; THICK (ICLR 2024, "
                              "openreview TjCDNssXKU; github.com/CognitiveModeling/THICK @ "
                              "934980a reference-only; executable JAX reference thix) was not "
                              "needed and no THICK arm is scheduled"},
        "evaluation": {"states": len(states), "evaluable_states": len(evaluable),
                       "unsupported_states": len(states) - len(evaluable),
                       "candidates": candidates,
                       "states_membership": states,
                       "endpoints": list(TIMES), "task_time": TASK_TIME,
                       "endpoint_units": "observed frames (1 frame = 50 native steps)",
                       "endpoint_semantics": "t=600 is the end-of-window replay cost on mostly "
                                             "right-censored branches; NOT a settled cost",
                       "estimand": "normalized ranking regret against the t=600 end-of-window "
                                   "replay cost (right-censored), grid.ranked over each state's "
                                   "admissible candidates",
                       "conditions": "zero-shot only: both baselines are trained on the frozen N1 "
                                     "corpus and evaluated without any N2 fitting; zero-shot and "
                                     "adapted conditions are never pooled (no adapted condition "
                                     "is scheduled in this ticket)",
                       "candidate_inventory": "frozen 13-candidate set (a00 reference + 12 "
                                              "angular a01-a12), per-state counts reduced by "
                                              "typed branch failures exactly as recorded in the "
                                              "#77 plans",
                       "adaptive_curve_capture": "the adaptive arm queries the controller with "
                                                 "the global remaining budget (600 - elapsed, as "
                                                 "the repo continuous adaptive arm); recursive "
                                                 "outputs are recorded only at transition "
                                                 "boundaries that coincide with the reporting "
                                                 "endpoints, so its curve errors are a sparse "
                                                 "subset; the t=600 task endpoint is always "
                                                 "exact (the controller masks horizons exceeding "
                                                 "remaining)",
                       "bootstrap": {"draws": 10000, "seed": 7201, "unit": "state",
                                     "note": "seed differences averaged before resampling; "
                                             "DESCRIPTIVE only - at 4 N1 + 17 N2 independent "
                                             "states no confirmatory claim is made"}},
        "contrasts": {
            "direction": "differences are mean-over-seeds per-state reference minus tested "
                         "regret; positive favors the tested system",
            "q1_method_class": [
                {"tested": f"{arm}_h{h}", "reference": f"continuous_h{h}",
                 "kind": "external_vs_fixed"} for arm in ARMS for h in HORIZONS],
            "q1_decay_signature": [
                {"tested": f"{arm}_h1", "reference": f"{arm}_h15", "kind": "decay_signature"}
                for arm in ARMS]
            + [{"tested": "continuous_h1", "reference": "continuous_h15",
                "kind": "decay_signature_reference"}],
            "q2_work_reported": [
                {"tested": name, "reference": "continuous_h5", "kind": "work_reported_comparison"}
                for name in sorted(SYSTEMS)],
            "q3_training_effect": [
                {"tested": f"{arm}_h1", "reference": "continuous_h1",
                 "kind": "training_effect_analog"} for arm in ARMS],
            "disclosure": "every contrast is DESCRIPTIVE (paired bootstrap, 10000 draws, seed "
                          "7201); contrasts involving continuous_h5 disclose its issue-74 "
                          "selection optimism",
            "reference_issue77_contrasts": "quoted read-only from the frozen issue-77 N1 "
                                           "diagnostic summary at publication"},
        "disposition_rules": {
            "tokens": ["supported", "not_supported_by_this_experiment",
                       "readiness_or_precision_insufficient"],
            "pre_declared": "2026-09-21, before any outcome was computed",
            "general": "if any scheduled cell is a typed terminal failure (missing/invalid "
                       "checkpoint or evaluation record) the affected questions take "
                       "readiness_or_precision_insufficient",
            "q1_method_class": "the decay-signature interval for arm a is the paired state "
                               "differences of a_h15 regret minus a_h1 regret (positive = h1 "
                               "better, decay present); arm status is positive if the interval "
                               "low > 0, negative if high < 0, else indeterminate. supported "
                               "iff both arms positive; not_supported_by_this_experiment if at "
                               "least one arm is negative and none indeterminate; else "
                               "readiness_or_precision_insufficient. A negative result is a "
                               "VALID outcome: it localizes the boundary to our implementation",
            "q2_work_reported": "supported if the full frontier (regret and per-state wall for "
                                "all seven baseline systems against the frozen continuous_h5 "
                                "records) is computable from executed cells; else "
                                "readiness_or_precision_insufficient",
            "q3_training_effect": "for arm a the training-effect analog interval is the paired "
                                  "state differences of continuous_h1 regret minus a_h1 regret "
                                  "(positive = the temporal-adaptation mechanism shows the h=1 "
                                  "advantage). supported iff at least one arm positive; "
                                  "not_supported_by_this_experiment if both negative; else "
                                  "readiness_or_precision_insufficient. The symbolic-EXECUTION "
                                  "effect is structurally absent for external arms (no symbolic "
                                  "heads), so distinctness from it holds by construction and is "
                                  "recorded, with #77's hybrid symbolic contrasts quoted for "
                                  "reference"},
        "compute": {"allowance": {"gpu_seconds": ALLOWANCE_GPU_SECONDS,
                                  "derived_artifact_bytes": ALLOWANCE_DERIVED_BYTES,
                                  "accounting": "cumulative across --train and --run-evaluation "
                                                "via allowance-ledger.json (per-cell measured "
                                                "deltas, resumed cells conservatively over-count "
                                                "load time); exceeding the total stops the ticket "
                                                "with readiness_or_precision_insufficient",
                                  "exceeded": "stop the ticket with "
                                              "readiness_or_precision_insufficient"},
                    "equalization": "equal training updates (9000) and examples (576000 windows "
                                    "per cell); inference NOT equalized - the full per-state "
                                    "wall / active-parameter-step / candidate-count frontier is "
                                    "reported per arm as #74 does",
                    "accounting": "per-state wall time, linear MACs, controller/transition "
                                  "calls, executed operator parameter counts, and candidate "
                                  "counts per arm; reference-arm walls come from the frozen #77 "
                                  "records measured 2026-09-20 and are labelled as such",
                    "gpu_lock": "--train and --run-evaluation hold an exclusive fcntl flock on "
                                "/tmp/novphy-addexp-gpu.lock for the whole phase (single shared "
                                "RTX 3090); dry-run/smoke/publish/validate need no lock",
                    "training_compute_matched": False},
        "operator": {"dry_run": "no-write plan summary",
                     "smoke": "bounded real training + one state, two candidates, all systems "
                              "in a temporary directory",
                     "progress": "foreground logs with ETA",
                     "resume": "atomic progress checkpoints every 90 updates and per-candidate "
                               "evaluation records; resume without outcome-conditioned "
                               "replacement",
                     "integrity": "no new content hashes or full-corpus integrity passes during "
                                  "execution"},
        "stop_rules": ["compute allowance exceeded -> typed terminal failure, disposition "
                       "readiness_or_precision_insufficient",
                       "nonfinite training loss -> typed terminal failure for that cell, run "
                       "stops, cell retained as failed",
                       "no outcome-conditioned exclusion, retry, replacement, or re-freeze"],
        "claim_boundary": "mechanism-class comparison under one frozen contract; no claim of "
                          "exact reproduction of TAWM/VLWM numbers on their original "
                          "environments; descriptive intervals only; no inferential superiority "
                          "claim; no new captures; not the #64/#65 sealed benchmark; prior "
                          "dispositions (#15, #72, #74, #75, #77) unchanged",
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                   text=True).strip(),
        "source_text": {p: (ROOT / p).read_text() for p in FILES},
        "archived_release": False, "fresh_evaluation_opened": False,
        "final_evaluation_opened": False, "issue_64_authorized": False}


def load_plan(args):
    plan = read(args.output / "plan.json")
    current = make_plan(args)
    current["source_revision"] = plan["source_revision"]
    if plan != current:
        raise ValueError("frozen issue-78 source/settings differ; retain old output and "
                         "explicitly version any change")
    return plan


# ---------------------------------------------------------------- checkpoints

def cell(args, seed, arm):
    return args.output / f"seed-{seed}" / arm


def binding(plan, seed, arm, component):
    return {"plan_identity": plan["identity"], "contract": plan["contract"],
            "capacity": plan["capacity"], "seed": seed, "arm": arm, "component": component,
            "training": plan["training"], "controller": plan["controller"]}


def check_binding(value, expected):
    if value["binding"] != expected:
        raise ValueError("issue-78 checkpoint binding differs")


def new_model(plan):
    return ContinuousDynamics(plan["capacity"]["continuous_width"])


def horizon_counts(plan, arm):
    raw = (eb.tawm_horizon_counts(plan["training"]["steps"]) if arm == "tawm"
           else eb.vlwm_horizon_counts(plan["training"]["steps"]))
    return {f"h{h}": n for h, n in raw.items()}


def train_cell(args, plan, plan77, source, seed, arm, stop_after=None):
    """One baseline predictor fit; resume exactly. stop_after is for smoke/tests only."""
    root = cell(args, seed, arm)
    target = root / "predictor.pt"
    if target.exists():
        load_predictor(args, plan, seed, arm)
        log(f"seed={seed} arm={arm} predictor reused")
        return
    torch.manual_seed(seed)
    model = new_model(plan).to(args.device)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    recipe = plan["training"]
    optim = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"],
                              weight_decay=recipe["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    expected = binding(plan, seed, arm, "predictor")
    start, prior, gradients, counts = 0, 0., set(), Counter()
    progress = root / "predictor-progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=True)
        check_binding(saved, expected)
        model.load_state_dict(saved["model"], strict=True)
        optim.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        start, prior = saved["step"], saved["wall_seconds"]
        gradients, counts = set(saved["gradient_parameters"]), Counter(saved["pair_counts"])
    began = time.monotonic()
    cached = None
    finish = min(recipe["steps"], stop_after if stop_after is not None else recipe["steps"])
    for step in range(start, finish):
        group = train.fitting_lineage_group(plan77, step)
        if group != cached:
            rows = [train.load_shard_entry(dynamics_args(args), plan77, i, source, True)
                    for i in group]
            data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            cached = group
        batch = old.sample(data, recipe["batch_size"], generator, args.device)
        if arm == "tawm":
            horizon = eb.tawm_horizon(step)
            loss = continuous_loss(model, batch["z"], batch["action"], batch["length"],
                                   PredictionPair(horizon, Abstraction.CONTINUOUS))
        else:
            horizon = eb.vlwm_horizon(step, recipe["steps"])
            loss = eb.vlwm_loss(model, batch["z"], batch["action"], batch["length"], horizon)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"nonfinite training loss seed={seed} arm={arm} step={step + 1}")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        gradients.update(n for n, p in model.named_parameters()
                         if p.grad is not None and bool((p.grad != 0).any()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
        optim.step()
        counts[f"h{horizon}"] += 1
        if (step + 1) % 90 == 0 or step + 1 == finish:
            elapsed = prior + time.monotonic() - began
            saved = {"binding": expected, "model": model.state_dict(),
                     "optimizer": optim.state_dict(), "generator": generator.get_state(),
                     "step": step + 1, "wall_seconds": elapsed,
                     "gradient_parameters": sorted(gradients), "pair_counts": dict(counts),
                     "changed_parameters": [n for n, p in model.named_parameters()
                                            if not torch.equal(p.detach().cpu(), initial[n])],
                     "optimizer_examples": (step + 1) * recipe["batch_size"],
                     "memory": memory(args.device)}
            old.repair.atomic_torch(progress, saved)
            log(f"train seed={seed} arm={arm} step={step + 1}/{recipe['steps']} h={horizon} "
                f"loss={float(loss.detach()):.6f} elapsed={elapsed:.1f}s "
                f"eta={elapsed / (step + 1) * (recipe['steps'] - step - 1):.1f}s "
                f"memory={saved['memory']}")
    if finish == recipe["steps"]:
        old.repair.atomic_torch(target, saved)


def load_predictor(args, plan, seed, arm):
    value = torch.load(cell(args, seed, arm) / "predictor.pt", map_location="cpu",
                       weights_only=True)
    check_binding(value, binding(plan, seed, arm, "predictor"))
    model = new_model(plan).to(args.device)
    model.load_state_dict(value["model"], strict=True)
    names = set(dict(model.named_parameters()))
    if (value["step"] != plan["training"]["steps"] or value["pair_counts"] != horizon_counts(plan, arm)
            or set(value["gradient_parameters"]) != names
            or set(value["changed_parameters"]) != names):
        raise ValueError(f"incomplete {arm} predictor budget/coverage/gradients")
    return model.eval(), value


def train_control(args, plan, plan77, source, seed, indices=None):
    """TAWM horizon controller: the repo `continuous adaptive` machinery, new fit."""
    root = cell(args, seed, "tawm")
    if (root / "controller.pt").exists():
        load_control(args, plan, seed)
        log(f"seed={seed} arm=tawm controller reused")
        return
    model, _ = load_predictor(args, plan, seed, "tawm")
    torch.manual_seed(seed + CONTROLLER_SEED_OFFSET)
    control = MatchedController(pure=True).to(args.device)
    initial = {n: p.detach().cpu().clone() for n, p in control.named_parameters()}
    expected = binding(plan, seed, "tawm", "controller")
    recipe = plan["controller"]
    pool = tuple(list(range(5, 3001, 5)) + plan77["n1_lineages"]["controller"])
    if indices is None:
        if len(pool) != recipe["pool_size"]:
            raise ValueError("controller pool disagrees with the frozen plan")
    else:
        pool = tuple(indices)
    gradients = set()
    saved = None
    for round_index in (0, 1):
        checkpoint = root / f"controller-round-{round_index}.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            check_binding(saved, expected)
            control.load_state_dict(saved["model"], strict=True)
            gradients.update(saved["gradient_parameters"])
            continue
        began = time.monotonic()
        for ordinal, entry in enumerate(pool, 1):
            path = train.label_path(root, round_index, entry)
            record = (source["records"][entry - 1] if isinstance(entry, int)
                      else plan77["n1_lineages"]["records"][entry])
            if not path.exists():
                tensors = train.label_lineage(dynamics_args(args), plan77, source, model,
                                              control, entry, bool(round_index))
                old.repair.atomic_torch(path, {"binding": expected, "record": record,
                                               "round": round_index, "tensors": tensors,
                                               "wall_seconds": time.monotonic() - began})
            else:
                cached = torch.load(path, weights_only=True, map_location="cpu")
                check_binding(cached, expected)
                if cached["record"] != record or cached["round"] != round_index:
                    raise ValueError("controller labels source/round differs")
            if ordinal % 100 == 0 or ordinal == len(pool):
                elapsed = time.monotonic() - began
                log(f"labels seed={seed} arm=tawm round={round_index + 1}/2 "
                    f"lineage={ordinal}/{len(pool)} elapsed={elapsed:.1f}s "
                    f"eta={elapsed / ordinal * (len(pool) - ordinal):.1f}s")
        optim = torch.optim.AdamW(control.parameters(), lr=recipe["learning_rate"])
        generator = torch.Generator().manual_seed(seed + CONTROLLER_SEED_OFFSET + round_index)
        progress = root / f"controller-progress-{round_index}.pt"
        start, prior = 0, 0.
        if progress.exists():
            saved = torch.load(progress, map_location="cpu", weights_only=True)
            check_binding(saved, expected)
            control.load_state_dict(saved["model"], strict=True)
            optim.load_state_dict(saved["optimizer"])
            generator.set_state(saved["generator"])
            start, prior = saved["step"], saved["wall_seconds"]
            gradients.update(saved["gradient_parameters"])
        began = time.monotonic()
        for step in range(start, recipe["steps_per_round"]):
            entry = pool[step % len(pool)]
            rows = [torch.load(train.label_path(root, r, entry), weights_only=True,
                               map_location="cpu")["tensors"] for r in range(round_index + 1)]
            data = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            batch = old.sample(data, recipe["batch_size"], generator, args.device)
            logits = control(batch["z"], batch["action"], batch["remaining"])
            loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite controller loss")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            gradients.update(n for n, p in control.named_parameters()
                             if p.grad is not None and bool((p.grad != 0).any()))
            optim.step()
            if (step + 1) % 60 == 0 or step + 1 == recipe["steps_per_round"]:
                elapsed = prior + time.monotonic() - began
                saved = {"binding": expected, "model": control.state_dict(),
                         "optimizer": optim.state_dict(), "generator": generator.get_state(),
                         "round": round_index, "step": step + 1, "wall_seconds": elapsed,
                         "gradient_parameters": sorted(gradients),
                         "changed_parameters": [n for n, p in control.named_parameters()
                                                if not torch.equal(p.detach().cpu(), initial[n])],
                         "memory": memory(args.device)}
                old.repair.atomic_torch(progress, saved)
                log(f"controller seed={seed} arm=tawm round={round_index + 1}/2 "
                    f"step={step + 1}/{recipe['steps_per_round']} elapsed={elapsed:.1f}s "
                    f"eta={elapsed / (step + 1) * (recipe['steps_per_round'] - step - 1):.1f}s")
        old.repair.atomic_torch(checkpoint, saved)
    if saved is None or saved["step"] != recipe["steps_per_round"]:
        raise ValueError("controller training did not reach the frozen budget")
    old.repair.atomic_torch(root / "controller.pt", saved)


def load_control(args, plan, seed):
    value = torch.load(cell(args, seed, "tawm") / "controller.pt", map_location="cpu",
                       weights_only=True)
    check_binding(value, binding(plan, seed, "tawm", "controller"))
    control = MatchedController(pure=True).to(args.device)
    control.load_state_dict(value["model"], strict=True)
    names = set(dict(control.named_parameters()))
    if (value["round"] != 1 or value["step"] != plan["controller"]["steps_per_round"]
            or set(value["gradient_parameters"]) != names
            or set(value["changed_parameters"]) != names):
        raise ValueError("tawm controller is incomplete or lacks task-trained parameters")
    return control.eval(), value


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def adaptive_curve(model, controller, context, action):
    """Deployment-boundary adaptive rollout; controller sees only current carrier."""
    device = next(model.parameters()).device
    if context.shape != (236,):
        raise ValueError("adaptive rollout requires a 236-value carrier")
    z = context[None].to(device)
    a = diag.n1_action_tensor(action, device)
    outputs, segments = {}, []
    elapsed, transition_wall, linear_macs = 0, 0., 0
    controller_calls = transition_calls = 0
    for target in TIMES:
        grid.synchronize(device)
        began = time.monotonic()
        while elapsed < target:
            remaining = TASK_TIME - elapsed
            pair = CONTINUOUS_PAIRS[int(controller(z, a, torch.tensor([remaining],
                                                                     device=device)).argmax(-1))]
            controller_calls += 1
            z = model.carrier(z, a, pair)
            elapsed += pair.delta
            transition_calls += 1
            linear_macs += work(model, pair, controller)
            segments.append({"start_fixed_step": elapsed - pair.delta, "horizon": pair.delta})
            if elapsed in TIMES:
                outputs[str(elapsed)] = z[0].cpu().tolist()
            if not bool(torch.isfinite(z).all()):
                grid.synchronize(device)
                return {"outputs": outputs, "failure": "nonfinite_recursive_carrier",
                        "completed_steps": elapsed, "segments": segments,
                        "linear_macs": linear_macs, "controller_calls": controller_calls,
                        "transition_calls": transition_calls,
                        "transition_wall_seconds": transition_wall + time.monotonic() - began}
        grid.synchronize(device)
        transition_wall += time.monotonic() - began
    if elapsed != TASK_TIME:
        raise ValueError("adaptive rollout did not land on the task endpoint")
    return {"outputs": outputs, "failure": None, "completed_steps": elapsed,
            "segments": segments, "linear_macs": linear_macs,
            "controller_calls": controller_calls, "transition_calls": transition_calls,
            "transition_wall_seconds": transition_wall}


def curve_path(args, seed, state, system, ordinal):
    return args.output / "fixed" / f"seed-{seed}" / state["state"] / system / \
        f"candidate-{ordinal:02d}.json"


def check_record(record, plan, state, source, seed, system, model, control, objective):
    arm, pair = SYSTEMS[system]
    keys = {"plan_identity": plan["identity"], "seed": seed, "system": system,
            "state": state["state"], "candidate_identity": source["identity"],
            "ordinal": source["ordinal"], "action": source["action"]}
    if any(record[k] != v for k, v in keys.items()):
        raise ValueError("issue-78 record binding differs")
    pred = record["prediction"]
    if pred["failure"]:
        if record["cost"] is not None:
            raise ValueError("failed rollout has a task cost")
        return
    if pred["completed_steps"] != TASK_TIME or "600" not in pred["outputs"]:
        raise ValueError("missing common physical endpoint")
    for values in pred["outputs"].values():
        if len(values) != 236 or not all(math.isfinite(v) for v in values):
            raise ValueError("invalid predicted carrier")
    if objective(torch.tensor(pred["outputs"]["600"])) != record["cost"]:
        raise ValueError("predicted task cost differs from endpoint carrier")
    if pair is None:
        if pred["controller_calls"] != len(pred["segments"]) or \
                any(s["horizon"] not in HORIZONS for s in pred["segments"]):
            raise ValueError("adaptive work accounting differs")
        if pred["linear_macs"] != sum(work(model, PredictionPair(s["horizon"],
                                                                 Abstraction.CONTINUOUS),
                                           control) for s in pred["segments"]):
            raise ValueError("adaptive segment work differs")
    else:
        if pred["linear_macs"] != work(model, pair) * (pred["completed_steps"] // pair.delta):
            raise ValueError("fixed execution work differs")
        if set(pred["outputs"]) != {str(t) for t in TIMES}:
            raise ValueError("fixed endpoints differ")
        if record["local"]["linear_macs"] != record["local"]["calls"] * work(model, pair):
            raise ValueError("local diagnostic work differs")
        if set(record["local"]["outputs"]) != {str(t) for t in TIMES}:
            raise ValueError("local endpoints differ")


def stored_record(args, plan, state, targets, seed, system, source, model, control, objective):
    path = curve_path(args, seed, state, system, source["ordinal"])
    if path.exists():
        record = read(path)
    else:
        context = torch.tensor(targets["context"])
        if system == ADAPTIVE:
            prediction = adaptive_curve(model, control, context, source["action"])
            record = {"plan_identity": plan["identity"], "seed": seed, "system": system,
                      "state": state["state"], "candidate_identity": source["identity"],
                      "ordinal": source["ordinal"], "action": source["action"],
                      "prediction": prediction, "cost": None,
                      "work_scope": "instrumented adaptive rollout with prefix synchronization; "
                                    "controller queries use the global remaining budget"}
            if not prediction["failure"]:
                record["cost"] = objective(torch.tensor(prediction["outputs"]["600"]))
        else:
            pair = SYSTEMS[system][1]
            prediction = diag.fixed_curve(model, context, source["action"], pair)
            target = next(t for t in targets["candidates"]
                          if t["source"]["ordinal"] == source["ordinal"])
            local = diag.local_predictions(model, target, context, source["action"], pair)
            cost = None if prediction["failure"] else \
                objective(torch.tensor(prediction["outputs"][str(TASK_TIME)]))
            record = {"plan_identity": plan["identity"], "seed": seed, "system": system,
                      "state": state["state"], "candidate_identity": source["identity"],
                      "ordinal": source["ordinal"], "action": source["action"],
                      "prediction": prediction, "local": local, "cost": cost,
                      "work_scope": "instrumented fixed-rollout wall with prefix "
                                    "synchronization; target parsing/local probes separate; "
                                    "MACs not full FLOPs"}
        check_record(record, plan, state, source, seed, system, model, control, objective)
        write(path, record)
    check_record(record, plan, state, source, seed, system, model, control, objective)
    return record


def check_targets(value, state, plan):
    if value["state"] != state["state"] or len(value["context"]) != 236:
        raise ValueError("frozen target state binding differs")
    if value["plan_identity"] not in (N1_CORPUS, N2_CORPUS):
        raise ValueError("frozen target corpus identity differs")
    if sorted(value["dropped_candidates"]) != sorted(state["dropped_candidates"]):
        raise ValueError("frozen dropped-candidate accounting differs")
    sources = {s["source"]["ordinal"]: s for s in value["candidates"]}
    if len(value["candidates"]) != len(state["candidates"]):
        raise ValueError("frozen candidate membership differs")
    for candidate in state["candidates"]:
        target = sources.get(candidate["ordinal"])
        if target is None or target["source"]["identity"] != candidate["identity"] \
                or target["source"]["action"] != candidate["action"]:
            raise ValueError("frozen candidate identity/ordinal/action differs")
        for offset in diag.needed_offsets():
            vector = target["carriers"].get(str(offset))
            if vector is None or len(vector) != 236 \
                    or not all(math.isfinite(x) for x in vector):
                raise ValueError("frozen target availability/representation differs")


def artifact_inventory(args, plan):
    states = [s for s in plan["evaluation"]["states_membership"] if not s["unsupported"]]
    expected = sum(len(s["candidates"]) for s in states) * len(SEEDS) * len(SYSTEMS)
    records = sum(curve_path(args, seed, s, name, c["ordinal"]).exists()
                  for seed in SEEDS for s in states for name in SYSTEMS
                  for c in s["candidates"])
    predictors = sum((cell(args, seed, arm) / "predictor.pt").exists()
                     for seed in SEEDS for arm in ARMS)
    controllers = sum((cell(args, seed, "tawm") / "controller.pt").exists() for seed in SEEDS)
    return {"evaluable_states": len(states),
            "target_states": sum(Path(s["target_path"]).exists() for s in states),
            "expected_target_states": len(states),
            "predictors": predictors, "expected_predictors": len(SEEDS) * len(ARMS),
            "tawm_controllers": controllers, "expected_tawm_controllers": len(SEEDS),
            "candidate_records": records, "expected_candidate_records": expected}


def run_evaluation(args, plan):
    inventory = artifact_inventory(args, plan)
    if inventory["candidate_records"] == inventory["expected_candidate_records"]:
        log("complete evaluation inventory already exists; no new inference; "
            "use --publish/--validate")
        return
    if inventory["predictors"] != inventory["expected_predictors"] \
            or inventory["tawm_controllers"] != inventory["expected_tawm_controllers"]:
        raise ValueError("run --train before --run-evaluation")
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    states = [s for s in plan["evaluation"]["states_membership"] if not s["unsupported"]]
    total = inventory["expected_candidate_records"]
    began = time.monotonic()
    completed = 0
    for seed in SEEDS:
        models = {arm: load_predictor(args, plan, seed, arm)[0].requires_grad_(False)
                  for arm in ARMS}
        control = load_control(args, plan, seed)[0].requires_grad_(False)
        for n, state in enumerate(states, 1):
            targets = read(target_path(args, state))
            check_targets(targets, state, plan)
            for system in SYSTEMS:
                arm = SYSTEMS[system][0]
                for source in state["candidates"]:
                    record = stored_record(args, plan, state, targets, seed, system, source,
                                           models[arm], control, objective)
                    completed += 1
                    elapsed = time.monotonic() - began
                    log(f"eval={completed}/{total} seed={seed} state={n}/{len(states)} "
                        f"system={system} candidate={source['ordinal']} "
                        f"failure={record['prediction']['failure']} elapsed={elapsed:.1f}s "
                        f"eta={elapsed / completed * (total - completed):.1f}s")
    log("evaluation complete")


# ---------------------------------------------------------------- publication

def _reference_curve_path(seed, state, system, ordinal):
    if state["corpus"] == N1_CORPUS:
        return N1_DIAG / "fixed" / f"seed-{seed}" / f"state-{state['state_ordinal']:03d}" / \
            system / f"candidate-{ordinal:02d}.json"
    return N2_EVAL / "fixed" / "zero-shot" / f"seed-{seed}" / f"state-{state['state']}" / system / \
        f"candidate-{ordinal:02d}.json"


def reference_records(args, plan, system):
    """Frozen #77 records for one reference system, keyed seed -> state -> ordinal."""
    out = {}
    for seed in SEEDS:
        rows = {}
        for state in plan["evaluation"]["states_membership"]:
            if state["unsupported"]:
                continue
            rows[state["state"]] = {c["ordinal"]: read(_reference_curve_path(
                seed, state, system, c["ordinal"])) for c in state["candidates"]}
        out[str(seed)] = rows
    return out


def reference_rows(args, plan, targets_by_state, ref_records):
    """Minimal per-state rows for the frozen continuous_h{1,5,15} reference systems.

    Built from the frozen #77 candidate records so contrast specs can pair the
    baseline systems against them exactly as they pair against each other.
    """
    rows = {}
    for system, by_seed in ref_records.items():
        delta = REFERENCE_SYSTEMS[system].delta
        for seed, per_state in by_seed.items():
            out = rows.setdefault(str(seed), {}).setdefault(system, {})
            for state, records in per_state.items():
                candidates = targets_by_state[state]["candidates"]
                realized = [t["realized"] for t in candidates]
                costs = [records[c["source"]["ordinal"]]["cost"] for c in candidates]
                out[state] = {"task": grid.ranked(costs, [{"accepted": True,
                                                           "realized_count_cost": r}
                                                          for r in realized]),
                              "recursive": {}, "local": {},
                              "work": {"linear_macs": sum(r["prediction"]["linear_macs"]
                                                          for r in records.values()),
                                       "wall_seconds": sum(r["prediction"]["transition_wall_seconds"]
                                                           for r in records.values()),
                                       "controller_calls": 0,
                                       "transition_calls": sum(r["prediction"]["completed_steps"]
                                                               // delta
                                                               for r in records.values()),
                                       "perception_seconds":
                                           targets_by_state[state]["perception"]["wall_seconds"]}}
    return rows


def paired_difference(all_states, tested, reference, ids=None):
    """Mean-over-seeds per-state (reference - tested) regret differences."""
    if ids is None:
        ids = sorted(all_states[str(SEEDS[0])][tested])
    values = []
    for state in ids:
        across = [all_states[str(seed)][reference][state]["task"]["regret"]
                  - all_states[str(seed)][tested][state]["task"]["regret"] for seed in SEEDS]
        values.append(float(np.mean(across)))
    return values


def interval_status(interval):
    low, high = interval["descriptive_95_percent_interval"]
    return "positive" if low > 0 else ("negative" if high < 0 else "indeterminate")


def dispositions(result, cells_complete):
    if not cells_complete:
        return {q: "readiness_or_precision_insufficient"
                for q in ("q1_method_class", "q2_work_reported", "q3_training_effect")}
    rules = {}
    decay = {c["tested"].rsplit("_h", 1)[0]: c for c in result["contrasts"]
             if c["kind"] == "decay_signature" and "corpus" not in c}
    statuses = {arm: interval_status(decay[arm]["regret"]) for arm in ARMS}
    if all(s == "positive" for s in statuses.values()):
        rules["q1_method_class"] = "supported"
    elif any(s == "negative" for s in statuses.values()) \
            and not any(s == "indeterminate" for s in statuses.values()):
        rules["q1_method_class"] = "not_supported_by_this_experiment"
    else:
        rules["q1_method_class"] = "readiness_or_precision_insufficient"
    rules["q2_work_reported"] = ("supported" if result.get("frontier_complete")
                                 else "readiness_or_precision_insufficient")
    training = {c["tested"].rsplit("_h", 1)[0]: c for c in result["contrasts"]
                if c["kind"] == "training_effect_analog" and "corpus" not in c}
    training_status = {arm: interval_status(training[arm]["regret"]) for arm in ARMS}
    if any(s == "positive" for s in training_status.values()):
        rules["q3_training_effect"] = "supported"
    elif all(s == "negative" for s in training_status.values()):
        rules["q3_training_effect"] = "not_supported_by_this_experiment"
    else:
        rules["q3_training_effect"] = "readiness_or_precision_insufficient"
    return rules


def publication(args, plan):
    inv = artifact_inventory(args, plan)
    failures = read(args.output / "failures.json") \
        if (args.output / "failures.json").exists() else []
    complete = (inv["target_states"] == inv["expected_target_states"]
                and inv["predictors"] == inv["expected_predictors"]
                and inv["tawm_controllers"] == inv["expected_tawm_controllers"]
                and inv["candidate_records"] == inv["expected_candidate_records"]
                and not failures)
    common = {"schema": "issue_78_external_temporal_baselines_report_v1",
              "identity": "issue-78-external-temporal-baselines-report-v1",
              "plan_identity": plan["identity"], "inventory": inv,
              "typed_failures": failures, "diagnostics_complete": complete,
              "definitions": plan["evaluation"], "deviations": plan["deviations"],
              "fidelity_gate": plan["fidelity_gate"],
              "comparator_disclosure": {"policy": "continuous_h5", "selection_optimism": True,
                                        "selection": "development evidence (issue-74 readiness)",
                                        "disclosed_in_every_contrast": True},
              "claim_boundary": plan["claim_boundary"],
              "limitations": [
                  "t=600 is the end-of-window replay cost on mostly right-censored branches, "
                  "not a settled cost",
                  "descriptive paired bootstrap intervals only; no inferential superiority claim",
                  "inference is NOT equalized across arms; the full frontier is reported",
                  "continuous_h5 comparator retains issue-74 selection optimism (selected on "
                  "development evidence); disclosed in every contrast that uses it",
                  "reference-arm walls come from frozen #77 records measured 2026-09-20; "
                  "baseline walls were measured under this ticket's exclusive GPU lock",
                  "linear MACs, not full FLOPs",
                  "the adaptive arm's recursive curve is captured sparsely (transition "
                  "boundaries only); its t=600 task endpoint is exact"],
              "archived_release": False, "fresh_evaluation_opened": False,
              "final_evaluation_opened": False, "issue_64_authorized": False}
    if not complete:
        return {**common, "dispositions": dispositions({}, False)}
    objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
    states = [s for s in plan["evaluation"]["states_membership"] if not s["unsupported"]]
    targets_by_state = {}
    for state in states:
        value = read(target_path(args, state))
        check_targets(value, state, plan)
        targets_by_state[state["state"]] = {
            **value, "candidates": [{**t, "realized": objective(torch.tensor(
                t["carriers"][str(TASK_TIME)]))} for t in value["candidates"]]}
    all_states, summary, headroom, frontier = {}, {}, [], {}
    for seed in SEEDS:
        models = {arm: load_predictor(args, plan, seed, arm)[0] for arm in ARMS}
        control = load_control(args, plan, seed)[0]
        seed_rows = {name: {} for name in SYSTEMS}
        for state in states:
            targets = targets_by_state[state["state"]]
            realized = [t["realized"] for t in targets["candidates"]]
            flags = [diag.outcome_flags(t["timing"]) for t in targets["candidates"]]
            if seed == SEEDS[0]:
                regrets, informative = grid.normalized_regrets(
                    [{"accepted": True, "realized_count_cost": r} for r in realized])
                prior = next((i for i, t in enumerate(targets["candidates"])
                              if t["source"]["ordinal"] == diag.PRIOR_ORDINAL), None)
                headroom.append({"state": state["state"], "family": state["family"],
                                 "corpus": state["corpus"], "candidates": len(realized),
                                 "informative": informative,
                                 "prior_ordinal09_regret":
                                     None if prior is None else regrets[prior],
                                 "uniform_expected_regret": float(np.mean(regrets)),
                                 "dropped_candidates": len(state["dropped_candidates"]),
                                 "opportunities": {k: any(f[k] for f in flags) for k in
                                                   ("pig_removed", "pig_contact",
                                                    "block_contact")}})
            for system in SYSTEMS:
                arm, pair = SYSTEMS[system]
                records = [stored_record(args, plan, state, targets, seed, system, source,
                                         models[arm], control, objective)
                           for source in state["candidates"]]
                row = {"family": state["family"], "corpus": state["corpus"],
                       "task": diag.task_metrics([r["cost"] for r in records], realized),
                       "local_prediction_failures": 0,
                       "recursive": {}, "local": {},
                       "work": {"linear_macs": sum(r["prediction"]["linear_macs"]
                                                   for r in records),
                                "wall_seconds": sum(r["prediction"]["transition_wall_seconds"]
                                                    for r in records),
                                "controller_calls": sum(r["prediction"].get("controller_calls", 0)
                                                        for r in records),
                                "transition_calls": sum(
                                    r["prediction"]["transition_calls"] if pair is None else
                                    r["prediction"]["completed_steps"] // pair.delta
                                    for r in records),
                                "perception_seconds": targets["perception"]["wall_seconds"]}}
                if pair is not None:
                    row["work"]["local_linear_macs"] = sum(r["local"]["linear_macs"]
                                                           for r in records)
                    row["work"]["local_transition_calls"] = sum(r["local"]["calls"]
                                                                for r in records)
                    row["work"]["local_wall_seconds"] = sum(r["local"]["wall_seconds"]
                                                            for r in records)
                    row["local_prediction_failures"] = sum(
                        v == "nonfinite_local_prediction" for r in records
                        for v in r["local"]["unavailable"].values())
                    by_ordinal = {s["ordinal"]: t for s, t in
                                  zip(state["candidates"], targets["candidates"], strict=True)}
                    for t in TIMES:
                        row["recursive"][str(t)] = diag.average_errors(
                            [diag.field_errors(r["prediction"]["outputs"].get(str(t)),
                                               by_ordinal[s["ordinal"]]["carriers"].get(str(t)),
                                               objective)
                             for r, s in zip(records, state["candidates"], strict=True)])
                        row["local"][str(t)] = diag.average_errors(
                            [diag.field_errors(r["local"]["outputs"].get(str(t)),
                                               by_ordinal[s["ordinal"]]["carriers"].get(str(t)),
                                               objective)
                             for r, s in zip(records, state["candidates"], strict=True)])
                seed_rows[system][state["state"]] = row
        all_states[str(seed)] = seed_rows
        summary[str(seed)] = {name: diag.aggregate_states(list(rows.values()))
                              for name, rows in seed_rows.items()}
        totals = {}
        probe = torch.zeros(1, 236, device=args.device)
        probe_action = torch.zeros(1, 5, device=args.device)
        for system, (arm, pair) in SYSTEMS.items():
            pairs = ([PredictionPair(h, Abstraction.CONTINUOUS) for h in HORIZONS]
                     if pair is None else [pair])
            executed = {str(p.delta): active_capacity(models[arm], probe, probe_action, p)
                        for p in pairs}
            state_rows = seed_rows[system]
            totals[system] = {"arm": arm,
                              "parameters": parameter_count(models[arm]),
                              "controller_parameters": parameter_count(control)
                              if arm == "tawm" else 0,
                              "executed_operator_parameters_by_horizon": executed,
                              "per_seed": {str(seed): {
                                  "linear_macs": sum(r["work"]["linear_macs"]
                                                     for r in state_rows.values()),
                                  "mean_per_state_wall_seconds": float(np.mean(
                                      [r["work"]["wall_seconds"] + r["work"]["perception_seconds"]
                                       for r in state_rows.values()])),
                                  "total_wall_seconds_all_states": sum(
                                      r["work"]["wall_seconds"] for r in state_rows.values()),
                                  "transition_calls": sum(r["work"]["transition_calls"]
                                                          for r in state_rows.values()),
                                  "controller_calls": sum(r["work"]["controller_calls"]
                                                          for r in state_rows.values()),
                                  "candidates": sum(len(s["candidates"])
                                                    for s in states if not s["unsupported"]),
                                  "prediction_failure_states": sum(
                                      r["task"]["prediction_failure"]
                                      for r in state_rows.values())}}}
        for system, entry in totals.items():
            if system in frontier:
                frontier[system]["per_seed"].update(entry["per_seed"])
            else:
                frontier[system] = entry
    contrasts = []
    disclosure = ("descriptive paired bootstrap; continuous_h5 comparator retains issue-74 "
                  "selection optimism")
    ref_records = {name: reference_records(args, plan, name) for name in REFERENCE_SYSTEMS}
    for seed, rows in reference_rows(args, plan, targets_by_state, ref_records).items():
        all_states[seed].update(rows)
    ref_h5_records = ref_records["continuous_h5"]
    ref_walls = {seed: {state: targets_by_state[state]["perception"]["wall_seconds"]
                        + sum(record["prediction"]["transition_wall_seconds"]
                              for record in rows.values())
                        for state, rows in rows_by_state.items()}
                 for seed, rows_by_state in ref_h5_records.items()}
    corpus_ids = {corpus: [s["state"] for s in states if s["corpus"] == corpus]
                  for corpus in (N1_CORPUS, N2_CORPUS)}
    for spec in (plan["contrasts"]["q1_method_class"] + plan["contrasts"]["q1_decay_signature"]
                 + plan["contrasts"]["q2_work_reported"]
                 + plan["contrasts"]["q3_training_effect"]):
        if spec["tested"] not in all_states[str(SEEDS[0])] \
                or spec["reference"] not in all_states[str(SEEDS[0])]:
            raise ValueError("contrast spec references a system without per-state rows")
        variants = [(None, None)] + [(corpus, ids) for corpus, ids in corpus_ids.items()]
        for corpus, ids in variants:
            differences = paired_difference(all_states, spec["tested"], spec["reference"], ids)
            entry = {**spec, "positive_is_improvement": True,
                     "regret": grid.paired_interval(differences),
                     "paired_states": len(differences),
                     "scope": disclosure if spec["reference"] == "continuous_h5"
                     else "descriptive paired bootstrap over states"}
            if corpus is not None:
                entry["corpus"] = corpus
            if spec["kind"] == "work_reported_comparison" and corpus is None:
                tested_walls = [np.mean([all_states[str(seed)][spec["tested"]][s]["work"]
                                         ["wall_seconds"] + all_states[str(seed)][spec["tested"]][s]
                                         ["work"]["perception_seconds"]
                                         for s in all_states[str(seed)][spec["tested"]]])
                                for seed in SEEDS]
                entry["mean_per_state_wall_seconds"] = {
                    "tested": float(np.mean(tested_walls)),
                    "reference": float(np.mean([np.mean(list(ref_walls[str(seed)].values()))
                                                for seed in SEEDS])),
                    "note": "reference wall from frozen #77 records measured 2026-09-20; tested "
                            "walls measured under this ticket's exclusive GPU lock"}
            contrasts.append(entry)
    ref_regret_mean = float(np.mean([
        row["task"]["regret"] for seed in SEEDS
        for row in all_states[str(seed)]["continuous_h5"].values()]))
    result = {**common, "per_seed": summary, "per_state": all_states, "contrasts": contrasts,
              "headroom": headroom, "frontier": frontier,
              "reference_frontier_continuous_h5": {
                  "system": "continuous_h5",
                  "source": "frozen #77 records (N1 diagnostic + N2 zero-shot)",
                  "wall_source": "measured 2026-09-20 during the issue-77 runs; no GPU-lock "
                                 "regime was in force then; disclosed",
                  "mean_regret": ref_regret_mean,
                  "per_seed": {seed: {
                      "linear_macs": sum(rec["prediction"]["linear_macs"]
                                         for rows in rows_by_state.values()
                                         for rec in rows.values()),
                      "transition_calls": sum(rec["prediction"]["completed_steps"] // 5
                                              for rows in rows_by_state.values()
                                              for rec in rows.values()),
                      "mean_per_state_wall_seconds": float(np.mean(
                          list(ref_walls[seed].values()))),
                      "candidates": sum(len(rows) for rows in rows_by_state.values())}
                      for seed, rows_by_state in ref_h5_records.items()}},
              "reference_issue77_contrasts": read(N1_DIAG / "summary.json")["contrasts"],
              "frontier_complete": True,
              "independent_state_counts": {
                  "n1": sum(1 for s in states if s["corpus"] == N1_CORPUS),
                  "n2": sum(1 for s in states if s["corpus"] == N2_CORPUS)},
              "mean_prior_regret": float(np.mean([r["prior_ordinal09_regret"] for r in headroom
                                                  if r["prior_ordinal09_regret"] is not None]))
              if any(r["prior_ordinal09_regret"] is not None for r in headroom) else None,
              "mean_uniform_expected_regret": float(np.mean([r["uniform_expected_regret"]
                                                             for r in headroom]))}
    result["dispositions"] = dispositions(result, True)
    return result


def compact_report(result):
    return {k: v for k, v in result.items() if k != "per_state"}


def comparisons_csv(result):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(("seed", "system", "elapsed_observed_frames", "kind", "metric", "value",
                     "available_states"))
    for seed, systems in result.get("per_seed", {}).items():
        for name, system in systems.items():
            for kind, key in (("recursive", "curves"), ("local", "local_curves")):
                for t, metrics in system.get(key, {}).items():
                    for metric, value in metrics.items():
                        if metric != "available_states":
                            writer.writerow((seed, name, t, kind, metric, value,
                                             metrics["available_states"]))
    writer.writerow(())
    writer.writerow(("seed", "system", "mean_regret", "top1_fraction", "top3_fraction",
                     "prediction_failure_states", "transition_linear_macs"))
    for seed, systems in result.get("per_seed", {}).items():
        for name, system in systems.items():
            writer.writerow((seed, name, system["mean_regret"], system["top1_fraction"],
                             system["top3_fraction"], system["prediction_failure_states"],
                             system["transition_linear_macs"]))
    writer.writerow(())
    writer.writerow(("contrast_tested", "reference", "kind", "corpus", "mean_regret_difference",
                     "interval_low", "interval_high", "paired_states", "scope"))
    for contrast in result.get("contrasts", []):
        interval = contrast["regret"]["descriptive_95_percent_interval"]
        writer.writerow((contrast["tested"], contrast["reference"], contrast["kind"],
                         contrast.get("corpus", "pooled"), contrast["regret"]["mean"],
                         interval[0], interval[1], contrast.get("paired_states"),
                         contrast["scope"]))
    writer.writerow(())
    writer.writerow(("disposition_question", "token"))
    for question, token in result.get("dispositions", {}).items():
        writer.writerow((question, token))
    return stream.getvalue()


def findings_md(result):
    lines = ["# Issue-78 external temporal-adaptation baselines - findings", ""]
    lines.append(f"Diagnostics complete: {result['diagnostics_complete']}. "
                 f"Claim boundary: {result['claim_boundary']}.")
    lines.append("")
    lines.append("Declared endpoint semantics: " + result["definitions"]["endpoint_semantics"])
    lines.append("")
    lines.append("Comparator disclosure: continuous_h5 retains issue-74 selection optimism "
                 "(selected_continuous_policy=continuous_h5, selection_optimism=true, selected "
                 "on development evidence); disclosed in every contrast that uses it. Inference "
                 "is NOT equalized across arms; the full frontier is reported.")
    lines.append("")
    lines.append("Ports: TAWM (arXiv 2506.08441; reference code consulted at pinned commit "
                 f"{TAWM_PIN}, never run at its original scale) with declared deviations "
                 "(h-conditioning on the frozen grid {1,5,15}, uniform mixed-h sampler at "
                 "per-update granularity, existing learned horizon controller for the adaptive "
                 "evaluation); VLWM (arXiv 2606.21775) as a paper-fidelity port of Eqs. 3-6, no "
                 "reference code consulted (7-way no-code search in plan.json), with the "
                 "single-shot mapping deviation (launch action + k-1 null actions). Structural "
                 "equivalence disclosure: on this carrier the TAWM cell shares the #77 continuous "
                 "reference cell's recipe (same seed-initialized carrier, minibatch stream, loss, "
                 "optimizer, horizon counts) and differs only in within-cycle horizon ordering "
                 "(round-robin 1,5,15 versus the reference's blocked 1,1,1,5,5,5,15,15,15); the "
                 "tawm vs continuous contrasts therefore estimate horizon-schedule-ordering "
                 "effects under an identical budget, and the vlwm arm carries the independent "
                 "mechanism contrast.")
    lines.append("")
    if not result.get("per_seed"):
        lines.append("Diagnostics incomplete; no outcome numbers reported.")
        for failure in result.get("typed_failures", []):
            lines.append(f"- typed terminal failure: {failure}")
        return "\n".join(lines) + "\n"
    lines.append("## Headroom (frozen states)")
    lines.append("")
    lines.append("| State | Corpus | Family | Candidates | Informative | Prior(ordinal09) "
                 "regret | Uniform regret | Dropped |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in result["headroom"]:
        prior = "n/a" if row["prior_ordinal09_regret"] is None else \
            f"{row['prior_ordinal09_regret']:.4f}"
        lines.append(f"| {row['state']} | {row['corpus']} | {row['family']} | "
                     f"{row['candidates']} | {row['informative']} | {prior} | "
                     f"{row['uniform_expected_regret']:.4f} | {row['dropped_candidates']} |")
    lines.append("")
    lines.append("## Action ranking (mean over the frozen states; seeds listed individually)")
    lines.append("")
    lines.append("| System | Seed | Mean regret | Top1 | Top3 | Prediction failure states |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for seed, systems in result["per_seed"].items():
        for name, system in systems.items():
            lines.append(f"| {name} | {seed} | {system['mean_regret']:.4f} | "
                         f"{system['top1_fraction']:.3f} | {system['top3_fraction']:.3f} | "
                         f"{system['prediction_failure_states']} |")
    lines.append("")
    lines.append("## Paired contrasts (positive favors the tested system; DESCRIPTIVE)")
    lines.append("")
    lines.append("| Tested | Reference | Kind | Mean regret difference | Descriptive 95% interval |")
    lines.append("| --- | --- | --- | --- | --- |")
    for contrast in result["contrasts"]:
        interval = contrast["regret"]["descriptive_95_percent_interval"]
        lines.append(f"| {contrast['tested']} | {contrast['reference']} | {contrast['kind']} | "
                     f"{contrast['regret']['mean']:+.4f} | [{interval[0]:+.4f}, "
                     f"{interval[1]:+.4f}] |")
    lines.append("")
    lines.append("## Work-reported frontier (inference NOT equalized)")
    lines.append("")
    lines.append("| System | Parameters | Controller params | Mean per-state wall s | "
                 "Linear MACs (seed mean) | Transition calls (seed mean) | Candidates/seed |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, entry in result["frontier"].items():
        per_seed = entry["per_seed"]
        wall = float(np.mean([v["mean_per_state_wall_seconds"] for v in per_seed.values()]))
        macs = float(np.mean([v["linear_macs"] for v in per_seed.values()]))
        calls = float(np.mean([v["transition_calls"] for v in per_seed.values()]))
        candidates = max(v["candidates"] for v in per_seed.values())
        lines.append(f"| {name} | {entry['parameters']} | {entry['controller_parameters']} | "
                     f"{wall:.3f} | {macs:.0f} | {calls:.0f} | {candidates} |")
    ref = result["reference_frontier_continuous_h5"]
    per_seed = ref["per_seed"]
    lines.append(f"| continuous_h5 (frozen #77 records) | #77 checkpoints | #77 checkpoints | "
                 f"{float(np.mean([v['mean_per_state_wall_seconds'] for v in per_seed.values()])):.3f}"
                 f" | {float(np.mean([v['linear_macs'] for v in per_seed.values()])):.0f} | "
                 f"{float(np.mean([v['transition_calls'] for v in per_seed.values()])):.0f} | "
                 f"{max(v['candidates'] for v in per_seed.values())} |")
    lines.append("")
    lines.append("Wall provenance: " + result["limitations"][4])
    lines.append("")
    lines.append("## Per-corpus contrasts (normal-mechanics N1 vs novelty N2; DESCRIPTIVE)")
    lines.append("")
    lines.append("| Corpus | Tested | Reference | Kind | Mean difference | Descriptive 95% interval |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for contrast in result["contrasts"]:
        if "corpus" not in contrast or contrast["kind"] not in \
                ("decay_signature", "training_effect_analog", "external_vs_fixed"):
            continue
        interval = contrast["regret"]["descriptive_95_percent_interval"]
        corpus = "N1 normal-mechanics" if contrast["corpus"] == "issue-77-n1-diagnostic-v1" \
            else "N2 novelty"
        lines.append(f"| {corpus} | {contrast['tested']} | {contrast['reference']} | "
                     f"{contrast['kind']} | {contrast['regret']['mean']:+.4f} | "
                     f"[{interval[0]:+.4f}, {interval[1]:+.4f}] |")
    lines.append("")
    lines.append("## Dispositions")
    lines.append("")
    for question, token in result["dispositions"].items():
        lines.append(f"- **{question}: `{token}`**")
    lines.append("")
    lines.append("Reference #77 hybrid contrasts (frozen, quoted for context): the h=1 training "
                 "effect hybrid_continuous_h1 vs continuous_h1 = +0.0887 [+0.0155, +0.2118]; "
                 "symbolic-execution effects at h=1 straddle zero (micro +0.0348 [-0.0102, "
                 "+0.1147]; macro +0.0239 [-0.0199, +0.0938]). External temporal-only arms have "
                 "no symbolic execution by construction.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    for item in result["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- operator modes

def smoke(args):
    plan = make_plan(args)
    plan77 = train.load_plan(dynamics_args(args))
    source = old.load_plan(train.source_args(dynamics_args(args)))
    began = time.monotonic()
    result = {"production_evidence": False, "records": 0, "cells": {}}
    with tempfile.TemporaryDirectory(prefix="novphy-issue78-smoke-") as temporary:
        test = copy.copy(args)
        test.output = Path(temporary)
        plan = copy.deepcopy(plan)
        plan["identity"] += ":smoke"
        plan["training"]["steps"] = 18
        plan["controller"]["steps_per_round"] = 6
        for arm in ARMS:
            train_cell(test, plan, plan77, source, SEEDS[0], arm, stop_after=9)
            train_cell(test, plan, plan77, source, SEEDS[0], arm, stop_after=18)
            saved = torch.load(cell(test, SEEDS[0], arm) / "predictor-progress.pt",
                               weights_only=True, map_location="cpu")
            model = new_model(plan).to(args.device)
            model.load_state_dict(saved["model"], strict=True)
            names = set(dict(model.named_parameters()))
            if set(saved["gradient_parameters"]) != names \
                    or set(saved["changed_parameters"]) != names:
                raise ValueError("smoke found untrained parameter tensors")
            if saved["pair_counts"] != horizon_counts(plan, arm):
                raise ValueError(f"smoke {arm} horizon schedule differs from the frozen counts")
            result["cells"][arm] = {"resume_step": saved["step"],
                                    "pair_counts": saved["pair_counts"]}
        train_control(test, plan, plan77, source, SEEDS[0], indices=(5,))
        control, _ = load_control(test, plan, SEEDS[0])
        models = {arm: load_predictor(test, plan, SEEDS[0], arm)[0] for arm in ARMS}
        objective = base.TaskObjective.from_vocabulary(plan["contract"]["vocabulary"])
        state = copy.deepcopy(next(s for s in plan["evaluation"]["states_membership"]
                                   if s["candidates"]))
        state["candidates"] = state["candidates"][:2]
        targets = copy.deepcopy(read(target_path(args, state)))
        keep = {c["ordinal"] for c in state["candidates"]}
        targets["candidates"] = [t for t in targets["candidates"]
                                 if t["source"]["ordinal"] in keep]
        check_targets(targets, state, plan)
        for system in SYSTEMS:
            arm = SYSTEMS[system][0]
            for candidate in state["candidates"]:
                record = stored_record(test, plan, state, targets, SEEDS[0], system, candidate,
                                       models[arm], control, objective)
                result["records"] += 1
                log(f"smoke state={state['state']} system={system} "
                    f"candidate={candidate['ordinal']} failure={record['prediction']['failure']}")
        result["wall_seconds"] = time.monotonic() - began
        result["memory"] = memory(args.device)
    write(args.output / "smoke.json", result)
    log(f"real smoke complete records={result['records']} wall={result['wall_seconds']:.1f}s; "
        "temporary weights removed, no production training")


def record_failure(args, cell_name, error):
    path = args.output / "failures.json"
    failures = read(path) if path.exists() else []
    failures.append({"cell": cell_name, "kind": "typed_terminal_failure", "error": str(error)})
    write(path, failures)


def ledger_add(args, key, seconds):
    """Cumulative active-GPU-time ledger; the frozen allowance is global, not per phase."""
    path = args.output / "allowance-ledger.json"
    entries = read(path) if path.exists() else {}
    entries[key] = float(entries.get(key, 0.0)) + float(seconds)
    write(path, entries)
    return entries


def ledger_total(args):
    entries = read(args.output / "allowance-ledger.json") \
        if (args.output / "allowance-ledger.json").exists() else {}
    return float(sum(entries.values())), entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("dry-run", "smoke-test", "prepare", "train", "run-evaluation", "publish",
                 "validate"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dynamics", type=Path, default=DYNAMICS)
    args = parser.parse_args()
    args.issue71 = old.ROOT
    args.parser = old.repair.ROOT
    torch.set_num_threads(2)
    try:
        if args.dry_run:
            plan = make_plan(args)
            states = plan["evaluation"]
            log(f"no-write dry run: arms={plan['arms']} seeds={plan['seeds']} "
                f"steps={plan['training']['steps']} systems={len(plan['systems']['inventory'])} "
                f"states={states['states']} ({states['evaluable_states']} evaluable, "
                f"{states['unsupported_states']} unsupported) candidates={states['candidates']}")
            log(f"tawm schedule counts={plan['training']['horizon_schedule']['tawm']['counts']}; "
                f"vlwm curriculum counts={plan['training']['horizon_schedule']['vlwm']['counts']}")
            log(f"allowance={plan['compute']['allowance']}; no files written")
            return 0
        if args.smoke_test:
            smoke(args)
            return 0
        if args.prepare:
            if (args.output / "plan.json").exists():
                load_plan(args)
            else:
                write(args.output / "plan.json", make_plan(args))
            log("issue-78 plan frozen; no training or scoring started")
            return 0
        plan = load_plan(args)
        if args.train:
            plan77 = train.load_plan(dynamics_args(args))
            source = old.load_plan(train.source_args(dynamics_args(args)))
            with eb.gpu_exclusive():
                for seed in SEEDS:
                    for arm in ARMS:
                        log(f"training cell seed={seed} arm={arm} start")
                        try:
                            cell_began = time.monotonic()
                            train_cell(args, plan, plan77, source, seed, arm)
                            if arm == "tawm":
                                train_control(args, plan, plan77, source, seed)
                            cell_wall = time.monotonic() - cell_began
                        except (ValueError, OSError) as error:
                            record_failure(args, f"train/{seed}/{arm}", error)
                            raise
                        ledger_add(args, f"train/{seed}/{arm}", cell_wall)
                        total, _ = ledger_total(args)
                        if total > ALLOWANCE_GPU_SECONDS:
                            message = (f"cumulative active GPU time {total:.0f}s exceeded the "
                                       f"frozen {ALLOWANCE_GPU_SECONDS}s allowance")
                            record_failure(args, f"train/{seed}/{arm}", ValueError(message))
                            raise ValueError(message)
            log(f"baseline fits complete memory={memory(args.device)}")
        elif args.run_evaluation:
            with eb.gpu_exclusive():
                eval_began = time.monotonic()
                run_evaluation(args, plan)
                ledger_add(args, "run-evaluation", time.monotonic() - eval_began)
                total, _ = ledger_total(args)
                if total > ALLOWANCE_GPU_SECONDS:
                    message = (f"cumulative active GPU time {total:.0f}s exceeded the frozen "
                               f"{ALLOWANCE_GPU_SECONDS}s allowance")
                    record_failure(args, "run-evaluation", ValueError(message))
                    raise ValueError(message)
            log(f"evaluation complete memory={memory(args.device)}")
        elif args.publish:
            result = publication(args, plan)
            write(args.output / "summary.json", compact_report(result))
            (args.output / "comparisons.csv").write_text(comparisons_csv(result))
            (args.output / "findings.md").write_text(findings_md(result))
            log(f"published diagnostics_complete={result['diagnostics_complete']} "
                f"dispositions={result['dispositions']}")
        else:
            result = publication(args, plan)
            if (compact_report(result) != read(args.output / "summary.json")
                    or comparisons_csv(result).encode("utf-8")
                    != (args.output / "comparisons.csv").read_bytes()
                    or findings_md(result) != (args.output / "findings.md").read_text()):
                raise ValueError("published issue-78 tables differ from bound source evidence")
            log("exact saved-evidence validation passed")
        return 0
    except (ValueError, OSError) as error:
        log(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
