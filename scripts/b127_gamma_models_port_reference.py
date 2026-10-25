"""#134 T3 γ-models G-port: the official jannerm/gamma-models code (the Colab notebook's training cell body) vs the
NovPhy adapter (``world_model.baselines.gamma_models.novphy``).

Both sides run in the ``novphy`` env (torch 2.13, CPU, float32, 4 threads), in separate processes:

``--reference WORK``
    Imports the official clone (``GAMMA_REF``, default /mnt/array/sukaih/novphy-baselines-src/gamma-models@bb23e06)
    as the notebook does (``sys.path`` = the clone; ``import gamma``). Per arm: ``torch.manual_seed(SEED_INIT)``, the
    notebook's model cell (``make_conditional_flow``, ``copy.deepcopy`` target + the torch 2.13 prior shim,
    ``BootstrapTarget``), the training cell's optimizer and criterion, then the training cell's loop body verbatim at
    loop indices ``STEP_INDICES`` = (0, 2000) (the first burn-in update, then the first post-burn-in update), torch
    seeded before each. The batch is built the official way: ``format_batch(batch, policy)`` on a dict of
    ``observations / actions / next_observations`` with the arm's policy. Recorded: initial state, every step's
    bootstrapped samples, target and model log-densities, loss, gradients, parameters, target parameters and Adam
    slots after the step; then the log-likelihood of fixed points and ``model.sample`` with a seeded global generator.

``--compare WORK``
    The adapter: ``novphy.Trainer`` (seeded the same way; its initial state must equal the official one), batches from
    ``novphy.transition_batch`` for the carrier arms (rows from ``novphy.transition_rows``), ``Trainer.train_step`` at
    the same indices, ``model.log_prob`` and ``novphy.sample_occupancy`` with the same uniform draw from a seeded CPU
    generator. Tolerances: #123's (``run_hjepa_comparison.GPORT_TOLERANCE``: loss / grad-norm relative 1e-4, per-tensor
    gradient relative L2 1e-3, parameters after each step |Δ| ≤ 1e-7 on ≥ 99.9 % of elements and ≤ 2 lr + 1e-7
    everywhere), forward / sampling max abs ≤ 1e-5. Then three injected defects on the carrier arm g0.99 (bootstrap
    discount used as the sample discount after burn-in; the launch as the next action; τ = 0.05) must each fail.
    Writes ``OUTPUT/gport/gport.json`` (schema ``issue_127_gport_v1``).

``--run``: both, as subprocesses (WORK = OUTPUT/gport/work, deleted after a pass).

Arms: ``pendulum-official`` (the notebook's geometry: 3-d state, 1-d action, the released replay pool's first rows,
a fixed stand-in policy for the next actions; rlkit is not in novphy), and ``car-g0.95`` / ``car-g0.978`` /
``car-g0.99`` (236-d carriers, 5-d launch, the three pre-declared discounts; synthetic carrier-shaped episodes:
5 shots x 46 states, values in [0, 1] with half the dimensions constant, as the E99 carriers; batch 1024).
The notebook has no learning-rate schedule: both steps run at lr 1e-4 (the step-0 LR is the positive LR).
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = Path(os.environ.get("GAMMA_REF", "/mnt/array/sukaih/novphy-baselines-src/gamma-models@bb23e06"))
COMMIT = "bb23e06753717a9255065355b5f6ab77278f3305"
UPSTREAM = "https://github.com/jannerm/gamma-models"
SCHEMA = "issue_127_gport_v1"
OUTPUT = ROOT / ".local-artifacts" / "issue-134-gamma-models-v1"
VENDOR = ROOT / "world_model/baselines/gamma_models/vendor"
ADAPTER = ROOT / "world_model/baselines/gamma_models/novphy.py"
SEED_INIT, SEED_INPUT, SEED_STEP1, SEED_STEP2, SEED_SAMPLE = 20261025, 7, 11, 13, 17
STEP_INDICES = (0, 2000)
THREADS = 4
N_SAMPLE_CHECK = 256
LAUNCHES = [[-0.25, 0.10, 1.0, 0.0, 1.0], [-0.08, 0.01, 1.0, 0.0, 1.0], [-0.17, 0.06, 1.0, 0.0, 1.0],
            [-0.33, 0.17, 1.0, 0.0, 1.0], [-0.12, 0.11, 1.0, 0.0, 1.0]]
NOTEBOOK_ARGS = {"hidden_dims": [256, 256, 256], "sigma": 0.1, "batch_size": 1024, "lr": 1e-4, "decay": 1e-5,
                 "tau": 0.005, "discount": 0.99, "sample_discount": 0.9, "burnin_discount": 0.5, "n_burnin": 2000}
ARMS = {
    "pendulum-official": {"dim": 3, "act": 1, "discount": 0.99, "data": "pendulum"},
    "car-g0.95": {"dim": 236, "act": 5, "discount": 0.95, "configuration": "g0.95", "data": "car"},
    "car-g0.978": {"dim": 236, "act": 5, "discount": 1.0 - 1.0 / 45.0, "configuration": "g0.978", "data": "car"},
    "car-g0.99": {"dim": 236, "act": 5, "discount": 0.99, "configuration": "g0.99", "data": "car"},
}
TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
             "grad_floor_of_global_norm": 1e-6, "param_close_abs": 1e-7, "param_close_share": 0.999,
             "param_max_abs_over_lr": 2.0, "forward_max_abs": 1e-5, "sample_max_abs": 1e-5}
DEFECTS = ("sample_discount_equals_discount", "launch_as_next_action", "tau_0.05")
DEFECT_ARM = "car-g0.99"


def sha256_of(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def discount_of(arm):
    return float(ARMS[arm]["discount"])


# ---------------------------------------------------------------------------------------------------------------
# fixed inputs (identical on both sides; numpy only)
# ---------------------------------------------------------------------------------------------------------------

def car_episodes():
    """[5, 46, 236] carrier-shaped episodes (values in [0, 1]; dims 118.. constant 0 as empty slots) and launches."""
    rng = np.random.default_rng(SEED_INPUT)
    base = rng.random((5, 1, 236)).astype(np.float32)
    drift = np.cumsum(rng.normal(0, 0.02, (5, 46, 236)), axis=1).astype(np.float32)
    episodes = np.clip(base + drift, 0, 1).astype(np.float32)
    episodes[:, :, 118:] = 0.0
    episodes[2, 30:] = episodes[2, 29]  # a shot ending early: final carrier repeated
    return episodes, np.asarray(LAUNCHES, dtype=np.float32)


def car_rows(batch=1024):
    rng = np.random.default_rng(SEED_INPUT + 1)
    flat = rng.integers(0, 5 * 46, size=batch)
    flat[:8] = [0, 46, 92, 45, 91, 137, 1, 47]  # launches (k = 0) and absorbing rows (k = 45) present
    return flat // 46, flat % 46


def pendulum_batch(batch=1024):
    with open(OFFICIAL / "data/pools/pendulum.pkl", "rb") as f:
        fields = pickle.load(f)
    obs = fields["observations"][:batch].astype(np.float32)
    act = fields["actions"][:batch].astype(np.float32)
    nxt = fields["next_observations"][:batch].astype(np.float32)
    return obs, act, nxt


def pendulum_policy(observations):
    """Fixed stand-in next-action policy (both sides): tanh(0.5 θdot) as a [B, 1] array."""
    observations = np.asarray(observations.detach().cpu() if torch.is_tensor(observations) else observations)
    return np.tanh(0.5 * observations[:, 2:3]).astype(np.float32)


def fixed_points(dim, act, n=64):
    g = torch.Generator().manual_seed(SEED_INPUT + 2)
    x = torch.rand(n, dim, generator=g)
    s = torch.rand(n, dim, generator=g)
    a = torch.rand(n, act, generator=g) * 2 - 1
    return x, s, a


# ---------------------------------------------------------------------------------------------------------------
# --reference (the official clone)
# ---------------------------------------------------------------------------------------------------------------

def reference(work):
    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(OFFICIAL))
    from gamma.flows import make_conditional_flow
    from gamma.td.distributions import BootstrapTarget
    from gamma.td.utils import soft_update_from_to, format_batch
    from gamma.utils import set_device
    set_device("cpu")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for arm, spec in ARMS.items():
        discount = discount_of(arm)
        args = dict(NOTEBOOK_ARGS, discount=discount)
        if spec["data"] == "pendulum":
            obs, act, nxt = pendulum_batch()
            batch = {"observations": obs, "actions": act, "next_observations": nxt}
            policy = pendulum_policy
        else:
            episodes, launches = car_episodes()
            shot, k = car_rows()
            batch = {"observations": episodes[shot, k],
                     "actions": launches[shot] * (k == 0).astype(np.float32)[:, None],
                     "next_observations": episodes[shot, np.minimum(k + 1, 45)]}

            def policy(observations):  # the NovPhy policy after the launch: the zero no-op
                return np.zeros((len(observations), 5), dtype=np.float32)
        observation_dim, action_dim = spec["dim"], spec["act"]
        # ---- model cell ----
        torch.manual_seed(SEED_INIT)
        condition_dims = {"s": observation_dim, "a": action_dim}
        model = make_conditional_flow(observation_dim, args["hidden_dims"], condition_dims)
        target_model = copy.deepcopy(model)
        target_model.prior = model.prior  # torch 2.13 deepcopy shim (VENDOR.json runtime note)
        bootstrap = BootstrapTarget(target_model, args["discount"])
        init_state = {k: v.clone() for k, v in model.state_dict().items()}
        # ---- training cell ----
        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["decay"])
        criterion = torch.nn.MSELoss()
        steps = []
        for seed, i in zip((SEED_STEP1, SEED_STEP2), STEP_INDICES):
            torch.manual_seed(seed)
            if i < args["n_burnin"]:
                bootstrap.update_discount(args["burnin_discount"])
                sample_discount = args["burnin_discount"]
            else:
                bootstrap.update_discount(args["discount"])
                sample_discount = args["sample_discount"]
            condition_dict, next_condition_dict = format_batch(batch, policy)
            bootstrap.update_p(next_condition_dict["s"], sigma=args["sigma"])
            samples = bootstrap.sample(args["batch_size"], condition_dict, next_condition_dict,
                                       discount=sample_discount)
            log_prob_target = bootstrap.log_prob(samples, condition_dict, next_condition_dict)
            log_prob_model = model.log_prob(samples, condition_dict)
            loss = criterion(log_prob_model, log_prob_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            soft_update_from_to(model, target_model, args["tau"])
            steps.append({"index": i, "lr": optimizer.param_groups[0]["lr"],
                          "condition": {k: v.clone() for k, v in condition_dict.items()},
                          "next_condition": {k: v.clone() for k, v in next_condition_dict.items()},
                          "samples": samples.clone(), "log_prob_target": log_prob_target.detach().clone(),
                          "log_prob_model": log_prob_model.detach().clone(), "loss": float(loss),
                          "grads": {n: p.grad.detach().clone() for n, p in model.named_parameters()},
                          "post_state": {k: v.clone() for k, v in model.state_dict().items()},
                          "target_state": {k: v.clone() for k, v in target_model.state_dict().items()},
                          "adam": {n: {k: v.clone() for k, v in optimizer.state[p].items()}
                                   for n, p in model.named_parameters()}})
        x, s, a = fixed_points(observation_dim, action_dim)
        with torch.no_grad():
            log_prob = model.log_prob(x, {"s": s, "a": a})
            target_log_prob = target_model.log_prob(x, {"s": s, "a": a})
            cond = {"s": s[:1].repeat(N_SAMPLE_CHECK, 1), "a": a[:1].repeat(N_SAMPLE_CHECK, 1)}
            torch.manual_seed(SEED_SAMPLE)
            sampled = model.sample(N_SAMPLE_CHECK, cond)
        torch.save({"arm": arm, "discount": discount, "args": args, "init_state": init_state, "steps": steps,
                    "log_prob": log_prob, "target_log_prob": target_log_prob, "sampled": sampled,
                    "parameters": sum(p.numel() for p in model.parameters()), "torch": torch.__version__},
                   work / f"ref-{arm}.pt")
        print(f"[reference] {arm}: losses {[s_['loss'] for s_ in steps]}", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------------------------
# --compare (the adapter)
# ---------------------------------------------------------------------------------------------------------------

def _max_abs(a, b):
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


def check_step(ref, out, trainer, lr):
    tol = TOLERANCE
    model = trainer.model
    loss_rel = abs(float(out["loss"]) - ref["loss"]) / max(1.0, abs(ref["loss"]))
    grads = {n: p.grad.detach() for n, p in model.named_parameters()}
    ref_norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in ref["grads"].values())))
    our_norm = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads.values())))
    floor = tol["grad_floor_of_global_norm"] * ref_norm
    rel = [float((grads[k] - v).norm() / v.norm()) for k, v in ref["grads"].items() if float(v.norm()) > floor]
    null = [float((grads[k] - v).norm()) for k, v in ref["grads"].items() if float(v.norm()) <= floor]
    post = model.state_dict()
    deltas = torch.cat([(post[k].double() - v.double()).abs().flatten() for k, v in ref["post_state"].items()
                        if v.is_floating_point() and v.numel()])
    target = trainer.target.state_dict()
    target_delta = max(_max_abs(target[k], v) for k, v in ref["target_state"].items() if v.is_floating_point())
    adam = trainer.optimizer
    named = dict(model.named_parameters())
    slot = max(_max_abs(adam.state[named[n]][k], v) for n, st in ref["adam"].items() for k, v in st.items()
               if torch.is_tensor(v) and v.is_floating_point())
    forward = max(_max_abs(out["samples"], ref["samples"]), _max_abs(out["log_prob_target"], ref["log_prob_target"]),
                  _max_abs(out["log_prob_model"], ref["log_prob_model"]))
    close_share = float((deltas <= tol["param_close_abs"]).double().mean())
    max_delta = float(deltas.max())
    result = {"index": ref["index"], "lr": lr, "lr_equal": lr == ref["lr"], "loss": [ref["loss"], float(out["loss"])],
              "loss_rel": loss_rel, "grad_norm": [ref_norm, our_norm],
              "grad_norm_rel": abs(our_norm - ref_norm) / max(ref_norm, 1e-30), "grad_max_rel_l2": max(rel or [0.0]),
              "null_gradient_max_abs_l2": max(null or [0.0]), "forward_max_abs": forward,
              "param_close_share": close_share, "param_max_abs_delta": max_delta,
              "target_param_max_abs_delta": target_delta, "adam_slot_max_abs_delta": slot}
    result["pass"] = bool(result["lr_equal"] and loss_rel <= tol["loss_relative"]
                          and result["grad_norm_rel"] <= tol["grad_norm_relative"]
                          and result["grad_max_rel_l2"] <= tol["grad_tensor_relative_l2"]
                          and forward <= tol["forward_max_abs"]
                          and close_share >= tol["param_close_share"]
                          and max_delta <= tol["param_max_abs_over_lr"] * lr + tol["param_close_abs"]
                          and target_delta <= tol["param_max_abs_over_lr"] * lr + tol["param_close_abs"])
    return result


def adapter_batches(arm, novphy, defect=None):
    spec = ARMS[arm]
    if spec["data"] == "pendulum":
        obs, act, nxt = pendulum_batch()
        cond = {"s": torch.from_numpy(obs), "a": torch.from_numpy(act)}
        nxt_t = torch.from_numpy(nxt)
        return cond, {"s": nxt_t, "a": torch.from_numpy(pendulum_policy(nxt))}
    episodes, launches = car_episodes()
    shot, k = car_rows()
    cond, nxt = novphy.transition_batch(torch.from_numpy(episodes), torch.from_numpy(launches), shot, k)
    if defect == "launch_as_next_action":
        nxt["a"] = torch.from_numpy(launches)[torch.as_tensor(shot)] * torch.as_tensor(k == 0).float()[:, None]
    return cond, nxt


def run_adapter(arm, ref, novphy, defect=None):
    spec = ARMS[arm]
    config = (novphy.config_of(spec["configuration"]) if "configuration" in spec
              else novphy.config_of("g0.99", {"discount": spec["discount"]}))
    if config["discount"] != ref["discount"]:
        raise ValueError(f"{arm}: adapter discount {config['discount']} != reference {ref['discount']}")
    if defect == "tau_0.05":
        config["tau"] = 0.05
    torch.manual_seed(SEED_INIT)
    trainer = novphy.Trainer(config, "cpu", spec["dim"], {"s": spec["dim"], "a": spec["act"]})
    if defect == "sample_discount_equals_discount":
        trainer.discounts = lambda step: ((config["burnin_discount"],) * 2 if step < config["n_burnin"]
                                          else (config["discount"],) * 2)
    init = trainer.model.state_dict()
    init_equal = (set(init) == set(ref["init_state"])
                  and all(torch.equal(init[k], v) for k, v in ref["init_state"].items()))
    cond, nxt = adapter_batches(arm, novphy, defect)
    steps = []
    batch_equal = []
    for seed, i, r in zip((SEED_STEP1, SEED_STEP2), STEP_INDICES, ref["steps"]):
        batch_equal.append(all(torch.equal(cond[k], r["condition"][k]) for k in ("s", "a"))
                           and all(torch.equal(nxt[k], r["next_condition"][k]) for k in ("s", "a")))
        torch.manual_seed(seed)
        out = trainer.train_step(cond, nxt, i)
        steps.append(check_step(r, out, trainer, trainer.optimizer.param_groups[0]["lr"]))
    x, s, a = fixed_points(spec["dim"], spec["act"])
    with torch.no_grad():
        log_prob = trainer.model.log_prob(x, {"s": s, "a": a})
        target_log_prob = trainer.target.log_prob(x, {"s": s, "a": a})
    u = torch.rand(N_SAMPLE_CHECK, spec["dim"], generator=torch.Generator().manual_seed(SEED_SAMPLE))
    sampled, failed = novphy.sample_occupancy(trainer.model, s[:1], a[:1], u)
    forward = {"log_prob_max_abs": _max_abs(log_prob, ref["log_prob"]),
               "target_log_prob_max_abs": _max_abs(target_log_prob, ref["target_log_prob"]),
               "sample_max_abs": _max_abs(sampled[0], ref["sampled"]), "sampling_failed": bool(failed.any())}
    forward["pass"] = bool(forward["log_prob_max_abs"] <= TOLERANCE["forward_max_abs"]
                           and forward["target_log_prob_max_abs"] <= TOLERANCE["forward_max_abs"]
                           and forward["sample_max_abs"] <= TOLERANCE["sample_max_abs"] and not failed.any())
    result = {"arm": arm, "discount": ref["discount"], "config": {k: v for k, v in config.items()},
              "official_parameters": ref["parameters"], "adapter_parameters": novphy.parameter_count(trainer.model),
              "init_state_equal": bool(init_equal), "batch_equal": batch_equal, "steps": steps, "forward": forward}
    result["pass"] = bool(init_equal and all(batch_equal) and all(s_["pass"] for s_ in steps) and forward["pass"]
                          and result["official_parameters"] == result["adapter_parameters"])
    return result


def compare(work):
    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(ROOT))
    from world_model.baselines.gamma_models import novphy
    work = Path(work)
    arms = {}
    for arm in ARMS:
        ref = torch.load(work / f"ref-{arm}.pt", map_location="cpu", weights_only=False)
        arms[arm] = run_adapter(arm, ref, novphy)
        print(f"[compare] {arm}: pass {arms[arm]['pass']}; losses {[s['loss'] for s in arms[arm]['steps']]}; "
              f"param share {[s['param_close_share'] for s in arms[arm]['steps']]}", flush=True)
    ref = torch.load(work / f"ref-{DEFECT_ARM}.pt", map_location="cpu", weights_only=False)
    defects = {}
    for defect in DEFECTS:
        r = run_adapter(DEFECT_ARM, ref, novphy, defect)
        defects[defect] = {"detected": not r["pass"], "failed_checks": [
            f"step{j}" for j, s in enumerate(r["steps"]) if not s["pass"]] + (["batch"] if not all(r["batch_equal"])
                                                                              else [])}
        print(f"[compare] injected defect {defect}: detected {defects[defect]['detected']}", flush=True)
    vendor = json.loads((VENDOR / "VENDOR.json").read_text())
    vendor_ok = {rel: sha256_of(VENDOR / rel) == entry["vendored_sha256"] for rel, entry in vendor["files"].items()}
    official = {rel: sha256_of(OFFICIAL / entry["upstream_path"]) == entry["upstream_sha256"]
                for rel, entry in vendor["files"].items() if entry["upstream_path"]}
    gport = {"schema": SCHEMA, "issue": 134, "arm": "T3 γ-model-car", "checked_at": utc_now(),
             "device": "cpu float32", "threads": THREADS, "torch": torch.__version__,
             "reference_torch": torch.__version__, "same_torch": True, "upstream": UPSTREAM, "commit": COMMIT,
             "tolerance": TOLERANCE,
             "G_port_rule": "issue_123_exact (run_hjepa_comparison.GPORT_TOLERANCE param rule: |delta| <= 1e-7 on >= "
                            "99.9 % of elements, <= 2 lr + 1e-7 everywhere; also the target network) + #123 loss / "
                            "grad tolerances; forward log-likelihood and sampling max abs <= 1e-5",
             "step_indices": list(STEP_INDICES), "vendor_hashes_match": vendor_ok, "upstream_hashes_match": official,
             "reference_script_sha256": sha256_of(Path(__file__)), "adapter_sha256": sha256_of(ADAPTER),
             "arms": arms, "injected_defects": defects,
             "notes": ["Both sides: novphy env, torch 2.13 CPU float32, separate processes; the reference imports the "
                       "pristine clone, the adapter the vendored copy (hashes above).",
                       "The reference runs the Colab notebook's model cell and training-cell body verbatim (with the "
                       "torch 2.13 deepcopy prior shim) at loop indices 0 (burn-in: discount 0.5, sample discount "
                       "0.5) and 2000 (discount = the arm's γ, sample discount 0.9); constant lr 1e-4.",
                       "Carrier arms: the official format_batch with the NovPhy no-op policy vs the adapter's "
                       "transition_batch on the same (shot, k) rows (launches at k = 0; absorbing k = 45 rows).",
                       "Sampling: official model.sample after torch.manual_seed vs the adapter's sample_occupancy on "
                       "the same uniform draw from a seeded CPU generator.",
                       "pendulum-official uses a fixed stand-in next-action policy (rlkit is not in novphy); the "
                       "released SAC policy is exercised by G-fid."]}
    gport["pass"] = bool(all(a["pass"] for a in arms.values()) and all(vendor_ok.values()) and all(official.values())
                         and all(d["detected"] for d in defects.values()))
    path = OUTPUT / "gport" / "gport.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gport, indent=1) + "\n")
    print(f"[compare] G-port pass {gport['pass']} -> {path}", flush=True)
    return 0 if gport["pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--reference", metavar="WORK")
    modes.add_argument("--compare", metavar="WORK")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.reference:
        return reference(args.reference)
    if args.compare:
        return compare(args.compare)
    work = OUTPUT / "gport" / "work"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(THREADS))
    for mode in ("--reference", "--compare"):
        code = subprocess.run([sys.executable, "-W", "ignore", __file__, mode, str(work)], env=env).returncode
        if code != 0:
            return code
    shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
