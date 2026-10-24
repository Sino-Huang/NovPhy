"""Issue-130 (C1 DINO-WM) DINOv2 feature cache and the readiness smoke (fit / policy splits only).

Modes (``novphy`` env; ``--extract`` / ``--readiness`` on a GPU through the night-window scheduler):
    --plan        CPU: the cache layout, its size and the GPU-time estimate -> features/plan.json.
    --extract     DINOv2 ViT-S/14 patch features (fp16 [391, 384] per frame; novphy.preprocess) of
                    fit:    every fit shot at frame-skip-5 positions 0, 5, .., <= min(225, end) (training) and at the
                            #123 R1 probe positions (every 15th to 285, clamped to the end; rc.probe_positions),
                    policy: the 120 policy anchors (frame 0 of the anchor branch, eb.policy_anchor_frames) and every
                            complete policy shot at its true endpoint min(225, end) (the #123 policy pixel store holds
                            every 15th position, clamped, so min(225, end) is always stored).
                  Resumable: rows are written in index order into preallocated memmaps on local NVMe; SIGTERM / SIGUSR1
                  -> finish the current batch, flush, write progress, exit 75. At the end: per-file sha256 in
                  features/MANIFEST.json.
    --readiness   R1 (novphy.PatchReadout) fitted on the fit probe frames only (E99 carrier targets of the same
                  frames), seed 20260908, under deterministic_scoring(); then the ceiling = pooled policy grid AUC of
                  the tie-free cost of R1(true endpoint features) vs R1(policy anchor features) on the policy grid
                  mixed cells (#123 policy_aucs "ceiling"). Threshold 0.65 (G-readout): below -> C1-dino stops with
                  typed reason ``readiness_ceiling_below_threshold`` (C1-car is unaffected). Writes readiness/.
No evaluation-split data is read by any mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUTPUT = ROOT / ".local-artifacts/issue-130-dino-wm-v1"
CACHE = Path(os.environ.get("ISSUE130_FEATURE_CACHE", Path.home() / ".cache" / "novphy-issue-130-dino-features"))
FEATURES = OUTPUT / "features"
READINESS = OUTPUT / "readiness"
BATCH = 256
WORKERS = 12
THRESHOLD = 0.65
TYPED_STOP = "readiness_ceiling_below_threshold"
STOP = []


def log(message):
    print(f"[issue-130] {message}", flush=True)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def mark(phase, seconds):
    path = OUTPUT / "ledger" / f"{phase}.json"
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    block["wall_seconds"] += seconds
    block["gpu_seconds"] += seconds
    block["updated_at"] = utc_now()
    write_json(path, block)


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def training_positions(end):
    """Frame-skip-5 positions 0, 5, .. <= min(225, end) (C1 training windows and rollout targets)."""
    from world_model.baselines.dino_wm import novphy as dn
    return list(range(0, min(dn.HORIZON_FRAME, int(end)) + 1, dn.FRAMESKIP))


def fit_positions(end):
    from scripts import run_hjepa_comparison as rc
    return sorted(set(training_positions(end)) | set(rc.probe_positions(int(end))))


def fit_layout(ends):
    """first [S+1] row offsets and the concatenated positions of every fit shot (frozen eb.FitShots order)."""
    first, positions = [0], []
    for end in ends:
        p = fit_positions(end)
        positions += p
        first.append(first[-1] + len(p))
    return np.asarray(first, dtype=np.int64), np.asarray(positions, dtype=np.int16)


class FitIndex:
    """(shot, position) -> row of the fit feature memmap; raises for anything that is not a cached fit frame."""

    def __init__(self, first, positions):
        self.first, self.positions = first, positions

    def rows(self, shots, positions):
        shots, positions = np.asarray(shots, dtype=np.int64), np.asarray(positions)
        out = np.empty(len(shots), dtype=np.int64)
        for i, (s, p) in enumerate(zip(shots, positions)):
            lo, hi = int(self.first[s]), int(self.first[s + 1])
            k = int(np.searchsorted(self.positions[lo:hi], p))
            if k >= hi - lo or int(self.positions[lo + k]) != int(p):
                raise KeyError(f"fit shot {s} position {p} is not in the feature cache")
            out[i] = lo + k
        return out


def policy_layout():
    """Policy items in frozen order: ("anchor", member, 0) for the 120 anchors, then ("endpoint", key, q) for every
    complete policy shot at q = min(225, end)."""
    from scripts import external_baselines as eb
    anchors = eb.policy_anchors()["keys"]
    shots = eb.policy_shots()
    items = [("anchor", m, 0) for m in sorted(anchors)]
    items += [("endpoint", k, int(min(225, int(shots[k]["end"])))) for k in sorted(shots)]
    return items


def paths():
    return {"fit": CACHE / "fit-features.f16", "policy": CACHE / "policy-features.f16",
            "fit_index": FEATURES / "fit-index.npz", "policy_index": FEATURES / "policy-index.json",
            "progress": FEATURES / "progress.json", "manifest": FEATURES / "MANIFEST.json"}


def open_features(kind, mode="r"):
    from world_model.baselines.dino_wm import novphy as dn
    p = paths()
    rows = (int(np.load(p["fit_index"])["first"][-1]) if kind == "fit" else len(read_json(p["policy_index"])["items"]))
    return np.memmap(p[kind], dtype=np.float16, mode=mode, shape=(rows, dn.NUM_PATCHES, dn.FEATURE_DIM))


def plan():
    from scripts import external_baselines as eb
    from world_model.baselines.dino_wm import novphy as dn
    fit = eb.FitShots()
    first, positions = fit_layout(fit.ends)
    p = paths()
    FEATURES.mkdir(parents=True, exist_ok=True)
    np.savez(p["fit_index"], keys=np.array(fit.keys), first=first, positions=positions,
             ends=np.asarray(fit.ends, dtype=np.int16))
    items = policy_layout()
    write_json(p["policy_index"], {"schema": "issue_130_policy_feature_index_v1", "items": items})
    frame_bytes = dn.NUM_PATCHES * dn.FEATURE_DIM * 2
    training = sum(len(training_positions(e)) for e in fit.ends)
    from scripts import run_hjepa_comparison as rc
    probe = sum(len(rc.probe_positions(int(e))) for e in fit.ends)
    n_fit = int(first[-1])
    out = {"schema": "issue_130_feature_plan_v1", "written_at": utc_now(),
           "encoder": {"model": dn.DINOV2_NAME, "weights_sha256": dn.DINOV2_WEIGHTS_SHA256,
                       "repo": str(dn.DINOV2_REPO.relative_to(ROOT)), "feature_key": "x_norm_patchtokens"},
           "preprocess": {"input": "uint8 [3, 240, 320] (#123 pixel stores)", "scale": "/255, Normalize(0.5, 0.5)",
                          "resize": f"bilinear antialias to {dn.DINO_HW} (full frame, aspect 1.353 vs 1.333)",
                          "grid": list(dn.GRID), "patches": dn.NUM_PATCHES, "dim": dn.FEATURE_DIM, "dtype": "float16"},
           "fit": {"shots": len(fit.keys), "frames": n_fit, "training_frames": training, "r1_probe_frames": probe,
                   "bytes": n_fit * frame_bytes},
           "policy": {"anchors": sum(1 for i in items if i[0] == "anchor"),
                      "endpoints": sum(1 for i in items if i[0] == "endpoint"), "bytes": len(items) * frame_bytes},
           "cache_dir": str(CACHE), "estimate": {
               "gpu_minutes_extract": 45, "gpu_minutes_readiness": 45,
               "basis": "ViT-S/14 at 392 tokens ~ 20 GFLOP / frame fp32 (no TF32) => ~10 min GPU for "
                        f"{n_fit + len(items)} frames; zstd decode on {WORKERS} workers, ~{(n_fit + len(items)) * frame_bytes / 1e9:.0f} GB "
                        "NVMe writes and the sha256 pass dominate [INFERENCE until measured]"}}
    write_json(FEATURES / "plan.json", out)
    log(f"plan: fit {n_fit} frames ({training} training, {probe} probe), policy {len(items)}; "
        f"{(n_fit + len(items)) * frame_bytes / 1e9:.1f} GB")
    return 0


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

class FitFrames(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items, self.store = items, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from scripts import run_hjepa_comparison as rc
        if self.store is None:
            self.store = rc.PixelStore(rc.OUTPUT, "fit")
        shot, position = self.items[i]
        return self.store.frame(int(shot), int(position))


def policy_frames(items):
    from scripts import external_baselines as eb
    from scripts import run_hjepa_comparison as rc
    anchors = eb.policy_anchor_frames()
    store = rc.PixelStore(rc.OUTPUT, "policy")
    return torch.stack([anchors[name] if kind == "anchor" else store.frame(store.row[name], q)
                        for kind, name, q in items])


def request_stop(signum, _frame):
    STOP.append(signum)


def extract(device="cuda"):
    from world_model.baselines.dino_wm import novphy as dn
    from world_model.planning import scoring_harness as harness
    p = paths()
    if not p["fit_index"].is_file():
        plan()
    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, request_stop)
    CACHE.mkdir(parents=True, exist_ok=True)
    index = np.load(p["fit_index"])
    first, positions = index["first"], index["positions"]
    shots = np.repeat(np.arange(len(first) - 1), np.diff(first))
    n_fit = int(first[-1])
    items = read_json(p["policy_index"])["items"]
    progress = read_json(p["progress"]) if p["progress"].is_file() else {"fit_rows": 0, "policy_done": False}
    for kind, rows in (("fit", n_fit), ("policy", len(items))):
        if not p[kind].is_file():
            np.memmap(p[kind], dtype=np.float16, mode="w+", shape=(rows, dn.NUM_PATCHES, dn.FEATURE_DIM)).flush()
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        encoder = dn.load_dinov2(device)
        if not progress["policy_done"]:
            mm = open_features("policy", "r+")
            frames = policy_frames(items)
            for i in range(0, len(frames), BATCH):
                mm[i:i + BATCH] = dn.patch_features(encoder, dn.preprocess(frames[i:i + BATCH].to(device))).half().cpu().numpy()
            mm.flush()
            progress["policy_done"] = True
            write_json(p["progress"], progress)
            log(f"policy features: {len(items)} frames")
        mm = open_features("fit", "r+")
        start = progress["fit_rows"]
        todo = list(zip(shots[start:].tolist(), positions[start:].tolist()))
        loader = torch.utils.data.DataLoader(FitFrames(todo), batch_size=BATCH, num_workers=WORKERS,
                                             prefetch_factor=4, persistent_workers=False)
        row = start
        for k, frames in enumerate(loader):
            n = frames.shape[0]
            mm[row:row + n] = dn.patch_features(encoder, dn.preprocess(frames.to(device, non_blocking=True))).half().cpu().numpy()
            row += n
            if k % 100 == 99 or STOP:
                mm.flush()
                progress["fit_rows"] = row
                write_json(p["progress"], progress)
                log(f"fit features: {row}/{n_fit} ({time.monotonic() - began:.0f}s)")
                if STOP:
                    mark("features", time.monotonic() - began)
                    log("stop requested: progress saved, exiting 75")
                    return 75
        mm.flush()
        progress["fit_rows"] = row
        write_json(p["progress"], progress)
    if row != n_fit:
        raise RuntimeError(f"fit features incomplete: {row}/{n_fit}")
    log(f"fit features done ({time.monotonic() - began:.0f}s); hashing")
    manifest = {"schema": "issue_130_feature_manifest_v1", "written_at": utc_now(), "scoring_policy": policy,
                "plan_sha256": sha256_of(FEATURES / "plan.json"),
                "files": {kind: {"path": str(p[kind]), "bytes": p[kind].stat().st_size, "sha256": sha256_of(p[kind])}
                          for kind in ("fit", "policy")},
                "indexes": {kind: {"path": str(p[kind].relative_to(ROOT)), "sha256": sha256_of(p[kind])}
                            for kind in ("fit_index", "policy_index")}}
    write_json(p["manifest"], manifest)
    mark("features", time.monotonic() - began)
    log("feature cache complete")
    return 0


def verify_manifest(hash_files=False):
    p = paths()
    manifest = read_json(p["manifest"])
    for kind in ("fit_index", "policy_index"):
        if sha256_of(ROOT / manifest["indexes"][kind]["path"]) != manifest["indexes"][kind]["sha256"]:
            raise ValueError(f"{kind} differs from the feature manifest")
    for kind in ("fit", "policy"):
        entry = manifest["files"][kind]
        if Path(entry["path"]).stat().st_size != entry["bytes"] or (hash_files and sha256_of(entry["path"]) != entry["sha256"]):
            raise ValueError(f"{kind} feature file differs from the manifest")
    return manifest


# ---------------------------------------------------------------------------
# readiness smoke
# ---------------------------------------------------------------------------

def r1_items(ends):
    """The #123 R1 frame set: every fit shot at rc.probe_positions(end)."""
    from scripts import run_hjepa_comparison as rc
    return [(shot, p) for shot, end in enumerate(ends) for p in rc.probe_positions(int(end))]


def readiness(device="cuda"):
    from scripts import external_baselines as eb
    from scripts import run_lookahead_control as wlc
    from world_model.baselines.dino_wm import novphy as dn
    from world_model.planning import scoring_harness as harness
    manifest = verify_manifest()
    p = paths()
    began = time.monotonic()
    fit = eb.FitShots()
    index = np.load(p["fit_index"])
    if list(map(str, index["keys"])) != fit.keys:
        raise ValueError("feature index and eb.FitShots disagree on the shot order")
    fit_index = FitIndex(index["first"], index["positions"])
    items = r1_items(fit.ends)
    shots = np.array([s for s, _ in items])
    positions = np.array([q for _, q in items])
    rows = fit_index.rows(shots, positions)
    targets = fit.carrier_rows(shots, positions)
    features = open_features("fit")
    policy_items = read_json(p["policy_index"])["items"]
    policy_row = {(kind, name): i for i, (kind, name, _) in enumerate(policy_items)}
    policy_features = open_features("policy")
    READINESS.mkdir(parents=True, exist_ok=True)
    with harness.deterministic_scoring() as scoring_policy:
        probe, fit_report = dn.fit_readout(features, rows, targets, eb.SELECTION_SEED, device=device, log=log)
        torch.save({"schema": "issue_130_r1_v1", "state": probe.cpu().state_dict(), "seed": eb.SELECTION_SEED,
                    "fit": fit_report, "feature_manifest_sha256": sha256_of(p["manifest"])}, READINESS / "r1.pt")
        probe.to(device)
        costs = harness.EndpointCosts(wlc.load_objective())
        cells = eb.policy_cells("grid")
        members = sorted(cells)
        anchors = eb.policy_anchors()
        shots_policy = eb.policy_shots()
        c0 = dn.apply_readout(probe, policy_features[[policy_row[("anchor", m)] for m in members]], device)
        flat = [(m, item) for m in members for item in cells[m]["inventory"]]
        c_true = dn.apply_readout(probe, policy_features[[policy_row[("endpoint", item["key"])] for _, item in flat]], device)
        e99_true, e99_anchor = [], {}
        for m, item in flat:
            shot = shots_policy[item["key"]]
            q = min(225, int(shot["end"]))
            e99_true.append(shot["carriers"][(shot["positions"] == q).nonzero()[0, 0]])
        e99_true = torch.stack(e99_true).float()
        e99_anchor = {m: anchors["carriers"][m].float() for m in members}
        rows_r1, rows_e99, k = {}, {}, 0
        c0_by = dict(zip(members, c0))
        for m in members:
            inv = cells[m]["inventory"]
            rows_r1[m] = eb.endpoint_rows(inv, c_true[k:k + len(inv)], c0_by[m], costs)
            rows_e99[m] = eb.endpoint_rows(inv, e99_true[k:k + len(inv)], e99_anchor[m], costs)
            k += len(inv)
        ceiling = eb.policy_grid_auc(rows_r1, harness.PRIMARY_COST)
        e99_ceiling = eb.policy_grid_auc(rows_e99, harness.PRIMARY_COST)
        presence, center = costs.presence, list(costs.center)
        readout_error = {
            "pig_presence_mae": float((c_true[:, presence] - e99_true[:, presence]).abs().mean()),
            "pig_center_mae": float((c_true[:, center] - e99_true[:, center]).abs().mean()),
            "carrier_mae": float((c_true - e99_true).abs().mean())}
    passed = ceiling is not None and ceiling >= THRESHOLD
    report = {"schema": "issue_130_readiness_v1", "arm": "C1-dino", "written_at": utc_now(),
              "scoring_policy": scoring_policy, "threshold": THRESHOLD, "ceiling_auc": ceiling, "pass": bool(passed),
              "typed_reason": None if passed else TYPED_STOP,
              "consequence": "C1-dino proceeds" if passed else "C1-dino stops (typed); C1-car still runs",
              "policy_grid_cells": len(members), "policy_candidates": len(flat),
              "e99_true_carrier_ceiling_auc": e99_ceiling,
              "readout_error_vs_e99_on_policy_endpoints": readout_error,
              "r1": {"artifact": str((READINESS / "r1.pt").relative_to(ROOT)), "sha256": sha256_of(READINESS / "r1.pt"),
                     "fit_frames": len(items), "fit_shots": len(fit.keys), "seed": eb.SELECTION_SEED, **fit_report},
              "data": {"r1_fit": "fit split only: eb.FitShots x rc.probe_positions, E99 targets from the #123 fit memmap",
                       "ceiling": "policy split: eb.policy_cells('grid') mixed cells, eb.policy_anchor_frames, #123 "
                                  "policy pixel store at min(225, end)", "evaluation": "not read"},
              "feature_manifest_sha256": sha256_of(p["manifest"]), "feature_files": manifest["files"],
              "seconds": time.monotonic() - began}
    write_json(READINESS / "readiness.json", report)
    mark("readiness", time.monotonic() - began)
    log(f"readiness: ceiling {ceiling} (threshold {THRESHOLD}) -> {'PASS' if passed else TYPED_STOP}; "
        f"E99 true-carrier ceiling {e99_ceiling}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("plan", "extract", "readiness"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    if args.plan:
        return plan()
    if args.extract:
        return extract()
    return readiness()


if __name__ == "__main__":
    sys.exit(main())
