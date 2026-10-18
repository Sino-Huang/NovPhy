"""Issue-123: H-JEPA (arXiv 2610.06805) reproduced on NovPhy and compared with the Gate-B unit.

Binding runner module: scripts/run_hjepa_comparison.py
Exact validation command: python -u -m scripts.run_hjepa_comparison --validate

#123 asks whether H-JEPA (Zhang et al., arXiv 2610.06805; official code kevinghst/H-JEPA @ 97054ec, vendored with
per-file hashes in world_model/hjepa/vendor/), reproduced on NovPhy, ranks the Gate-B grid candidates of the sealed
#109 evaluation split better or worse than the Gate-B unit U (LC : F-15-macro, hybrid, E99). This runner

1. builds the fit / policy data only (--data-pixels / --assemble-pixels / --mirror-pixels: 320 x 240 agent frames,
   zstd shards; --fit-carriers: the #112 retrain shot cache; --policy-data: E99 policy carriers and anchors;
   --data-labels: engine slot labels at every 15th fit position); the evaluation stream is never read before the
   freeze (stream_rows refuses it);
2. --gport: the vendored modules plus this port's loader / optimizer / scheduler reproduce the official code's forward
   pass and one optimizer step (scripts/hjepa_port_reference.py in the hjepa env);
3. --selection / --selection-readouts: the one pre-declared selection per input type (SIGReg lambda in {0.09, 0.18},
   HJ-px and HJ-car-m at B1, seed 20260908, policy-split pooled grid AUC under R1; ties at 4 d.p. -> 0.09);
4. --smoke: data accounting, G-port, tests, throughput, the projection against 260 GPU-h / 72 h (STOP RULE) and the
   lambda choice; --prepare freezes plan.json (every input hash, data manifest, arms, budgets, queues, readouts,
   scoring, contrasts, guards, dispositions) before any main-arm training or evaluation read;
5. --train (per GPU queue) and --bench-train (CLEVRER / Physion, EXPLORATORY); --readouts fits R1 / R2 / R3 / R4 /
   P_c and the collapse / action / readout-ceiling diagnostics on fit (policy) data; --freeze-readouts hashes every
   model and readout before any evaluation anchor is encoded;
6. --score, once, in a process that did not train, under deterministic_scoring(): every arm-seed on the Gate-B mixed
   cells of every inventory, the S10 wrong-anchor control, the position error and the U / C* position curves;
   --bench-score the exploratory benchmark rollouts;
7. --publish: records-manifest.json, compute.json, summary.json, findings.md (guards, tokens by the delta rule,
   dispositions); --validate re-checks every hash, re-derives every table byte for byte and re-scores one record.

G-fid (FourRoom Tier E / Tier T) runs in scripts/run_hjepa_fidelity.py and gates every H-JEPA token at publication.
Every interval is a member-clustered percentile bootstrap (10000 draws, PCG64 7201), DESCRIPTIVE. Engine seconds 0.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import gc
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts import run_decision_dynamics_retrain as rt
from scripts import run_decision_relevant_dynamics as v1
from scripts import run_slot_encoder_fix as s99

ROOT = v1.ROOT
OUTPUT = ROOT / ".local-artifacts/issue-123-hjepa-comparison-v1"
RETRAIN = rt.OUTPUT
CAMPAIGN = rt.CAMPAIGN
STREAMS = {"fit": "issue109-fit", "policy": "issue109-policy", "evaluation": "issue109-evaluation"}
MAX_POSITION = s99.MAX_POSITION  # 285, the retrain's carrier window
PROBE_STRIDE = 15
ZSTD_LEVEL = 3

read_json = v1.read_json
write_json = v1.write_json
json_text = v1.json_text
sha256_of = v1.sha256_of
atomic_torch = v1.atomic_torch


def log(message):
    print(f"[issue-123] {message}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# membership: the #109 fit / policy streams of the #104 campaign (evaluation only after the freeze)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def stream_rows(split):
    """Campaign report rows of one stream only (rows of other streams are dropped at load)."""
    if split == "evaluation":
        raise ValueError("the evaluation stream is read only by the post-freeze scoring modes")
    return {row["identity"]: row for row in read_json(CAMPAIGN / "branches.json") if row["stream"] == STREAMS[split]}


def split_levels(split):
    return [level for family in rt.families() for level in rt.cohort()["splits"][split][family]]


def split_branches(split):
    """Every complete branch of a fit / policy stream in frozen order (families, roster order, candidate
    suffix order); the retrain's rule for the fit split."""
    if split == "fit":
        return rt.complete_branches()
    rows = stream_rows(split)
    out = []
    for level in split_levels(split):
        for candidate in rt.candidates():
            key = f"{level['identity']}-{candidate['suffix']}"
            row = rows[key]
            if row["status"] != "complete":
                continue
            out.append({"key": key, "member": level["identity"], "family": level["generator_family"],
                        "candidate": candidate, "status": row["status"], "failure_class": row["failure_class"],
                        "attempt": str(CAMPAIGN / STREAMS[split] / "attempts" / key)})
    return out


def split_verdicts(split):
    return {key: bool(row["pig_removed"]) for key, row in stream_rows(split).items() if row["status"] == "complete"}


def store_positions(split, end):
    """Stored frame positions of a shot with carrier positions 0..end (end = min(last, 285))."""
    if split == "fit":
        return list(range(end + 1))
    return sorted({min(p, end) for p in range(0, MAX_POSITION + 1, PROBE_STRIDE)})


def probe_positions(end):
    """The readout / slot-probe frames of a fit shot: every 15th position, clamped to its end."""
    return sorted({min(p, end) for p in range(0, MAX_POSITION + 1, PROBE_STRIDE)})


def shot_end(attempt):
    _, _, frames, _ = s99.shot_frames(attempt)
    return min(len(frames) - 1, MAX_POSITION), len(frames)


# ---------------------------------------------------------------------------
# pixel store: per-worker shards of zstd-compressed 320 x 240 RGB frames (s99.image_tensor: PIL bilinear,
# the E99 parser's input); index rows (key, position, shard, offset, length)
# ---------------------------------------------------------------------------

def pixel_dir(output, split):
    return Path(output) / "pixels" / split


def shard_paths(output, split, worker):
    base = pixel_dir(output, split)
    return base / f"shard-{worker:02d}.bin", base / f"shard-{worker:02d}.jsonl"


def build_pixel_shard(output, split, workers, worker):
    """Encode this worker's shots (index % workers == worker); resumable per shot."""
    import zstandard
    torch.set_num_threads(1)
    blob_path, index_path = shard_paths(output, split, worker)
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    done, size = {}, 0
    if index_path.is_file():
        for line in index_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                done[entry["key"]] = entry
                size = max(size, entry["offsets"][-1] + entry["lengths"][-1])
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    branches = [b for i, b in enumerate(split_branches(split)) if i % workers == worker]
    began = time.monotonic()
    with open(blob_path, "ab") as blob, open(index_path, "a") as index:
        blob.truncate(size)
        blob.seek(size)
        for count, branch in enumerate(branches):
            if branch["key"] in done:
                continue
            _, observation_root, frames, _ = s99.shot_frames(branch["attempt"])
            end = min(len(frames) - 1, MAX_POSITION)
            positions = store_positions(split, end)
            offsets, lengths = [], []
            for p in positions:
                image = s99.image_tensor(observation_root / frames[p]["agent_observation"]["relative_path"])
                data = compressor.compress(image.permute(1, 2, 0).contiguous().numpy().tobytes())
                offsets.append(size)
                lengths.append(len(data))
                blob.write(data)
                size += len(data)
            blob.flush()
            os.fsync(blob.fileno())
            index.write(json.dumps({"key": branch["key"], "end": end, "frame_count": len(frames),
                                    "positions": positions, "offsets": offsets, "lengths": lengths}) + "\n")
            index.flush()
            if count % 100 == 0:
                log(f"pixels {split} worker {worker}: {count + 1}/{len(branches)} shots, "
                    f"{size / 1e9:.1f} GB, {time.monotonic() - began:.0f}s")
    log(f"pixels {split} worker {worker}: done ({len(branches)} shots, {size / 1e9:.1f} GB)")
    return 0

# The shards are read from a byte-identical local-NVMe mirror (random 17 KB reads through the mergerfs FUSE array
# measured ~33 ms per frame under load); --mirror-pixels writes pixels/<split>/MANIFEST.json (per-shard sha256 of
# the canonical shards) and verifies every mirrored shard against it.
PIXEL_MIRROR = Path.home() / ".cache" / "novphy-issue-123-pixels"


def mirror_dir(split):
    return PIXEL_MIRROR / split


def mirror_pixels(output, split):
    import shutil
    source = pixel_dir(output, split)
    target = mirror_dir(split)
    target.mkdir(parents=True, exist_ok=True)
    shards = sorted(source.glob("shard-*.bin"))
    manifest = {"schema": "issue_123_pixel_manifest_v1", "split": split,
                "shards": {p.name: {"sha256": sha256_of(p), "bytes": p.stat().st_size} for p in shards}}
    for p in shards:
        copy = target / p.name
        if not copy.is_file() or copy.stat().st_size != p.stat().st_size:
            shutil.copyfile(p, copy)
        if sha256_of(copy) != manifest["shards"][p.name]["sha256"]:
            raise ValueError(f"mirror of {p.name} differs from the canonical shard")
    write_json(source / "MANIFEST.json", manifest)
    log(f"pixels {split}: {len(shards)} shards mirrored to {target} and verified")
    return 0



class PixelStore:
    """Random access to the stored frames of a split (uint8 [3, 240, 320] tensors)."""

    def __init__(self, output, split):
        self.dir = pixel_dir(output, split)
        index = np.load(self.dir / "index.npz", allow_pickle=False)
        self.keys = [str(k) for k in index["keys"]]
        self.row = {k: i for i, k in enumerate(self.keys)}
        self.first = index["first"]
        self.positions = index["positions"]
        self.shard = index["shard"]
        self.offset = index["offset"]
        self.length = index["length"]
        self.ends = index["ends"]
        self._fds = {}
        self._decompressor = None

    def _fd(self, shard):
        fd = self._fds.get(shard)
        if fd is None:
            fd = self._fds[shard] = os.open(mirror_dir(self.dir.name) / f"shard-{shard:02d}.bin", os.O_RDONLY)
        return fd

    def frame_row(self, shot, position):
        lo, hi = int(self.first[shot]), int(self.first[shot + 1])
        rows = self.positions[lo:hi]
        k = int(np.searchsorted(rows, position))
        if k >= len(rows) or rows[k] != position:
            raise KeyError(f"position {position} not stored for {self.keys[shot]}")
        return lo + k

    def frame(self, shot, position):
        import zstandard
        if self._decompressor is None:
            self._decompressor = zstandard.ZstdDecompressor()
        r = self.frame_row(shot, position)
        data = os.pread(self._fd(int(self.shard[r])), int(self.length[r]), int(self.offset[r]))
        raw = self._decompressor.decompress(data)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(240, 320, 3).permute(2, 0, 1)


def assemble_pixels(output, split, workers):
    """Merge the worker shard indices into index.npz (frozen branch order)."""
    entries = {}
    for worker in range(workers):
        _, index_path = shard_paths(output, split, worker)
        for line in index_path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                entries[entry["key"]] = (worker, entry)
    keys = [b["key"] for b in split_branches(split)]
    missing = [k for k in keys if k not in entries]
    if missing:
        raise ValueError(f"{len(missing)} {split} shots missing from the pixel shards")
    first, positions, shard, offset, length, ends, counts = [0], [], [], [], [], [], []
    for key in keys:
        worker, entry = entries[key]
        positions += entry["positions"]
        shard += [worker] * len(entry["positions"])
        offset += entry["offsets"]
        length += entry["lengths"]
        first.append(len(positions))
        ends.append(entry["end"])
        counts.append(entry["frame_count"])
    np.savez(pixel_dir(output, split) / "index.npz", keys=np.array(keys), first=np.array(first, dtype=np.int64),
             positions=np.array(positions, dtype=np.int16), shard=np.array(shard, dtype=np.int8),
             offset=np.array(offset, dtype=np.int64), length=np.array(length, dtype=np.int32),
             ends=np.array(ends, dtype=np.int16), frame_counts=np.array(counts, dtype=np.int16))
    log(f"pixels {split}: {len(keys)} shots, {len(positions)} frames, "
        f"{sum(length) / 1e9:.1f} GB compressed")
    return {"shots": len(keys), "frames": len(positions), "bytes": int(sum(length))}


# ---------------------------------------------------------------------------
# engine labels (slot presence and engine-projected centres, the #100 metric's labels) at chosen frames;
# only the native chunks holding those frames are read
# ---------------------------------------------------------------------------

def engine_labels(attempt, positions):
    """presence bool [P, 18] and centres float [P, 18, 2] (run_issue_70_parser_repair.targets, x/width,
    y/height) at positions (clamped to the last frame), from the shot's native trace."""
    from scripts.canonical_native_trace import NativeTrace
    from scripts.run_issue_70_parser_repair import targets
    _, _, frames, segment = s99.shot_frames(attempt)
    last = len(frames) - 1
    wanted = {frames[min(p, last)]["fixed_step"] for p in positions}
    trace = NativeTrace(segment["native_root"])
    samples = {}
    for ordinal, descriptor in enumerate(trace.manifest["chunks"], 1):
        if not any(descriptor["first_fixed_step"] <= s <= descriptor["last_fixed_step"] for s in wanted):
            continue
        with gzip.open(trace.root / descriptor["path"], "rb") as stream:
            chunk = json.loads(stream.read())
        if chunk.get("capture_id") != trace.manifest["capture_id"] or chunk.get("shot_id") != trace.manifest["shot_id"]:
            raise ValueError("native chunk belongs to a different capture")
        for sample in chunk["fixed_step_samples"]:
            if sample["fixed_step"] in wanted:
                samples[sample["fixed_step"]] = sample
    if set(samples) != wanted:
        raise ValueError(f"native samples missing for {attempt}")
    names = s99.vocabulary()
    slots = set(names)
    unavailable = {"availability": "unavailable"}
    micro = {"predicates": {"contact": unavailable, "supports": unavailable}}
    macro = {"predicates": {"steady-state": unavailable, "structure-unstable": unavailable}}
    presence, centers = [], []
    for p in positions:
        q = min(p, last)
        sample = samples[frames[q]["fixed_step"]]
        filtered = {**sample, "entities": [e for e in sample["entities"] if e["scenario_object_id"] in slots]}
        row = targets(filtered, frames[q]["capture_metadata"], micro, macro, names)
        presence.append(row["presence"].bool())
        centers.append(row["centers"].float())
    return torch.stack(presence), torch.stack(centers)


def labels_path(output, split, worker):
    return Path(output) / "labels" / split / f"labels-{worker:02d}.pt"


def build_label_shard(output, split, workers, worker):
    """Engine labels at the probe positions of this worker's fit shots; resumable (saved every 200 shots)."""
    torch.set_num_threads(1)
    path = labels_path(output, split, worker)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = torch.load(path, weights_only=True) if path.is_file() else {"shots": {}}
    branches = [b for i, b in enumerate(split_branches(split)) if i % workers == worker]
    began = time.monotonic()
    for count, branch in enumerate(branches):
        if branch["key"] in store["shots"]:
            continue
        end, _ = shot_end(branch["attempt"])
        positions = probe_positions(end)
        presence, centers = engine_labels(branch["attempt"], positions)
        store["shots"][branch["key"]] = {"positions": torch.tensor(positions), "presence": presence, "centers": centers}
        if count % 200 == 0:
            atomic_torch(path, store)
            log(f"labels {split} worker {worker}: {count + 1}/{len(branches)} ({time.monotonic() - began:.0f}s)")
    atomic_torch(path, store)
    log(f"labels {split} worker {worker}: done ({len(branches)} shots)")
    return 0



# ---------------------------------------------------------------------------
# G-port: the vendored modules + this port's sparse planned encode, optimizer and scheduler against the
# official code (scripts/hjepa_port_reference.py, hjepa env) on fixed inputs; CPU float32
# ---------------------------------------------------------------------------

GPORT_ARMS = ("HJ-px", "HWM-px", "LeWM-px", "HJ-car-m", "HJ-car-m-fs5")
GPORT_CROPS = {"H3": (7, 3, None), "L1": (None,), "H2fs5": (2, None)}
GPORT_LAMBDA = 0.09
# Per-tensor gradients: relative L2 on every tensor whose official gradient norm exceeds GRAD_FLOOR x the global
# norm; below it (tensors whose exact gradient is 0: key biases under the softmax, biases feeding BatchNorm) the
# absolute L2 difference must stay under GRAD_FLOOR x the global norm (float round-off only).
GPORT_TOLERANCE = {"loss_relative": 1e-4, "grad_norm_relative": 1e-4, "grad_tensor_relative_l2": 1e-3,
                   "grad_floor_of_global_norm": 1e-6,
                   "embed_max_abs": 1e-4, "param_close_share": 0.999, "param_close_abs": 1e-7,
                   "param_max_abs_over_lr": 2.0, "lr_abs": 1e-12}


def gport_dir(output):
    return Path(output) / "gport"


def gport_configs(output):
    from world_model.hjepa import novphy as hn
    configs = {}
    for arm in GPORT_ARMS:
        geometry = hn.Geometry(arm)
        configs[arm] = {"config": hn.arm_config(arm, GPORT_LAMBDA), "input": hn.ARMS[arm]["input"],
                        "level1_steps": geometry.level1_steps,
                        "crop_starts": list(GPORT_CROPS[hn.ARMS[arm]["layout"]])}
    path = gport_dir(output) / "configs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(configs))
    log(f"G-port configs written: {path}")
    return 0


def gport_arm(reference):
    from world_model.hjepa import novphy as hn
    arm = reference["arm"]
    geometry = hn.Geometry(arm)
    starts = {level: s for level, s in enumerate(reference["crop_starts"], start=1)}
    needed = geometry.needed(starts)
    cfg = hn.arm_config(arm, GPORT_LAMBDA)
    model, losses = hn.build_model(arm, GPORT_LAMBDA)
    model.load_state_dict(reference["init_state"])
    model.plan = (geometry.level1_steps, starts, needed)
    wrapper = hn.Wrapper(model, losses).train()
    x = reference["inputs"]["x"]
    batch = {"pixels_level1": x[:, needed].clone(), "action_level1": reference["inputs"]["action"].clone()}
    optimizer = hn.make_optimizer(model, cfg)
    scheduler = hn.WarmupCosine(optimizer, hn.warmup_steps(reference["total_steps"]), reference["total_steps"])
    torch.manual_seed(reference["seeds"][2])
    output = hn.hjepa_utils.hjepa_forward(wrapper, batch, "fit", cfg,
                                          normalize_batch=hn.normalize_uint8_images_on_device)
    scalars = {k: float(v) for k, v in output.items() if "loss" in k and torch.is_tensor(v) and v.ndim == 0}
    output["loss"].backward()
    params = [p for group in optimizer.param_groups for p in group["params"]]
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    norm = float(torch.nn.utils.clip_grad_norm_(params, hn.GRAD_CLIP))
    optimizer.step()
    scheduler.step()
    tol = GPORT_TOLERANCE
    loss_rel = max(abs(scalars[k] - v) / max(1.0, abs(v)) for k, v in reference["scalars"].items())
    embed_abs = max(float((output[k].detach() - v).abs().max()) for k, v in reference["embeds"].items()
                    if k in output and output[k].shape == v.shape)
    floor = tol["grad_floor_of_global_norm"] * reference["grad_norm"]
    grad_rel = max(float((grads[k] - v).norm() / v.norm()) for k, v in reference["grads"].items()
                   if float(v.norm()) > floor)
    grad_null_abs = max([float((grads[k] - v).norm()) for k, v in reference["grads"].items()
                         if float(v.norm()) <= floor] or [0.0])
    null_tensors = sum(1 for v in reference["grads"].values() if float(v.norm()) <= floor)
    lr_max = max(group["lr"] for group in optimizer.param_groups)
    post = model.state_dict()
    deltas = torch.cat([(post[k].float() - v.float()).abs().flatten() for k, v in reference["post_state"].items()
                        if v.is_floating_point() and v.numel() and post[k].shape == v.shape])
    close_share = float((deltas <= tol["param_close_abs"]).float().mean())
    lrs = [group["lr"] for group in optimizer.param_groups]
    out = {"arm": arm, "needed_frames": needed, "official_needed_frames": reference["official_needed_frames"],
           "needed_equal": needed == reference["official_needed_frames"],
           "drawn_crop_starts": reference["drawn"], "loss": scalars.get("loss"),
           "official_loss": reference["scalars"]["loss"],
           "loss_max_relative_delta": loss_rel, "embed_max_abs_delta": embed_abs,
           "grad_norm": norm, "official_grad_norm": reference["grad_norm"],
           "grad_norm_relative_delta": abs(norm - reference["grad_norm"]) / max(reference["grad_norm"], 1e-12),
           "grad_tensor_max_relative_l2": grad_rel, "null_gradient_tensors": null_tensors,
           "null_gradient_max_abs_l2": grad_null_abs, "null_gradient_floor": floor, "param_close_share": close_share,
           "param_max_abs_delta": float(deltas.max()), "lr_max": lr_max, "lrs_after_step": lrs,
           "official_lrs_after_step": reference["lrs_after_step"],
           "optimizer_defaults": {k: optimizer.defaults[k] for k in ("lr", "betas", "eps", "weight_decay", "amsgrad")},
           "official_optimizer_defaults": {k: reference["optimizer_defaults"].get(k)
                                           for k in ("lr", "betas", "eps", "weight_decay", "amsgrad")},
           "official_torch": reference["torch"], "torch": torch.__version__}
    out["optimizer_equal"] = all(
        tuple(out["optimizer_defaults"][k]) == tuple(out["official_optimizer_defaults"][k] or ())
        if isinstance(out["optimizer_defaults"][k], tuple)
        else out["optimizer_defaults"][k] == out["official_optimizer_defaults"][k] for k in out["optimizer_defaults"])
    out["pass"] = bool(out["needed_equal"] and out["optimizer_equal"] and loss_rel <= tol["loss_relative"]
                       and embed_abs <= tol["embed_max_abs"]
                       and out["grad_norm_relative_delta"] <= tol["grad_norm_relative"]
                       and grad_rel <= tol["grad_tensor_relative_l2"] and grad_null_abs <= floor
                       and close_share >= tol["param_close_share"]
                       and out["param_max_abs_delta"] <= tol["param_max_abs_over_lr"] * lr_max + tol["param_close_abs"]
                       and max(abs(a - b) for a, b in zip(lrs, reference["lrs_after_step"])) <= tol["lr_abs"])
    return out


def gport(output):
    """Compare every reference arm; writes gport/gport.json (G-port passes iff every arm passes)."""
    torch.set_num_threads(8)
    vendor = read_json(ROOT / "world_model/hjepa/vendor/VENDOR.json")
    hashes = {rel: sha256_of(ROOT / "world_model/hjepa/vendor" / rel) == entry["vendored_sha256"]
              for rel, entry in vendor["files"].items()}
    arms = {}
    for arm in GPORT_ARMS:
        reference = torch.load(gport_dir(output) / f"reference-{arm}.pt", map_location="cpu", weights_only=False)
        arms[arm] = gport_arm(reference)
        log(f"G-port {arm}: pass {arms[arm]['pass']} (loss {arms[arm]['loss']:.6f} vs {arms[arm]['official_loss']:.6f})")
    result = {"schema": "issue_123_gport_v1", "checked_at": utc_now(), "tolerance": GPORT_TOLERANCE,
              "vendor_hashes_match": hashes, "arms": arms, "device": "cpu float32",
              "pass": bool(all(hashes.values()) and all(a["pass"] for a in arms.values()))}
    write_json(gport_dir(output) / "gport.json", result)
    log(f"G-port: {'PASS' if result['pass'] else 'FAIL'}")
    return 0 if result["pass"] else 1


# ---------------------------------------------------------------------------
# fit carriers (the retrain shot cache, positions 0..end; the E99 carriers the Gate-B unit trained on) and the
# policy-split E99 data (anchors, carriers at the stored positions), parsed with the frozen #99 encoder
# ---------------------------------------------------------------------------

def carriers_dir(output):
    return Path(output) / "carriers"


def build_fit_carriers(output):
    """carriers/fit-carriers.bin (float32 rows, frozen shot order, positions 0..end) + fit-index.npz."""
    from concurrent.futures import ThreadPoolExecutor
    branches = split_branches("fit")
    path = carriers_dir(output) / "fit-carriers.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    ends, first, actions = [], [0], []

    def load(branch):
        shot = torch.load(rt.shot_path(RETRAIN, branch["key"]), map_location="cpu", weights_only=True)
        if shot["key"] != branch["key"] or shot["carriers"].shape != (shot["end"] + 1, 236):
            raise ValueError(f"retrain shot {branch['key']} binding differs")
        return shot["end"], shot["action"].float(), shot["carriers"].float().numpy()

    with open(path, "wb") as blob, ThreadPoolExecutor(8) as pool:
        for i, (end, action, carriers) in enumerate(pool.map(load, branches)):
            blob.write(np.ascontiguousarray(carriers).tobytes())
            ends.append(end)
            actions.append(action)
            first.append(first[-1] + end + 1)
            if i % 2000 == 0:
                log(f"fit carriers: {i + 1}/{len(branches)}")
    np.savez(carriers_dir(output) / "fit-index.npz", keys=np.array([b["key"] for b in branches]),
             members=np.array([b["member"] for b in branches]), ends=np.array(ends, dtype=np.int16),
             first=np.array(first, dtype=np.int64), actions=torch.stack(actions).numpy())
    log(f"fit carriers: {len(branches)} shots, {first[-1]} positions")
    return 0


def policy_data_path(output):
    return carriers_dir(output) / "policy.pt"


def build_policy_data(output):
    """Policy-split E99 data under deterministic_scoring(): the level anchors (rt.anchor_carrier: batch-1 parse
    of frame 0 of the first complete candidate) and every shot's carriers at the stored positions."""
    from world_model.planning import scoring_harness as harness
    branches = split_branches("policy")
    first = rt.anchor_branches(branches)
    began = time.monotonic()
    with harness.deterministic_scoring() as policy:
        adapter = s99.load_adapter(s99.OUTPUT, "E")
        anchors = {member: rt.anchor_carrier(adapter, b["attempt"]) for member, b in first.items()}
        shots = {}
        for i, branch in enumerate(branches):
            end, frame_count = shot_end(branch["attempt"])
            positions = store_positions("policy", end)
            shots[branch["key"]] = {"end": end, "frame_count": frame_count, "positions": torch.tensor(positions),
                                    "carriers": s99.carriers_for(adapter, branch["attempt"], positions).float().cpu(),
                                    "action": wlc_action(branch["candidate"]["action"])}
            if i % 500 == 0:
                log(f"policy carriers: {i + 1}/{len(branches)} ({time.monotonic() - began:.0f}s)")
    atomic_torch(policy_data_path(output), {"schema": "issue_123_policy_data_v1", "scoring_policy": policy,
                                            "anchors": anchors, "anchor_frames": {m: b["key"] for m, b in first.items()},
                                            "shots": shots, "gpu_seconds": time.monotonic() - began})
    log(f"policy data: {len(anchors)} anchors, {len(shots)} shots ({time.monotonic() - began:.0f}s)")
    return 0


def wlc_action(action):
    """The 5-d launch vector [drag_x/480, drag_y/480, release/1000, tap/1000, 1] (wlc.action_tensor)."""
    return torch.tensor([action["drag_x"] / 480., action["drag_y"] / 480., action["release_time_ms"] / 1000.,
                         action["tap_time_ms"] / 1000., 1.], dtype=torch.float32)


class FitData:
    """The fit split: frozen shot order, ends, launch vectors, carriers (memmap), pixels (lazy store)."""

    def __init__(self, output):
        index = np.load(carriers_dir(output) / "fit-index.npz", allow_pickle=False)
        self.output = output
        self.keys = [str(k) for k in index["keys"]]
        self.members = [str(m) for m in index["members"]]
        self.ends = index["ends"].astype(np.int64)
        self.first = index["first"]
        self.actions = torch.from_numpy(index["actions"]).float()
        self.carriers = np.memmap(carriers_dir(output) / "fit-carriers.bin", dtype=np.float32, mode="r",
                                  shape=(int(self.first[-1]), 236))
        self._pixels = None

    @property
    def pixels(self):
        if self._pixels is None:
            self._pixels = PixelStore(self.output, "fit")
            if list(self._pixels.keys) != self.keys or not np.array_equal(self._pixels.ends.astype(np.int64), self.ends):
                raise ValueError("pixel store and carrier store disagree on the fit shots")
        return self._pixels

    def carrier(self, shot, position):
        return torch.from_numpy(np.array(self.carriers[int(self.first[shot]) + int(position)]))

    def carrier_rows(self, shots, positions):
        rows = self.first[np.asarray(shots)] + np.asarray(positions)
        return torch.from_numpy(np.array(self.carriers[rows]))

    def frame(self, shot, position):
        return self.pixels.frame(shot, position)


# ---------------------------------------------------------------------------
# training: batches (one Dataset item = one planned batch), the H-JEPA step, checkpoints
# ---------------------------------------------------------------------------

SEEDS = tuple(v1.SEEDS)  # 20260908, 20260909, 20260910
SELECTION_SEED = SEEDS[0]
CHECKPOINT_EVERY = 2000
LOG_EVERY = 50
PIXEL_LOADER = {"num_workers": 5, "prefetch_factor": 2}


def valid_clips(arm, ends):
    from world_model.hjepa import novphy as hn
    geometry = hn.Geometry(arm)
    return int(sum(geometry.valid_starts(int(e)) for e in ends))


class BatchSource(torch.utils.data.Dataset):
    """Batch b of a run: hn.plan_batch with PCG64([seed, b]) (deterministic, resumable at any step)."""

    def __init__(self, output, arm, seed, steps, batch):
        from world_model.hjepa import novphy as hn
        self.hn = hn
        self.output, self.arm, self.seed, self.steps, self.batch = output, arm, seed, steps, batch
        self.spec = hn.ARMS[arm]
        self.geometry = hn.Geometry(arm)
        self._data = FitData(output)
        self.ends = self._data.ends
        valid = np.array([self.geometry.valid_starts(int(e)) for e in self.ends], dtype=np.int64)
        self.cumulative = np.concatenate([[0], np.cumsum(valid)])

    def __getstate__(self):
        """Loader workers reopen the stores (the carrier memmap and pixel file handles are not pickled)."""
        state = dict(self.__dict__)
        if isinstance(state.get("_data"), FitData):
            state["_data"] = None
        return state

    @property
    def data(self):
        if self._data is None:
            self._data = FitData(self.output)
        return self._data

    def __len__(self):
        return self.steps

    def plan(self, b):
        rng = np.random.Generator(np.random.PCG64([self.seed, b]))
        return self.hn.plan_batch(rng, self.ends, self.cumulative, self.geometry, self.spec["sampler"], self.batch)

    def __getitem__(self, b):
        clips, starts, needed = self.plan(b)
        fs = self.geometry.frameskip
        persist = self.spec["action"] == "persist"
        if self.spec["input"] == "pixels":
            x = torch.stack([torch.stack([self.data.frame(shot, start + k * fs) for k in needed])
                             for shot, start in clips])
        else:
            shots = np.repeat([c[0] for c in clips], len(needed))
            positions = np.array([start + k * fs for _, start in clips for k in needed])
            x = self.data.carrier_rows(shots, positions).view(len(clips), len(needed), 236)
        action = torch.stack([self.hn.level1_actions(self.data.actions[shot], start, self.geometry, persist)
                              for shot, start in clips])
        return {"b": b, "x": x, "action": action, "starts": [starts[level] for level in sorted(starts)],
                "needed": needed, "clips": clips}


def run_dir(output, arm, seed, lam=None):
    base = Path(output) / ("selection" if lam is not None else "models") / arm
    return base / (f"lam-{lam}" if lam is not None else "") / f"seed-{seed}"


def train_run(output, arm, seed, lam, steps, batch, out_dir, device="cuda", max_steps=None, source=None):
    """Train one arm-seed (resumable from out_dir/resume.pt). A nonfinite loss is a typed terminal failure:
    model.pt is written with complete False and the run is never retried."""
    from world_model.hjepa import novphy as hn
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "model.pt"
    if final.is_file():
        log(f"{arm} seed {seed}: model.pt exists; skipped")
        return read_json(out_dir / "report.json")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg = hn.arm_config(arm, lam)
    torch.manual_seed(seed)
    model, losses = hn.build_model(arm, lam)
    wrapper = hn.Wrapper(model, losses).to(device).train()
    optimizer = hn.make_optimizer(model, cfg)
    scheduler = hn.WarmupCosine(optimizer, hn.warmup_steps(steps), steps)
    start, elapsed = 0, 0.0
    resume = out_dir / "resume.pt"
    if resume.is_file():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start, elapsed = state["step"], state["elapsed"]
        log(f"{arm} seed {seed}: resumed at step {start}")
    source = source or BatchSource(output, arm, seed, steps, batch)
    sampler = list(range(start, steps if max_steps is None else min(steps, start + max_steps)))
    pixels = hn.ARMS[arm]["input"] == "pixels"
    loader = torch.utils.data.DataLoader(source, batch_size=None, sampler=sampler, pin_memory=pixels,
                                         **(PIXEL_LOADER if pixels else {"num_workers": 2, "prefetch_factor": 4}))
    geometry = hn.Geometry(arm)
    log_path = out_dir / "train-log.jsonl"
    began = time.monotonic()
    failure = None
    step = start
    trace = []
    with open(log_path, "a") as log_file:
        for item in loader:
            step = item["b"]
            starts = {level: s for level, s in enumerate(item["starts"], start=1)}
            model.plan = (geometry.level1_steps, starts, item["needed"])
            batch_gpu = {"pixels_level1": item["x"].to(device, non_blocking=True),
                         "action_level1": item["action"].to(device, non_blocking=True)}
            try:
                stats = hn.train_step(wrapper, cfg, optimizer, scheduler, batch_gpu)
            except ValueError as error:
                if "NaN loss" not in str(error):
                    raise
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": str(error)}
                break
            if not np.isfinite(stats["loss"]):
                failure = {"typed_failure": "nonfinite_loss", "step": step, "message": "nonfinite total loss"}
                break
            if step % LOG_EVERY == 0 or step == steps - 1:
                gc.collect()  # reference cycles can hold a step's graph until a rare generation-2 collection
                entry = {"step": step, "lr": [g["lr"] for g in optimizer.param_groups],
                         "seconds": elapsed + time.monotonic() - began,
                         "allocated_gib": torch.cuda.memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
                         **stats}
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                trace.append(entry)
            if (step + 1) % CHECKPOINT_EVERY == 0 and step + 1 < steps:
                atomic_torch(resume, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                      "scheduler": scheduler.state_dict(), "cpu_rng": torch.get_rng_state(),
                                      "cuda_rng": torch.cuda.get_rng_state(), "step": step + 1,
                                      "elapsed": elapsed + time.monotonic() - began})
    seconds = elapsed + time.monotonic() - began
    if failure is None and max_steps is not None and step + 1 < steps:
        return {"partial": True, "steps_done": step + 1 - start, "seconds": time.monotonic() - began}
    complete = failure is None
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
    payload = {"schema": "issue_123_model_v1", "arm": arm, "seed": seed, "lambda": lam, "steps": steps, "batch": batch,
               "complete": complete, "typed_failure": failure, "state_dict": model.state_dict() if complete else None,
               "parameters": sum(p.numel() for p in model.parameters())}
    atomic_torch(final, payload)
    report = {"arm": arm, "seed": seed, "lambda": lam, "steps": steps, "batch": batch, "complete": complete,
              "typed_failure": failure, "wall_seconds": seconds, "gpu_seconds": seconds, "peak_gib": peak,
              "model_sha256": sha256_of(final), "parameters": payload["parameters"],
              "final": trace[-1] if trace else None, "finished_at": utc_now()}
    write_json(out_dir / "report.json", report)
    if resume.is_file() and complete:
        resume.unlink()
    log(f"{arm} seed {seed} lam {lam}: {'complete' if complete else failure['typed_failure']} after {seconds:.0f}s")
    return report


def load_trained(path, device="cuda"):
    """(model, payload) of a model.pt, or (None, payload) for a typed failure."""
    from world_model.hjepa import novphy as hn
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "issue_123_model_v1":
        raise ValueError(f"{path}: schema differs")
    if not payload["complete"]:
        return None, payload
    model, _ = hn.build_model(payload["arm"], payload["lambda"], checkpointing=False)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


# ---------------------------------------------------------------------------
# readouts and diagnostics (fit split only; policy split for the guard / selection AUC); deterministic
# ---------------------------------------------------------------------------

def fit_labels(output):
    """Engine labels of the fit probe frames, in FitData shot order."""
    shots = {}
    for path in sorted((Path(output) / "labels" / "fit").glob("labels-*.pt")):
        shots.update(torch.load(path, weights_only=True)["shots"])
    return shots


def grid_items():
    """The 16 grid candidates (ordinal, action) of the frozen training candidate set."""
    return sorted(({"ordinal": e["ordinal"], "action": c["action"]} for c in rt.candidates()
                   for e in c["inventory_entries"] if e["inventory"] == "grid"), key=lambda item: item["ordinal"])


@torch.no_grad()
def encode_top(model, x, device="cuda", chunk=512):
    """Per-level encodings of frames x (uint8 pixels or float carriers) in chunks; returns a list per level."""
    from world_model.hjepa import novphy as hn
    outs = None
    for i in range(0, len(x), chunk):
        levels = hn.encode_levels(model, x[i:i + chunk].to(device))
        outs = [[] for _ in levels] if outs is None else outs
        for k, z in enumerate(levels):
            outs[k].append(z.float().cpu())
    return [torch.cat(o) for o in outs]


@torch.no_grad()
def rollout(model, arm, z0_top, launches, device="cuda", chunk=4096):
    """Top-level states 0..T [N, T+1, D] of every (anchor encoding, launch) row."""
    from world_model.hjepa import novphy as hn
    geometry = hn.Geometry(arm)
    persist = hn.ARMS[arm]["action"] == "persist"
    out = []
    for i in range(0, len(launches), chunk):
        act = hn.top_action_embeddings(model, launches[i:i + chunk].to(device), geometry, persist)
        out.append(hn.rollout_top(model, z0_top[i:i + chunk].to(device), act).float().cpu())
    return torch.cat(out)


class FrameItems(torch.utils.data.Dataset):
    def __init__(self, output, split, items):
        self.output, self.split, self.items, self.store = output, split, items, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.store is None:
            self.store = PixelStore(self.output, self.split)
        shot, position = self.items[i]
        return self.store.frame(shot, position)


def load_frames(output, split, items, workers=8):
    loader = torch.utils.data.DataLoader(FrameItems(output, split, items), batch_size=256, num_workers=workers)
    return torch.cat(list(loader)) if items else torch.empty(0, 3, 240, 320, dtype=torch.uint8)


def encode_items(output, model, arm, split, items, carriers, device="cuda"):
    """Per-level encodings of (shot, position) items: pixels from the split's store, or the given carriers."""
    from world_model.hjepa import novphy as hn
    if hn.ARMS[arm]["input"] == "carriers":
        return encode_top(model, carriers, device)
    out = None
    for i in range(0, len(items), 16384):
        frames = load_frames(output, split, items[i:i + 16384])
        levels = encode_top(model, frames, device)
        out = [[z] for z in levels] if out is None else [o + [z] for o, z in zip(out, levels)]
    return [torch.cat(o) for o in out]


def policy_split(output):
    """Policy-split membership: grid mixed cells (gb.inventory_entries rule), anchors and per-shot data."""
    from scripts import run_gate_b as gb
    data = torch.load(policy_data_path(output), map_location="cpu", weights_only=True)
    branches = split_branches("policy")
    verdicts = split_verdicts("policy")
    entries = gb.inventory_entries(split_levels("policy"), branches, verdicts, "grid")
    cells = {m: e for m, e in entries.items() if set(e["verdicts"].values()) == {True, False}}
    store = PixelStore(output, "policy")
    return {"data": data, "cells": cells, "store": store}


def cost_rows(items, z_end_carrier, z0_carrier, costs, extra=None):
    """EndpointCosts rows (tie-free primary) of R1 carriers; extra: {key: [N] costs} added per row."""
    rows = []
    for slot, item in enumerate(items):
        row = costs.row(item["ordinal"], z_end_carrier[slot], z0_carrier)
        for key, values in (extra or {}).items():
            value = values[slot]
            row[key] = None if row["excluded"] or value is None or not np.isfinite(value) else float(value)
        rows.append(row)
    return rows


def pooled_auc(entries, rows_by_member, key):
    from scripts import run_decision_chain_attribution as m99
    values = [m99.view(entries[m], rows_by_member[m], key)["auc"] for m in sorted(entries)]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def policy_aucs(output, model, arm, readouts, policy, device="cuda"):
    """Policy-split grid AUCs: predicted endpoints under R1 (and R2 / R3 / R4), and the readout ceiling (R1 on the
    true frame min(225, last) encodings of every candidate's shot)."""
    from world_model.hjepa import novphy as hn
    from world_model.planning import scoring_harness as harness
    from scripts import run_lookahead_control as wlc
    costs = harness.EndpointCosts(wlc.load_objective())
    cells, data, store = policy["cells"], policy["data"], policy["store"]
    members = sorted(cells)
    pixels = hn.ARMS[arm]["input"] == "pixels"
    anchor_keys = [data["anchor_frames"][m] for m in members]
    if pixels:
        z0 = encode_top(model, torch.stack([store.frame(store.row[k], 0) for k in anchor_keys]), device)[-1]
    else:
        z0 = encode_top(model, torch.stack([data["anchors"][m] for m in members]), device)[-1]
    z0_by = dict(zip(members, z0))
    flat, launches, z0s = [], [], []
    for m in members:
        for item in cells[m]["inventory"]:
            flat.append((m, item))
            launches.append(wlc_action(item["action"]))
            z0s.append(z0_by[m])
    states = rollout(model, arm, torch.stack(z0s), torch.stack(launches), device)
    z_end = states[:, -1]
    true_items = []
    for m, item in flat:
        shot = data["shots"][item["key"]]
        q = min(225, shot["end"])
        true_items.append((item["key"], q, shot))
    if pixels:
        z_true = encode_top(model, torch.stack([store.frame(store.row[k], q) for k, q, _ in true_items]), device)[-1]
    else:
        z_true = encode_top(model, torch.stack([s["carriers"][(s["positions"] == q).nonzero()[0, 0]]
                                                for _, q, s in true_items]), device)[-1]
    out = {}
    with torch.no_grad():
        r1 = readouts["R1"].to(device)
        c_end, c_true, c0 = r1(z_end.to(device)).cpu(), r1(z_true.to(device)).cpu(), r1(z0.to(device)).cpu()
        c0_by = dict(zip(members, c0))
        variants = {"R1": c_end}
        for name in ("R2", "R4"):
            if name in readouts:
                variants[name] = readouts[name].to(device)(z_end.to(device)).cpu()
                variants[name + "_anchor"] = dict(zip(members, readouts[name].to(device)(z0.to(device)).cpu()))
        r3 = ((z_end - readouts["R3"]) ** 2).sum(1) if "R3" in readouts else None
    rows = {"R1": {}, "ceiling": {}, "R2": {}, "R3": {}, "R4": {}}
    k = 0
    for m in members:
        n = len(cells[m]["inventory"])
        items = cells[m]["inventory"]
        rows["R1"][m] = cost_rows(items, c_end[k:k + n], c0_by[m], costs)
        rows["ceiling"][m] = cost_rows(items, c_true[k:k + n], c0_by[m], costs)
        for name in ("R2", "R4"):
            if name in variants:
                rows[name][m] = cost_rows(items, variants[name][k:k + n], variants[name + "_anchor"][m], costs)
        if r3 is not None:
            rows["R3"][m] = [{"ordinal": it["ordinal"], "excluded": False, "goal": float(r3[k + j])}
                             for j, it in enumerate(items)]
        k += n
    for name, key in (("R1", harness.PRIMARY_COST), ("ceiling", harness.PRIMARY_COST), ("R2", harness.PRIMARY_COST),
                      ("R4", harness.PRIMARY_COST), ("R3", "goal")):
        out[name] = pooled_auc(cells, rows[name], key) if rows[name] else None
    out["cells"] = len(members)
    return out


def readouts_path(run):
    return Path(run) / "readouts.pt"


def fit_readouts(output, run, device="cuda", data=None, policy=None, labels=None):
    """R1 / R2 / R3 / R4 / P_c on fit frames, the collapse and action diagnostics on fit shots and anchors, and
    the policy-split AUCs; all under deterministic_scoring(). Writes readouts.pt and readouts.json in run."""
    from world_model.hjepa import novphy as hn
    from world_model.planning import scoring_harness as harness
    run = Path(run)
    if (run / "readouts.json").is_file():
        return read_json(run / "readouts.json")
    began = time.monotonic()
    data = data or FitData(output)
    policy = policy or policy_split(output)
    labels = labels if labels is not None else fit_labels(output)
    with harness.deterministic_scoring() as scoring_policy:
        model, payload = load_trained(run / "model.pt", device)
        if model is None:
            report = {"typed_failure": payload["typed_failure"], "scoring_policy": scoring_policy}
            write_json(run / "readouts.json", report)
            return report
        arm, seed = payload["arm"], payload["seed"]
        # 1. fit probe frames: every 15th position of every fit shot
        items, episodes = [], []
        for shot, end in enumerate(data.ends):
            for p in probe_positions(int(end)):
                items.append((shot, p))
                episodes.append(shot)
        shots_np = np.array([s for s, _ in items])
        positions_np = np.array([p for _, p in items])
        targets = data.carrier_rows(shots_np, positions_np)
        levels = encode_items(output, model, arm, "fit", items, targets, device)
        z_top = levels[-1]
        presence = torch.cat([labels[data.keys[s]]["presence"] for s in range(len(data.keys))]).float()
        centers = torch.cat([labels[data.keys[s]]["centers"] for s in range(len(data.keys))])
        if presence.shape[0] != z_top.shape[0]:
            raise ValueError("fit labels and probe frames are misaligned")
        r1, r1_fit = hn.fit_probe(z_top, targets, seed, device=device)
        pc, pc_fit = hn.fit_slot_probe(z_top, presence, centers, seed, device=device)
        dim = z_top.shape[1]
        collapse = [{"level": k + 1, "effective_rank": hn.effective_rank(z), "within_episode_share":
                     hn.within_episode_share(z, episodes), "dim": int(z.shape[1])} for k, z in enumerate(levels)]
        # 2. fit anchors (the retrain's anchors: frame 0 of each level's first complete candidate)
        fit_anchor_store = torch.load(rt.anchors_path(RETRAIN), map_location="cpu", weights_only=True)
        members = sorted(fit_anchor_store["anchors"])
        row_of = {k: i for i, k in enumerate(data.keys)}
        if hn.ARMS[arm]["input"] == "pixels":
            z0 = encode_top(model, torch.stack([data.frame(row_of[fit_anchor_store["frames"][m]], 0) for m in members]),
                            device)[-1]
        else:
            z0 = encode_top(model, torch.stack([fit_anchor_store["anchors"][m] for m in members]), device)[-1]
        z0_by = dict(zip(members, z0))
        grid = grid_items()
        launches = torch.stack([wlc_action(item["action"]) for item in grid])
        ends = rollout(model, arm, z0.repeat_interleave(len(grid), 0), launches.repeat(len(members), 1), device)[:, -1]
        sensitivity = hn.action_sensitivity(ends.view(len(members), len(grid), -1))
        # 3. R2: predicted endpoints of every fit shot from its level anchor under its executed launch
        shot_z0 = torch.stack([z0_by[m] for m in data.members])
        states = rollout(model, arm, shot_z0, data.actions, device)
        z_end = states[:, -1]
        end_positions = np.minimum(225, data.ends)
        end_targets = data.carrier_rows(np.arange(len(data.keys)), end_positions)
        r2, r2_fit = hn.fit_probe(z_end, end_targets, seed, device=device)
        # 4. R3: success prototype of the true frame-min(225, end) top-level encodings of pig-removing fit shots
        verdicts = read_json(RETRAIN / "fit-verdicts.json")["branches"]
        removed = {b["key"]: bool(b["pig_removed"]) for b in verdicts}
        winners = [i for i, k in enumerate(data.keys) if removed[k]]
        win_items = [(i, int(end_positions[i])) for i in winners]
        z_win = encode_items(output, model, arm, "fit", win_items,
                             data.carrier_rows(np.array(winners), end_positions[winners]), device)[-1]
        prototype = z_win.mean(0)
        # 5. R4: R2 + softplus pairwise ranking on the fit mixed cells (retrain cell rule), 8 cells per step
        cells = rt.ranking_cells(split_branches("fit"), {k: removed[k] for k in data.keys},
                                 {m: torch.zeros(236) for m in members})
        key_of = {(b["member"], e["inventory"], e["ordinal"]): b["key"] for b in split_branches("fit")
                  for e in b["candidate"]["inventory_entries"]}
        cell_rows = []
        for cell in cells:
            idx = [row_of[key_of[(cell["member"], cell["inventory"], o)]] for o in cell["ordinals"]]
            verdict = cell["verdict"][:len(idx)]
            cell_rows.append((torch.tensor(idx), verdict.bool(), cell["member"]))
        z_end_dev = z_end.to(device)
        z0_dev = {m: z0_by[m].to(device) for m in members}
        from world_model.planning.scoring_harness import EndpointCosts
        from scripts import run_lookahead_control as wlc
        costs = EndpointCosts(wlc.load_objective())
        rank_gen = torch.Generator().manual_seed(seed + 4)

        def ranking_loss(probe):
            picks = torch.randint(len(cell_rows), (8,), generator=rank_gen).tolist()
            total = 0.0
            for p in picks:
                idx, verdict, member = cell_rows[p]
                out = probe(z_end_dev[idx.to(device)])
                anchor = probe(z0_dev[member][None])[0]
                pres = out[:, costs.presence]
                disp = torch.sqrt((out[:, costs.center[0]] - anchor[costs.center[0]]) ** 2
                                  + (out[:, costs.center[1]] - anchor[costs.center[1]]) ** 2 + 1e-12)
                c = harness.tie_free(pres, disp)
                diff = c[verdict.to(device)][:, None] - c[~verdict.to(device)][None, :]
                total = total + torch.nn.functional.softplus(diff / 0.1).mean()
            return 0.1 * total / len(picks)

        r4, r4_fit = hn.fit_probe(z_end, end_targets, seed, extra_loss=ranking_loss, device=device)
        readouts = {"R1": r1.cpu(), "R2": r2.cpu(), "R3": prototype, "R4": r4.cpu()}
        aucs = policy_aucs(output, model, arm, readouts, policy, device)
    atomic_torch(readouts_path(run), {"schema": "issue_123_readouts_v1", "arm": arm, "seed": seed,
                                      "model_sha256": sha256_of(run / "model.pt"), "dim": dim,
                                      "R1": r1.cpu().state_dict(), "R2": r2.cpu().state_dict(),
                                      "R4": r4.cpu().state_dict(), "R3_prototype": prototype,
                                      "P_c": pc.cpu().state_dict()})
    report = {"schema": "issue_123_readouts_report_v1", "arm": arm, "seed": seed, "lambda": payload["lambda"],
              "scoring_policy": scoring_policy, "typed_failure": None, "dim": dim,
              "fit_frames": len(items), "fit_shots": len(data.keys), "R1_fit": r1_fit, "P_c_fit": pc_fit,
              "R2_fit": r2_fit, "R4_fit": r4_fit, "R3_winners": len(winners), "ranking_cells": len(cell_rows),
              "collapse": collapse, "action_sensitivity": sensitivity, "fit_anchors": len(members),
              "policy_auc": aucs, "readouts_sha256": sha256_of(readouts_path(run)),
              "model_sha256": sha256_of(run / "model.pt"), "seconds": time.monotonic() - began, "fitted_at": utc_now()}
    write_json(run / "readouts.json", report)
    log(f"readouts {arm} seed {seed}: policy R1 AUC {aucs['R1']}, ceiling {aucs['ceiling']}, "
        f"erank {collapse[-1]['effective_rank']:.1f}/{dim}, chi {collapse[-1]['within_episode_share']:.3f}, "
        f"sensitivity {sensitivity:.3f} ({report['seconds']:.0f}s)")
    return report


def load_readouts(run, device="cuda"):
    from world_model.hjepa import novphy as hn
    payload = torch.load(readouts_path(run), map_location="cpu", weights_only=True)
    dim = payload["dim"]
    out = {}
    for name in ("R1", "R2", "R4"):
        state = payload[name]
        probe = hn.Probe(dim, 236, state["mean"], state["std"])
        probe.load_state_dict(state)
        out[name] = probe.to(device).eval()
    pc = hn.SlotProbe(dim)
    pc.load_state_dict(payload["P_c"])
    out["P_c"] = pc.to(device).eval()
    out["R3"] = payload["R3_prototype"].to(device)
    return out


# ---------------------------------------------------------------------------
# smoke (pre-freeze; fit / policy only): throughput, projection, SIGReg lambda selection
# ---------------------------------------------------------------------------

SMOKE_STEPS = {"pixels": 40, "carriers": 200}
CAP_GPU_HOURS = 260.0
CAP_WALL_HOURS = 72.0
GPUS = 4


def measure_throughput(output, arm, batch, steps, device="cuda"):
    """Seconds per optimizer step of a scratch run (nothing kept): the median of the timed steps after warm-up."""
    from world_model.hjepa import novphy as hn
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cfg = hn.arm_config(arm, hn.SIGREG_LAMBDAS[0])
    torch.manual_seed(0)
    model, losses = hn.build_model(arm, hn.SIGREG_LAMBDAS[0])
    wrapper = hn.Wrapper(model, losses).to(device).train()
    optimizer = hn.make_optimizer(model, cfg)
    scheduler = hn.WarmupCosine(optimizer, 10, 100000)
    source = BatchSource(output, arm, 0, steps, batch)
    pixels = hn.ARMS[arm]["input"] == "pixels"
    loader = torch.utils.data.DataLoader(source, batch_size=None, sampler=list(range(steps)), pin_memory=pixels,
                                         **(PIXEL_LOADER if pixels else {"num_workers": 2, "prefetch_factor": 4}))
    geometry = hn.Geometry(arm)
    times = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    last = time.monotonic()
    for item in loader:
        starts = {level: s for level, s in enumerate(item["starts"], start=1)}
        model.plan = (geometry.level1_steps, starts, item["needed"])
        hn.train_step(wrapper, cfg, optimizer, scheduler, {"pixels_level1": item["x"].to(device),
                                                           "action_level1": item["action"].to(device)})
        torch.cuda.synchronize()
        now = time.monotonic()
        times.append(now - last)
        last = now
    warm = times[len(times) // 4:]
    return {"arm": arm, "batch": batch, "steps_timed": len(warm), "seconds_per_step": float(np.median(warm)),
            "mean_seconds_per_step": float(np.mean(warm)), "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}


def smoke_throughput(output, device="cuda"):
    """Steps/s of every arm at its budget's batch on one RTX 5090 (scratch runs, nothing kept)."""
    from world_model.hjepa import novphy as hn
    data = FitData(output)
    out = {}
    for arm, spec in hn.ARMS.items():
        steps, batch = hn.steps_for(arm, valid_clips(arm, data.ends))
        measured = measure_throughput(output, arm, batch, SMOKE_STEPS[spec["input"]], device)
        out[arm] = {**measured, "budget": spec["budget"], "steps": steps, "valid_clips": valid_clips(arm, data.ends),
                    "projected_gpu_hours_per_seed": steps * measured["seconds_per_step"] / 3600.0}
        log(f"throughput {arm}: {measured['seconds_per_step']:.3f} s/step at batch {batch}, {steps} steps -> "
            f"{out[arm]['projected_gpu_hours_per_seed']:.2f} GPU-h per seed, peak {measured['peak_gib']:.1f} GiB")
        torch.cuda.empty_cache()
    return out


def selection_runs():
    """The one pre-declared selection per input type: SIGReg lambda in {0.09, 0.18} on HJ-px / HJ-car-m at B1,
    seed 20260908, by policy-split grid AUC under R1 (ties -> 0.09)."""
    from world_model.hjepa import novphy as hn
    return [(arm, lam) for arm in ("HJ-px", "HJ-car-m") for lam in hn.SIGREG_LAMBDAS]


def selection(output, gpu, gpus):
    """Train this GPU's selection runs (fit split only)."""
    from world_model.hjepa import novphy as hn
    runs = [r for i, r in enumerate(selection_runs()) if i % gpus == gpu]
    for arm, lam in runs:
        run = run_dir(output, arm, SELECTION_SEED, lam)
        if (run / "model.pt").is_file():
            continue
        report = train_run(output, arm, SELECTION_SEED, lam, hn.B1["steps"], hn.B1["batch"], run)
        mark(output, f"selection-train-{arm}-lam{lam}", report["wall_seconds"], report["gpu_seconds"])
    return 0


def selection_readouts(output, gpu, gpus):
    """Fit the readouts of this GPU's selection runs (fit split; policy split for the selection AUC only)."""
    runs = [r for i, r in enumerate(selection_runs()) if i % gpus == gpu]
    for arm, lam in runs:
        run = run_dir(output, arm, SELECTION_SEED, lam)
        if (run / "readouts.json").is_file():
            continue
        t0 = time.monotonic()
        fit_readouts(output, run)
        mark(output, f"selection-readouts-{arm}-lam{lam}", time.monotonic() - t0, time.monotonic() - t0)
    return 0


def selection_result(output):
    from world_model.hjepa import novphy as hn
    out = {}
    for input_kind, arm in (("pixels", "HJ-px"), ("carriers", "HJ-car-m")):
        aucs = {}
        for lam in hn.SIGREG_LAMBDAS:
            report = read_json(run_dir(output, arm, SELECTION_SEED, lam) / "readouts.json")
            aucs[str(lam)] = None if report.get("typed_failure") else report["policy_auc"]["R1"]
        valid = {k: v for k, v in aucs.items() if v is not None}
        best = max(valid.values()) if valid else None
        chosen = None if best is None else float(min(float(k) for k, v in valid.items() if round(v, 4) == round(best, 4)))
        out[input_kind] = {"arm": arm, "policy_grid_auc_R1": aucs, "selected_lambda": chosen,
                           "rule": "largest policy-split pooled grid AUC under R1 at B1, seed 20260908; ties (4 d.p.) -> 0.09"}
    return out


def ledger_dir(output):
    return Path(output) / "ledger"


def mark(output, phase, wall, gpu, **extra):
    path = ledger_dir(output) / f"{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    block = read_json(path) if path.is_file() else {"wall_seconds": 0.0, "gpu_seconds": 0.0}
    write_json(path, {**block, "wall_seconds": block["wall_seconds"] + wall, "gpu_seconds": block["gpu_seconds"] + gpu,
                      "last_at": utc_now(), **extra})


def ledger_totals(output):
    entries = {p.stem: read_json(p) for p in sorted(ledger_dir(output).glob("*.json"))} if ledger_dir(output).is_dir() else {}
    return entries, sum(e["gpu_seconds"] for e in entries.values()), sum(e["wall_seconds"] for e in entries.values())


def lpt_queues(jobs, gpus=GPUS):
    """Longest-processing-time assignment of (name, hours) jobs to GPUs; returns queues and their loads."""
    loads = [0.0] * gpus
    queues = [[] for _ in range(gpus)]
    for name, hours in sorted(jobs, key=lambda j: -j[1]):
        k = int(np.argmin(loads))
        queues[k].append(name)
        loads[k] += hours
    return queues, loads


def fidelity_projection():
    """Measured Tier E / Tier T GPU seconds so far plus the remaining runs at the measured (or smoke) rate."""
    from scripts import run_hjepa_fidelity as fid
    entries = []
    if fid.ledger_path().is_file():
        entries = [json.loads(line) for line in fid.ledger_path().read_text().splitlines() if line.strip()]
    done = {(e["tier"], e["model"], e["seed"]): e for e in entries if e["exit_code"] == 0}
    spent = sum(e["gpu_seconds"] for e in entries)
    per = {}
    for tier in ("tier-e", "tier-t"):
        for model in fid.MODELS:
            measured = [e["gpu_seconds"] for (t, m, _), e in done.items() if t == tier and m == model]
            per[(tier, model)] = float(np.mean(measured)) if measured else None
    return {"spent_gpu_seconds": spent, "per_run_measured_gpu_seconds": {f"{t}/{m}": v for (t, m), v in per.items()},
            "runs_done": len(done), "remaining": [f"{t}/{m}/{s}" for t in ("tier-e", "tier-t") for m in fid.MODELS
                                                  for s in fid.SEEDS if (t, m, s) not in done]}


def projection(output, throughput, readout_seconds, fidelity_remaining_hours):
    """GPU-h and wall projection of the unconditional work against the 260 GPU-h / 72 h cap."""
    from world_model.hjepa import novphy as hn
    jobs = [(f"{arm}/{seed}", throughput[arm]["projected_gpu_hours_per_seed"]) for arm in hn.ARMS for seed in SEEDS]
    train_hours = sum(h for _, h in jobs)
    readout_hours = readout_seconds * len(jobs) / 3600.0
    scoring_hours = 0.15 * len(jobs) + 1.0  # anchors / floors / rows: projected from the readout encode rates
    bench_hours = 2.0
    _, spent_gpu, spent_wall = ledger_totals(output)
    queues, loads = lpt_queues(jobs)
    total = (train_hours + readout_hours + scoring_hours + bench_hours + fidelity_remaining_hours["gpu_hours"]
             + fidelity_remaining_hours["spent_gpu_hours"] + spent_gpu / 3600.0)
    wall = (max(loads) + (readout_hours + scoring_hours + bench_hours) / GPUS
            + fidelity_remaining_hours["wall_hours"])
    return {"training_gpu_hours": train_hours, "readout_gpu_hours": readout_hours, "scoring_gpu_hours": scoring_hours,
            "benchmark_gpu_hours": bench_hours, "fidelity": fidelity_remaining_hours,
            "spent_issue123_gpu_hours": spent_gpu / 3600.0, "total_gpu_hours": total,
            "training_queue_hours": loads, "projected_wall_hours": wall, "cap_gpu_hours": CAP_GPU_HOURS,
            "cap_wall_hours": CAP_WALL_HOURS,
            "within_cap": bool(total <= CAP_GPU_HOURS and wall <= CAP_WALL_HOURS)}


def smoke_tests():
    """Run the uncertain-code tests (clip sampler, action encoding, readout fit isolation, port check)."""
    import subprocess
    proc = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_issue_123_hjepa_comparison.py"], cwd=ROOT,
                          capture_output=True, text=True)
    return {"command": "pytest -q tests/test_issue_123_hjepa_comparison.py", "exit_code": proc.returncode,
            "tail": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-400:]}


def data_manifest(output):
    """Content hashes of every data input the training / readout / scoring modes read (fit and policy only)."""
    output = Path(output)
    files = [pixel_dir(output, s) / name for s in ("fit", "policy") for name in ("index.npz", "MANIFEST.json")]
    files += sorted(pixel_dir(output, "fit").glob("shard-*.jsonl")) + sorted(pixel_dir(output, "policy").glob("shard-*.jsonl"))
    files += [carriers_dir(output) / "fit-index.npz", carriers_dir(output) / "fit-carriers.bin", policy_data_path(output)]
    files += sorted((output / "labels" / "fit").glob("labels-*.pt"))
    return [{"artifact": str(p.relative_to(ROOT)), "sha256": sha256_of(p), "bytes": p.stat().st_size} for p in files]


def smoke(output):
    """Pre-freeze smoke on the fit / policy splits: data accounting, G-port, tests, throughput, projection and the
    lambda selection (the selection runs must be complete). Writes smoke.json."""
    from world_model.hjepa import novphy as hn
    output = Path(output)
    began = time.monotonic()
    data = FitData(output)
    fit_pixels = PixelStore(output, "fit")
    policy = torch.load(policy_data_path(output), map_location="cpu", weights_only=True)
    labels = fit_labels(output)
    accounting = {"fit_shots": len(data.keys), "fit_positions": int(data.first[-1]),
                  "fit_levels": len(set(data.members)), "fit_pixel_frames": int(fit_pixels.first[-1]),
                  "fit_pixel_shots_match": list(fit_pixels.keys) == data.keys,
                  "fit_label_shots": len(labels), "policy_shots": len(policy["shots"]),
                  "policy_levels": len(policy["anchors"]),
                  "streams_read": sorted({STREAMS["fit"], STREAMS["policy"]}),
                  "valid_clips": {arm: valid_clips(arm, data.ends) for arm in hn.ARMS},
                  "parameters": {arm: hn.parameter_count(arm) for arm in hn.ARMS},
                  "gate_b_unit_parameters": 1837690}
    gport_result = read_json(gport_dir(output) / "gport.json")
    tests = smoke_tests()
    throughput = smoke_throughput(output)
    selected = selection_result(output)
    readout_seconds = float(np.mean([read_json(run_dir(output, arm, SELECTION_SEED, lam) / "readouts.json")["seconds"]
                                     for arm, lam in selection_runs()]))
    fidelity = fidelity_projection()
    # remaining fidelity: Tier E evals and Tier T runs at their measured per-run GPU time (smoke-based where absent)
    measured = fidelity["per_run_measured_gpu_seconds"]
    fallback = {"tier-e": 3.0 * 3600, "tier-t": 6.0 * 3600}
    remaining_seconds = sum(measured.get("/".join(r.split("/")[:2])) or fallback[r.split("/")[0]]
                            for r in fidelity["remaining"])
    fid_hours = {"spent_gpu_hours": fidelity["spent_gpu_seconds"] / 3600.0, "gpu_hours": remaining_seconds / 3600.0,
                 "wall_hours": remaining_seconds / 3600.0 / GPUS, "remaining_runs": fidelity["remaining"],
                 "rule": "remaining runs at their measured mean GPU time (else 3 h per Tier-E eval, 6 h per Tier-T run)"}
    project = projection(output, throughput, readout_seconds, fid_hours)
    gpu_free = all(v["selected_lambda"] is not None for v in selected.values())
    ok = bool(gport_result["pass"] and tests["exit_code"] == 0 and accounting["fit_pixel_shots_match"]
              and accounting["fit_label_shots"] == len(data.keys) and gpu_free and project["within_cap"])
    evidence = {"schema": "issue_123_smoke_v1", "ok": ok, "at": utc_now(), "accounting": accounting,
                "gport": {"pass": gport_result["pass"], "tolerance": gport_result["tolerance"],
                          "arms": {a: {k: r[k] for k in ("loss_max_relative_delta", "embed_max_abs_delta",
                                                         "grad_norm_relative_delta", "grad_tensor_max_relative_l2",
                                                         "param_close_share", "param_max_abs_delta", "pass")}
                                   for a, r in gport_result["arms"].items()}},
                "tests": tests, "throughput": throughput, "selection": selected,
                "selection_reports": {f"{arm}/lam{lam}": read_json(run_dir(output, arm, SELECTION_SEED, lam) / "readouts.json")
                                      for arm, lam in selection_runs()},
                "projection": project, "wall_seconds": time.monotonic() - began}
    write_json(output / "smoke.json", evidence)
    mark(output, "smoke", evidence["wall_seconds"], evidence["wall_seconds"])
    log(f"smoke ok={ok}: projection {project['total_gpu_hours']:.0f} GPU-h, {project['projected_wall_hours']:.0f} h wall "
        f"(cap {CAP_GPU_HOURS:.0f} / {CAP_WALL_HOURS:.0f}); lambda {selected}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# plan (frozen before any training of the main arms and before any evaluation-split read)
# ---------------------------------------------------------------------------

IDENTITY = "issue-123-hjepa-comparison-v1"
SCHEMA_PLAN = "issue_123_plan_v1"
SCHEMA_RECORD = "issue_123_record_v1"
RUNNER = "scripts/run_hjepa_comparison.py"
VALIDATION_COMMAND = "python -u -m scripts.run_hjepa_comparison --validate"
MODULE = "world_model/hjepa/novphy.py"
DELTA = v1.DELTA
STOP_TOKEN = v1.STOP_TOKEN
GUARD_THRESHOLDS = {"effective_rank_fraction": 0.1, "within_episode_share": 0.05, "action_sensitivity": 0.05,
                    "readout_ceiling_auc": 0.65, "nonfinite_share": 0.10}
GATE_B = ROOT / ".local-artifacts/issue-108-gate-b-v1"
ISSUE_101 = ROOT / ".local-artifacts/issue-101-decision-aware-policy-v1"
J_ARMS = {"J": "J-pe-1e-04", "J_dec": "J-dec-1e-01"}
CONTRASTS = {
    "PC1": ("HJ-px", "U"), "PC2": ("HJ-car-m", "U"),
    "S1_pixels": ("HJ-px", "LeWM-px"), "S2_carriers": ("HJ-car-m", "LeWM-car-m"),
    "S3_pixels": ("HJ-px", "HWM-px"), "S4_carriers": ("HJ-car-m", "HWM-car-m"),
    "S5_pixels": ("HJ-px", "C*"), "S5_carriers": ("HJ-car-m", "C*"),
    "S6_pixels": ("HJ-px", "P"), "S6_carriers": ("HJ-car-m", "P"),
    "S7": ("HJ-px-B1", "U"), "S8": ("HJ-car-native", "U"),
    "S9_PC1_R2": ("HJ-px:R2", "U"), "S9_PC2_R2": ("HJ-car-m:R2", "U"),
    "S11_pixels_J": ("HJ-px", "J"), "S11_pixels_J_dec": ("HJ-px", "J_dec"),
    "S11_carriers_J": ("HJ-car-m", "J"), "S11_carriers_J_dec": ("HJ-car-m", "J_dec"),
}
WRONG_ANCHOR_ARMS = ("HJ-px", "HJ-car-m", "HJ-px-B1", "HJ-car-native")


def frozen_inputs(output):
    from scripts import run_hjepa_fidelity as fid  # noqa: F401  (bound by hash below)
    output = Path(output)
    vendor = ROOT / "world_model/hjepa/vendor"
    names = {"runner": ROOT / RUNNER, "module": ROOT / MODULE, "package_init": ROOT / "world_model/hjepa/__init__.py",
             "fidelity_runner": ROOT / "scripts/run_hjepa_fidelity.py", "tier_t_launcher": ROOT / "scripts/hjepa_train_ckpt.py",
             "port_reference": ROOT / "scripts/hjepa_port_reference.py",
             "tests": ROOT / "tests/test_issue_123_hjepa_comparison.py",
             "gport": gport_dir(output) / "gport.json", "smoke": output / "smoke.json",
             "scoring_harness": ROOT / "world_model/planning/scoring_harness.py",
             "retrain_runner": ROOT / rt.RUNNER, "v1_runner": ROOT / v1.RUNNER,
             "slot_encoder_fix_runner": ROOT / "scripts/run_slot_encoder_fix.py",
             "gate_b_runner": ROOT / "scripts/run_gate_b.py",
             "retrain_fit_verdicts": RETRAIN / "fit-verdicts.json", "retrain_fit_anchors": rt.anchors_path(RETRAIN),
             "retrain_handoff": RETRAIN / "gate_b_handoff.json", "issue_99_encoder": s99.OUTPUT / "encoder" / "encoder.pt",
             "cohort_plan": rt.COHORT, "power_plan": rt.POWER_PLAN, "campaign_branches": CAMPAIGN / "branches.json",
             "gate_b_plan": GATE_B / "plan.json", "gate_b_summary": GATE_B / "summary.json",
             "gate_b_anchors": GATE_B / "data" / "anchors-evaluation.pt",
             "issue_101_plan": ISSUE_101 / "plan.json", "issue_101_selection": ISSUE_101 / "selection.json"}
    for name in ("unit", "c_star"):
        for seed in SEEDS:
            names[f"gate_b_record_{name}_{seed}"] = GATE_B / "records" / f"{name}--seed{seed}.json"
    for label, arm in J_ARMS.items():
        for seed in SEEDS:
            names[f"issue_101_record_{label}_{seed}"] = ISSUE_101 / "records" / f"{arm}--seed{seed}.json"
    for rel in sorted(read_json(vendor / "VENDOR.json")["files"]):
        names[f"vendor_{rel}"] = vendor / rel
    names["vendor_manifest"] = vendor / "VENDOR.json"
    return [{"name": name, "artifact": str(Path(path).relative_to(ROOT)), "sha256": sha256_of(path)}
            for name, path in names.items()]


def frozen_plan(output, frozen_at):
    from world_model.hjepa import novphy as hn
    output = Path(output)
    evidence = read_json(output / "smoke.json")
    lam = {kind: evidence["selection"][kind]["selected_lambda"] for kind in ("pixels", "carriers")}
    data = FitData(output)
    arms = {}
    for arm, spec in hn.ARMS.items():
        steps, batch = hn.steps_for(arm, valid_clips(arm, data.ends))
        arms[arm] = {**spec, "lambda": lam[spec["input"]], "steps": steps, "batch": batch,
                     "valid_clips": valid_clips(arm, data.ends), "parameters": hn.parameter_count(arm),
                     "config": hn.arm_config(arm, lam[spec["input"]]),
                     "projected_gpu_hours_per_seed": evidence["throughput"][arm]["projected_gpu_hours_per_seed"]}
    jobs = [(f"{arm}/{seed}", arms[arm]["projected_gpu_hours_per_seed"]) for arm in arms for seed in SEEDS]
    queues, loads = lpt_queues(jobs)
    return {
        "schema": SCHEMA_PLAN, "identity": IDENTITY, "issue": 123, "related_issues": [108, 104, 109, 112, 101, 117],
        "frozen_at": frozen_at, "frozen_before_training_and_evaluation": True, "runner": RUNNER,
        "validation_command": VALIDATION_COMMAND, "engine_seconds": 0,
        "question": ("does H-JEPA (arXiv 2610.06805, kevinghst/H-JEPA @ 97054ec), reproduced on NovPhy, rank the "
                     "Gate-B grid candidates of the sealed #109 evaluation split better or worse than the Gate-B unit?"),
        "seeds": list(SEEDS), "selected_lambda": lam, "selection": evidence["selection"],
        "arms": arms, "existing_arms": {
            "U": "Gate-B unit LC : F-15-macro (hybrid, E99), its published Gate-B records (rows) and checkpoints",
            "C*": "LC : C-F-15, its published Gate-B records and checkpoints",
            "J": f"ICLR J ({J_ARMS['J']}), #101 published evaluation records",
            "J_dec": f"#101 J_dec ({J_ARMS['J_dec']}), #101 published evaluation records",
            "P": "the no-model action prior (Gate-B rule over the retrain fit-verdicts.json)"},
        "training": {"queues": queues, "queue_projected_hours": loads, "checkpoint_every": CHECKPOINT_EVERY,
                     "loader": {"pixels": PIXEL_LOADER, "carriers": {"num_workers": 2, "prefetch_factor": 4}},
                     "typed_failure": "a NaN / nonfinite loss ends the run (model.pt complete False); never retried",
                     "precision": "bf16-mixed autocast, TF32 on, cudnn.benchmark on (main_hjepa runtime settings)",
                     "recipe": {"optimizer": "AdamW, per-level lr (5e-5 level 1, 1e-5 upper), wd 1e-3",
                                "schedule": "linear warm-up over max(1, int(0.01 x steps)) then cosine to 0, per step",
                                "clip": "global grad-norm 1.0", "objective": "vendor hjepa_forward (unchanged)"}},
        "readouts": {"probe": dict(hn.PROBE), "fit_frames": "every 15th position 0..min(last, 285) of every fit shot",
                     "R1": "MLP(D->512, LN, GELU->236) frozen top-level encodings of real fit frames -> E99 carriers (primary)",
                     "R2": "same MLP on predicted endpoint latents of every fit shot (level anchor, executed launch) -> carrier at min(225, end)",
                     "R3": "goal latent: mean top-level encoding of true frame min(225, end) of pig-removing fit shots; cost = squared distance",
                     "R4": "R2 + 0.1 x softplus((c_success - c_failure) / 0.1) over 8 random fit mixed cells (grid, angle) per step",
                     "P_c": "MLP(D->512, LN, GELU->18x3) slot presence (BCE) + engine-projected centres (MSE on present slots)",
                     "policy": "every readout fit, encode and rollout under deterministic_scoring(); fit split only; "
                               "policy split only for the selection AUC and the G-readout ceiling",
                     "freeze": "readouts-manifest.json (sha256 of every model.pt and readouts.pt) before any evaluation anchor is encoded"},
        "scoring": {"metric": "Gate-B pooled grid AUC: 311 mixed grid cells x 3 seeds, tie-free cost of R1 carriers "
                              "(EndpointCosts.row(R1(z_end), R1(z_0))), m99.view, member-clustered bootstrap 10000 PCG64 7201",
                    "endpoint": "top-level rollout of 15 steps of 15 frames (H3 / H2-fs5) or 225 steps (L1) from the "
                                "level anchor; launch at frame 0, no-ops after (persist: launch every frame)",
                    "anchors": "the Gate-B anchors: carriers = its E99 batch-1 parses (anchors-evaluation.pt); pixels = "
                               "frame 0 of the same anchor branches (s99.image_tensor), encoded after the readout freeze",
                    "process": "one process that did not train, deterministic_scoring(), once",
                    "descriptive": ["offset / angle / power AUC", "top-1", "position MSE at t in (15, 30, 60, 150, 225) "
                                    "via P_c with its readout floor (P_c on the true frame encodings)",
                                    "R3 / R4 rankers", "CLEVRER / Physion action-free rollout error (exploratory)"]},
        "contrasts": {name: {"first": a, "second": b} for name, a, b in [(k, *v) for k, v in CONTRASTS.items()]},
        "wrong_anchor": {"arms": list(WRONG_ANCHOR_ARMS), "pool": "the 480 fit anchors (retrain anchors-fit.pt / frame 0 "
                                                                  "of the same branches)", "token": "S10 right - wrong, delta rule"},
        "token_rule": f"supported iff mean >= {DELTA} and lower > 0; not_supported_by_this_experiment iff upper < {DELTA}; "
                      f"otherwise {STOP_TOKEN}; primary contrasts in both directions",
        "guards": {"G_fid": "fidelity/gfid.json G_fid true (Tier E and Tier T pass)", "G_port": "gport/gport.json pass",
                   "G_collapse": "per arm and seed: top-level effective rank >= 0.1 D and within-episode variance share >= 0.05",
                   "G_action": "per arm and seed: endpoint action sensitivity >= 0.05 on the fit anchors (16 grid launches)",
                   "G_readout": "per arm: R1 on the true frame-min(225, last) encodings reaches pooled policy grid AUC >= 0.65",
                   "G_nonfinite": "per arm: <= 10 % nonfinite verdict slots on the grid cells",
                   "G_isolation": "training / readouts read fit (and policy) only; scored branches are evaluation-roster "
                                  "captures; wrong anchors are fit levels",
                   "G_chronology": "plan frozen_at <= readouts frozen_at <= anchors encoded_at <= first record scored_at",
                   "G_records": "records-manifest.json exists; --validate re-checks it, re-scores >= 1 record and exits 0",
                   "failure": "a failed guard turns every token involving the arm into readiness_or_precision_insufficient",
                   "thresholds": GUARD_THRESHOLDS},
        "dispositions": {
            "1": ("H-JEPA ranks better than our Gate-B model (scoped) iff PC1 or PC2 is supported in the H-JEPA - U "
                  "direction, G-fid, G-port and that arm's guards pass, and S10 is supported for that arm; names the "
                  "arm; triggers the HJ-LC follow-up ticket"),
            "2": ("our Gate-B model ranks better than H-JEPA (scoped) iff U - HJ-px and U - HJ-car-m are both supported "
                  "and G-fid, G-port and every guard of both arms pass; scope: #109 sealed split, Gate-B grid cells, "
                  "tie-free cost, readout R1, budgets B3 (pixels) / B1 (carriers), layout H3-Delta, Gate-B unit"),
            "3": "otherwise per-contrast tokens only (delta-rule wording)",
            "4": "S5 and S11 reported next to every primary contrast (C* 0.783 vs U 0.744 at Gate B)",
            "5": ("if PC1 holds only at B3 and S7 does not: 'H-JEPA's advantage requires a larger training budget than "
                  "ours'; the conditional arm U-B3 runs before any architectural reading")},
        "conditional": {"U-B3": "trigger: disposition 5; our LC recipe at the B3 update count (steps x batch), 3 seeds",
                        "closed_loop": "trigger: PC1 or PC2 supported for H-JEPA; GD planner live in the #104 player "
                                       "(#102 machinery); own engine-hour projection reported before launch"},
        "benchmarks": {"arms": ["HJ-bench (H3, matched carriers, uniform sampler, null action)", "LeWM-bench"],
                       "data": "#79 CLEVRER / #84 Physion-Dominoes annotation carriers, 12 scenes each, fold all",
                       "budget": "B1", "metric": "#79 field_errors position_mse at t in (15, 30, 60, 120) from contexts "
                                                 "0-3 through an R1 probe; next to #117 item-5 J (LC, full) rows",
                       "status": "EXPLORATORY, no ranking (no candidate actions)"},
        "caps": {"gpu_hours": CAP_GPU_HOURS, "wall_hours": CAP_WALL_HOURS, "stop": "over the cap: stop and report"},
        "disclosures": [
            "the evaluation split was scored by earlier tickets (#101-#121) for our models; it is pristine for H-JEPA",
            "H-JEPA's LeWM is its 4-layer re-implementation; HWM is its identity-encoder re-implementation",
            "anchor-mixture sampler (half of every batch starts at the launch frame) is an adaptation for action-sparse "
            "data; the uniform sampler is an ablation",
            "the level-1 ViT uses activation checkpointing (identical gradients) on 32 GB RTX 5090s",
            "320 x 240 frames, patch 16; the 20 x 20 ViT position table is bicubic-interpolated to 15 x 20 by the "
            "official interpolate_pos_encoding path",
            "RTX 5090 instead of H200; torch 2.13 (novphy) for the NovPhy arms, 2.7.1 (hjepa env) for the fidelity check"],
        "data_manifest": data_manifest(output),
        "inputs": frozen_inputs(output),
        "smoke_evidence": {k: evidence[k] for k in ("ok", "at", "accounting", "gport", "tests", "projection")},
    }


def load_plan(output):
    path = Path(output) / "plan.json"
    if not path.is_file():
        raise ValueError("plan.json missing; run --smoke and --prepare first")
    plan = read_json(path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("identity") != IDENTITY or not plan["smoke_evidence"]["ok"]:
        raise ValueError("plan.json is not the issue-123 protocol with a passed smoke")
    from scripts import post_freeze_amendments as amendments
    for entry in plan["inputs"] + plan["data_manifest"]:
        actual = sha256_of(ROOT / entry["artifact"])
        if not amendments.matches(entry["artifact"], entry["sha256"], actual):
            raise ValueError(f"frozen input {entry['artifact']} changed after the freeze")
    return plan


def prepare(output):
    output = Path(output)
    if (output / "plan.json").is_file():
        load_plan(output)
        log("existing frozen plan validated")
        return 0
    if (output / "models").exists() or (output / "records").exists():
        raise ValueError("main-arm models or records exist before the freeze; refusing")
    evidence = read_json(output / "smoke.json")
    if not evidence.get("ok"):
        raise ValueError("smoke failed or projects over the cap; STOP RULE: no freeze")
    plan = frozen_plan(output, utc_now())
    (output / "plan.json").write_text(json_text(plan))
    load_plan(output)
    log(f"frozen plan published: lambda {plan['selected_lambda']}, queues {plan['training']['queue_projected_hours']}")
    return 0


def train(output, gpu):
    """Run this GPU's frozen training queue (one arm-seed after another)."""
    plan = load_plan(output)
    for name in plan["training"]["queues"][gpu]:
        arm, seed = name.split("/")
        spec = plan["arms"][arm]
        report = train_run(output, arm, int(seed), spec["lambda"], spec["steps"], spec["batch"],
                           run_dir(output, arm, int(seed)))
        if not report.get("partial"):
            mark(output, f"train-{arm}-seed{seed}", report["wall_seconds"], report["gpu_seconds"])
    return 0


def readouts(output, gpu, gpus=GPUS):
    """Fit the readouts of every main arm-seed assigned to this GPU (round robin over the frozen order)."""
    plan = load_plan(output)
    names = [f"{arm}/{seed}" for arm in plan["arms"] for seed in SEEDS]
    data, policy, labels = FitData(output), policy_split(output), fit_labels(output)
    for i, name in enumerate(names):
        if i % gpus != gpu:
            continue
        arm, seed = name.split("/")
        run = run_dir(output, arm, int(seed))
        if (run / "readouts.json").is_file():
            continue
        t0 = time.monotonic()
        fit_readouts(output, run, data=data, policy=policy, labels=labels)
        mark(output, f"readouts-{arm}-seed{seed}", time.monotonic() - t0, time.monotonic() - t0)
    return 0


def freeze_readouts(output):
    """readouts-manifest.json: every main model.pt / readouts.pt hash, frozen before any evaluation anchor encode."""
    plan = load_plan(output)
    path = Path(output) / "readouts-manifest.json"
    if path.is_file():
        raise ValueError("readouts are already frozen")
    entries = {}
    for arm in plan["arms"]:
        for seed in SEEDS:
            run = run_dir(output, arm, seed)
            report = read_json(run / "readouts.json")
            entries[f"{arm}/{seed}"] = {"model": str((run / "model.pt").relative_to(ROOT)),
                                        "model_sha256": sha256_of(run / "model.pt"),
                                        "readouts": str(readouts_path(run).relative_to(ROOT)) if readouts_path(run).is_file() else None,
                                        "readouts_sha256": sha256_of(readouts_path(run)) if readouts_path(run).is_file() else None,
                                        "report_sha256": sha256_of(run / "readouts.json"),
                                        "typed_failure": report.get("typed_failure")}
    write_json(path, {"schema": "issue_123_readouts_manifest_v1", "plan_identity": IDENTITY,
                      "frozen_at": utc_now(), "entries": entries})
    log(f"readouts frozen: {len(entries)} arm-seeds")
    return 0


# ---------------------------------------------------------------------------
# scoring: once, after the readout freeze, in a process that did not train, under deterministic_scoring()
# ---------------------------------------------------------------------------

EVAL_INVENTORIES = ("grid", "offset", "angle", "power")
POSITION_TIMES = (15, 30, 60, 150, 225)
ISSUE_113_CARRIERS = ROOT / ".local-artifacts/issue-113-temporal-macro-head-v1/carriers"


def record_path(output, arm, seed):
    return Path(output) / "records" / f"{arm}--seed{seed}.json"


def curve_path(output, name, seed):
    return Path(output) / "records" / f"curves-{name}--seed{seed}.json"


def eval_attempt(key):
    return CAMPAIGN / STREAMS["evaluation"] / "attempts" / key


def evaluation_set():
    from scripts import run_gate_b as gb
    return gb.evaluation_set()


def position_branches(cells):
    """The position-error branches: every complete grid candidate of every mixed grid cell (frozen order)."""
    return [(member, item) for member, entry in sorted(cells["grid"]["mixed"].items()) for item in entry["inventory"]]


def _position_entry(key):
    import zstandard
    torch.set_num_threads(1)
    attempt = eval_attempt(key)
    _, observation_root, frames, _ = s99.shot_frames(attempt)
    last = len(frames) - 1
    presence, centers = engine_labels(attempt, list(POSITION_TIMES))
    compressor = zstandard.ZstdCompressor(level=ZSTD_LEVEL)
    blobs = [compressor.compress(s99.image_tensor(observation_root / frames[min(t, last)]["agent_observation"]
                                                  ["relative_path"]).permute(1, 2, 0).contiguous().numpy().tobytes())
             for t in POSITION_TIMES]
    carriers = torch.load(ISSUE_113_CARRIERS / "evaluation" / f"{key}.pt", map_location="cpu", weights_only=True)
    if carriers["key"] != key:
        raise ValueError(f"#113 carrier binding differs for {key}")
    rows = carriers["carriers"][[min(t, len(carriers["carriers"]) - 1) for t in POSITION_TIMES]].float()
    return key, {"last": last, "presence": presence, "centers": centers, "frames": blobs, "carriers": rows}


def eval_cache_path(output):
    return Path(output) / "data" / "evaluation-positions.pt"


def build_eval_cache(output, cells, workers=16):
    """Engine labels, true frames (zstd) and #113 carriers at t in POSITION_TIMES of every position branch."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    keys = [item["key"] for _, item in position_branches(cells)]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        entries = dict(pool.map(_position_entry, keys, chunksize=8))
    atomic_torch(eval_cache_path(output), {"schema": "issue_123_eval_positions_v1", "built_at": utc_now(),
                                           "times": list(POSITION_TIMES), "entries": entries})


def decode_frame(blob):
    import zstandard
    raw = zstandard.ZstdDecompressor().decompress(blob)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(240, 320, 3).permute(2, 0, 1)


def pixel_anchors_path(output):
    return Path(output) / "data" / "anchors-evaluation-pixels.pt"


def read_pixel_anchors(output, policy):
    """Frame 0 of the Gate-B anchor branches (s99.image_tensor), read after the readout freeze."""
    store = torch.load(GATE_B / "data" / "anchors-evaluation.pt", map_location="cpu", weights_only=True)
    frames = {}
    for member, key in sorted(store["frames"].items()):
        _, observation_root, shot, _ = s99.shot_frames(eval_attempt(key))
        frames[member] = s99.image_tensor(observation_root / shot[0]["agent_observation"]["relative_path"])
    atomic_torch(pixel_anchors_path(output), {"schema": "issue_123_eval_pixel_anchors_v1", "scoring_policy": policy,
                                              "read_at": utc_now(), "frames": frames, "keys": dict(store["frames"])})


def top_steps_for(arm, times):
    from world_model.hjepa import novphy as hn
    return [t // hn.Geometry(arm).frames_per_top_step for t in times]


@torch.no_grad()
def score_arm_seed(output, plan, manifest, arm, seed, cells, anchors, cache, costs, device="cuda"):
    from world_model.hjepa import novphy as hn
    from world_model.planning import scoring_harness as harness
    began = time.monotonic()
    run = run_dir(output, arm, seed)
    entry = manifest["entries"][f"{arm}/{seed}"]
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "arm": arm, "seed": seed, "engine_seconds": 0}
    if entry["typed_failure"] is not None:
        return {**base, "typed_failure": entry["typed_failure"], "inventories": {}, "wrong_anchor": None, "positions": {}}
    if sha256_of(run / "model.pt") != entry["model_sha256"] or sha256_of(readouts_path(run)) != entry["readouts_sha256"]:
        raise ValueError(f"{arm}/{seed}: model or readouts differ from the frozen readout manifest")
    model, _ = load_trained(run / "model.pt", device)
    probes = load_readouts(run, device)
    pixels = hn.ARMS[arm]["input"] == "pixels"
    members = sorted(anchors["carriers"])
    x0 = torch.stack([anchors["pixels"][m] if pixels else anchors["carriers"][m] for m in members])
    z0 = dict(zip(members, encode_top(model, x0, device)[-1]))
    # every mixed cell of every inventory, batched
    flat = [(inventory, member, item) for inventory in EVAL_INVENTORIES
            for member, cell in sorted(cells[inventory]["mixed"].items()) for item in cell["inventory"]]
    states = rollout(model, arm, torch.stack([z0[m] for _, m, _ in flat]),
                     torch.stack([wlc_action(item["action"]) for _, _, item in flat]), device)
    z_end = states[:, -1].to(device)
    z0_mat = torch.stack([z0[m] for m in members]).to(device)
    anchor_index = {m: i for i, m in enumerate(members)}
    carriers = {name: (probes[name](z_end).cpu(), probes[name](z0_mat).cpu()) for name in ("R1", "R2", "R4")}
    goal = ((z_end - probes["R3"]) ** 2).sum(1).cpu()
    inventories = {inventory: {} for inventory in EVAL_INVENTORIES}
    for k, (inventory, member, item) in enumerate(flat):
        a = anchor_index[member]
        row = costs.row(item["ordinal"], carriers["R1"][0][k], carriers["R1"][1][a])
        for name in ("R2", "R4"):
            other = costs.row(item["ordinal"], carriers[name][0][k], carriers[name][1][a])
            row[f"{harness.PRIMARY_COST}_{name}"] = other[harness.PRIMARY_COST]
        row["goal_R3"] = None if row["excluded"] or not np.isfinite(float(goal[k])) else float(goal[k])
        inventories[inventory].setdefault(member, {"rows": []})["rows"].append(row)
    # wrong-anchor control: the 16 grid launches from every fit anchor (computed once per anchor)
    wrong = None
    if arm in WRONG_ANCHOR_ARMS:
        fit_store = torch.load(rt.anchors_path(RETRAIN), map_location="cpu", weights_only=True)
        fit_members = sorted(fit_store["anchors"])
        if pixels:
            data = FitData(output)
            row_of = {k: i for i, k in enumerate(data.keys)}
            fx = torch.stack([data.frame(row_of[fit_store["frames"][m]], 0) for m in fit_members])
        else:
            fx = torch.stack([fit_store["anchors"][m] for m in fit_members])
        fz0 = encode_top(model, fx, device)[-1]
        grid = grid_items()
        launches = torch.stack([wlc_action(item["action"]) for item in grid])
        w_end = rollout(model, arm, fz0.repeat_interleave(len(grid), 0), launches.repeat(len(fit_members), 1),
                        device)[:, -1].to(device)
        w_carrier = probes["R1"](w_end).cpu()
        w_anchor = probes["R1"](fz0.to(device)).cpu()
        wrong = {}
        for a, member in enumerate(fit_members):
            wrong[member] = {"rows": [costs.row(item["ordinal"], w_carrier[a * len(grid) + j], w_anchor[a])
                                      for j, item in enumerate(grid)]}
    # position error on the grid cells' branches through P_c, with the readout floor
    branches = position_branches(cells)
    p_states = rollout(model, arm, torch.stack([z0[m] for m, _ in branches]),
                       torch.stack([wlc_action(item["action"]) for _, item in branches]), device)
    steps = top_steps_for(arm, POSITION_TIMES)
    positions = {}
    true_x = []
    for _, item in branches:
        e = cache["entries"][item["key"]]
        true_x.append(torch.stack([decode_frame(b) for b in e["frames"]]) if pixels else e["carriers"])
    true_z = encode_top(model, torch.cat(true_x), device)[-1].to(device)
    _, floor_centers = probes["P_c"](true_z)
    pred_flat = p_states[:, steps].reshape(-1, p_states.shape[-1]).to(device)
    _, pred_centers = probes["P_c"](pred_flat)
    pred_centers = pred_centers.view(len(branches), len(steps), -1, 2).cpu()
    floor_centers = floor_centers.view(len(branches), len(steps), -1, 2).cpu()
    for b, (member, item) in enumerate(branches):
        e = cache["entries"][item["key"]]
        positions[item["key"]] = {str(t): {"pred": hn.position_mse(pred_centers[b, j], e["presence"][j], e["centers"][j]),
                                           "floor": hn.position_mse(floor_centers[b, j], e["presence"][j], e["centers"][j])}
                                  for j, t in enumerate(POSITION_TIMES)}
    torch.cuda.synchronize()
    return {**base, "typed_failure": None, "model_sha256": entry["model_sha256"],
            "readouts_sha256": entry["readouts_sha256"], "inventories": inventories, "wrong_anchor": wrong,
            "positions": positions, "seconds": time.monotonic() - began}


@torch.no_grad()
def score_reference_curves(output, name, seed, cells, cache, device="cuda"):
    """Predicted-carrier position error of U / C* (their F-15 rollouts from the Gate-B anchors) on the same branches."""
    from scripts import run_gate_b as gb
    from scripts import run_tau_ad_within_checkpoint as t96
    from world_model.training.matched_dynamics import pairs_for
    model, failure = gb.load_model(name, seed)
    base = {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "reference": name, "seed": seed, "engine_seconds": 0}
    if model is None:
        return {**base, "typed_failure": failure, "positions": {}}
    store = torch.load(GATE_B / "data" / "anchors-evaluation.pt", map_location="cpu", weights_only=True)
    branches = position_branches(cells)
    pair = pairs_for(model)[t96.arm_spec(gb.UNITS[name]["request"])["pair_index"]]
    z = torch.stack([store["anchors"][m] for m, _ in branches]).to(device)
    actions = torch.stack([wlc_action(item["action"]) for _, item in branches]).to(device)
    position, at = 0, {}
    while position < max(POSITION_TIMES):
        z = model.carrier(z, actions, pair)
        position += pair.delta
        if position in POSITION_TIMES:
            at[position] = z.cpu()
    positions = {}
    for b, (_, item) in enumerate(branches):
        e = cache["entries"][item["key"]]
        positions[item["key"]] = {}
        for j, t in enumerate(POSITION_TIMES):
            pred = at[t][b]
            present = e["presence"][j]
            value = None
            if bool(torch.isfinite(pred).all()) and bool(present.any()):
                p = pred[2:2 + 13 * 18].reshape(18, 13)[:, 5:7]
                value = float((p[present] - e["centers"][j][present]).square().mean())
            positions[item["key"]][str(t)] = {"pred": value, "floor": None}
    return {**base, "typed_failure": None, "request": gb.UNITS[name]["request"], "positions": positions}


def load_manifest(output):
    path = Path(output) / "readouts-manifest.json"
    if not path.is_file():
        raise ValueError("readouts are not frozen; run --freeze-readouts first")
    manifest = read_json(path)
    for name, entry in manifest["entries"].items():
        if entry["typed_failure"] is None and (sha256_of(ROOT / entry["model"]) != entry["model_sha256"]
                                               or sha256_of(ROOT / entry["readouts"]) != entry["readouts_sha256"]):
            raise ValueError(f"{name}: frozen readouts changed")
    return manifest


def score(output):
    """Score every arm-seed once (resumable per record) and the U / C* position curves; one GPU process."""
    from world_model.planning import scoring_harness as harness
    from scripts import run_lookahead_control as wlc
    output = Path(output)
    plan = load_plan(output)
    manifest = load_manifest(output)
    began = time.monotonic()
    _, cells = evaluation_set()
    if not eval_cache_path(output).is_file():
        t0 = time.monotonic()
        build_eval_cache(output, cells)
        mark(output, "score-eval-cache", time.monotonic() - t0, 0.0, cpu_workers=16)
    cache = torch.load(eval_cache_path(output), map_location="cpu", weights_only=False)
    with harness.deterministic_scoring() as policy:
        if not pixel_anchors_path(output).is_file():
            read_pixel_anchors(output, policy)
        carriers = torch.load(GATE_B / "data" / "anchors-evaluation.pt", map_location="cpu", weights_only=True)
        anchors = {"carriers": carriers["anchors"],
                   "pixels": torch.load(pixel_anchors_path(output), map_location="cpu", weights_only=True)["frames"]}
        costs = harness.EndpointCosts(wlc.load_objective())
        for arm in plan["arms"]:
            for seed in SEEDS:
                path = record_path(output, arm, seed)
                if path.is_file():
                    continue
                t0 = time.monotonic()
                record = score_arm_seed(output, plan, manifest, arm, seed, cells, anchors, cache, costs)
                record.update({"scoring_policy": policy, "scored_at": utc_now()})
                write_json(path, record)
                torch.cuda.empty_cache()
                mark(output, f"score-{arm}-seed{seed}", time.monotonic() - t0, time.monotonic() - t0)
                log(f"scored {arm} seed {seed} ({time.monotonic() - t0:.0f}s)")
        for name in ("unit", "c_star"):
            for seed in SEEDS:
                path = curve_path(output, name, seed)
                if not path.is_file():
                    record = score_reference_curves(output, name, seed, cells, cache)
                    record.update({"scoring_policy": policy, "scored_at": utc_now()})
                    write_json(path, record)
    log(f"scoring complete ({time.monotonic() - began:.0f}s)")
    return 0


# ---------------------------------------------------------------------------
# CLEVRER / Physion-Dominoes (EXPLORATORY): action-free H-JEPA and LeWM on the #117 item-5 annotation carriers
# ---------------------------------------------------------------------------

BENCH_ARMS = {"HJ-bench": "HJ-car-m", "LeWM-bench": "LeWM-car-m"}
BENCH_TIMES = (15, 30, 60, 120)
BENCH_CONTEXTS = (0, 1, 2, 3)


class BenchData:
    """One benchmark's 12 scene clips as 'shots' (positions 0..n-1), the null action, in-sample (fold all)."""

    def __init__(self, bench):
        from scripts import run_request_interference_benchmarks as rib
        self.bench = bench
        self.scenes = rib.scenes(bench)
        clips = [torch.load(p, map_location="cpu", weights_only=True)["clip"].float() for p in rib.shard_paths(bench)]
        self.clips = dict(zip(self.scenes, clips))
        self.keys = list(self.scenes)
        self.ends = np.array([len(c) - 1 for c in clips], dtype=np.int64)
        self.first = np.concatenate([[0], np.cumsum([len(c) for c in clips])])
        self.flat = torch.cat(clips)
        self.actions = torch.zeros(len(clips), 5)

    def carrier_rows(self, shots, positions):
        return self.flat[torch.as_tensor(self.first[np.asarray(shots)] + np.asarray(positions))]


class BenchSource(BatchSource):
    def __init__(self, bench, arm, seed, steps, batch):
        from world_model.hjepa import novphy as hn
        self.hn = hn
        self.arm, self.seed, self.steps, self.batch = arm, seed, steps, batch
        self.spec = {**hn.ARMS[arm], "sampler": "uniform", "action": "sparse"}
        self.geometry = hn.Geometry(arm)
        self._data = BenchData(bench)
        self.ends = self._data.ends
        valid = np.array([self.geometry.valid_starts(int(e)) for e in self.ends], dtype=np.int64)
        self.cumulative = np.concatenate([[0], np.cumsum(valid)])


def bench_dir(output, bench, name, seed):
    return Path(output) / "benchmarks" / bench / name / f"seed-{seed}"


def bench_jobs():
    return [(bench, name, seed) for bench in ("clevrer", "physion") for name in BENCH_ARMS for seed in SEEDS]


def bench_train(output, gpu, gpus=GPUS):
    from world_model.hjepa import novphy as hn
    plan = load_plan(output)
    for i, (bench, name, seed) in enumerate(bench_jobs()):
        if i % gpus != gpu:
            continue
        arm = BENCH_ARMS[name]
        source = BenchSource(bench, arm, seed, hn.B1["steps"], hn.B1["batch"])
        report = train_run(output, arm, seed, plan["selected_lambda"]["carriers"], hn.B1["steps"], hn.B1["batch"],
                           bench_dir(output, bench, name, seed), source=source)
        mark(output, f"bench-train-{bench}-{name}-seed{seed}", report["wall_seconds"], report["gpu_seconds"])
    return 0


@torch.no_grad()
def bench_score(output, device="cuda"):
    """R1 probe on every clip frame (in-sample, as the #79 / #84 / #117 in-sample cohort), then the top-level rollout
    from contexts 0-3 with the null action; #79 field_errors position_mse at t in (15, 30, 60, 120)."""
    from world_model.hjepa import novphy as hn
    from world_model.planning import scoring_harness as harness
    from scripts import run_request_interference_benchmarks as rib
    load_plan(output)
    with harness.deterministic_scoring() as policy:
        for bench, name, seed in bench_jobs():
            arm = BENCH_ARMS[name]
            path = bench_dir(output, bench, name, seed) / "record.json"
            if path.is_file():
                continue
            data = BenchData(bench)
            model, payload = load_trained(bench_dir(output, bench, name, seed) / "model.pt", device)
            if model is None:
                write_json(path, {"typed_failure": payload["typed_failure"]})
                continue
            levels = encode_top(model, data.flat, device)
            with torch.enable_grad():
                r1, r1_fit = hn.fit_probe(levels[-1], data.flat, seed, device=device)
            units = [(scene, c) for scene in data.scenes for c in BENCH_CONTEXTS]
            z0 = encode_top(model, torch.stack([data.clips[s][c] for s, c in units]), device)[-1]
            states = rollout(model, arm, z0, torch.zeros(len(units), 5), device)
            steps = [t // hn.Geometry(arm).frames_per_top_step for t in BENCH_TIMES]
            pred = r1(states[:, steps].reshape(-1, states.shape[-1]).to(device)).cpu().view(len(units), len(steps), -1)
            errors = {}
            for u, (scene, c) in enumerate(units):
                errors.setdefault(scene, []).append(
                    {"context": c, "errors": {str(t): rib.unit_errors(pred[u, j], data.clips[scene][c + t])
                                              for j, t in enumerate(BENCH_TIMES)}})
            collapse = {"effective_rank": hn.effective_rank(levels[-1]),
                        "within_episode_share": hn.within_episode_share(
                            levels[-1], np.repeat(np.arange(len(data.scenes)), np.diff(data.first)))}
            write_json(path, {"schema": SCHEMA_RECORD, "plan_identity": IDENTITY, "benchmark": bench, "arm": name,
                              "seed": seed, "scoring_policy": policy, "scored_at": utc_now(), "R1_fit": r1_fit,
                              "collapse": collapse, "rollout": errors, "model_sha256": sha256_of(
                                  bench_dir(output, bench, name, seed) / "model.pt")})
    return 0


def bench_table(output):
    """Mean position_mse over scenes x contexts x seeds per (benchmark, arm, t), next to #117 item-5 J (LC, full,
    fold all) for F-15-continuous and F-5-continuous. EXPLORATORY, no tokens."""
    from scripts import run_request_interference_benchmarks as rib
    table = {}
    for bench in ("clevrer", "physion"):
        rows = {}
        for name in BENCH_ARMS:
            values = {str(t): [] for t in BENCH_TIMES}
            collapse = []
            for seed in SEEDS:
                record = read_json(bench_dir(output, bench, name, seed) / "record.json")
                if record.get("typed_failure"):
                    continue
                collapse.append(record["collapse"])
                for units in record["rollout"].values():
                    for unit in units:
                        for t, err in unit["errors"].items():
                            if err is not None and err.get(rib.METRIC) is not None:
                                values[t].append(err[rib.METRIC])
            rows[name] = {"position_mse": {t: float(np.mean(v)) if v else None for t, v in values.items()},
                          "collapse": collapse}
        for request in ("F-15-continuous", "F-5-continuous"):
            values = {str(t): [] for t in BENCH_TIMES}
            for seed in SEEDS:
                record = read_json(rib.record_path(rib.OUTPUT, bench, "LC", "J", "full", "all", seed))
                if record.get("typed_failure"):
                    continue
                for units in record["rollout"][request].values():
                    for unit in units:
                        for t, err in unit["errors"].items():
                            if err is not None and err.get(rib.METRIC) is not None:
                                values[t].append(err[rib.METRIC])
            rows[f"J (#117 item 5, LC, full) {request}"] = {
                "position_mse": {t: float(np.mean(v)) if v else None for t, v in values.items()}}
        table[bench] = rows
    return table


# ---------------------------------------------------------------------------
# publication: tables, guards, tokens, dispositions; validation
# ---------------------------------------------------------------------------

def records_manifest(output):
    files = sorted((Path(output) / "records").glob("*.json")) + sorted((Path(output) / "benchmarks").glob("*/*/*/record.json"))
    return {"schema": "issue_123_records_manifest_v1", "plan_identity": IDENTITY,
            "records": [{"artifact": str(p.relative_to(ROOT)), "sha256": sha256_of(p), "bytes": p.stat().st_size}
                        for p in files]}


def gate_b_rows(name, seed):
    return read_json(GATE_B / "records" / f"{name}--seed{seed}.json")


def view_rows(plan, cells, inventory, records, references, prior_stats):
    """One row per (member, seed) of the inventory's mixed cells; views: U, C*, P, J, J_dec and every arm (R1), with
    arm:R2 / arm:R3 / arm:R4."""
    from scripts import run_decision_chain_attribution as m99
    from scripts import run_gate_b as gb
    from world_model.planning import scoring_harness as harness
    primary = harness.PRIMARY_COST
    rows = []
    for member, entry in sorted(cells[inventory]["mixed"].items()):
        prior = m99.view(entry, gb.prior_rows(entry, prior_stats), primary)
        for seed in SEEDS:
            views = {"P": prior}
            for label, record in (("U", references[("unit", seed)]), ("C*", references[("c_star", seed)]),
                                  ("J", references[("J", seed)]), ("J_dec", references[("J_dec", seed)])):
                found = None if record.get("typed_failure") else record["inventories"][inventory][member]["rows"]
                views[label] = v1.cost_view(entry, found, primary)
            for arm in plan["arms"]:
                record = records[(arm, seed)]
                found = None if record["typed_failure"] else record["inventories"][inventory][member]["rows"]
                views[arm] = v1.cost_view(entry, found, primary)
                for name, key in (("R2", f"{primary}_R2"), ("R4", f"{primary}_R4"), ("R3", "goal_R3")):
                    views[f"{arm}:{name}"] = v1.cost_view(entry, found, key)
            rows.append({"member": member, "seed": seed, "views": views})
    return rows


def reverse(block):
    return {**block, "mean": -block["mean"], "interval": [-block["interval"][1], -block["interval"][0]]}


def arm_of(view):
    return view.split(":")[0]


def arm_guards(output, plan, records):
    """Per-arm guard evaluation (seed-wise collapse / action, pooled readout ceiling, nonfinite share)."""
    from world_model.hjepa import novphy as hn
    out = {}
    for arm in plan["arms"]:
        seeds = {}
        ceilings = []
        for seed in SEEDS:
            report = read_json(run_dir(output, arm, seed) / "readouts.json")
            if report.get("typed_failure"):
                seeds[str(seed)] = {"typed_failure": report["typed_failure"], "pass": False}
                continue
            top = report["collapse"][-1]
            dim = top["dim"]
            collapse_ok = (top["effective_rank"] >= GUARD_THRESHOLDS["effective_rank_fraction"] * dim
                           and top["within_episode_share"] >= GUARD_THRESHOLDS["within_episode_share"])
            action_ok = report["action_sensitivity"] >= GUARD_THRESHOLDS["action_sensitivity"]
            seeds[str(seed)] = {"effective_rank": top["effective_rank"], "dim": dim,
                                "within_episode_share": top["within_episode_share"],
                                "action_sensitivity": report["action_sensitivity"], "G_collapse": bool(collapse_ok),
                                "G_action": bool(action_ok), "policy_ceiling_auc": report["policy_auc"]["ceiling"],
                                "policy_r1_auc": report["policy_auc"]["R1"]}
            if report["policy_auc"]["ceiling"] is not None:
                ceilings.append(report["policy_auc"]["ceiling"])
        ceiling = float(np.mean(ceilings)) if ceilings else None
        slots = nonfinite = 0
        for seed in SEEDS:
            record = records[(arm, seed)]
            if record["typed_failure"]:
                continue
            for rows in record["inventories"]["grid"].values():
                slots += len(rows["rows"])
                nonfinite += sum(1 for r in rows["rows"] if r["excluded"])
        share = nonfinite / slots if slots else None
        guards = {"G_collapse": all(s.get("G_collapse", False) for s in seeds.values()),
                  "G_action": all(s.get("G_action", False) for s in seeds.values()),
                  "G_readout": ceiling is not None and ceiling >= GUARD_THRESHOLDS["readout_ceiling_auc"],
                  "G_nonfinite": share is not None and share <= GUARD_THRESHOLDS["nonfinite_share"],
                  "G_trained": all(not records[(arm, s)]["typed_failure"] for s in SEEDS)}
        out[arm] = {"seeds": seeds, "policy_ceiling_auc": ceiling, "nonfinite_share": share, **guards,
                    "pass": all(guards.values()), "parameters": plan["arms"][arm]["parameters"],
                    "role": hn.ARMS[arm]["role"]}
    return out


def global_guards(output, plan, records, manifest):
    from scripts import run_gate_b as gb
    from world_model.planning import scoring_harness as harness
    gfid = Path(output) / "fidelity" / "gfid.json"
    fid = read_json(gfid) if gfid.is_file() else {"G_fid": None}
    gport_result = read_json(gport_dir(output) / "gport.json")
    roster = {level["identity"] for level in gb.evaluation_levels()}
    fit_roster = {level["identity"] for level in rt.fit_levels()}
    scored = [r for r in records.values() if not r["typed_failure"]]
    pixel_store = torch.load(pixel_anchors_path(output), map_location="cpu", weights_only=True)
    isolation = (all(set(r["inventories"]["grid"]) <= roster for r in scored)
                 and all(set(r["wrong_anchor"] or {}) <= fit_roster for r in scored)
                 and all(k.startswith("issue-109-sc-fit") for k in FitData(output).keys))
    first_scored = min(r["scored_at"] for r in records.values())
    chronology = plan["frozen_at"] <= manifest["frozen_at"] <= pixel_store["read_at"] <= first_scored
    policies = all(r.get("scoring_policy") == harness.SCORING_POLICY for r in records.values())
    return {"G_fid": {"pass": fid["G_fid"] is True, "value": fid["G_fid"],
                      "tiers": {t: fid["tiers"][t]["pass"] for t in ("tier-e", "tier-t")} if "tiers" in fid else None},
            "G_port": {"pass": bool(gport_result["pass"])},
            "G_isolation": {"pass": bool(isolation)},
            "G_chronology": {"pass": bool(chronology), "plan_frozen_at": plan["frozen_at"],
                             "readouts_frozen_at": manifest["frozen_at"], "anchors_read_at": pixel_store["read_at"],
                             "first_record_scored_at": first_scored},
            "G_scoring_policy": {"pass": bool(policies)}}


def token_pair(block, guarded):
    from scripts import run_decision_dynamics_controls as c2
    if not guarded:
        return STOP_TOKEN, STOP_TOKEN
    return c2.token(block), c2.token(reverse(block))


def compute_tables(output, plan):
    from scripts import run_decision_chain_attribution as m99
    from scripts import run_decision_dynamics_controls as c2
    from scripts import run_gate_b as gb
    from scripts import run_hjepa_fidelity as fid
    from world_model.planning import scoring_harness as harness
    output = Path(output)
    manifest = load_manifest(output)
    _, cells = evaluation_set()
    records = {(arm, seed): read_json(record_path(output, arm, seed)) for arm in plan["arms"] for seed in SEEDS}
    for (arm, seed), record in records.items():
        if record.get("schema") != SCHEMA_RECORD or record.get("plan_identity") != IDENTITY:
            raise ValueError(f"record {arm}/{seed} binding differs")
    references = {}
    for seed in SEEDS:
        references[("unit", seed)] = gate_b_rows("unit", seed)
        references[("c_star", seed)] = gate_b_rows("c_star", seed)
        for label, arm in J_ARMS.items():
            references[(label, seed)] = read_json(ISSUE_101 / "records" / f"{arm}--seed{seed}.json")
    fit = read_json(RETRAIN / "fit-verdicts.json")["branches"]
    guards_global = global_guards(output, plan, records, manifest)
    guards_arm = arm_guards(output, plan, records)
    global_ok = all(g["pass"] for g in guards_global.values())
    rankers = ["U", "C*", "P", "J", "J_dec", *plan["arms"],
               *[f"{a}:{r}" for a in plan["arms"] for r in ("R2", "R3", "R4")]]
    tables = {}
    for inventory in EVAL_INVENTORIES:
        rows = view_rows(plan, cells, inventory, records, references, gb.prior_stats(fit, inventory))
        block = {"cells": len(rows), "members": len({r["member"] for r in rows}), "rankers": {}}
        for name in rankers:
            ranker = v1.ranker_block(rows, name)
            ranker["per_seed"] = ({str(s): m99.block_of([r for r in rows if r["seed"] == s], name) for s in SEEDS}
                                  if name != "P" else None)
            ranker["top1_rate"] = ranker["top1_hits"] / ranker["cells"] if ranker["cells"] else None
            chances = [r["views"][name]["chance_top1"] for r in rows if r["views"][name]["chance_top1"] is not None]
            ranker["chance_top1"] = float(np.mean(chances)) if chances else None
            block["rankers"][name] = ranker
        if inventory == "grid":
            contrasts = {}
            for label, (first, second) in CONTRASTS.items():
                paired = m99.paired(rows, [(1, first), (-1, second)])
                involved = [a for a in (arm_of(first), arm_of(second)) if a in plan["arms"]]
                guarded = global_ok and all(guards_arm[a]["pass"] for a in involved)
                forward, backward = token_pair(paired["estimate"], guarded)
                contrasts[label] = {"first": first, "second": second, **paired, "token": forward,
                                    "reverse_token": backward, "guards_pass": guarded,
                                    "raw_token": c2.token(paired["estimate"]),
                                    "raw_reverse_token": c2.token(reverse(paired["estimate"]))}
            block["contrasts"] = contrasts
            wrong = {}
            for arm in WRONG_ANCHOR_ARMS:
                wrows = []
                for member, entry in sorted(cells["grid"]["mixed"].items()):
                    for seed in SEEDS:
                        record = records[(arm, seed)]
                        if record["typed_failure"]:
                            wrows.append({"member": member, "seed": seed, "views": {arm: None}})
                            continue
                        right = record["inventories"]["grid"][member]["rows"]
                        others = [record["wrong_anchor"][a]["rows"] for a in sorted(record["wrong_anchor"])]
                        aucs = [a for a in (c2.auc(entry, w) for w in others) if a is not None]
                        rhos = [r for r in (c2.spearman_rows(right, w) for w in others) if r is not None]
                        wrows.append({"member": member, "seed": seed, "views": {arm: {
                            "right": c2.auc(entry, right), "wrong": float(np.mean(aucs)) if aucs else None,
                            "stability": float(np.mean(rhos)) if rhos else None}}})
                diff = c2.difference_block(wrows, [(1, arm, "right"), (-1, arm, "wrong")])
                guarded = global_ok and guards_arm[arm]["pass"]
                wrong[arm] = {"right": c2.field_block(wrows, arm, "right"), "wrong": c2.field_block(wrows, arm, "wrong"),
                              "right_minus_wrong": diff, "stability": c2.field_block(wrows, arm, "stability"),
                              "token": c2.token(diff) if guarded else STOP_TOKEN, "raw_token": c2.token(diff),
                              "guards_pass": guarded,
                              "wrong_anchors": len(records[(arm, SEEDS[0])]["wrong_anchor"] or {})}
            block["S10_wrong_anchor"] = wrong
        tables[inventory] = block
    positions = position_table(output, plan, records, cells)
    disposition = dispositions(tables["grid"])
    entries, gpu, wall = ledger_totals(output)
    fid_ledger = fid.compute_ledger()
    fid_gpu = sum(b["gpu_seconds"] for b in fid_ledger.values())
    return {
        "schema": "issue_123_compute_v1", "identity": IDENTITY, "frozen_at": plan["frozen_at"],
        "selected_lambda": plan["selected_lambda"], "guards": {"global": guards_global, "arms": guards_arm},
        "inventories": tables, "positions": positions, "disposition": disposition,
        "benchmarks": bench_table(output),
        "records": {"arm_seeds": len(records), "typed_failures": {f"{a}/{s}": r["typed_failure"]
                                                                 for (a, s), r in records.items() if r["typed_failure"]}},
        "compute": {"issue_123_phases": entries, "issue_123_gpu_seconds": gpu, "issue_123_wall_seconds": wall,
                    "fidelity": fid_ledger, "fidelity_gpu_seconds": fid_gpu, "total_gpu_hours": (gpu + fid_gpu) / 3600.0,
                    "engine_seconds": 0, "hardware": "4 x RTX 5090 (32 GB)", "cap_gpu_hours": CAP_GPU_HOURS},
        "primary_cost": harness.PRIMARY_COST,
    }


def position_table(output, plan, records, cells):
    """Mean #100-style engine position MSE per t (member-clustered bootstrap) for every arm (P_c) and its readout
    floor, and for U / C* (their predicted carriers)."""
    from scripts import run_decision_chain_attribution as m99
    branches = position_branches(cells)
    member_of = {item["key"]: member for member, item in branches}
    out = {}
    sources = {arm: [records[(arm, s)] for s in SEEDS] for arm in plan["arms"]}
    sources.update({"U": [read_json(curve_path(output, "unit", s)) for s in SEEDS],
                    "C*": [read_json(curve_path(output, "c_star", s)) for s in SEEDS]})
    for name, recs in sources.items():
        row = {}
        for t in POSITION_TIMES:
            for kind in ("pred", "floor"):
                values, clusters = [], []
                for record in recs:
                    if record.get("typed_failure"):
                        continue
                    for key, curve in record["positions"].items():
                        value = curve[str(t)][kind]
                        if value is not None:
                            values.append(value)
                            clusters.append(member_of[key])
                row[f"{kind}_{t}"] = m99.bootstrap(values, clusters) if values else None
        out[name] = row
    return out


def dispositions(grid):
    c = grid["contrasts"]
    wrong = grid["S10_wrong_anchor"]
    hj_better = [arm for label, arm in (("PC1", "HJ-px"), ("PC2", "HJ-car-m"))
                 if c[label]["token"] == "supported" and wrong[arm]["token"] == "supported"]
    ours_better = c["PC1"]["reverse_token"] == "supported" and c["PC2"]["reverse_token"] == "supported"
    if hj_better:
        outcome = {"disposition": 1, "label": "H-JEPA ranks better than our Gate-B model (scoped)", "arms": hj_better,
                   "branch": "open the HJ-LC follow-up ticket"}
    elif ours_better:
        outcome = {"disposition": 2, "label": "Our Gate-B model ranks better than H-JEPA (scoped)",
                   "scope": ("#109 sealed evaluation split; Gate-B mixed grid cells (311 x 3 seeds); tie-free cost; "
                             "readout R1; budgets B3 (pixels) and B1 (carriers); layout H3-Delta; the Gate-B unit "
                             "LC : F-15-macro"), "branch": "scoped claim recorded for the manuscript",
                   "sota_wording": "better than H-JEPA, LeWM and HWM as instantiated here (only for supported arms)"}
    else:
        outcome = {"disposition": 3, "label": "per-contrast tokens only", "branch": "no scoped ranking claim"}
    budget_only = c["PC1"]["token"] == "supported" and c["S7"]["token"] != "supported"
    outcome["disposition_5_budget_asymmetry"] = bool(budget_only)
    outcome["conditional_triggers"] = {"U-B3": bool(budget_only),
                                       "closed_loop": bool(c["PC1"]["token"] == "supported"
                                                           or c["PC2"]["token"] == "supported")}
    outcome["note_unit_not_best"] = "S5 / S11 are reported next to every primary contrast; at Gate B C* 0.783 vs U 0.744"
    return outcome


def fmt_block(block, digits=4):
    if block is None or block.get("mean") is None:
        return "n/a"
    lo, hi = block["interval"]
    return f"{block['mean']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def fmt_auc(block, digits=4):
    if block is None or block.get("mean") is None:
        return "n/a"
    lo, hi = block["interval"]
    return f"{block['mean']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def fmt_num(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def findings_md(plan, compute):
    grid = compute["inventories"]["grid"]
    d = compute["disposition"]
    lines = ["# Issue #123 findings: H-JEPA vs the Gate-B unit on the sealed #109 evaluation split", "",
             f"Plan frozen {plan['frozen_at']}; selected SIGReg lambda {plan['selected_lambda']}. Intervals are "
             "member (level)-clustered percentile bootstraps (10 000 draws, PCG64 7201), DESCRIPTIVE. delta = 0.02.", "",
             f"**Disposition {d['disposition']}: {d['label']}** (outcome branch: {d['branch']}).", "",
             "## Guards", "", "| guard | pass |", "| --- | --- |"]
    for name, g in compute["guards"]["global"].items():
        lines.append(f"| {name} | {g['pass']} |")
    lines += ["", "| arm | role | params | G-collapse | G-action | G-readout (policy ceiling) | G-nonfinite | trained | pass |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, g in compute["guards"]["arms"].items():
        lines.append(f"| {arm} | {g['role']} | {g['parameters']:,} | {g['G_collapse']} | {g['G_action']} | "
                     f"{g['G_readout']} ({fmt_num(g['policy_ceiling_auc'])}) | {g['G_nonfinite']} "
                     f"({fmt_num(g['nonfinite_share'], 3)}) | {g['G_trained']} | {g['pass']} |")
    lines += ["", "Per seed (top level): effective rank / D, within-episode share chi, action sensitivity, policy AUCs.", "",
              "| arm | seed | erank / D | chi | sensitivity | policy R1 AUC | policy ceiling |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for arm, g in compute["guards"]["arms"].items():
        for seed, s in g["seeds"].items():
            if "typed_failure" in s:
                lines.append(f"| {arm} | {seed} | typed failure {s['typed_failure']} | | | | |")
                continue
            lines.append(f"| {arm} | {seed} | {s['effective_rank']:.1f} / {s['dim']} | {s['within_episode_share']:.3f} | "
                         f"{s['action_sensitivity']:.3f} | {fmt_num(s['policy_r1_auc'])} | {fmt_num(s['policy_ceiling_auc'])} |")
    lines += ["", "## Grid rankers (311 mixed cells x 3 seeds)", "",
              "| ranker | pooled AUC | seed 20260908 | seed 20260909 | seed 20260910 | top-1 | chance top-1 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in grid["rankers"].items():
        seeds = r["per_seed"]
        per = " | ".join(fmt_auc(seeds.get(str(s))) for s in SEEDS) if seeds else "seed-free | | "
        lines.append(f"| {name} | {fmt_auc(r['auc'])} | {per} | {fmt_num(r['top1_rate'], 3)} | {fmt_num(r['chance_top1'], 3)} |")
    lines += ["", "## Contrasts (grid, paired per (level, seed))", "",
              "| contrast | first - second | estimate | token (first - second) | token (second - first) | guards |",
              "| --- | --- | --- | --- | --- | --- |"]
    for label, c in grid["contrasts"].items():
        lines.append(f"| {label} | {c['first']} - {c['second']} | {fmt_block(c['estimate'])} | {c['token']} | "
                     f"{c['reverse_token']} | {c['guards_pass']} |")
    lines += ["", "S10 wrong-anchor control (right - wrong, 480 fit anchors):", "",
              "| arm | right | wrong | right - wrong | token |", "| --- | --- | --- | --- | --- |"]
    for arm, w in grid["S10_wrong_anchor"].items():
        lines.append(f"| {arm} | {fmt_auc(w['right'])} | {fmt_auc(w['wrong'])} | {fmt_block(w['right_minus_wrong'])} | "
                     f"{w['token']} |")
    lines += ["", "## Descriptive inventories (no tokens)", "", "| ranker | offset | angle | power |", "| --- | --- | --- | --- |"]
    for name in grid["rankers"]:
        lines.append(f"| {name} | " + " | ".join(fmt_auc(compute["inventories"][i]["rankers"][name]["auc"])
                                                for i in ("offset", "angle", "power")) + " |")
    lines += ["", "## Engine position MSE (#100 metric; P_c for H-JEPA arms, predicted carriers for U / C*)", "",
              "| model | " + " | ".join(f"t = {t}" for t in POSITION_TIMES) + " | floor t = 225 |",
              "| --- | " + " | ".join("---" for _ in POSITION_TIMES) + " | --- |"]
    for name, row in compute["positions"].items():
        lines.append(f"| {name} | " + " | ".join(fmt_auc(row[f'pred_{t}']) for t in POSITION_TIMES)
                     + f" | {fmt_auc(row['floor_225'])} |")
    lines += ["", "## CLEVRER / Physion-Dominoes (EXPLORATORY, in-sample, null action; position_mse)", ""]
    for bench, rows in compute["benchmarks"].items():
        lines += [f"**{bench}**", "", "| model | " + " | ".join(f"t = {t}" for t in BENCH_TIMES) + " |",
                  "| --- | " + " | ".join("---" for _ in BENCH_TIMES) + " |"]
        for name, row in rows.items():
            lines.append(f"| {name} | " + " | ".join("n/a" if row["position_mse"][str(t)] is None
                                                   else f"{row['position_mse'][str(t)]:.4g}" for t in BENCH_TIMES) + " |")
        lines.append("")
    lines += ["## Disposition", "", f"- {d['label']} (rule {d['disposition']}); branch: {d['branch']}.",
              f"- Disposition 5 (budget asymmetry): {d['disposition_5_budget_asymmetry']}; conditional triggers "
              f"{d['conditional_triggers']}.", f"- {d['note_unit_not_best']}.", "",
              "## Compute", "", f"- issue-123 GPU seconds {compute['compute']['issue_123_gpu_seconds']:.0f}; fidelity "
              f"{compute['compute']['fidelity_gpu_seconds']:.0f}; total {compute['compute']['total_gpu_hours']:.1f} GPU-h "
              f"(cap {CAP_GPU_HOURS:.0f}); engine seconds 0."]
    return "\n".join(lines) + "\n"


def summary(plan, compute):
    grid = compute["inventories"]["grid"]
    return {"schema": "issue_123_summary_v1", "identity": IDENTITY, "frozen_at": plan["frozen_at"],
            "disposition": compute["disposition"], "selected_lambda": plan["selected_lambda"],
            "grid_auc": {name: r["auc"] for name, r in grid["rankers"].items()},
            "contrasts": {label: {"estimate": c["estimate"], "token": c["token"], "reverse_token": c["reverse_token"]}
                          for label, c in grid["contrasts"].items()},
            "S10": {arm: w["token"] for arm, w in grid["S10_wrong_anchor"].items()},
            "guards": {"global": {k: g["pass"] for k, g in compute["guards"]["global"].items()},
                       "arms": {a: g["pass"] for a, g in compute["guards"]["arms"].items()}},
            "engine_seconds": 0}


def rendered(plan, compute):
    return {"compute.json": json_text(compute), "summary.json": json_text(summary(plan, compute)),
            "findings.md": findings_md(plan, compute)}


def publish(output):
    from world_model.planning import scoring_harness as harness
    output = Path(output)
    plan = load_plan(output)
    write_json(output / "records-manifest.json", records_manifest(output))
    with harness.deterministic_scoring():
        compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        (output / name).write_text(text)
    log(f"published: disposition {compute['disposition']['disposition']} ({compute['disposition']['label']})")
    return 0


def spot_check(output, plan):
    """Re-score the first complete record of HJ-car-m (then HJ-px) and compare its R1 grid rows."""
    from world_model.planning import scoring_harness as harness
    from scripts import run_lookahead_control as wlc
    manifest = load_manifest(output)
    _, cells = evaluation_set()
    cache = torch.load(eval_cache_path(output), map_location="cpu", weights_only=False)
    carriers = torch.load(GATE_B / "data" / "anchors-evaluation.pt", map_location="cpu", weights_only=True)
    anchors = {"carriers": carriers["anchors"],
               "pixels": torch.load(pixel_anchors_path(output), map_location="cpu", weights_only=True)["frames"]}
    costs = harness.EndpointCosts(wlc.load_objective())
    for arm in ("HJ-car-m", "HJ-px"):
        for seed in SEEDS:
            record = read_json(record_path(output, arm, seed))
            if record["typed_failure"]:
                continue
            with harness.deterministic_scoring():
                fresh = score_arm_seed(output, plan, manifest, arm, seed, cells, anchors, cache, costs)
            worst, passed = dict.fromkeys(harness.PARSED_QUANTITIES, 0.0), True
            for member, block in record["inventories"]["grid"].items():
                check = harness.endpoint_replication(block["rows"], fresh["inventories"]["grid"][member]["rows"])
                passed &= check["pass"]
                for k in worst:
                    worst[k] = max(worst[k], check["max_abs_delta"][k])
            return {"arm": arm, "seed": seed, "cells": len(record["inventories"]["grid"]), "max_abs_delta": worst,
                    "pass": bool(passed)}
    raise ValueError("no complete HJ-car-m / HJ-px record to re-score")


def validate(output):
    from world_model.planning import scoring_harness as harness
    output = Path(output)
    began = time.monotonic()
    plan = load_plan(output)
    manifest = read_json(output / "records-manifest.json")
    for entry in manifest["records"]:
        if sha256_of(ROOT / entry["artifact"]) != entry["sha256"]:
            raise ValueError(f"record {entry['artifact']} differs from records-manifest.json")
    if manifest != records_manifest(output):
        raise ValueError("records-manifest.json does not list exactly the current records")
    with harness.deterministic_scoring():
        compute = compute_tables(output, plan)
    for name, text in rendered(plan, compute).items():
        if (output / name).read_bytes() != text.encode():
            raise ValueError(f"published {name} differs from the recomputation")
    spot = spot_check(output, plan)
    if not spot["pass"]:
        raise ValueError(f"re-scoring does not replicate the retained record: {spot}")
    log(f"validation passed ({time.monotonic() - began:.0f}s): {len(manifest['records'])} records hash-checked, every "
        f"table re-derived and byte-compared, {spot['arm']} seed {spot['seed']} re-scored on {spot['cells']} grid cells "
        f"(max delta {spot['max_abs_delta']})")
    return 0


def dry_run(output):
    from world_model.hjepa import novphy as hn
    data = FitData(output)
    for arm in hn.ARMS:
        steps, batch = hn.steps_for(arm, valid_clips(arm, data.ends))
        log(f"{arm}: {hn.parameter_count(arm):,} params, {valid_clips(arm, data.ends):,} valid clips, {steps} x {batch}")
    return 0


MODES = {"dry-run": dry_run, "smoke": smoke, "prepare": prepare, "freeze-readouts": freeze_readouts, "score": score,
         "bench-score": bench_score, "publish": publish, "validate": validate, "gport": gport,
         "gport-configs": gport_configs, "fit-carriers": build_fit_carriers, "policy-data": build_policy_data}
GPU_MODES = {"train": train, "readouts": readouts, "bench-train": bench_train}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for mode in (*MODES, *GPU_MODES):
        parser.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--selection", action="store_true")
    parser.add_argument("--selection-readouts", action="store_true")
    parser.add_argument("--data-pixels", choices=("fit", "policy"))
    parser.add_argument("--data-labels", choices=("fit",))
    parser.add_argument("--assemble-pixels", choices=("fit", "policy"))
    parser.add_argument("--mirror-pixels", choices=("fit", "policy"))
    parser.add_argument("--gpu", type=int, default=0, help="queue index (run with CUDA_VISIBLE_DEVICES set to it)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker", type=int, default=0)
    args = parser.parse_args()
    if args.data_pixels:
        return build_pixel_shard(OUTPUT, args.data_pixels, args.workers, args.worker)
    if args.data_labels:
        return build_label_shard(OUTPUT, args.data_labels, args.workers, args.worker)
    if args.assemble_pixels:
        assemble_pixels(OUTPUT, args.assemble_pixels, args.workers)
        return 0
    if args.mirror_pixels:
        return mirror_pixels(OUTPUT, args.mirror_pixels)
    if args.selection:
        return selection(OUTPUT, args.gpu, args.workers)
    if args.selection_readouts:
        return selection_readouts(OUTPUT, args.gpu, args.workers)
    for mode, fn in GPU_MODES.items():
        if getattr(args, mode.replace("-", "_")):
            return fn(OUTPUT, args.gpu)
    for mode, fn in MODES.items():
        if getattr(args, mode.replace("-", "_")):
            return fn(OUTPUT)
    parser.error("no mode given")


if __name__ == "__main__":
    raise SystemExit(main())
