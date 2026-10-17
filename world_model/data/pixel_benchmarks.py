"""Issue #106: CLEVRER and Physion-Dominoes from pixels: readers and frozen description definitions.

Everything here is read at training time only (annotation-derived parser targets and description
labels) except ``decode_clevrer_video`` / ``physion_images``, which are the only test-time inputs.

Carrier time base:
- CLEVRER: one carrier frame per video frame (128 frames, 5 s).
- Physion-Dominoes: one carrier frame per ``PHYSION_STRIDE`` = 3 native frames (native frame 3f). The
  OCP readout protocol (physopt / physics-models: STATE_LEN 7 at SUBSAMPLE_FACTOR 6) observes native
  frames 0..36, i.e. carrier frames 0..12; rollouts run ``PHYSION_ROLLOUT`` = 90 carrier frames past
  frame 12 (native 306, beyond the longest observed contact onset); frames past a trial's end repeat
  its last frame (terminal absorption: TDW ends a trial when every object is at rest).

Slots (18, the CNNHybridPredictor contract):
- CLEVRER: slot = COLORS.index(color) * 2 + MATERIALS.index(material) (16 keyed slots, 16-17 empty);
  member videos have pairwise distinct (color, material) keys (41 % of videos repeat a key and are
  excluded). Kind = shape.
- Physion: slot = position in ``static/object_ids`` (zone, target, probe, middle dominoes,
  distractors, occluders); kind = that role.

Descriptions (labels indexed by the frame f a request lands on; Delta in DELTAS):
- micro (channel 0 of the 18 x 18 x 2 relation tensor; channel 1 is masked out):
  CLEVRER collision event between objects i and j at frame f (the #79 contact definition);
  Physion pairwise contact (``collisions`` state enter/stay) on any native frame in (3f - 3, 3f].
  Mask: both slots declared by the scene. Delta-independent (the micro head reads one carrier).
- macro (two predicates per Delta, the #113 observable templates):
  CLEVRER ``enter-exit``: debounced in-view set V(g) (inside_camera_view on at least 2 of the frames
  g-2, g-1, g) differs between f - Delta and f (endpoint comparison); available iff f - Delta >= 2.
  CLEVRER ``all-at-rest``: with D = max(1, Delta), on every frame of (f - D, f] at least one object is
  in view and every in-view object has planar speed <= CLEVRER_REST_SPEED; available iff f >= D.
  Physion ``ocp-contact``: C(g) = the target touched the zone (``target_contacting_zone``) on some
  native frame <= 3g; label C(f) != C(f - Delta) (contact onset inside the request interval);
  available iff f >= Delta. The OCP readout label is C at the trial end (Physion ``label_fn``: any).
  Physion ``settled``: with D = max(1, Delta), the probe push happened at or before native 3(f - D)
  and every object has |v| <= PHYSION_REST_SPEED and |omega| <= PHYSION_REST_ANGULAR on every native
  frame of (3(f - D), 3f]; available iff f >= D.
"""
from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SLOTS = 18
DELTAS = (1, 5, 15)
MACRO_NAMES = {"clevrer": ("enter-exit", "all-at-rest"), "physion": ("ocp-contact", "settled")}

# ------------------------------------------------------------------ CLEVRER
CLEVRER = ROOT / "data/clevrer"
CLEVRER_VIDEOS = CLEVRER / "videos/extracted"
CLEVRER_ANNOTATIONS = {"train": CLEVRER / "annotations/annotation_train.zip",
                       "validation": CLEVRER / "annotations/annotation_validation.zip"}
CLEVRER_QUESTIONS = {"train": CLEVRER / "questions/train.json", "validation": CLEVRER / "questions/validation.json"}
CLEVRER_FRAMES = 128
CLEVRER_NATIVE = (320, 480)  # (H, W)
CLEVRER_IMAGE = (160, 240)
CLEVRER_ELAPSED = 5.0 / 128
COLORS = ("gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow")
MATERIALS = ("rubber", "metal")
SHAPES = ("cube", "cylinder", "sphere")
CLEVRER_REST_SPEED = 0.05
CLEVRER_VIEW_WINDOW = 3
CLEVRER_VIEW_MAJORITY = 2
# hue prototypes (degrees) of the CLEVR palette, used only to collect homography correspondences
HUES = {"red": 0, "brown": 26, "yellow": 55, "green": 113, "cyan": 180, "blue": 228, "purple": 277}

# ------------------------------------------------------------------ Physion
PHYSION = ROOT / "data/physion-dominoes"
PHYSION_SPLITS = {"dynamics": "Dominoes_dynamics_training_HDF5s", "readout": "Dominoes_readout_training_HDF5s",
                  "testing": "Dominoes_testing_HDF5s"}
PHYSION_EXTRACTED = PHYSION / "extracted"
PHYSION_GLITCHY = PHYSION / "manual_stim_evaluation_glitchy_test_stims.txt"
PHYSION_IMAGE = (256, 256)
PHYSION_STRIDE = 3
PHYSION_INPUT_NATIVE = 36
PHYSION_INPUT = PHYSION_INPUT_NATIVE // PHYSION_STRIDE  # carrier frame 12 = last observed frame
PHYSION_ROLLOUT = 90
PHYSION_ELAPSED = 0.1  # 3 native frames at a nominal 30 frames/s; scales the motion columns only
PHYSION_MIN_NATIVE = 151
PHYSION_REST_SPEED = 0.02
PHYSION_REST_ANGULAR = 0.1
PHYSION_MIN_PIXELS = 16
ROLES = ("zone", "target", "probe", "middle", "distractor", "occluder")


class PixelBenchmarkError(ValueError):
    """Typed data-contract failure."""


# =========================================================================== CLEVRER


def clevrer_slot(color, material):
    return COLORS.index(color) * len(MATERIALS) + MATERIALS.index(material)


def clevrer_member(payload):
    """Frozen membership: shapes in vocabulary, 128 contiguous frames, distinct (color, material)."""
    objects = payload["object_property"]
    keys = [(o["color"], o["material"]) for o in objects]
    return (len(objects) > 0 and len(set(keys)) == len(keys)
            and all(o["shape"] in SHAPES and o["color"] in COLORS and o["material"] in MATERIALS for o in objects)
            and [f["frame_id"] for f in payload["motion_trajectory"]] == list(range(CLEVRER_FRAMES)))


def clevrer_annotations(split):
    """(scene_index, payload) in ascending scene index for one split."""
    with zipfile.ZipFile(CLEVRER_ANNOTATIONS[split]) as archive:
        names = sorted((n for n in archive.namelist() if n.endswith(".json")),
                       key=lambda n: int(Path(n).stem.split("_")[-1]))
        for name in names:
            payload = json.loads(archive.read(name))
            yield int(payload["scene_index"]), payload


def clevrer_scene(payload):
    objects = payload["object_property"]
    n = len(objects)
    location = np.zeros((CLEVRER_FRAMES, n, 3), np.float32)
    velocity = np.zeros((CLEVRER_FRAMES, n, 3), np.float32)
    visible = np.zeros((CLEVRER_FRAMES, n), bool)
    for frame in payload["motion_trajectory"]:
        t = frame["frame_id"]
        seen = set()
        for entry in frame["objects"]:
            i = entry["object_id"]
            if not 0 <= i < n or i in seen:
                raise PixelBenchmarkError(f"scene {payload['scene_index']}: object {i} at frame {t}")
            seen.add(i)
            location[t, i] = entry["location"]
            velocity[t, i] = entry["velocity"]
            visible[t, i] = bool(entry["inside_camera_view"])
        if len(seen) != n:
            raise PixelBenchmarkError(f"scene {payload['scene_index']}: frame {t} declares {len(seen)} of {n}")
    collisions = []
    for event in payload["collision"]:
        a, b = event["object_ids"]
        if not (0 <= a < n and 0 <= b < n) or a == b or not 0 <= event["frame_id"] < CLEVRER_FRAMES:
            raise PixelBenchmarkError(f"scene {payload['scene_index']}: collision {event}")
        collisions.append((int(event["frame_id"]), int(a), int(b)))
    return {"scene_index": int(payload["scene_index"]), "video": payload["video_filename"],
            "slots": np.array([clevrer_slot(o["color"], o["material"]) for o in objects], np.int64),
            "shapes": np.array([SHAPES.index(o["shape"]) for o in objects], np.int64),
            "colors": [o["color"] for o in objects], "materials": [o["material"] for o in objects],
            "location": location, "velocity": velocity, "visible": visible, "collisions": collisions}


def clevrer_video_path(scene_index):
    split = "video_{:05d}-{:05d}".format(scene_index // 1000 * 1000, scene_index // 1000 * 1000 + 1000)
    return CLEVRER_VIDEOS / split / f"video_{scene_index:05d}.mp4"


def decode_clevrer_video(scene_index, size=CLEVRER_IMAGE):
    """uint8 [128, H, W, 3]; ffmpeg area resampling (test-time input)."""
    h, w = size
    command = ["ffmpeg", "-v", "error", "-threads", "1", "-i", str(clevrer_video_path(scene_index))]
    if (h, w) != CLEVRER_NATIVE:
        command += ["-vf", f"scale={w}:{h}:flags=area"]
    raw = subprocess.run(command + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True,
                         check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if len(frames) != CLEVRER_FRAMES:
        raise PixelBenchmarkError(f"video {scene_index}: {len(frames)} frames")
    return frames


def project(homography, points):
    """Ground-plane (x, y) -> native pixel (u, v)."""
    x = np.concatenate([points, np.ones(points.shape[:-1] + (1,), points.dtype)], -1) @ homography.T
    return x[..., :2] / x[..., 2:]


def _hsv(rgb):
    x = rgb.astype(np.float32) / 255
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, d / (mx + 1e-6), mx


def color_correspondences(frames, scene, every=8):
    """Correspondences (x, y, u, v) from native frames: centroid of the largest component of an in-view
    object's palette hue when no other in-view object shares its colour (gray excluded)."""
    from scipy import ndimage
    out = []
    for t in range(0, CLEVRER_FRAMES, every):
        h, s, v = _hsv(frames[t])
        inview = np.nonzero(scene["visible"][t])[0]
        for i in inview:
            color = scene["colors"][i]
            if color not in HUES or sum(scene["colors"][j] == color for j in inview) > 1:
                continue
            mask = (np.abs((h - HUES[color] + 180) % 360 - 180) < 12) & (s > 0.45) & (v > 0.15)
            labels, count = ndimage.label(mask)
            if not count:
                continue
            sizes = ndimage.sum(mask, labels, range(1, count + 1))
            k = int(np.argmax(sizes)) + 1
            if sizes[k - 1] < 30:
                continue
            yy, xx = np.nonzero(labels == k)
            out.append((*scene["location"][t, i, :2], xx.mean(), yy.mean()))
    return np.array(out, np.float64).reshape(-1, 4)


def _dlt(world, image):
    rows = []
    for (x, y), (p, q) in zip(world, image):
        rows += [[x, y, 1, 0, 0, 0, -p * x, -p * y, -p], [0, 0, 0, x, y, 1, -q * x, -q * y, -q]]
    _, _, vt = np.linalg.svd(np.array(rows))
    return vt[-1].reshape(3, 3) / vt[-1][-1]


def fit_homography(points, *, seed=10601, iterations=2000, tolerance=6.0):
    """RANSAC (4-point DLT, inlier tolerance in native pixels) then least-squares DLT on the inliers."""
    rng = np.random.default_rng(seed)
    world, image = points[:, :2], points[:, 2:]
    best = None
    for _ in range(iterations):
        pick = rng.choice(len(points), 4, replace=False)
        hom = _dlt(world[pick], image[pick])
        count = int((np.linalg.norm(project(hom, world) - image, axis=1) < tolerance).sum())
        if best is None or count > best[0]:
            best = (count, hom)
    inliers = np.linalg.norm(project(best[1], world) - image, axis=1) < tolerance
    hom = _dlt(world[inliers], image[inliers])
    residual = np.linalg.norm(project(hom, world) - image, axis=1)
    return hom, {"points": int(len(points)), "inlier_fraction": float(inliers.mean()),
                 "residual_px_quantiles": {q: float(np.quantile(residual, float(q))) for q in ("0.5", "0.9", "0.95")}}


def clevrer_targets(scene, homography):
    """Parser targets: presence [128, 18] (in camera view), centers [128, 18, 2] (normalized image
    coordinates of the projected ground-plane location), kinds [128, 18] (shape; -1 empty), and the
    center-valid mask (in view and projected inside the image)."""
    presence = np.zeros((CLEVRER_FRAMES, SLOTS), bool)
    centers = np.zeros((CLEVRER_FRAMES, SLOTS, 2), np.float32)
    kinds = np.full((CLEVRER_FRAMES, SLOTS), -1, np.int64)
    uv = project(homography, scene["location"][..., :2].astype(np.float64))
    uv = uv / np.array(CLEVRER_NATIVE[::-1], np.float64)
    for i, slot in enumerate(scene["slots"]):
        presence[:, slot] = scene["visible"][:, i]
        centers[:, slot] = uv[:, i]
        kinds[:, slot] = scene["shapes"][i]
    valid = presence & (centers >= 0).all(-1) & (centers <= 1).all(-1)
    centers = np.where(valid[..., None], centers, 0).astype(np.float32)
    return {"presence": presence, "centers": centers, "kinds": kinds, "center_valid": valid}


def clevrer_micro(scene):
    contact = np.zeros((CLEVRER_FRAMES, SLOTS, SLOTS), bool)
    for t, a, b in scene["collisions"]:
        sa, sb = scene["slots"][a], scene["slots"][b]
        contact[t, sa, sb] = contact[t, sb, sa] = True
    declared = np.zeros(SLOTS, bool)
    declared[scene["slots"]] = True
    mask = declared[:, None] & declared[None, :] & ~np.eye(SLOTS, dtype=bool)
    return contact, mask


def clevrer_macro(scene):
    """values, available [128, 2, 3] (predicate x Delta)."""
    n = CLEVRER_FRAMES
    visible = scene["visible"]
    counts = np.zeros(visible.shape, np.int64)
    for k in range(CLEVRER_VIEW_WINDOW):
        counts[k:] += visible[:n - k]
    debounced = counts >= CLEVRER_VIEW_MAJORITY
    view_ok = np.arange(n) >= CLEVRER_VIEW_WINDOW - 1
    speed = np.linalg.norm(scene["velocity"][..., :2], axis=-1)
    rest = visible.any(1) & ((speed <= CLEVRER_REST_SPEED) | ~visible).all(1)
    values = np.zeros((n, 2, len(DELTAS)), bool)
    available = np.zeros((n, 2, len(DELTAS)), bool)
    for j, delta in enumerate(DELTAS):
        for f in range(n):
            g = f - delta
            if g >= CLEVRER_VIEW_WINDOW - 1 and view_ok[f]:
                available[f, 0, j] = True
                values[f, 0, j] = bool((debounced[f] != debounced[g]).any())
            span = max(1, delta)
            if f >= span:
                available[f, 1, j] = True
                values[f, 1, j] = bool(rest[f - span + 1:f + 1].all())
    return values, available


# =========================================================================== Physion


def physion_glitchy():
    return frozenset(line.strip() for line in PHYSION_GLITCHY.read_text().splitlines() if line.strip())


def physion_files(split):
    """HDF5 trials of a split. The testing archive holds every stimulus twice: ``hdf5s`` (random agent /
    patient colours, the human version) and ``hdf5s-redyellow`` (red agent, yellow patient: the version the
    Physion README says is passed to models); only the latter is used."""
    files = sorted((PHYSION_EXTRACTED / split).rglob("*.hdf5"))
    if split == "testing":
        files = [f for f in files if f.parent.name == "hdf5s-redyellow"]
    return files


def _decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def physion_roles(static):
    ids = list(static["object_ids"][()])
    if not (ids.index(static["zone_id"][()]) == 0 and ids.index(static["target_id"][()]) == 1
            and ids.index(static["probe_id"][()]) == 2):
        raise PixelBenchmarkError("object order is not zone, target, probe")
    distractors = len(static["distractors"][()]) if "distractors" in static else 0
    occluders = len(static["occluders"][()]) if "occluders" in static else 0
    middles = len(ids) - 3 - distractors - occluders
    if middles < 0:
        raise PixelBenchmarkError("negative middle-object count")
    roles = [0, 1, 2] + [3] * middles + [4] * distractors + [5] * occluders
    return np.array(roles, np.int64)


def physion_header(path):
    """Cheap membership facts without images."""
    import h5py
    with h5py.File(path, "r") as h:
        static = h["static"]
        return {"name": Path(path).stem, "stimulus": _decode(static["stimulus_name"][()]),
                "objects": len(static["object_ids"][()]), "native_frames": len(h["frames"])}


def physion_trial(path, *, images=True, segmentation=True):
    """One trial at the carrier stride. Native-frame physics for labels; images [T, 256, 256, 3] at
    native frames 3f (stored frames of another size, the 512 x 512 testing set, are resized with the box
    filter); segmentation-derived presence/centers (training targets / annotation truth) at the stored
    resolution, with the presence pixel threshold scaled to the 256 x 256 area."""
    import h5py
    from PIL import Image
    with h5py.File(path, "r") as h:
        static = h["static"]
        frames = sorted(h["frames"].keys())
        n_native = len(frames)
        roles = physion_roles(static)
        n = len(roles)
        if n > SLOTS:
            raise PixelBenchmarkError(f"{path}: {n} objects exceed {SLOTS} slots")
        ids = list(static["object_ids"][()])
        velocity = np.zeros((n_native, n), np.float32)
        angular = np.zeros((n_native, n), np.float32)
        contact_zone = np.zeros(n_native, bool)
        contacts = np.zeros((n_native, n, n), bool)
        for k, name in enumerate(frames):
            frame = h["frames"][name]
            velocity[k] = np.linalg.norm(frame["objects/velocities"][()], axis=-1)
            angular[k] = np.linalg.norm(frame["objects/angular_velocities"][()], axis=-1)
            contact_zone[k] = bool(frame["labels/target_contacting_zone"][()])
            pairs = frame["collisions/object_ids"][()].reshape(-1, 2)
            states = frame["collisions/states"][()].reshape(-1)
            for (a, b), state in zip(pairs, states):
                if _decode(state) in ("enter", "stay") and a in ids and b in ids and a != b:
                    i, j = ids.index(a), ids.index(b)
                    contacts[k, i, j] = contacts[k, j, i] = True
        carrier_native = list(range(0, n_native, PHYSION_STRIDE))
        out = {"name": Path(path).stem, "stimulus": _decode(static["stimulus_name"][()]), "roles": roles,
               "native_frames": n_native, "push_time": int(static["push_time"][()]) if "push_time" in static else 0,
               "velocity": velocity, "angular": angular, "contact_zone": contact_zone, "contacts": contacts,
               "carrier_native": np.array(carrier_native, np.int64)}
        if images:
            def rgb(k):
                image = Image.open(io.BytesIO(h["frames"][frames[k]]["images/_img"][()])).convert("RGB")
                if image.size != PHYSION_IMAGE[::-1]:
                    image = image.resize(PHYSION_IMAGE[::-1], Image.BOX)
                return np.asarray(image)
            out["images"] = np.stack([rgb(k) for k in carrier_native])
        if segmentation:
            colors = static["object_segmentation_colors"][()].astype(np.int64)
            key = colors @ np.array([65536, 256, 1])
            presence = np.zeros((len(carrier_native), SLOTS), bool)
            centers = np.zeros((len(carrier_native), SLOTS, 2), np.float32)
            for t, k in enumerate(carrier_native):
                seg = np.asarray(Image.open(io.BytesIO(h["frames"][frames[k]]["images/_id"][()])).convert("RGB"))
                code = seg.astype(np.int64) @ np.array([65536, 256, 1])
                hgt, wid = code.shape
                minimum = PHYSION_MIN_PIXELS * hgt * wid / (PHYSION_IMAGE[0] * PHYSION_IMAGE[1])
                for i in range(n):
                    yy, xx = np.nonzero(code == key[i])
                    if len(yy) >= minimum:
                        presence[t, i] = True
                        centers[t, i] = ((xx.mean() + 0.5) / wid, (yy.mean() + 0.5) / hgt)
            kinds = np.full((len(carrier_native), SLOTS), -1, np.int64)
            kinds[:, :n] = roles[None]
            out.update(presence=presence, centers=centers, kinds=kinds, center_valid=presence.copy())
    return out


def physion_ocp_label(trial):
    return bool(trial["contact_zone"].any())


def _absorbed(native, last):
    return min(native, last)


def physion_micro(trial, frames):
    """contact [frames, 18, 18] over native (3f - 3, 3f] (frame 0: native 0); native frames past the end
    repeat the last native frame (terminal absorption); mask = both slots declared."""
    n = len(trial["roles"])
    last = trial["native_frames"] - 1
    contact = np.zeros((frames, SLOTS, SLOTS), bool)
    for f in range(frames):
        lo, hi = max(0, PHYSION_STRIDE * f - PHYSION_STRIDE + 1), PHYSION_STRIDE * f
        window = sorted({_absorbed(k, last) for k in range(lo, hi + 1)})
        contact[f, :n, :n] = trial["contacts"][window].any(0)
    declared = np.zeros(SLOTS, bool)
    declared[:n] = True
    mask = declared[:, None] & declared[None, :] & ~np.eye(SLOTS, dtype=bool)
    return contact, mask


def physion_macro(trial, frames):
    """values, available [frames, 2, 3]; native frames past the end repeat the last native frame."""
    last = trial["native_frames"] - 1

    def at(k):
        return _absorbed(k, last)
    cumulative = np.maximum.accumulate(trial["contact_zone"])
    rest = (trial["velocity"] <= PHYSION_REST_SPEED).all(1) & (trial["angular"] <= PHYSION_REST_ANGULAR).all(1)
    values = np.zeros((frames, 2, len(DELTAS)), bool)
    available = np.zeros((frames, 2, len(DELTAS)), bool)
    for j, delta in enumerate(DELTAS):
        for f in range(frames):
            if f >= delta:
                available[f, 0, j] = True
                values[f, 0, j] = bool(cumulative[at(PHYSION_STRIDE * f)] != cumulative[at(PHYSION_STRIDE * (f - delta))])
            span = max(1, delta)
            if f >= span:
                available[f, 1, j] = True
                begin = PHYSION_STRIDE * (f - span)
                window = [at(k) for k in range(begin + 1, PHYSION_STRIDE * f + 1)]
                values[f, 1, j] = bool(trial["push_time"] <= begin and rest[window].all())
    return values, available
