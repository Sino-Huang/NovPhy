"""#131 D1 THICK G-fid: VisualPinPad, the env wrappers, the driver, the replay and the random agent.

``PinPad`` is the vendored upstream file ``vendor/thick/common/pinpad_env.py`` executed verbatim (it needs only
``gym.spaces`` for its space declarations; when gym is not installed a minimal ``gym.spaces`` stand-in is provided).
``OneHotAction``, ``TimeLimit`` (``common/envs.py``), ``Driver`` (``common/driver.py``), ``Replay``
(``common/replay.py``) and ``RandomAgent`` (``common/other.py``) are ports of the upstream code, kept line by line,
with these implementation differences (no semantic change):

- every random stream is a seeded ``np.random.RandomState`` held by the object (upstream: unseeded ``RandomState()``
  objects, and the global ``np.random`` for the replay's length jitter), so a run is reproducible and resumable;
- the replay's chunk generator keeps its pending sequence as (episode, start, length) and is checkpointable;
  batches are drawn synchronously (upstream: ``tf.data`` with ``prefetch(5)``, i.e. up to a few batches drawn
  ahead of the update that uses them);
- episodes loaded from another run's directory (the Plan2Explore data) are uncompressed once into ``.npy`` files
  by a separate step (``build_cache``: temporary directory, fsync, rename, ``MANIFEST.json`` with per-array sha256)
  and opened read-only and memory-mapped by every run (``open_cache`` refuses an incomplete or mismatching cache),
  so concurrent runs share them through the page cache (upstream: each process decompresses every episode into its
  own memory); saved episodes use the upstream ``.npz`` format and file names.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import uuid

import numpy as np

VENDOR = Path(__file__).resolve().parent / "vendor"


# ---------------------------------------------------------------------------
# VisualPinPad (vendored file, verbatim)
# ---------------------------------------------------------------------------


class _Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, tuple(shape or ()), dtype


class _Discrete:
    def __init__(self, n):
        self.n, self.shape, self.dtype = n, (), np.int64


def _gym_module():
    try:
        import gym  # noqa: F401
        return None
    except ImportError:
        gym = types.ModuleType("gym")
        gym.spaces = types.SimpleNamespace(Box=_Box, Discrete=_Discrete)
        return gym


PINPAD_MODULE = "_thick_vendored_pinpad"


def pinpad_class():
    """The vendored ``PinPad`` class (module registered as ``_thick_vendored_pinpad`` so env objects pickle)."""
    if PINPAD_MODULE in sys.modules:
        return sys.modules[PINPAD_MODULE].PinPad
    stand_in = _gym_module()
    spec = importlib.util.spec_from_file_location(PINPAD_MODULE, VENDOR / "thick/common/pinpad_env.py")
    module = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("gym")
    if stand_in is not None:
        sys.modules["gym"] = stand_in
    try:
        spec.loader.exec_module(module)
    finally:
        if stand_in is not None:
            if saved is None:
                sys.modules.pop("gym", None)
            else:
                sys.modules["gym"] = saved
    sys.modules[PINPAD_MODULE] = module
    return module.PinPad


class OneHotAction:
    """``common.envs.OneHotAction``."""

    def __init__(self, env, key="action", seed=None):
        assert hasattr(env.act_space[key], "n")
        self._env = env
        self._key = key
        self._random = np.random.RandomState(seed)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def act_space(self):
        shape = (self._env.act_space[self._key].n,)
        space = _Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.n = shape[0]
        return {**self._env.act_space, self._key: space}

    def step(self, action):
        index = np.argmax(action[self._key]).astype(int)
        reference = np.zeros_like(action[self._key])
        reference[index] = 1
        if not np.allclose(reference, action[self._key]):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self._env.step({**action, self._key: index})

    def reset(self):
        return self._env.reset()


class TimeLimit:
    """``common.envs.TimeLimit`` (duration 0 = no limit)."""

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()


def make_env(task, seed, time_limit=0):
    """``common.env_register.make_env`` for ``pinpad_<layout>``: PinPad -> OneHotAction -> TimeLimit."""
    suite, name = task.split("_", 1)
    if suite != "pinpad":
        raise NotImplementedError(suite)
    env = pinpad_class()(name)
    env.random = np.random.RandomState(seed)
    env = OneHotAction(env, seed=seed + 1)
    return TimeLimit(env, time_limit)


# ---------------------------------------------------------------------------
# driver and random agent
# ---------------------------------------------------------------------------


def _convert(value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    if np.issubdtype(value.dtype, np.signedinteger):
        return value.astype(np.int32)
    if np.issubdtype(value.dtype, np.uint8):
        return value.astype(np.uint8)
    return value


class Driver:
    """``common.driver.Driver``; ``__call__`` also stops early when ``stop()`` is true at a step boundary and returns
    the (steps, episodes) it ran, so a caller can resume the remainder."""

    def __init__(self, envs, **kwargs):
        self._envs = envs
        self._kwargs = kwargs
        self._on_steps, self._on_resets, self._on_episodes = [], [], []
        self._act_spaces = [env.act_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def state_dict(self):
        return {"obs": self._obs, "eps": self._eps, "state": self._state}

    def load_state_dict(self, state):
        self._obs, self._eps, self._state = state["obs"], state["eps"], state["state"]

    def __call__(self, policy, steps=0, episodes=0, stop=None):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            if stop is not None and stop():
                break
            obs = {i: self._envs[i].reset() for i, ob in enumerate(self._obs) if ob is None or ob["is_last"]}
            for i, ob in obs.items():
                self._obs[i] = ob() if callable(ob) else ob
                act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
                tran = {k: _convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
                self._eps[i] = [tran]
            obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
            policy_outs, self._state = policy(obs, self._state, **self._kwargs)
            actions = {"action": policy_outs["action"]}
            actions = [{k: np.array(actions[k][i]) for k in actions} for i in range(len(self._envs))]
            assert len(actions) == len(self._envs)
            obs = [e.step(a) for e, a in zip(self._envs, actions)]
            obs = [ob() if callable(ob) else ob for ob in obs]
            for i, (act, ob) in enumerate(zip(actions, obs)):
                tran = {k: _convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)
                step += 1
                if ob["is_last"]:
                    ep = self._eps[i]
                    ep = {k: _convert([t[k] for t in ep]) for k in ep[0]}
                    [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
            self._obs = obs
        return step, episode


class RandomAgent:
    """``common.other.RandomAgent`` for one-hot action spaces (uniform one-hot actions)."""

    def __init__(self, act_space, seed):
        self.n = act_space["action"].n
        self._random = np.random.RandomState(seed)

    def __call__(self, obs, state=None, mode=None):
        index = self._random.randint(0, self.n, len(obs["is_first"]))
        return {"action": np.eye(self.n, dtype=np.float32)[index]}, None


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def eplen(episode):
    return len(episode["action"]) - 1


def count_episodes(directory):
    filenames = list(Path(directory).glob("*.npz"))
    return len(filenames), sum(int(str(n).split("-")[-1][:-4]) - 1 for n in filenames)


def save_episode(directory, episode):
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    identifier = str(uuid.uuid4().hex)
    filename = Path(directory) / f"{timestamp}-{identifier}-{eplen(episode)}.npz"
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        tmp = filename.with_suffix(".tmp")
        tmp.write_bytes(f1.read())
        os.replace(tmp, filename)
    return filename


CACHE_SCHEMA = "issue_131_replay_cache_v1"
PROBE_KEYS = ("action", "reward")  # arrays re-hashed at every run start (small)


class CacheError(RuntimeError):
    pass


def _sha256(array):
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def build_cache(load_directory, cache_root):
    """Uncompress every ``.npz`` episode of ``load_directory`` once into ``cache_root`` (``<episode>/<key>.npy``)
    with ``MANIFEST.json`` (per-episode npz name, bytes, length, and per-array shape / dtype / sha256). Built in a
    temporary directory (every file fsynced), then renamed into place; refuses to overwrite an existing cache.
    Returns the manifest. Never called by a training run (runs only open a complete cache, ``open_cache``)."""
    load_directory, cache_root = Path(load_directory), Path(cache_root)
    if cache_root.exists():
        raise CacheError(f"{cache_root} exists; a cache is built once (remove it explicitly to rebuild)")
    building = cache_root.with_name(cache_root.name + f".building-{os.getpid()}")
    building.mkdir(parents=True)
    episodes = []
    for filename in sorted(load_directory.glob("*.npz")):
        target = building / filename.stem
        target.mkdir()
        arrays = {}
        with np.load(filename) as data:
            for key in data.files:
                value = data[key]
                with open(target / f"{key}.npy", "wb") as f:
                    np.save(f, value)
                    f.flush()
                    os.fsync(f.fileno())
                arrays[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _sha256(value)}
        _fsync_dir(target)
        episodes.append({"npz": filename.name, "npz_bytes": filename.stat().st_size,
                         "length": arrays["action"]["shape"][0] - 1, "arrays": arrays})
    manifest = {"schema": CACHE_SCHEMA, "source": str(load_directory), "episodes": episodes,
                "steps": sum(e["length"] for e in episodes), "complete": True}
    with open(building / "MANIFEST.json", "w") as f:
        f.write(json.dumps(manifest, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(building)
    os.rename(building, cache_root)
    _fsync_dir(cache_root.parent)
    return manifest


def read_manifest(cache_root):
    path = Path(cache_root) / "MANIFEST.json"
    if not path.is_file():
        raise CacheError(f"{cache_root}: no MANIFEST.json (cache missing or incomplete; run --build-cache)")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("complete") is not True:
        raise CacheError(f"{cache_root}: manifest incomplete or of another schema")
    return manifest


def open_cache(cache_root, filenames):
    """Read-only memory maps of the cached episodes ``filenames`` (upstream .npz paths). Refuses unless the cache's
    manifest covers exactly the source directory's episodes with matching npz sizes, every array has its recorded
    shape / dtype, and the probe arrays (``PROBE_KEYS``) re-hash to their recorded sha256."""
    cache_root = Path(cache_root)
    manifest = read_manifest(cache_root)
    entries = {e["npz"]: e for e in manifest["episodes"]}
    filenames = [Path(f) for f in filenames]
    source = {p.name for p in filenames[0].parent.glob("*.npz")} if filenames else set()
    if source != set(entries):
        raise CacheError(f"{cache_root}: manifest episodes differ from {filenames[0].parent}")
    out = {}
    for filename in filenames:
        entry = entries[filename.name]
        if filename.stat().st_size != entry["npz_bytes"]:
            raise CacheError(f"{filename.name}: npz size differs from the manifest")
        arrays = {}
        for key, spec in entry["arrays"].items():
            value = np.load(cache_root / filename.stem / f"{key}.npy", mmap_mode="r")
            if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
                raise CacheError(f"{filename.stem}/{key}: shape or dtype differs from the manifest")
            if key in PROBE_KEYS and _sha256(value) != spec["sha256"]:
                raise CacheError(f"{filename.stem}/{key}: sha256 differs from the manifest")
            arrays[key] = value
        out[str(filename)] = arrays
    return out


def verify_cache(load_directory, cache_root):
    """Every array of every source episode equals its cached copy and its manifest sha256 (full read)."""
    manifest = read_manifest(cache_root)
    filenames = sorted(Path(load_directory).glob("*.npz"))
    opened = open_cache(cache_root, filenames)
    mismatches, arrays = [], 0
    for filename in filenames:
        spec = {e["npz"]: e for e in manifest["episodes"]}[filename.name]["arrays"]
        with np.load(filename) as data:
            if set(data.files) != set(spec):
                mismatches.append(f"{filename.name}: keys")
            for key in data.files:
                arrays += 1
                cached = opened[str(filename)][key]
                if not np.array_equal(data[key], cached) or _sha256(cached) != spec[key]["sha256"]:
                    mismatches.append(f"{filename.name}/{key}")
    return {"episodes": len(filenames), "arrays": arrays, "steps": manifest["steps"], "mismatches": mismatches,
            "pass": not mismatches and len(filenames) == len(manifest["episodes"])}


class Replay:
    """``common.replay.Replay`` (uniform episodes, ``minlen`` / ``maxlen`` sequences, ``prioritize_ends``)."""

    def __init__(self, directory, capacity=0, ongoing=False, minlen=1, maxlen=0, prioritize_ends=False,
                 load_directory=None, cache=None, seed=0):
        self._directory = Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)
        self._capacity, self._ongoing = capacity, ongoing
        self._minlen, self._maxlen, self._prioritize_ends = minlen, maxlen, prioritize_ends
        self._random = np.random.RandomState(seed)
        self._length_random = np.random.RandomState(seed + 1)
        self._load_directory, self._cache = load_directory, cache
        self._complete_eps = {}
        if load_directory is not None:
            filenames = sorted(Path(load_directory).expanduser().glob("*.npz"))
            if capacity:
                steps = episodes = 0
                for filename in reversed(filenames):
                    steps += int(str(filename).split("-")[-1][:-4])
                    episodes += 1
                    if steps >= capacity:
                        break
                filenames = filenames[-episodes:]
            if cache is not None:
                self._complete_eps.update(open_cache(cache, filenames))
            else:
                for filename in filenames:
                    with np.load(filename) as data:
                        self._complete_eps[str(filename)] = {k: data[k] for k in data.files}
            self._total_episodes, self._total_steps = count_episodes(load_directory)
        else:
            self._total_episodes, self._total_steps = count_episodes(self._directory)
            for filename in sorted(self._directory.glob("*.npz")):
                with np.load(filename) as data:
                    self._complete_eps[str(filename)] = {k: data[k] for k in data.files}
        self._ongoing_eps = collections.defaultdict(lambda: collections.defaultdict(list))
        self._loaded_episodes = len(self._complete_eps)
        self._loaded_steps = sum(eplen(x) for x in self._complete_eps.values())
        self._pending = None

    @property
    def stats(self):
        return {"total_steps": self._total_steps, "total_episodes": self._total_episodes,
                "loaded_steps": self._loaded_steps, "loaded_episodes": self._loaded_episodes}

    def add_step(self, transition, worker=0):
        episode = self._ongoing_eps[worker]
        for key, value in transition.items():
            episode[key].append(value)
        if transition["is_last"]:
            self.add_episode(episode)
            episode.clear()

    def add_episode(self, episode):
        length = eplen(episode)
        if length < self._minlen:
            return
        self._total_steps += length
        self._loaded_steps += length
        self._total_episodes += 1
        self._loaded_episodes += 1
        episode = {key: _convert(value) for key, value in episode.items()}
        filename = save_episode(self._directory, episode)
        self._complete_eps[str(filename)] = episode
        self._enforce_limit()

    def _enforce_limit(self):
        if not self._capacity:
            return
        while self._loaded_episodes > 1 and self._loaded_steps > self._capacity:
            oldest, episode = next(iter(self._complete_eps.items()))
            self._loaded_steps -= eplen(episode)
            self._loaded_episodes -= 1
            del self._complete_eps[oldest]

    def _sample_sequence(self):
        assert not self._ongoing, "ported: ongoing False"
        keys = list(self._complete_eps.keys())
        key = keys[self._random.choice(len(keys))]
        total = len(self._complete_eps[key]["action"])
        length = total
        if self._maxlen:
            length = min(length, self._maxlen)
        length -= self._length_random.randint(self._minlen)
        length = max(self._minlen, length)
        upper = total - length + 1
        if self._prioritize_ends:
            upper += self._minlen
        index = min(self._random.randint(upper), total - length)
        return [key, index, length]

    def _materialize(self, ref, take):
        key, index, length = ref
        episode = self._complete_eps[key]
        out = {k: _convert(np.asarray(v[index:index + take])) for k, v in episode.items() if not k.startswith("log_")}
        return out

    def chunk(self, length):
        """One ``_generate_chunks`` chunk (the pending sequence is kept across calls, as the upstream generator)."""
        if self._pending is None:
            self._pending = self._sample_sequence()
        parts = collections.defaultdict(list)
        added = 0
        while added < length:
            needed = length - added
            key, index, seq_len = self._pending
            take = min(needed, seq_len)
            adding = self._materialize(self._pending, take)
            first = np.zeros(take, bool)
            if self._pending_is_fresh():
                first[0] = True
            adding["is_first"] = first
            self._pending = [key, index + take, seq_len - take]
            self._fresh = False
            for k, v in adding.items():
                parts[k].append(v)
            added += take
            if self._pending[2] < 1:
                self._pending = self._sample_sequence()
                self._fresh = True
        return {k: np.concatenate(v) for k, v in parts.items()}

    def _pending_is_fresh(self):
        return getattr(self, "_fresh", True)

    def batch(self, batch, length):
        chunks = [self.chunk(length) for _ in range(batch)]
        return {k: np.stack([c[k] for c in chunks]) for k in chunks[0]}

    def state_dict(self):
        return {"keys": list(self._complete_eps.keys()), "random": self._random.get_state(),
                "length_random": self._length_random.get_state(), "pending": self._pending,
                "fresh": self._pending_is_fresh(), "totals": [self._total_steps, self._total_episodes,
                                                             self._loaded_steps, self._loaded_episodes],
                "ongoing": {w: {k: list(v) for k, v in ep.items()} for w, ep in self._ongoing_eps.items()}}

    def load_state_dict(self, state):
        """Restore after a restart: the loaded directory is re-read in the constructor; this run's own episodes
        are re-read from disk and the episode order restored."""
        own = {str(p): p for p in self._directory.glob("*.npz")}
        for key in state["keys"]:
            if key not in self._complete_eps:
                if key not in own:
                    raise FileNotFoundError(f"replay episode {key} missing on resume")
                with np.load(own[key]) as data:
                    self._complete_eps[key] = {k: data[k] for k in data.files}
        self._complete_eps = {k: self._complete_eps[k] for k in state["keys"]}
        self._random.set_state(state["random"])
        self._length_random.set_state(state["length_random"])
        self._pending, self._fresh = state["pending"], state["fresh"]
        (self._total_steps, self._total_episodes, self._loaded_steps, self._loaded_episodes) = state["totals"]
        self._ongoing_eps = collections.defaultdict(lambda: collections.defaultdict(list))
        for w, ep in state["ongoing"].items():
            for k, v in ep.items():
                self._ongoing_eps[w][k] = list(v)
