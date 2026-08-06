# NovPhy Episode Selection

Date: 2026-06-24

NovPhy episodes are selected by writing `sciencebirdsgames/Linux/config.xml` before starting the Science Birds Java interface. Runtime arbitrary level loading through bridge requests is unreliable in this environment, so the stable path is:

1. Choose an episode directory under `sciencebirdsgames/Linux/9001_Data/StreamingAssets/Levels/`.
2. Generate `sciencebirdsgames/Linux/config.xml` with `sciencebirdsagents/Utils/PrepareTestConfig.py`.
3. Start a fresh engine for collection or evaluation.

## Episode Layout

The installed level tree follows this structure:

```text
sciencebirdsgames/Linux/9001_Data/StreamingAssets/Levels/
  novelty_level_0/
    type010101/Levels/*.xml
    type010102/Levels/*.xml
    ...
  novelty_level_1/
    type010101/Levels/*.xml
    ...
  novelty_level_8/
    type010805/Levels/*.xml
```

`novelty_level_0` contains the normal task templates. `novelty_level_1` through `novelty_level_8` correspond to the eight NovPhy novelty categories described in the project README: objects, agents, actions, interactions, relations, environments, goals, and events.

The last two digits of `type010xyz` identify the physical scenario:

```text
...01 = single force
...02 = multiple forces
...03 = rolling
...04 = falling
...05 = sliding
```

For example:

```text
novelty_level_0/type010101 = normal single-force tasks
novelty_level_3/type010303 = action-novelty rolling tasks
novelty_level_8/type010805 = event-novelty sliding tasks
```

Observed available directories on 2026-06-24:

- `novelty_level_0`: 40 type directories, including normal and novelty-coded templates.
- `novelty_level_1` through `novelty_level_8`: 5 type directories each, matching their five scenarios.
- Most type directories contain 350 level XML files; `novelty_level_0/type010104` contains 301.

## Loading an Episode

Use `PrepareTestConfig.py` from the repo root. Always initialize the project environment first:

```sh
source ~/cd_novphy && \
python3 sciencebirdsagents/Utils/PrepareTestConfig.py \
  --os Linux \
  --novelty-level novelty_level_3 \
  --level-type type010303 \
  --max-levels 20
```

This writes `sciencebirdsgames/Linux/config.xml` with the first 20 matching XML levels. The game reads that file at startup.

Important: `config.xml` is global mutable state. If you are running a temporary collection, either restore it afterward or intentionally leave it pointing at the desired episode.

The previous local default before the 2026-06-24 review collection was restored to:

```sh
source ~/cd_novphy && \
python3 sciencebirdsagents/Utils/PrepareTestConfig.py \
  --os Linux \
  --novelty-level novelty_level_0 \
  --level-type type010102 \
  --max-levels 20
```

## Safe Multi-Episode Rollout Recipe

On this server, protocol screenshots can be uniform gray. Use Xvnc desktop capture with fresh-engine-per-rollout mode:

```sh
source ~/cd_novphy && \
Xvnc :149 -geometry 1024x768 -depth 24 -SecurityTypes None -rfbport 0 >/tmp/novphy_rollout_xvnc_149.log 2>&1 &
```

For each selected episode:

```sh
source ~/cd_novphy && \
python3 sciencebirdsagents/Utils/PrepareTestConfig.py \
  --os Linux \
  --novelty-level novelty_level_3 \
  --level-type type010303 \
  --max-levels 20

source ~/cd_novphy && \
DISPLAY=:149 LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}" \
python scripts/collect_rollouts.py \
  --output-dir data/review_rollouts_YYYYMMDD_novelty3_type010303 \
  --capture-source desktop \
  --fresh-engine-per-rollout \
  --ui-level 1 \
  --ui-settle-seconds 5 \
  --count 2 \
  --fps 6 \
  --duration 5 \
  --connect-timeout 40 \
  --prepare-timeout 80 \
  --read-timeout 360
```

After collection, stop the Xvnc process and restore `config.xml` if needed.

## 2026-06-24 Review Collection

Collected a compact review range across three different episode buckets:

- `data/review_rollouts_20260624_novelty0_type010101/manifest.json`
- `data/review_rollouts_20260624_novelty3_type010303/manifest.json`
- `data/review_rollouts_20260624_novelty8_type010805/manifest.json`

Each manifest uses:

- `capture_source: capture_desktop_rollout`
- `replay_mode: fresh-engine-per-rollout`
- `target_fps: 6.0`
- `duration_seconds: 5.0`
- `ui_level: 1`
- 2 rollouts with distinct drag/release actions

Verification summary:

- `novelty0/type010101`: 30 PNG frames per rollout, `pre_shot.png` present, nonzero pre-shot and frame deltas.
- `novelty3/type010303`: 30 PNG frames per rollout, `pre_shot.png` present, nonzero pre-shot and frame deltas.
- `novelty8/type010805`: 19 and 22 PNG frames, `pre_shot.png` present, nonzero pre-shot and frame deltas.
- Every rollout had `pre_shot_sample.state == PLAYING`, `pre_shot_sample.score == 0`, and a resolved `slingshot_reference`.
- No leftover `game_playing_interface`, Unity, or Xvnc process was observed after collection.
