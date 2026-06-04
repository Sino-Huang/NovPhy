# Root Science Birds Linux Engine Startup

The root NovPhy `sciencebirdsgames/Linux` bundle contains NovPhy Unity data and levels. Runtime startup also requires real Science Birds Java interface assets at `sciencebirdsgames/Linux/game_playing_interface.jar` and `sciencebirdsgames/Linux/DB/`, plus a generated top-level `config.xml`.

The Java interface assets are expected to be provisioned into the root runtime engine directory outside `modules/`. Do not use benchmark module levels/config as the runtime source for NovPhy startup.

Use `sciencebirdsagents/Utils/PrepareTestConfig.py` to prepare the root engine. It validates the root `game_playing_interface.jar` and `DB/` assets, then writes a NovPhy-compatible root `sciencebirdsgames/Linux/config.xml` using existing root level paths such as:

```text
9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml
```

Default command:

```sh
python3 sciencebirdsagents/Utils/PrepareTestConfig.py --os Linux --novelty-level novelty_level_0 --level-type type010101 --max-levels 20
```

Do not overwrite root `sciencebirdsgames/Linux/9001_Data` with non-NovPhy level data.
