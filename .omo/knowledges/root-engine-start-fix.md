# Root Science Birds Linux Engine Startup

The root NovPhy `sciencebirdsgames/Linux` bundle contains NovPhy Unity data and levels, but its extracted `Linux.zip` does not include `game_playing_interface.jar`, top-level `config.xml`, or `DB/`.

The benchmark module bundle `modules/benchmark/sciencebirdsgames/Linux` includes the Java interface assets, but its config and levels are Phy-Q-specific (`type1`, `type2`, etc.) and must not be copied wholesale into the root NovPhy engine.

Use `sciencebirdsagents/Utils/PrepareTestConfig.py` to prepare the root engine. It copies only `game_playing_interface.jar` and `DB/` from the benchmark bundle and writes a NovPhy-compatible root `sciencebirdsgames/Linux/config.xml` using existing root level paths such as:

```text
9001_Data/StreamingAssets/Levels/novelty_level_0/type010101/Levels/00001_0_1_010101_0_1.xml
```

Default command:

```sh
python3 sciencebirdsagents/Utils/PrepareTestConfig.py --os Linux --novelty-level novelty_level_0 --level-type type010101 --max-levels 20
```

Do not overwrite root `sciencebirdsgames/Linux/9001_Data` with the benchmark module's `9001_Data`.
