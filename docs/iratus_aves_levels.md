# IratusAves Level Generation

`modules/IratusAves` generates Science Birds XML levels from a `parameters.txt` file. The generator reads `parameters.txt` from its working directory and writes files named `level-XX.xml`.

Generate one batch into the root Linux engine level tree:

```sh
python3 sciencebirdsagents/Utils/GenerateIratusAvesLevels.py --levels 20 --pig-range 3,6
```

Generated levels are copied to:

```text
sciencebirdsgames/Linux/9001_Data/StreamingAssets/Levels/iratus_aves/Levels/
```

Load those generated levels through the game engine config:

```sh
python3 sciencebirdsagents/Utils/PrepareGeneratedLevelsConfig.py
```

Expected result: `sciencebirdsgames/Linux/config.xml` contains `game_levels` entries pointing at `9001_Data/StreamingAssets/Levels/iratus_aves/Levels/*.xml`.

To use a custom IratusAves parameter file instead of CLI defaults:

```sh
python3 sciencebirdsagents/Utils/GenerateIratusAvesLevels.py --parameters modules/IratusAves/parameters.txt
python3 sciencebirdsagents/Utils/PrepareGeneratedLevelsConfig.py
```

After writing the config, start the normal game interface or WebUI. To switch back to NovPhy levels, rerun:

```sh
python3 sciencebirdsagents/Utils/PrepareTestConfig.py --os Linux --novelty-level novelty_level_0 --level-type type010101 --max-levels 20
```
