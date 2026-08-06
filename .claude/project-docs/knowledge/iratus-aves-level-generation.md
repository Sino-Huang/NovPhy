# IratusAves Level Generation

`modules/IratusAves/generator_competition.py` reads `parameters.txt` from its current working directory and writes `level-XX.xml` files there. The root integration script is `sciencebirdsagents/Utils/GenerateIratusAvesLevels.py`; it runs the generator in a temporary directory, copies generated levels into `sciencebirdsgames/<OS>/9001_Data/StreamingAssets/Levels/iratus_aves/Levels/`, and normalizes the XML for the NovPhy Science Birds engine.

Normalization is required because the IratusAves generator emits XML that is not directly loadable in this engine build: it declares `utf-16` while writing UTF-8 text, leaves `Camera` and `Slingshot` unclosed, omits `Score highScore="0"`, and omits numeric `rotation`, `scaleX`, and `scaleY` attributes on some `Platform` elements.

Generate and load generated levels from the repo root:

```sh
python3 sciencebirdsagents/Utils/GenerateIratusAvesLevels.py --levels 20 --pig-range 3,6
python3 sciencebirdsagents/Utils/PrepareGeneratedLevelsConfig.py
```

The generated-level config can take longer to reach `PLAYING` than NovPhy levels; WebUI `AppState.readiness_timeout` is 60 seconds for this reason. Verified live smoke loaded a generated level with `numberOfLevels: 1`, frame `640 x 480`, state `PLAYING`, and RGB base64 length `1228800`.
