#!/bin/bash
# Implemented the IratusAves integration.
# Added root-repo scripts:
python3 sciencebirdsagents/Utils/GenerateIratusAvesLevels.py --levels 20 --pig-range 3,6

python3 sciencebirdsagents/Utils/PrepareGeneratedLevelsConfig.py
# The generator script runs modules/IratusAves in a temp workdir, copies levels into sciencebirdsgames/Linux/9001_Data/StreamingAssets/Levels/iratus_aves/Levels/, and normalizes IratusAves XML so the NovPhy Science Birds engine can load it. The config script writes sciencebirdsgames/Linux/config.xml to point at those generated levels. 