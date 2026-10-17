launch1-wrapper-pids.txt: PIDs recorded at the first (and only kept) training launch were the nohup wrapper shells, not the python workers.
The real workers are 2013294/2013343/2013392/2013441 (train-pids.txt). Believing them dead, a second launch (setsid wrappers 2014741-7,
python workers 2014742/2014744/2014746/2014748) ran for about 4 minutes on the same GPUs; it was killed before any of its units reached a
progress checkpoint or completed. The claim files prevented any unit from being trained twice; its claims became stale and are re-claimed
by the first launch. train-worker*.log here are the killed duplicate workers' logs.
