## Decision P-crash (2026-10-24): selection runs relaunched once after a co-scheduling OOM

**Trigger.** The three selection jobs (`129-select-c1/c2/c3`) were launched together on GPU 1 at 20:40 as
`shares_gpu` jobs. All three died within their first steps with CUDA out-of-memory (the logs show about 8.3 GB per
stride-15 run and over 14.8 GB for the stride-5 run, so they could not share the 32 GB GPU). No optimizer state,
checkpoint or model was written. This is an infrastructure crash caused by my memory estimate, not a typed
failure of any configuration.

**Choice.** Relaunch once (P-crash), changing only the co-scheduling, never a configuration:
- `129-select-c1` and `129-select-c3` (both stride 15) run together on GPU 1, about 17 GB;
- `129-select-c2` (stride 5) runs alone on GPU 1 after both.

The runs restart from step 0 with the same seed and batch stream. The relaunch started at 20:50, and both
stride-15 runs passed step 200 with finite, decreasing loss. Disclosed in the child doc.
