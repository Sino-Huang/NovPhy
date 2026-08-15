# Bounded scorer worker report

Implementation surface:

- `world_model/training/scoring_torch.py`: `TorchCatalogPredictor` now decodes each shot through positional frame batches, encodes online and EMA-target latents in the same bounded batches, caches only detached CPU latent tensors/action per shot, and gathers requested context/terminal-clamped target rows into bounded predictor batches.
- `world_model/training/real_data.py`: added `shot_frame_count`, `shot_action`, and bounded `shot_frame_batch` adapter methods.
- `tests/test_world_model_scoring.py`: failing-first synthetic regression with an instrumented backbone/data adapter.

Evidence:

- Scenario: synthetic four-frame shot, `batch_size=2`, all partitions, contexts 0/1/2, requested deltas 1/5/15. Invocation: `python -m unittest -v tests.test_world_model_scoring.ExhaustiveScoringTests.test_real_catalog_scoring_bounds_decode_and_encode_batches`. Observable: test passed; decoder calls were exactly `[2, 2]`, all online/target encoder and predictor batch sizes were `<= 2`, and each latent MSE matched the independent scalar reference with `min(context + requested_delta, terminal)` for terminal states.
- Scenario: focused scoring, real-data adapter, and grid-run suite. Invocation: `python -m unittest -v tests.test_world_model_scoring tests.test_world_model_real_data tests.test_world_model_grid_run`. Observable: `Ran 25 tests ... OK`.
- Scenario: CUDA checkpoint round-trip when CUDA is available. Invocation: `python -m unittest -v tests.test_world_model_grid_run.GridRunTests.test_checkpoint_round_trip_restores_cpu_rng_state_on_cuda`. Observable: test passed (`Ran 1 test ... OK`).
- Scenario: changed Python syntax and whitespace. Invocation: `python -m py_compile world_model/training/scoring_torch.py world_model/training/real_data.py tests/test_world_model_scoring.py && git diff --check`. Observable: both commands passed.

Tool note: `ruff` was not installed (`command not found`), so no ruff result is claimed.
