# Pytest Characterization and File-Probe Guidance

Research baseline: Python 3.14.6 standard-library docs and pytest stable docs, checked 2026-07-28.

## Planning consequences for Todo 2

- Add characterization tests before refactoring. Exercise the existing public entry points and assert returned catalog/export values, ordering, generated paths, and exact failure behavior. Use ordinary `assert` so pytest assertion introspection reports mismatches clearly; use `pytest.raises` for invalid manifests and validation failures.
- Use pytest's function-scoped `tmp_path` for normal filesystem fixtures. It is already a unique `pathlib.Path` per test and is automatically managed. Use `monkeypatch` only when the existing code reads process globals such as the current directory or environment variables; pytest restores those changes after the test.
- Keep manual `tempfile.TemporaryDirectory()` probes small and explicit when the behavior under test is context-manager cleanup, named-directory lifetime, or code that cannot accept pytest's `tmp_path` directly. Prefer the context-manager form; the standard library removes the directory and contents on exit.
- Use `Path` operations for manifests and discovery: `/` for child paths, `read_text`/`write_text` with an explicit encoding, and `iterdir`/`glob`/`rglob` for candidate files. Sort discovered `Path` values before asserting or exporting; filesystem iteration order is not a contract.
- For JSON manifests, use `json.load`/`json.dump` with text files. Characterize malformed JSON as `json.JSONDecodeError` (or the existing wrapper's exception), missing files, wrong top-level shapes, missing keys, and invalid values. For stable serialized exports, assert the parsed object; if byte/text output is itself the contract, use explicit `sort_keys=True` and characterize indentation/separators rather than relying on incidental formatting.
- Keep export tests at the existing public boundary. Verify the exported catalog/list and any written manifest separately: discovery/validation tests should establish the normalized in-memory result, while export tests should establish file destination, JSON shape, ordering, and repeatability.

## Authoritative references

- Python `pathlib`: https://docs.python.org/3/library/pathlib.html
- Python `json`: https://docs.python.org/3/library/json.html
- Python `tempfile.TemporaryDirectory`: https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
- pytest assertions: https://docs.pytest.org/en/stable/how-to/assert.html
- pytest fixtures and teardown: https://docs.pytest.org/en/stable/how-to/fixtures.html
- pytest `tmp_path`: https://docs.pytest.org/en/stable/how-to/tmp_path.html
- pytest monkeypatch: https://docs.pytest.org/en/stable/how-to/monkeypatch.html
