# F2 Verifier Confinement Debug Journal

Started: 2026-08-06
Goal: Confine archive and provenance paths and permit extraction of only confined regular files and directories.

## Environment Snapshot

- Runtime: CPython 3.13.9 at `/home/sukai/miniconda3/bin/python`
- Entry point: `scripts/verify_physics_player.py`
- Test module: `tests/test_verify_physics_player.py`
- Governing plan: `.omo/plans/world-model-physics-instrumentation.md`
- References read: debugging Python runtime, setup, investigation; programming Python README and typed error handling
- Owned source files were unmodified at the initial scoped status check.

## Hypotheses

1. [OPEN] `archive_from_stage` accepts absolute, parent-traversal, and nested checksum filenames because `stage / checksum[1]` does not parse a single confined filename. Distinguishing evidence: CLI tests can point the receipt at readable files outside or below the stage and reach hashing instead of returning a path diagnostic. If true, fix is: parse component.
2. [OPEN] `verify_payload` accepts absolute, parent-traversal, nested, and symlink-escaped payload paths because `output / relative` plus `is_file()` follows filesystem resolution without proving confinement. Distinguishing evidence: rebuilt archives can make outside or nested files satisfy the declared digest. If true, fix is: resolve confinement.
3. [OPEN] `safe_unpack` permits FIFO, device, socket, and other special tar members because it denies only path traversal and links before `extractall`. Distinguishing evidence: crafted tar member types are not rejected with a typed unsupported-member diagnostic. If true, fix is: allowlist types.
4. [OPEN] Member-by-member extraction would leave partial output when a later member is unsafe, so complete archive preflight is required. Distinguishing evidence: a direct `safe_unpack` test with a valid member before an unsafe member observes whether the valid file appears. If true, fix is: preflight first.
5. [OPEN] Overly strict path validation could reject the existing package layout, which includes root directory entries and nested regular files. Distinguishing evidence: the original valid CLI test and a confined directory/file extraction test remain green. If true, fix is: preserve layout.

## Artifacts And Cleanup Obligations

- [ ] Source edits: `scripts/verify_physics_player.py` and `tests/test_verify_physics_player.py`; retain as the requested remediation.
- [ ] Test and probe temporary directories; create only through `tempfile` or `/tmp`, then remove.
- [ ] Python bytecode caches produced by `py_compile`; remove before completion.
- [ ] Evidence logs under this verifier directory; retain final evidence only.
- [ ] No debugger process, listener, environment override, or published-stage mutation is permitted.

## Findings

- Initial source inspection: checksum and provenance paths are joined directly; tar extraction rejects absolute paths, `..`, symlinks, and hardlinks but does not allowlist regular files/directories.
- Failing-first selector after completing the attack matrix: 6 tests ran with 15 intended assertion failures and no fixture/import errors.
- H1 confirmed: absolute, parent, slash-nested, backslash-ambiguous, and symlink-escaped receipt paths all returned CLI exit 0.
- H2 confirmed: absolute, parent, slash-nested, backslash-ambiguous, and symlink-escaped provenance paths raised no `VerificationError`.
- H3 confirmed: FIFO, character-device, block-device, and socket members reached extraction without the typed unsafe-member diagnostic.
- H4 confirmed: both `first` and `fifo` appeared in the output before the direct extraction test returned.
- H5 initially exposed an over-strict implementation: the real stage uses canonical nested payload paths. The rule was narrowed to reject ambiguous/noncanonical nested spelling while preserving canonical nested paths.
- H5 control passed: the real stage and a unit fixture with a canonical nested payload both verify successfully.

## Final Fix

Typed path diagnostics, tar member allowlisting, complete preflight, and filtered extraction are implemented. Focused/full tests, static gates, real stage verification, malformed/wrong-SHA probes, Oracle review, and cleanup are recorded in the sibling evidence files.
