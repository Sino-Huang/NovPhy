# Issues

## 2026-07-28 Resume at Todo 7
- Prior session was blocked because native delegation was unavailable; this session exposes task delegation.
- The existing Boulder file contains legacy records for another plan; preserve those records while selecting the world-model work as active.

## 2026-07-29 Todo 7 independent verification
- The active `novphy` interpreter still lacks torch, so the exact unqualified `python -m unittest ...` command fails during package import. The already-installed base interpreter provides torch and ran all native tests; no dependency was installed.
- Python LSP diagnostics remain unavailable because basedpyright is not installed and installation was previously declined. Compile, unittest, and source-rule checks pass.
- Repository `.gitignore` has a broad `data/` rule that also ignores `world_model/data/*`; scoped status does not display curriculum product files unless ignore state is queried explicitly.

## 2026-07-29 Todo 8 temporal ablations
- Python LSP diagnostics remain unavailable because basedpyright is not installed and installation was previously declined; compile, unittest, process-stability, source-rule, and literal runtime checks passed instead.
- `world_model/data/ablations.py` measures 249 pure lines, inside the 200-250 warning band. Split serialization/comparison from deterministic selection/accounting before the next substantive feature grows this module.
- The repository-wide `data/` ignore rule also hides `world_model/data/ablations.py` and the modified package exports; direct inspection and no-index stats were required to audit the scoped files.

## 2026-07-29 Todo 9 integration and active-root QA
- Python LSP diagnostics remain unavailable because basedpyright installation was previously declined; `py_compile`, 110 unit tests, import smoke checks, and source gates passed instead.
- The active root contains 6,490 partial episodes across train/dev; they are reported as typed `missing_artifact` rejections and are not repaired or removed.
- Sequential metadata fingerprinting is prohibitively slow on the array; ordered batched `lstat` preserved the exact digest algorithm and completed the after-check.

## 2026-07-30 Todo 9 independent-verification remediation
- Independent verification correctly rejected the prior DoneClaim despite green tests: live composition was lossy and the static gate had nine violations.
- LSP remains unavailable because basedpyright installation was previously declined; all changed Python files were requested and returned the same unavailable status.
- The final exact inspector remains bounded at 155 seconds while explicitly reporting noncanonical summary status.
