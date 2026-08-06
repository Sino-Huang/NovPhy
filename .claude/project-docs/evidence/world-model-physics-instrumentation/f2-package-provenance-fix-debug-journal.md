# Package provenance fix debug journal

## Hypotheses

1. The guard rejects the ignored package pair because it has no allowlist; distinguishing evidence is the exact `git status --porcelain=v1 --ignored=matching` output, and the fix is a path allowlist.
2. Allowing paths without bytes would permit stale or changed package content; distinguishing evidence is a digest mismatch after mutation, and the fix is digest binding.
3. A broad `Packages/*` exception would admit unrelated Unity inputs; distinguishing evidence is an ignored third package or asset path, and the fix is exact-path matching.

## Observations

- Target worktree HEAD: `6668b12f43f2c577c7f2446c98aedea0811f913e`.
- `manifest.json` is ignored with SHA-256 `05677cc3199d5fff4aac54096877e795518487918e53810f477a228e5d1e28fb`.
- `packages-lock.json` is ignored with SHA-256 `3101c351984e6a73a1be7ad76d1a67c1b7638a6616554e50000b9672175ebe50`.
- Current guard reports `untracked product source: !! tasks/task_template_designer/Packages/manifest.json`.

## Artifacts

- [ ] Focused test output: `.omo/evidence/world-model-physics-instrumentation/f2-package-provenance-fix-focused-1.log`.
- [ ] Focused test output: `.omo/evidence/world-model-physics-instrumentation/f2-package-provenance-fix-focused-2.log`.
- [ ] Preflight output: `.omo/evidence/world-model-physics-instrumentation/f2-package-provenance-fix-preflight.log`.
