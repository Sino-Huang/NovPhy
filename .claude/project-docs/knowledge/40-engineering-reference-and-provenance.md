# Engineering Reference and Provenance

## Testing and Characterization

Use focused characterization tests when changing file-backed behavior. Cover missing, malformed, escaping, symlinked, and partially written artifacts; make fail-closed behavior explicit; preserve accepted/rejected distinctions; and avoid turning dated runtime counts into timeless claims.

Keep dependency-free parsing and validation modules separate from Torch/runtime dependencies. Split a parser before adding unrelated collection behavior. At strict Python boundaries, require exact numeric types, reject booleans where integers are required, and coerce third-party string subclasses to plain `str` only at defined validator boundaries.

## Generated Levels and Root Runtime

IratusAves levels can create dense out-of-distribution structures and materially increase Unity physics/rendering cost. Use them deliberately for that regime; use configured NovPhy subsets for normal QA.

`modules/IratusAves/generator_competition.py` reads `parameters.txt` and writes `level-XX.xml` in its cwd. Use `sciencebirdsagents/Utils/GenerateIratusAvesLevels.py` from the repository root so generation occurs in a temporary directory, output is copied into the root runtime, and malformed XML is normalized. Required normalization fixes the false UTF-16 declaration, unclosed `Camera`/`Slingshot`, missing `Score highScore`, and missing numeric platform transform attributes. Prepare the generated-level config with `PrepareGeneratedLevelsConfig.py` and allow the 60-second readiness timeout.

The root Science Birds runtime depends on its root interface JAR, data tree, database/configuration, and compatible startup layout. Do not replace root assets with benchmark-module equivalents or overwrite the root `9001_Data` tree during diagnosis. Give launcher and engine processes explicit ownership, bounded logs, and cleanup; drain long-lived Java output.

## Evidence Retention

Tracked `.claude/project-docs` content is dormant until explicitly read. Stored size is not automatic context cost. Keep machine-readable evidence, current plans, high-level roadmap, ledger/provenance chains, active worktree recovery evidence, and a compact project-memory index.

Do not bulk-delete or physically deduplicate active Git worktree copies merely because their tracked content is identical.

Before any archive or deletion:

1. reconcile conflicting active plans and evidence;
2. identify a canonical successor for each source;
3. preserve original relative path and SHA-256 in a source-to-destination manifest;
4. verify copied digests;
5. remove only explicitly approved sources.

Do not relocate evidence roots with hard-coded code/documentation references without a separate reviewed migration.

## Context Management

Reduce always-injected global rules and hook output before deleting dormant repository evidence. Byte-based token figures are estimates, not model-exact counts. Recurring hooks should emit only matched instructions.

The context audit and storage inventory are dated operational evidence. Link their paths rather than copying mutable counts into stable knowledge.

## Evidence Link Format

Use:

> Claim: `<short invariant or status>`
> Evidence: `<relative receipt path>`, commit `<commit>`, SHA-256 `<digest>`

Current runtime, branch, editor, licensing, process, artifact, and test claims require dated receipts and should be revalidated before execution.
