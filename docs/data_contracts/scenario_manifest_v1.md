# Scenario manifest v1

`scenario_manifest_v1` is the canonical identity and provenance envelope for one NovPhy level instance. It establishes the hierarchy

`benchmark condition -> scenario template -> level instance -> scenario specification -> scenario lineage`

before scenario collection. The standard-library implementation is `scripts/scenario_manifest.py`.

## Artifact pairing

A canonical sidecar is adjacent to its XML and replaces `.xml` with `.scenario.json`. Loading a sidecar with `load_manifest(manifest_path, xml_path)` validates the manifest's closed record shape, generation/import mode consistency, nonempty declared identities, research-eligibility fields, and XML parseability. Identities are carried as declared provenance strings; loading does not re-derive them from file bytes.

Planner discovery validates an adjacent sidecar. XML without a sidecar is represented in memory through the legacy importer as `legacy_static`; it is never assigned a generator, generator version, generation seed, parameter realization, or available template identity that cannot be proven.

## Records

The manifest contains these immutable records:

- `benchmark_condition`: novelty level, novelty type, and their derived identity.
- `scenario_template`: an identity with `available` evidence, or an explicit `unavailable` state.
- `generation`: `generated` or `legacy_static` provenance. Generated records include generator identity/version, generation seed, declared inputs, and deterministic parameter realization. Legacy records include importer identity/version and source path.
- `level_instance`, `scenario_specification`, and `scenario_lineage`: the derived canonical identity hierarchy.
- `declared_initial_engine_state`: the versioned identity declared from the level-instance identity; supplied XML is parse-validated only.
- `research_eligibility`: `research_eligible` or `smoke_only`, with a required reason for `smoke_only`.

## Canonicalization and identities

Structured identity keys use UTF-8 JSON with sorted keys, no insignificant whitespace, and no non-finite numbers. `_identity(namespace, *keys)` converts structured keys to that compact JSON, converts `None` to `none`, percent-quotes each value with `safe="-._~"`, and joins the namespace and keys with `:`. XML declared-state projection separately preserves root and child order, every element tag, every attribute sorted by name, non-whitespace text, and unknown elements/attributes. Formatting-only indentation is not engine state.

Producers declare these semantic identity strings:

1. `benchmark-condition-v1:<novelty-level>:<novelty-type>`;
2. `scenario-declaration-v1:<benchmark-identity>:<template-identity-or-availability>:<mode>:<generator-or-importer>:<version>:<seed-or-none>:<declared-inputs-or-source-path>`;
3. `level-instance-v1:<quoted-scenario-declaration-identity>`;
4. `scenario-content-v1:<quoted-level-instance-identity>` and `scenario-specification-v1:<quoted-level-instance-identity>`;
5. `scenario-lineage-v1:<quoted-scenario-specification-identity>`;
6. `declared-initial-engine-state-v1:<quoted-level-instance-identity>`.

The current `scenario-content-v1` identity is a declared binding to the level-instance identity; it is not an XML-byte fingerprint. XML bytes remain the level artifact and must be parseable when supplied to `load_manifest()` or `verify_replay()`, but consumers do not recompute an identity from those bytes. Generator/importer identity and version, seed, template evidence, benchmark condition, and declared inputs or legacy source path are the semantic keys used by the producer's declaration format.

The staged legacy `type2` XML historically declares UTF-16 while containing UTF-8 bytes. The declared-state parser normalizes that known declaration mismatch so the existing engine-tolerated tree can be projected; the semantic identities remain those declared by the legacy importer provenance.

## Deterministic level-instance materialization and replay

`tasks/task_generator/canonical_materialization.py` exposes `materialize_level_instance()` for one explicit template and one generated level instance. It uses an operation-local `random.Random(generation_seed)`, explicit paths and constraints, deterministic XML serialization, and the historical variation transformation. Process-global RNG state, working directory, previous generation operations, and batch ordering do not participate.

The `scenario_parameter_realization_v1` record contains the complete canonical declared initial engine state after generation. It therefore captures every realized distraction count, type, material, position, transformed object, and retained unknown XML field without depending on an incomplete log of random draws.

A deterministic materialization is expected to reproduce the same XML bytes and parameter realization for the same generator version, template, benchmark condition, seed, and declared inputs. `verify_replay()` currently validates the existing manifest and confirms that the replayed XML is parseable; it does not recompute or compare the manifest's declared identities from those bytes. Observed pre-intervention reset equality and runtime-state clearing belong to the rollout reset contract, not `scenario_manifest_v1`.

## Planning and admission

`LevelEntry` carries the complete validated manifest plus its optional sidecar reference. Partition and collection-plan artifacts serialize the manifest and flattened hierarchy/provenance identities; reload and command generation revalidate them rather than reconstructing identity from paths.

Collection planning has an explicit purpose:

- `research` (default) requires a validated scenario manifest and calls `require_research_eligible()` before partition selection, plan publication, plan reload, and command generation;
- `smoke` permits bounded operational planning of `smoke_only` or path-only artifacts and records that purpose in the collection plan. Path-only artifacts can never be admitted after changing that purpose to `research`.

The planner's `planner_seed` controls deterministic partition/order only. It is separate from each manifest's `generation_seed`, which defines scenario generation.

`tasks/task_template_designer/Assets/StreamingAssets/Levels/novelty_level_0/type2/Levels/3_9_6_1.scenario.json` is explicitly `legacy_static`, has unavailable template and seed evidence, and is `smoke_only`. Sidecar-less `type2` XML receives the same `smoke_only` admission policy during legacy import. Its path or split name cannot make it research eligible. It remains usable only through an explicitly smoke-purpose, explicitly scoped `type2` plan.
