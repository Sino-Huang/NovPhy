import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collection_plan import (
    PLAN_COPY_FILENAME,
    REPORT_FILENAME,
    RuntimeResult,
    assert_plan_unchanged,
    create_collection_plan,
    execute_collection_plan,
    load_collection_plan,
    write_collection_plan,
)
from scripts.scenario_manifest import BenchmarkCondition, create_generated_manifest, scenario_manifest_projection


XML = b'''<?xml version="1.0" encoding="utf-8"?>
<Level width="2">
  <Camera maxWidth="30" minWidth="20" />
  <Birds><Bird type="BirdRed" /></Birds>
  <Slingshot x="-8" y="-2" />
  <GameObjects><Pig type="BasicSmall" x="1" y="-3" rotation="0" /></GameObjects>
</Level>
'''


def fixture_projection() -> dict[str, object]:
    manifest = create_generated_manifest(
        XML,
        benchmark_condition=BenchmarkCondition("novelty_level_1", "type0101"),
        template_identity="scenario-template-v1:fixture",
        generator_identity="novphy-task-generator",
        generator_version="canonical-v1",
        generation_seed=41,
        declared_inputs={"layout_choice": 0},
        parameter_realization={"shift_x": 0.25},
    )
    return scenario_manifest_projection(manifest, "fixtures/fixture.scenario.json")


def coverage_strata() -> dict[str, dict[str, object]]:
    return {
        "no-contact/miss": {"status": "targeted", "intervention_ids": ["miss-shot"]},
        "collision": {"status": "targeted", "intervention_ids": ["collision-shot"]},
        "persistent support": {"status": "inapplicable", "rationale": "support evidence capability unavailable"},
        "support change": {"status": "inapplicable", "rationale": "support evidence capability unavailable"},
        "destruction": {"status": "inapplicable", "rationale": "fixture has no destructible target"},
        "pig removal": {"status": "inapplicable", "rationale": "pig-removal event capability unavailable"},
        "explosion": {"status": "inapplicable", "rationale": "fixture has no explosive target"},
        "stability transitions": {"status": "inapplicable", "rationale": "stability event capability unavailable"},
        "level clear": {"status": "inapplicable", "rationale": "terminal completion capability unavailable"},
        "level fail": {"status": "inapplicable", "rationale": "terminal failure capability unavailable"},
    }


def plan_arguments() -> dict[str, object]:
    projection = fixture_projection()
    scenario = {
        "scenario_id": "scenario-a",
        "exposure_role": "training",
        **projection,
        "expected_initial_engine_state_identity": "runtime-initial-state-v1:fixture",
        "retry_policy": {
            "max_attempts": 2,
            "transient_failure_codes": ["transport_unavailable"],
            "stopping_rule": "execute_all_interventions",
        },
        "negative_specification": {
            "cap": 1,
            "intervention_ids": ["miss-shot"],
            "semantic_justification": "prospective no-contact trajectory outside authored geometry",
        },
        "interventions": [
            {
                "id": "collision-shot",
                "ordinal": 1,
                "intended_coverage_stratum": "collision",
                "source": "geometry_stratified",
                "interface_action": {
                    "action_type": "drag_hold_release",
                    "coordinate_frame": "slingshot_relative",
                    "drag_start": [100, 200],
                    "drag_release": [30, 50],
                    "tapTime": 70,
                    "releaseTime": 600,
                    "frame_height": 480,
                    "socket_command": {"x": 130, "y": 329, "tapTime": 70, "releaseTime": 600},
                },
                "engine_relative_action": {
                    "coordinate_frame": "slingshot_relative",
                    "release_offset": [30, 50],
                    "release_point": [130, 150],
                    "tap_time_ms": 70,
                    "release_time_ms": 600,
                },
                "mapping_version": "science-birds-slingshot-relative-v1",
                "slingshot_reference": {"gameX": 100, "gameY": 200},
                "source_provenance": {
                    "scenario_geometry_identity": "geometry-v1:fixture",
                    "stratum": "upper-target",
                    "feasibility_rule": "unobstructed-pull-v1",
                },
            },
            {
                "id": "miss-shot",
                "ordinal": 2,
                "intended_coverage_stratum": "no-contact/miss",
                "source": "targeted_rare",
                "interface_action": {
                    "action_type": "drag_hold_release",
                    "coordinate_frame": "slingshot_relative",
                    "drag_start": [100, 200],
                    "drag_release": [-20, 40],
                    "tapTime": 0,
                    "releaseTime": 600,
                    "frame_height": 480,
                    "socket_command": {"x": 80, "y": 319, "tapTime": 0, "releaseTime": 600},
                },
                "engine_relative_action": {
                    "coordinate_frame": "slingshot_relative",
                    "release_offset": [-20, 40],
                    "release_point": [80, 160],
                    "tap_time_ms": 0,
                    "release_time_ms": 600,
                },
                "mapping_version": "science-birds-slingshot-relative-v1",
                "slingshot_reference": {"gameX": 100, "gameY": 200},
                "source_provenance": {
                    "target_stratum": "no-contact/miss",
                    "selection_rule": "clearance-margin-v1",
                },
            },
        ],
        "source_dispositions": {
            "geometry_stratified": {"status": "included"},
            "targeted_rare": {"status": "included"},
            "benchmark_agent_replay": {
                "status": "unavailable",
                "rationale": "no benchmark agent trace exists",
            },
        },
        "coverage_strata": coverage_strata(),
    }
    return {"plan_version": 1, "scenarios": [scenario]}


def create_loaded_plan(root: Path, arguments: dict[str, object] | None = None):
    path = root / "source-plan.json"
    plan = create_collection_plan(**(arguments or plan_arguments()))
    write_collection_plan(plan, path)
    return load_collection_plan(path)


class CollectionPlanTests(unittest.TestCase):
    def test_round_trip_has_deterministic_identity_and_exact_frozen_bytes(self) -> None:
        arguments = plan_arguments()
        first = create_collection_plan(**arguments)
        second = create_collection_plan(**arguments)
        self.assertEqual(first, second)
        self.assertEqual(first.identity, second.identity)
        self.assertNotEqual(
            first.scenarios[0].expected_initial_engine_state_identity,
            first.scenarios[0].scenario_manifest_projection["declared_initial_engine_state_identity"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            write_collection_plan(first, path)
            loaded = load_collection_plan(path)
            self.assertEqual(loaded.plan, first)
            self.assertEqual(loaded.original_bytes, path.read_bytes())

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["plan_version"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity is stale"):
                load_collection_plan(path)
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                assert_plan_unchanged(loaded, path)

    def test_coverage_strata_are_complete_and_prospective(self) -> None:
        cases = []
        missing = plan_arguments()
        del missing["scenarios"][0]["coverage_strata"]["explosion"]
        cases.append((missing, "incomplete"))
        unknown = plan_arguments()
        del unknown["scenarios"][0]["coverage_strata"]["explosion"]
        unknown["scenarios"][0]["coverage_strata"]["unknown"] = {
            "status": "inapplicable",
            "rationale": "unknown",
        }
        cases.append((unknown, "incomplete"))
        no_rationale = plan_arguments()
        no_rationale["scenarios"][0]["coverage_strata"]["explosion"]["rationale"] = ""
        cases.append((no_rationale, "nonempty"))
        wrong_target = plan_arguments()
        wrong_target["scenarios"][0]["coverage_strata"]["collision"]["intervention_ids"] = ["miss-shot"]
        cases.append((wrong_target, "exactly its planned interventions"))

        for arguments, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                create_collection_plan(**arguments)

    def test_hybrid_sources_require_geometry_rare_and_available_replay(self) -> None:
        missing_geometry = plan_arguments()
        missing_geometry["scenarios"][0]["interventions"][0]["source"] = "targeted_rare"
        missing_geometry["scenarios"][0]["interventions"][0]["source_provenance"] = {
            "target_stratum": "collision",
            "selection_rule": "rare-collision-v1",
        }
        with self.assertRaisesRegex(ValueError, "geometry_stratified must be included"):
            create_collection_plan(**missing_geometry)

        replay_without_action = plan_arguments()
        replay_without_action["scenarios"][0]["source_dispositions"]["benchmark_agent_replay"] = {
            "status": "included"
        }
        with self.assertRaisesRegex(ValueError, "requires an intervention"):
            create_collection_plan(**replay_without_action)

        replay_without_reason = plan_arguments()
        replay_without_reason["scenarios"][0]["source_dispositions"]["benchmark_agent_replay"]["rationale"] = ""
        with self.assertRaisesRegex(ValueError, "nonempty"):
            create_collection_plan(**replay_without_reason)

        replay = plan_arguments()
        scenario = replay["scenarios"][0]
        replay_action = copy.deepcopy(scenario["interventions"][1])
        replay_action.update(
            id="replay-clear-shot",
            ordinal=3,
            intended_coverage_stratum="level clear",
            source="benchmark_agent_replay",
            source_provenance={
                "agent_identity": "benchmark-agent-v1",
                "trace_identity": "trace-7",
                "action_index": 4,
            },
        )
        scenario["interventions"].append(replay_action)
        scenario["source_dispositions"]["benchmark_agent_replay"] = {"status": "included"}
        scenario["coverage_strata"]["level clear"] = {
            "status": "targeted",
            "intervention_ids": ["replay-clear-shot"],
        }
        self.assertEqual(len(create_collection_plan(**replay).scenarios[0].interventions), 3)
        replay["scenarios"][0]["interventions"][2]["source_provenance"] = {}
        with self.assertRaisesRegex(ValueError, "nonempty object"):
            create_collection_plan(**replay)

    def test_scenario_projection_and_intervention_order_fail_closed(self) -> None:
        stale_projection = plan_arguments()
        stale_projection["scenarios"][0]["scenario_lineage_identity"] = "scenario-lineage-v1:stale"
        duplicate = plan_arguments()
        duplicate["scenarios"][0]["interventions"][1]["id"] = "collision-shot"
        noncontiguous = plan_arguments()
        noncontiguous["scenarios"][0]["interventions"][1]["ordinal"] = 3
        cases = (
            (stale_projection, "stale or missing scenario_lineage_identity"),
            (duplicate, "IDs must be unique"),
            (noncontiguous, "contiguous and ordered"),
        )
        for arguments, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                create_collection_plan(**arguments)

    def test_action_retry_negative_and_exposure_validation_fail_closed(self) -> None:
        cases = []
        for field in ("interface_action", "engine_relative_action", "slingshot_reference"):
            arguments = plan_arguments()
            arguments["scenarios"][0]["interventions"][0][field] = {}
            cases.append((arguments, "nonempty object"))
        no_mapping = plan_arguments()
        no_mapping["scenarios"][0]["interventions"][0]["mapping_version"] = ""
        cases.append((no_mapping, "nonempty string"))
        no_attempts = plan_arguments()
        no_attempts["scenarios"][0]["retry_policy"]["max_attempts"] = 0
        cases.append((no_attempts, "positive integer"))
        duplicate_codes = plan_arguments()
        duplicate_codes["scenarios"][0]["retry_policy"]["transient_failure_codes"] = [
            "transport_unavailable",
            "transport_unavailable",
        ]
        cases.append((duplicate_codes, "unique nonempty"))
        semantic_retry = plan_arguments()
        semantic_retry["scenarios"][0]["retry_policy"]["transient_failure_codes"] = ["schema_defect"]
        cases.append((semantic_retry, "non-transient"))
        adaptive_stopping = plan_arguments()
        adaptive_stopping["scenarios"][0]["retry_policy"]["stopping_rule"] = "fill_quota"
        cases.append((adaptive_stopping, "execute_all_interventions"))
        over_cap = plan_arguments()
        over_cap["scenarios"][0]["negative_specification"]["cap"] = 0
        cases.append((over_cap, "exceed cap"))
        omitted_negative = plan_arguments()
        omitted_negative["scenarios"][0]["negative_specification"]["intervention_ids"] = []
        cases.append((omitted_negative, "exactly the no-contact/miss"))
        mismatched_action = plan_arguments()
        mismatched_action["scenarios"][0]["interventions"][0]["engine_relative_action"]["release_point"] = [0, 0]
        cases.append((mismatched_action, "does not match interface_action"))
        mismatched_socket = plan_arguments()
        mismatched_socket["scenarios"][0]["interventions"][0]["interface_action"]["socket_command"]["y"] = 0
        cases.append((mismatched_socket, "socket_command does not match"))
        mismatched_rare = plan_arguments()
        mismatched_rare["scenarios"][0]["interventions"][1]["source_provenance"]["target_stratum"] = "collision"
        cases.append((mismatched_rare, "target_stratum must match"))
        bad_role = plan_arguments()
        bad_role["scenarios"][0]["exposure_role"] = "research"
        cases.append((bad_role, "exposure_role is unknown"))

        for arguments, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                create_collection_plan(**arguments)

    def test_execution_is_ordered_and_realized_outcomes_do_not_fill_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded = create_loaded_plan(root)
            output = root / "output"
            requests = []

            def runtime(request):
                self.assertEqual((output / PLAN_COPY_FILENAME).read_bytes(), loaded.original_bytes)
                requests.append(request)
                realized = ("no-contact/miss",) if request.intervention_id == "collision-shot" else ("collision",)
                return RuntimeResult("accepted", realized_coverage_strata=realized)

            report = execute_collection_plan(loaded, runtime, output)

            self.assertEqual([request.intervention_id for request in requests], ["collision-shot", "miss-shot"])
            self.assertEqual(len(report["attempt_ledger"]), 2)
            self.assertEqual(report["accepted_count"], 2)
            self.assertEqual(report["realized_coverage_stratum_counts"]["collision"], 1)
            self.assertEqual(report["realized_coverage_stratum_counts"]["no-contact/miss"], 1)
            self.assertEqual(report["unmet_slots"], [])
            self.assertEqual(len(report["realized_coverage_shortfalls"]), 2)
            self.assertEqual((output / PLAN_COPY_FILENAME).read_bytes(), loaded.original_bytes)
            self.assertEqual(json.loads((output / REPORT_FILENAME).read_text(encoding="utf-8")), report)

    def test_transient_retry_preserves_action_and_uses_distinct_attempt_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded = create_loaded_plan(root)
            requests = []

            def runtime(request):
                requests.append(request)
                if request.intervention_id == "collision-shot" and request.attempt_number == 1:
                    return RuntimeResult("failed", reason="transport unavailable", failure_code="transport_unavailable")
                stratum = "collision" if request.intervention_id == "collision-shot" else "no-contact/miss"
                return RuntimeResult("accepted", realized_coverage_strata=(stratum,))

            report = execute_collection_plan(loaded, runtime, root / "output")
            first, retry = requests[:2]
            self.assertEqual(
                [item.intervention_id for item in requests],
                ["collision-shot", "collision-shot", "miss-shot"],
            )
            self.assertNotEqual(first.attempt_id, retry.attempt_id)
            self.assertEqual(first.interface_action, retry.interface_action)
            self.assertEqual(first.engine_relative_action, retry.engine_relative_action)
            self.assertEqual(report["failed_count"], 1)
            self.assertEqual(report["accepted_count"], 2)

    def test_attempt_ledger_records_machine_readable_disposition_and_reason(self) -> None:
        cases = {
            "accepted_and_rejected": (
                lambda request: (
                    RuntimeResult("accepted", realized_coverage_strata=("collision",))
                    if request.intervention_id == "collision-shot"
                    else RuntimeResult("rejected", reason="artifact ineligible", eligible=False)
                ),
                [("accepted", "accept", "accepted"), ("rejected", "quarantine", "rejected")],
            ),
            "transient_retry": (
                lambda request: (
                    RuntimeResult("failed", reason="transport unavailable", failure_code="transport_unavailable")
                    if request.intervention_id == "collision-shot" and request.attempt_number == 1
                    else RuntimeResult(
                        "accepted",
                        realized_coverage_strata=("collision" if request.intervention_id == "collision-shot" else "no-contact/miss",),
                    )
                ),
                [
                    ("failed", "retry", "transient_failure"),
                    ("accepted", "accept", "accepted"),
                    ("accepted", "accept", "accepted"),
                ],
            ),
            "permanent_failure": (
                lambda request: (
                    RuntimeResult("failed", reason="schema defect", failure_code="permanent_schema_defect")
                    if request.intervention_id == "collision-shot"
                    else RuntimeResult("accepted", realized_coverage_strata=("no-contact/miss",))
                ),
                [
                    ("failed", "quarantine", "permanent_failure"),
                    ("accepted", "accept", "accepted"),
                ],
            ),
            "exhausted_transient_failure": (
                lambda request: (
                    RuntimeResult("failed", reason="transport unavailable", failure_code="transport_unavailable")
                    if request.intervention_id == "collision-shot"
                    else RuntimeResult("accepted", realized_coverage_strata=("no-contact/miss",))
                ),
                [
                    ("failed", "retry", "transient_failure"),
                    ("failed", "quarantine", "retry_exhausted"),
                    ("accepted", "accept", "accepted"),
                ],
            ),
        }
        for name, (runtime, expected) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                report = execute_collection_plan(create_loaded_plan(root), runtime, root / "output")
                persisted = json.loads((root / "output" / REPORT_FILENAME).read_text(encoding="utf-8"))
                actual = [
                    (entry["status"], entry["disposition"], entry["disposition_reason"])
                    for entry in report["attempt_ledger"]
                ]
                self.assertEqual(actual, expected)
                self.assertEqual(persisted["attempt_ledger"], report["attempt_ledger"])

    def test_permanent_and_exhausted_failures_never_replace_planned_actions(self) -> None:
        cases = {
            "permanent": ("permanent_schema_defect", 1),
            "exhausted": ("transport_unavailable", 2),
        }
        for name, (failure_code, expected_collision_attempts) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                loaded = create_loaded_plan(root)
                intervention_ids = []

                def runtime(request):
                    intervention_ids.append(request.intervention_id)
                    if request.intervention_id == "collision-shot":
                        return RuntimeResult("failed", reason=name, failure_code=failure_code)
                    return RuntimeResult("rejected", reason="artifact ineligible", eligible=False)

                report = execute_collection_plan(loaded, runtime, root / "output")
                self.assertEqual(intervention_ids.count("collision-shot"), expected_collision_attempts)
                self.assertEqual(intervention_ids[-1], "miss-shot")
                self.assertEqual(len(report["planned_slots"]), 2)
                self.assertEqual(report["rejected_count"], 1)
                self.assertEqual(len(report["attempt_ledger"]), expected_collision_attempts + 1)

    def test_runtime_inputs_are_immutable_and_use_frozen_initial_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded = create_loaded_plan(root)
            expected = loaded.plan.scenarios[0].expected_initial_engine_state_identity

            def runtime(request):
                self.assertEqual(request.expected_initial_engine_state_identity, expected)
                with self.assertRaises(TypeError):
                    request.interface_action["tapTime"] = 9
                with self.assertRaises(TypeError):
                    request.interface_action["drag_release"][0] = 9
                stratum = "collision" if request.intervention_id == "collision-shot" else "no-contact/miss"
                return RuntimeResult("accepted", realized_coverage_strata=(stratum,))

            execute_collection_plan(loaded, runtime, root / "output")

    def test_accounting_uses_all_and_only_eligible_accepted_realized_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded = create_loaded_plan(root)

            def runtime(request):
                if request.intervention_id == "collision-shot":
                    return RuntimeResult("accepted", realized_coverage_strata=("collision", "level clear"))
                return RuntimeResult("accepted", realized_coverage_strata=("no-contact/miss",), eligible=False)

            report = execute_collection_plan(loaded, runtime, root / "output")
            self.assertEqual(report["accepted_count"], 2)
            self.assertEqual(report["realized_coverage_stratum_counts"]["collision"], 1)
            self.assertEqual(report["realized_coverage_stratum_counts"]["level clear"], 1)
            self.assertEqual(report["realized_coverage_stratum_counts"]["no-contact/miss"], 0)
            self.assertFalse(report["attempt_ledger"][1]["eligible"])
            self.assertEqual(len(report["unmet_slots"]), 1)

    def test_execution_detects_source_or_copied_plan_mutation(self) -> None:
        for target in ("source", "copy"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                loaded = create_loaded_plan(root)
                output = root / "output"

                def runtime(request):
                    mutation_path = loaded.path if target == "source" else output / PLAN_COPY_FILENAME
                    mutation_path.write_bytes(b"mutated")
                    stratum = "collision" if request.intervention_id == "collision-shot" else "no-contact/miss"
                    return RuntimeResult("accepted", realized_coverage_strata=(stratum,))

                message = "bytes changed" if target == "source" else "Copied collection plan bytes changed"
                with self.assertRaisesRegex(ValueError, message):
                    execute_collection_plan(loaded, runtime, output)
                checkpoint = json.loads((output / REPORT_FILENAME).read_text(encoding="utf-8"))
                self.assertEqual(len(checkpoint["attempt_ledger"]), 2)

    def test_execution_rejects_existing_output_state_without_erasing_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded = create_loaded_plan(root)
            output = root / "output"

            def runtime(request):
                stratum = "collision" if request.intervention_id == "collision-shot" else "no-contact/miss"
                return RuntimeResult("accepted", realized_coverage_strata=(stratum,))

            first_report = execute_collection_plan(loaded, runtime, output)
            report_bytes = (output / REPORT_FILENAME).read_bytes()
            with self.assertRaisesRegex(ValueError, "already contains execution state"):
                execute_collection_plan(loaded, runtime, output)
            self.assertEqual((output / REPORT_FILENAME).read_bytes(), report_bytes)
            self.assertEqual(len(first_report["attempt_ledger"]), 2)

            other_arguments = plan_arguments()
            other_arguments["plan_version"] = 2
            other_loaded = create_loaded_plan(root / "other", other_arguments)
            with self.assertRaisesRegex(ValueError, "different frozen plan"):
                execute_collection_plan(other_loaded, runtime, output)
            self.assertEqual((output / REPORT_FILENAME).read_bytes(), report_bytes)

    def test_attempt_identities_bind_plan_scenario_intervention_and_attempt(self) -> None:
        arguments = plan_arguments()
        second_scenario = copy.deepcopy(arguments["scenarios"][0])
        second_scenario["scenario_id"] = "scenario-b"
        second_scenario["exposure_role"] = "calibration"
        arguments["scenarios"].append(second_scenario)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loaded = create_loaded_plan(root, arguments)

            def runtime(request):
                stratum = "collision" if request.intervention_id == "collision-shot" else "no-contact/miss"
                return RuntimeResult("accepted", realized_coverage_strata=(stratum,))

            report = execute_collection_plan(loaded, runtime, root / "output")
            attempt_ids = [entry["attempt_id"] for entry in report["attempt_ledger"]]
            self.assertEqual(len(attempt_ids), 4)
            self.assertEqual(len(set(attempt_ids)), 4)


if __name__ == "__main__":
    unittest.main()
