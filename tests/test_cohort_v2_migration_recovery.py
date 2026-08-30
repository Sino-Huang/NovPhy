from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.cohort_v2_migration_recovery import (
    MANIFEST_NAME,
    audit_surviving_public_release,
    build_migration_recovery_manifest,
    validate_migration_recovery_manifest,
)
from scripts.cohort_v2_production_plans_v5 import validate_plan_v5_evidence
from scripts.cohort_v2_scenarios import write_immutable_cohort_v2_json
from world_model.data import CohortV2IngestionError, CohortV2ReleaseReader
from world_model.training.grid_artifacts import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/runtime_evidence/issue-53-plan-v5"
PUBLIC = ROOT / "data/runtime_evidence/issue-53-mixed-termination-v5"
CAPABILITIES = ROOT / "docs/data_contracts/cohort_v2_capabilities_v1.json"


def _replace_json(path: Path, value: dict[str, object]) -> None:
    path.unlink()
    path.write_bytes(canonical_json_bytes(value))


class CohortV2MigrationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="cohort-v2-migration-recovery-tests-"
        )
        cls.authority = Path(cls.temporary.name) / "authority"
        manifest = build_migration_recovery_manifest(
            repository_root=ROOT,
            plan_root=PLAN,
            release_root=PUBLIC,
        )
        write_immutable_cohort_v2_json(
            manifest, cls.authority / MANIFEST_NAME
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _release_copy(self, root: Path) -> Path:
        release = root / "release"
        shutil.copytree(PUBLIC, release, copy_function=os.link)
        return release

    def test_explicit_recovery_authority_restores_public_reader(self) -> None:
        reader = CohortV2ReleaseReader(
            PUBLIC,
            capability_declaration_path=CAPABILITIES,
            production_plan_root=PLAN,
            workflow_kind="training",
            influence="learned_parameters",
            migration_recovery_authority=self.authority,
        )
        self.assertEqual(len(reader.rollouts), 6)
        self.assertEqual(
            sum(len(item.frame_records) for item in reader.rollouts), 2_156
        )
        with self.assertRaisesRegex(CohortV2IngestionError, "canonical observation"):
            reader.load_observation(reader.rollouts[0], observation_role="canonical")

    def test_normal_validation_remains_strict_without_recovery_authority(self) -> None:
        with self.assertRaises((FileNotFoundError, CohortV2IngestionError)):
            CohortV2ReleaseReader(
                PUBLIC,
                capability_declaration_path=CAPABILITIES,
                production_plan_root=PLAN,
                workflow_kind="training",
                influence="learned_parameters",
            )

    def test_recovery_plan_rejects_altered_v5_payload(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-artifacts") as temporary:
            plan = Path(temporary) / "plan"
            shutil.copytree(PLAN, plan, copy_function=os.link)
            collection_path = plan / "collection-plan.json"
            collection = json.loads(collection_path.read_bytes())
            collection["identity"] = "foreign-plan-v5-identity"
            _replace_json(collection_path, collection)
            with self.assertRaisesRegex(ValueError, "Plan-v5"):
                validate_plan_v5_evidence(
                    plan,
                    repository_root=ROOT,
                    migration_recovery=True,
                )

    def test_recovery_audit_rejects_missing_rollout_member(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-artifacts") as temporary:
            release = self._release_copy(Path(temporary))
            target = next((release / "primary-rollouts").rglob("physics_capture_v2.json"))
            target.unlink()
            with self.assertRaisesRegex(ValueError, "membership"):
                audit_surviving_public_release(release, plan_root=PLAN)

    def test_recovery_audit_rejects_foreign_release_identity(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-artifacts") as temporary:
            release = self._release_copy(Path(temporary))
            release_path = release / "cohort-v2-release.json"
            value = json.loads(release_path.read_bytes())
            value["identity"] = "representative-cohort-v2-release-v5:issue-99:mixed-termination"
            _replace_json(release_path, value)
            with self.assertRaisesRegex(ValueError, "release identities"):
                audit_surviving_public_release(release, plan_root=PLAN)

    def test_recovery_audit_rejects_broken_derivation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-artifacts") as temporary:
            release = self._release_copy(Path(temporary))
            index = json.loads(
                (release / "authoritative-derivation-index.json").read_bytes()
            )
            relative = index["artifacts"][0]["path"]
            derivation_path = release / relative
            value = json.loads(derivation_path.read_bytes())
            value["identity"] = "broken-derivation-identity"
            _replace_json(derivation_path, value)
            with self.assertRaisesRegex(ValueError, "derivation"):
                audit_surviving_public_release(release, plan_root=PLAN)

    def test_recovery_audit_rejects_cross_role_leakage(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / ".local-artifacts") as temporary:
            release = self._release_copy(Path(temporary))
            ledger_path = release / "production-attempt-accounting.json"
            ledger = json.loads(ledger_path.read_bytes())
            ledger["attempt_ledger"][0]["exposure_role"] = "calibration"
            _replace_json(ledger_path, ledger)
            with self.assertRaisesRegex(ValueError, "role binding"):
                audit_surviving_public_release(release, plan_root=PLAN)

    def test_unavailable_sealed_data_cannot_be_declared_available(self) -> None:
        stored = json.loads((self.authority / MANIFEST_NAME).read_bytes())
        changed = copy.deepcopy(stored)
        changed["unavailable_sources"][-1]["status"] = "available"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / MANIFEST_NAME
            path.write_bytes(canonical_json_bytes(changed))
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_migration_recovery_manifest(
                    path,
                    repository_root=ROOT,
                    plan_root=PLAN,
                    release_root=PUBLIC,
                )


if __name__ == "__main__":
    unittest.main()
