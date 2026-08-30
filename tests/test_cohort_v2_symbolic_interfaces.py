from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from world_model.data.cohort_v2 import COHORT_V2_RELEASE_IDENTITY
from world_model.model import (
    MicroTransitionBatch,
    MicroTransitionInput,
    RelationTransitionValue,
)
from world_model.training.cohort_v2_symbolic_interfaces import (
    INTERFACE_ORDER,
    MATERIAL_RELATION_F1_GAIN,
    MatchedMicroInterfaceAdapter,
    SymbolicInterface,
    build_interface_predictor,
    interface_compute_macs,
    load_symbolic_interface_checkpoint,
    save_symbolic_interface_checkpoint,
    select_symbolic_interface,
)
from world_model.training.cohort_v2_micro import (
    MICRO_PAIRS,
    CohortV2MicroConfig,
    CohortV2StateCodec,
)


def _batch(*, contact=(), supports=()):
    return MicroTransitionBatch((MicroTransitionInput(
        frame_record_identity="frame:fixture",
        contact=RelationTransitionValue("available", contact),
        supports=RelationTransitionValue("available", supports),
    ),))


class CohortV2SymbolicInterfaceTests(unittest.TestCase):
    def test_all_interfaces_have_exactly_matched_capacity(self) -> None:
        counts = {
            interface: sum(
                parameter.numel()
                for parameter in MatchedMicroInterfaceAdapter(16, interface).parameters()
            )
            for interface in INTERFACE_ORDER
        }

        self.assertEqual(len(set(counts.values())), 1)

    def test_no_symbol_ignores_relations_while_symbolic_variants_preserve_direction(
        self,
    ) -> None:
        hidden = torch.zeros(1, 16)
        forward = _batch(supports=(("entity:a", "entity:b"),))
        reverse = _batch(supports=(("entity:b", "entity:a"),))

        for interface in INTERFACE_ORDER:
            torch.manual_seed(9)
            adapter = MatchedMicroInterfaceAdapter(16, interface)
            first = adapter(hidden, forward)
            second = adapter(hidden, reverse)
            with self.subTest(interface=interface):
                if interface is SymbolicInterface.NO_SYMBOL:
                    self.assertTrue(torch.equal(first, second))
                else:
                    self.assertFalse(torch.equal(first, second))

    def test_contact_is_symmetric_for_every_symbolic_interface(self) -> None:
        hidden = torch.zeros(1, 16)
        forward = _batch(contact=(("entity:a", "entity:b"),))
        reverse = _batch(contact=(("entity:b", "entity:a"),))

        for interface in INTERFACE_ORDER[1:]:
            torch.manual_seed(4)
            adapter = MatchedMicroInterfaceAdapter(16, interface)
            with self.subTest(interface=interface):
                self.assertTrue(torch.equal(
                    adapter(hidden, forward), adapter(hidden, reverse)
                ))

    def test_spsg_uses_every_parameter_without_negative_evidence(self) -> None:
        adapter = MatchedMicroInterfaceAdapter(16, SymbolicInterface.SPSG)
        hidden = torch.randn(1, 16, requires_grad=True)

        adapter(
            hidden,
            _batch(
                contact=(("entity:a", "entity:b"),),
                supports=(("entity:a", "entity:b"),),
            ),
        ).sum().backward()

        self.assertEqual(adapter.role_dim * adapter.filler_dim, 16)
        self.assertTrue(all(
            parameter.grad is not None for parameter in adapter.parameters()
        ))
        self.assertFalse(any(
            "negative" in name for name, _parameter in adapter.named_parameters()
        ))

    def test_decision_promotes_only_material_held_out_macro_f1_gains(self) -> None:
        scores = {
            SymbolicInterface.NO_SYMBOL: 0.50,
            SymbolicInterface.ORDERED_FLAT: 0.53,
            SymbolicInterface.DIRECTED_GNN: 0.54,
            SymbolicInterface.SPSG: 0.549,
        }

        selected, comparisons = select_symbolic_interface(scores)

        self.assertEqual(selected, SymbolicInterface.ORDERED_FLAT)
        self.assertEqual(MATERIAL_RELATION_F1_GAIN, 0.02)
        self.assertTrue(comparisons[0]["promoted"])
        self.assertFalse(comparisons[1]["promoted"])
        self.assertFalse(comparisons[2]["promoted"])

    def test_compute_accounting_charges_graph_and_tensor_product_work(self) -> None:
        values = {
            interface: interface_compute_macs(
                interface,
                hidden_dim=16,
                contact_count=1,
                support_count=1,
                entity_count=2,
            )
            for interface in INTERFACE_ORDER
        }

        self.assertGreater(
            values[SymbolicInterface.DIRECTED_GNN],
            values[SymbolicInterface.ORDERED_FLAT],
        )
        self.assertGreater(values[SymbolicInterface.SPSG], values[SymbolicInterface.DIRECTED_GNN])

    def test_interface_checkpoint_round_trips_its_variant_and_parameter_accounting(self) -> None:
        config = CohortV2MicroConfig(
            steps=6,
            batch_size=1,
            latent_dim=32,
            hidden_dim=16,
            depth=1,
            max_entities=2,
            device="cpu",
        )

        class Reader:
            release_identity = COHORT_V2_RELEASE_IDENTITY
            partition_identity = "partition:fixture"

        interface = SymbolicInterface.SPSG
        predictor = build_interface_predictor(config, interface)
        trainer = SimpleNamespace(
            data=SimpleNamespace(reader=Reader()),
            config=config,
            interface=interface,
            predictor=predictor,
            codec=CohortV2StateCodec(latent_dim=32, max_entities=2),
            step_count=6,
            pair_counts={pair: 1 for pair in MICRO_PAIRS},
        )
        with tempfile.TemporaryDirectory() as directory:
            saved = save_symbolic_interface_checkpoint(
                Path(directory) / "checkpoint.pt", trainer
            )
            loaded_predictor, _codec, loaded = load_symbolic_interface_checkpoint(
                saved.path,
                reader=Reader(),
                config=config,
                interface=interface,
                device="cpu",
            )

        self.assertEqual(loaded.identity, saved.identity)
        self.assertEqual(loaded.interface, SymbolicInterface.SPSG)
        self.assertEqual(
            loaded.trainable_parameter_count,
            sum(parameter.numel() for parameter in loaded_predictor.parameters()),
        )


if __name__ == "__main__":
    unittest.main()
