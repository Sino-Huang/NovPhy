"""Deterministic level-instance materialization for research workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import random
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

from scripts.scenario_manifest import (
    BenchmarkCondition,
    ELIGIBLE,
    ScenarioManifest,
    canonical_xml_projection,
    create_generated_manifest,
    write_manifest,
)
from tasks.task_generator.utils.data_classes import Block, Pig, Tnt
from tasks.task_generator.utils.generate_variations import GenerateLevels


GENERATOR_IDENTITY = "novphy-task-generator"
GENERATOR_VERSION = "canonical_materialization_v1"
_NOVELTY_TYPES = {
    "PinkCircle",
    "Fan",
    "InverseAirTurbulence",
    "NonNovelAirTurbulence",
    "NovelAirTurbulence",
    "PinkRectFat",
    "PinkSquareHole",
    "Storm",
    "InverseGravity",
    "Magnet",
}


@dataclass(frozen=True, slots=True)
class CanonicalMaterializationRequest:
    template_path: Path
    output_xml_path: Path
    output_manifest_path: Path
    template_name: str
    benchmark_condition: BenchmarkCondition
    template_identity: str
    generation_seed: int
    reference_point: tuple[float, float]
    min_coordinate: tuple[float, float]
    max_coordinate: tuple[float, float]
    restricted_objects: tuple[str, ...]
    eligibility: str = ELIGIBLE
    eligibility_reason: str | None = None
    template_source_reference: str | None = None


@dataclass(frozen=True, slots=True)
class MaterializedLevelInstance:
    xml_content: bytes
    parameter_realization: dict[str, Any]
    manifest: ScenarioManifest


def _number(value: float) -> str:
    return str(round(value, 4))


def _read_template(xml_content: bytes) -> tuple[ET.Element, list[Block], list[Pig], list[Tnt], list[ET.Element], list[ET.Element], list[ET.Element]]:
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        normalized = xml_content.replace(b'encoding="utf-16"', b'encoding="utf-8"', 1)
        normalized = normalized.replace(b"encoding='utf-16'", b"encoding='utf-8'", 1)
        if normalized == xml_content:
            raise ValueError(f"Malformed scenario template XML: {exc}") from exc
        try:
            root = ET.fromstring(normalized)
        except ET.ParseError:
            raise ValueError(f"Malformed scenario template XML: {exc}") from exc
    game_objects = root.find("GameObjects")
    if game_objects is None:
        raise ValueError("Scenario template must contain GameObjects")

    blocks: list[Block] = []
    pigs: list[Pig] = []
    tnts: list[Tnt] = []
    block_nodes: list[ET.Element] = []
    pig_nodes: list[ET.Element] = []
    tnt_nodes: list[ET.Element] = []
    identifier = 0
    for node in game_objects:
        if node.tag == "Pig":
            identifier += 1
            pigs.append(Pig(identifier, node.attrib["type"], float(node.attrib["x"]), float(node.attrib["y"]), float(node.attrib.get("rotation", 0))))
            pig_nodes.append(node)
        elif node.tag == "TNT":
            identifier += 1
            tnts.append(Tnt(identifier, float(node.attrib["x"]), float(node.attrib["y"]), float(node.attrib.get("rotation", 0))))
            tnt_nodes.append(node)
        elif node.tag in {"Block", "Platform", "Novelty", "ExternalAgent"}:
            identifier += 1
            blocks.append(Block(
                identifier,
                node.attrib["type"],
                node.attrib.get("material", ""),
                float(node.attrib["x"]),
                float(node.attrib["y"]),
                float(node.attrib.get("rotation", 0)),
                float(node.attrib.get("scaleX", 1)),
                float(node.attrib.get("scaleY", 1)),
            ))
            block_nodes.append(node)
    return root, blocks, pigs, tnts, block_nodes, pig_nodes, tnt_nodes


def _update_node(node: ET.Element, game_object: Block | Pig | Tnt) -> None:
    node.set("x", _number(game_object.x))
    node.set("y", _number(game_object.y))
    node.set("rotation", _number(game_object.rotation))
    if isinstance(game_object, Block) and node.tag == "Platform":
        node.set("scaleX", _number(game_object.scale_x))
        node.set("scaleY", _number(game_object.scale_y))


def _append_generated_block(game_objects: ET.Element, block: Block) -> None:
    if block.type == "Platform":
        tag = "Platform"
    elif block.type in _NOVELTY_TYPES:
        tag = "Novelty"
    else:
        tag = "Block"
    attributes = {
        "type": block.type,
        "material": block.material,
        "x": _number(block.x),
        "y": _number(block.y),
        "rotation": _number(block.rotation),
    }
    if tag in {"Platform", "Novelty"}:
        attributes["scaleX"] = _number(block.scale_x)
        attributes["scaleY"] = _number(block.scale_y)
    ET.SubElement(game_objects, tag, attributes)


def _author_scenario_object_ids(root: ET.Element) -> None:
    slingshot = root.find("Slingshot")
    if slingshot is None:
        raise ValueError("Scenario template must contain Slingshot")
    slingshot.set("scenarioObjectId", "slingshot:0000")

    birds = root.find("Birds")
    if birds is None:
        raise ValueError("Scenario template must contain Birds")
    for index, bird in enumerate(birds):
        bird.set("scenarioObjectId", f"bird:{index:04d}")

    game_objects = root.find("GameObjects")
    assert game_objects is not None
    ordinals: dict[str, int] = {}
    for node in game_objects:
        kind = {
            "ExternalAgent": "external-agent",
            "Novelty": "novelty",
        }.get(node.tag, node.tag.lower())
        ordinal = ordinals.get(kind, 0)
        node.set("scenarioObjectId", f"{kind}:{ordinal:04d}")
        ordinals[kind] = ordinal + 1


def _serialize_generated_tree(
    root: ET.Element,
    generated_blocks: list[Block],
    generated_pigs: list[Pig],
    generated_tnts: list[Tnt],
    block_nodes: list[ET.Element],
    pig_nodes: list[ET.Element],
    tnt_nodes: list[ET.Element],
) -> bytes:
    for node, game_object in zip(block_nodes, generated_blocks):
        _update_node(node, game_object)
    for node, game_object in zip(pig_nodes, generated_pigs):
        _update_node(node, game_object)
    for node, game_object in zip(tnt_nodes, generated_tnts):
        _update_node(node, game_object)

    game_objects = root.find("GameObjects")
    assert game_objects is not None
    for block in generated_blocks[len(block_nodes):]:
        _append_generated_block(game_objects, block)

    _author_scenario_object_ids(root)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True) + b"\n"


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def materialize_level_instance(
    request: CanonicalMaterializationRequest,
    *,
    publish: bool = True,
) -> MaterializedLevelInstance:
    """Materialize exactly one level instance from explicit, path-stable inputs."""
    if isinstance(request.generation_seed, bool) or not isinstance(request.generation_seed, int):
        raise ValueError("generation_seed must be an integer")

    template_content = request.template_path.read_bytes()
    root, blocks, pigs, tnts, block_nodes, pig_nodes, tnt_nodes = _read_template(template_content)
    generator = GenerateLevels(random.Random(request.generation_seed))
    generated = generator.generate_levels_from_template(
        request.template_name,
        [blocks, pigs, tnts],
        [
            request.reference_point,
            request.min_coordinate,
            request.max_coordinate,
            list(request.restricted_objects),
        ],
        variant_count=1,
    )[0]
    xml_content = _serialize_generated_tree(
        root,
        generated[0],
        generated[1],
        generated[2],
        block_nodes,
        pig_nodes,
        tnt_nodes,
    )
    parameter_realization = {
        "schema": "scenario_parameter_realization_v1",
        "declared_initial_engine_state": canonical_xml_projection(xml_content),
    }
    template_source_reference = (
        request.template_source_reference or request.template_path.as_posix()
    )
    declared_inputs = {
        "template_content_identity": (
            f"xml_bytes_v1:{template_source_reference}:{GENERATOR_VERSION}"
        ),
        "template_name": request.template_name,
        "reference_point": list(request.reference_point),
        "min_coordinate": list(request.min_coordinate),
        "max_coordinate": list(request.max_coordinate),
        "restricted_objects": list(request.restricted_objects),
    }
    manifest = create_generated_manifest(
        xml_content,
        benchmark_condition=request.benchmark_condition,
        template_identity=request.template_identity,
        generator_identity=GENERATOR_IDENTITY,
        generator_version=GENERATOR_VERSION,
        generation_seed=request.generation_seed,
        declared_inputs=declared_inputs,
        parameter_realization=parameter_realization,
        eligibility=request.eligibility,
        eligibility_reason=request.eligibility_reason,
    )
    if publish:
        _atomic_write_bytes(request.output_xml_path, xml_content)
        write_manifest(manifest, request.output_manifest_path)
    return MaterializedLevelInstance(xml_content, parameter_realization, manifest)
