"""Cross-catalog source-provenance invariants."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from world_model.data.catalog_errors import DuplicateSourceKeyError

if TYPE_CHECKING:
    from world_model.data.catalog import EpisodeCatalog


def check_source_key_disjointness(catalogs: Sequence[EpisodeCatalog]) -> None:
    seen: dict[str, str] = {}
    for catalog in catalogs:
        if not catalog.provenance_available:
            continue
        for episode in catalog.episodes:
            if episode.source_level_key is None:
                continue
            if episode.source_level_key in seen:
                raise DuplicateSourceKeyError(
                    source_key=episode.source_level_key,
                    first_split=seen[episode.source_level_key],
                    second_split=catalog.split,
                )
            seen[episode.source_level_key] = catalog.split
