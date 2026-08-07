"""Typed catalog boundary failures."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequiredCapabilityError(ValueError):
    capability: str
    contract_name: str

    def __str__(self) -> str:
        return (
            f"required capability {self.capability!r} is not declared in "
            f"capture contract {self.contract_name!r}"
        )


@dataclass(frozen=True, slots=True)
class DuplicateSourceKeyError(ValueError):
    source_key: str
    first_split: str
    second_split: str

    def __str__(self) -> str:
        return (
            f"source key {self.source_key!r} appears in both split "
            f"{self.first_split!r} and {self.second_split!r}"
        )
