from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_name: str
    page: int
    section: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Chunk":
        return cls(**value)


@dataclass(slots=True)
class Evidence:
    chunk: Chunk
    score: float
