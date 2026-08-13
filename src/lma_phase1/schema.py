"""Canonical document schema shared by all corpus stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

CollectionType = Literal["manual", "downloaded"]


@dataclass(slots=True)
class Document:
    """A provenance-preserving monolingual text document.

    The text may change during preprocessing, but provenance fields must be
    copied unchanged through every pipeline stage.
    """

    document_id: str
    language: str
    text: str
    source_name: str
    collection_type: CollectionType
    collection_method: str
    license: str
    collected_at: str
    source_url: str | None = None
    source_document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def validate(self) -> None:
        """Raise ``ValueError`` when required schema invariants are violated."""

        required = {
            "document_id": self.document_id,
            "language": self.language,
            "text": self.text,
            "source_name": self.source_name,
            "collection_method": self.collection_method,
            "license": self.license,
            "collected_at": self.collected_at,
        }
        missing = [
            name for name, value in required.items() if not isinstance(value, str) or not value
        ]
        if missing:
            raise ValueError(f"Missing or empty required fields: {', '.join(missing)}")
        if self.collection_type not in {"manual", "downloaded"}:
            raise ValueError("collection_type must be 'manual' or 'downloaded'")
        if any(char.isspace() for char in self.document_id):
            raise ValueError("document_id must not contain whitespace")

    def to_dict(self) -> dict[str, Any]:
        """Validate and serialize the document to a JSON-compatible mapping."""

        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Document:
        """Construct and validate a document from a decoded JSON object."""

        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown document fields: {', '.join(unknown)}")
        document = cls(**value)
        document.validate()
        return document
