from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class Mode(StrEnum):
    LIST = "list"
    SUMMARIZE = "summarize"
    GENERATE = "generate"


@dataclass(slots=True)
class SearchRequest:
    query: str
    site: str | None = None
    mode: Mode = Mode.LIST
    previous_queries: tuple[str, ...] = ()
    max_results: int = 10
    min_score: int = 70
    canonical_query: str | None = None
    retrieval: str = "compare"
    ranking_model: str | None = None
    # How wide to cast, and how much of the catch to judge. Both default to the
    # server's configured values; overriding them is what makes an offline
    # evaluation possible -- retrieve far more than a user would see and judge
    # it with a stronger model, and the result is a relevance opinion that does
    # not come from the retrieval being measured.
    retrieval_count: int | None = None
    ranking_count: int | None = None
    # Return the judge's verdict on every candidate, exclusions included. A
    # served query wants only what passed; measuring agreement between judges
    # needs the rejections too, since two judges that reject the same record
    # agree about it just as much as two that accept it.
    include_excluded: bool = False

    @property
    def effective_query(self) -> str:
        return self.canonical_query or self.query


# Retrieval fields, most discriminative first. A query that names a model is
# answered by `name`; a query that describes one is answered by `description`;
# everything else is corroborating detail. Scoring all three as one flat blob is
# what let a record whose `name` matched the query byte-for-byte rank 34th,
# because its length -- 100 language codes and a dozen enriched bullets -- was
# penalised while a terse derivative repo scored higher on the same terms.
FIELD_NAME = "name"
FIELD_DESCRIPTION = "description"
FIELD_BODY = "body"
RETRIEVAL_FIELDS: tuple[str, ...] = (FIELD_NAME, FIELD_DESCRIPTION, FIELD_BODY)


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    url: str
    name: str
    site: str
    text: str
    schema_object: dict[str, Any]
    # Pairs rather than a mapping so the record stays hashable, and ordered so
    # the catalog can build its per-field statistics without re-deriving them.
    fields: tuple[tuple[str, str], ...] = ()

    def field_text(self) -> dict[str, str]:
        """Per-field retrieval text, falling back to one `body` field.

        Adapters that predate fielded retrieval, and any future source that has
        no useful field structure, still score correctly -- as a single body
        field with weight 1.0, which is exactly plain BM25.
        """
        if not self.fields:
            return {FIELD_BODY: self.text}
        return dict(self.fields)


@dataclass(frozen=True, slots=True)
class Candidate:
    document: Document
    vector_score: float | None
    bm25_score: float | None
    retrieval_score: float
    retrieval_source: str

    @property
    def similarity(self) -> float:
        return self.vector_score or 0.0


@dataclass(frozen=True, slots=True)
class Result:
    document: Document
    score: int
    description: str
    vector_score: float | None = None
    bm25_score: float | None = None
    retrieval_source: str | None = None
    relevance: str = "relevant"

    def wire(self) -> dict[str, Any]:
        return {
            "@type": "Item",
            "url": self.document.url,
            "name": self.document.name,
            "site": self.document.site,
            "score": self.score,
            "relevance": self.relevance,
            "description": self.description,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "retrieval_source": self.retrieval_source,
            "schema_object": self.document.schema_object,
        }


@dataclass(slots=True)
class Event:
    type: str
    data: Any = None
    query_id: str = ""
    sequence: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def wire(self) -> dict[str, Any]:
        return {
            "message_id": self.event_id,
            "message_type": self.type,
            "query_id": self.query_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "content": self.data,
        }

    def json(self) -> str:
        return json.dumps(self.wire(), separators=(",", ":"))
