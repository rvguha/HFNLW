"""Two validators, as required by design doc s8.1.

1. A JSON-LD/schema-shape validator: required identifiers, names, URLs, types,
   and property structures.
2. A corpus evidence validator: every subject claim in a published item is
   backed by a passage that really occurs in the versioned source card.

Schema.org's range expectations are not a substitute for either.
"""

from __future__ import annotations

from typing import Any

from .cards import CleanCard
from .evidence import locate

TOP_LEVEL_ALLOWED = {
    # schema.org
    "@context", "@type", "@id", "name", "url", "description", "creator",
    "dateCreated", "dateModified", "license", "inLanguage", "keywords", "citation",
    # hf: namespace -- identity and provenance
    "hf:repository", "hf:revision", "hf:modelCard",
    # what it is
    "hf:task", "hf:library", "hf:architecture", "hf:parameters", "hf:contextLength",
    "hf:format", "hf:mixtureOfExperts", "hf:category", "hf:capability", "hf:subject",
    # documented behaviour
    "hf:intendedUse", "hf:limitation", "hf:deploymentNote",
    # lineage and reach
    "hf:family", "hf:familyRelation", "hf:discoveryScope", "hf:canonical", "hf:variant",
    "hf:baseModel", "hf:trainedOnDataset",
    "hf:downloads", "hf:likes", "hf:trendingScore", "hf:gated",
    "hf:licenseStatus", "hf:cardStatus", "hf:declaredLicense",
    # evidence, grouped by the relation it supports
    "hf:trainedOn", "hf:fineTunedOn", "hf:evaluatedOn", "hf:intendedFor",
    "hf:architectureFor",
}

# Evidence properties hold verbatim quotes from the card and are checked against
# it. Kept as a separate set so adding a fact property cannot silently create an
# unverified evidence channel.
EVIDENCE_PROPERTIES = {
    "hf:trainedOn", "hf:fineTunedOn", "hf:evaluatedOn", "hf:intendedFor",
    "hf:architectureFor",
}

# Keys that would mean internal pipeline state leaked into a published item.
INTERNAL_MARKERS = {"internal", "score", "reason_code", "attempts", "enrichment_key",
                    "raw_path", "warnings", "quarantine"}


def validate_item(item: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    context = item.get("@context") or {}
    require(context.get("@vocab") == "https://schema.org/",
            "@context must set @vocab to https://schema.org/")
    require(bool(context.get("hf")), "@context must bind the hf: namespace")
    require(item.get("@type") == ["SoftwareApplication", "hf:Model"],
            "@type must be [SoftwareApplication, hf:Model]")
    require(isinstance(item.get("@id"), str) and item["@id"].startswith("https://"),
            "@id must be an absolute https URL")
    require(bool(item.get("name")), "name is required")
    require(bool(item.get("url")), "url is required")
    require(bool(item.get("description")), "description is required")
    require(bool(item.get("dateModified")), "dateModified is required for provenance")
    require(bool(item.get("hf:repository")), "hf:repository is required")
    require(bool(item.get("hf:revision")), "hf:revision is required for provenance")

    # YAML 1.1 resolves bare `no`, `yes` and `on` to booleans, and those are all
    # real language codes -- `no` is Norwegian. Fifty records once shipped a
    # language called "false" because of it, which no query could ever match.
    # cards.py stops that at the parser; this stops it at the door, whatever
    # the source.
    coerced = [
        value for value in (item.get("inLanguage") or [])
        if isinstance(value, bool) or str(value).lower() in ("true", "false")
    ]
    require(not coerced, f"inLanguage holds YAML-coerced booleans: {coerced}")

    unknown = set(item) - TOP_LEVEL_ALLOWED
    if unknown:
        errors.append(f"unexpected top-level keys: {sorted(unknown)}")
    leaked = set(item) & INTERNAL_MARKERS
    if leaked:
        errors.append(f"internal fields leaked into the published item: {sorted(leaked)}")

    creator = item.get("creator") or {}
    require(creator.get("@type") in ("Organization", "Person"),
            "creator must be an Organization or Person")
    require(bool(creator.get("name")), "creator.name is required")

    for key in ("hf:subject", "hf:capability", "hf:intendedUse", "hf:limitation",
                "hf:deploymentNote", "hf:format", "hf:variant", *EVIDENCE_PROPERTIES):
        value = item.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{key} must be a list")
        elif isinstance(value, list) and not all(isinstance(v, str) for v in value):
            errors.append(f"{key} must hold strings")

    for key in ("hf:canonical", "hf:baseModel", "hf:trainedOnDataset", "hf:modelCard"):
        for value in _as_list(item.get(key)):
            if not str(value).startswith("https://"):
                errors.append(f"{key} must hold absolute https URLs, got {value!r}")

    return errors


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def validate_evidence(item: dict[str, Any], card: CleanCard | None) -> list[str]:
    """Every evidence quote must occur in the card at the recorded revision.

    This is the only guarantee the corpus makes about evidence: the span is
    really in the card. Whether it supports the claim is the extractor's
    judgement, which is why relation is published and confidence is not.
    """
    errors: list[str] = []
    quotes = [q for key in EVIDENCE_PROPERTIES for q in _as_list(item.get(key))]
    if not quotes:
        return errors
    if card is None:
        return [f"{len(quotes)} evidence quotes but no cleaned card to verify them"]
    for quote in quotes:
        if locate(quote, card) is None:
            errors.append(f"evidence quote not found in source: {quote[:80]!r}")
    return errors
