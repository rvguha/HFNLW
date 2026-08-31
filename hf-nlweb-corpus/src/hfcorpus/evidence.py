"""Deterministic evidence validation (design doc s7.4, s8.1).

The extractor proposes; this module disposes. Every quote must actually occur in
the versioned source card, and the confidence a claim is allowed to carry is
capped by its evidence type -- regardless of what the model asserted.
Unsupported claims are dropped, never cosmetically downgraded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .cards import CleanCard
from .config import Taxonomy

WHITESPACE = re.compile(r"\s+")
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}

# Enough of an ISO-639-1 table to reconcile a card that declares `language: en`
# with an extractor that answered "English". Unknown codes fall through to the
# card-text check, which is the conservative branch.
LANGUAGE_NAMES = {
    "en": "english", "zh": "chinese", "es": "spanish", "fr": "french", "de": "german",
    "it": "italian", "pt": "portuguese", "ru": "russian", "ja": "japanese", "ko": "korean",
    "ar": "arabic", "hi": "hindi", "bn": "bengali", "ta": "tamil", "te": "telugu",
    "mr": "marathi", "ur": "urdu", "fa": "persian", "tr": "turkish", "nl": "dutch",
    "pl": "polish", "sv": "swedish", "da": "danish", "fi": "finnish", "no": "norwegian",
    "cs": "czech", "el": "greek", "he": "hebrew", "th": "thai", "vi": "vietnamese",
    "id": "indonesian", "ms": "malay", "sw": "swahili", "am": "amharic", "yo": "yoruba",
    "ha": "hausa", "ig": "igbo", "zu": "zulu", "uk": "ukrainian", "ro": "romanian",
    "hu": "hungarian", "bg": "bulgarian", "ca": "catalan", "eu": "basque", "gl": "galician",
}
LANGUAGE_CODES = {name: code for code, name in LANGUAGE_NAMES.items()}


def language_aliases(value: str) -> set[str]:
    key = value.strip().lower()
    aliases = {key}
    if key in LANGUAGE_NAMES:
        aliases.add(LANGUAGE_NAMES[key])
    if key in LANGUAGE_CODES:
        aliases.add(LANGUAGE_CODES[key])
    return aliases


@dataclass
class ValidatedEnrichment:
    short_description: str
    capabilities: list[dict[str, Any]]
    subject_areas: list[dict[str, Any]]
    intended_uses: list[str]
    languages: list[str]
    limitations: list[str]
    deployment_notes: list[str]
    dropped: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "short_description": self.short_description,
            "capabilities": self.capabilities,
            "subject_areas": self.subject_areas,
            "intended_uses": self.intended_uses,
            "languages": self.languages,
            "limitations": self.limitations,
            "deployment_notes": self.deployment_notes,
        }


def _norm(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip().lower()


def _clean_list(values: Any) -> list[str]:
    return [str(v).strip() for v in (values or []) if str(v).strip()]


# Emphasis and code markers are formatting, not content. A model reading
# "**Significantly enhanced** reasoning" quotes it back without the asterisks,
# which is a faithful quote of what the card says; comparing literally would
# discard real evidence as a fabrication. Matching ignores these on both sides,
# while the span that gets stored is the source text, markers and all.
MARKDOWN_NOISE = frozenset("*`~#_")

# Typographic variants of the same character. Models routinely "improve" ASCII
# punctuation on the way out -- gpt-oss renders "GPT-2" with a non-breaking
# hyphen and "1 M context" with a narrow no-break space -- and a byte comparison
# then reads a faithful quote as a fabrication. Folding these is not leniency:
# it makes the comparison about characters rather than about which codepoint a
# model happened to pick for a hyphen. Written as escapes so the table is
# readable and cannot be mangled by an editor.
PUNCTUATION_FOLD = {
    "\u2010": "-",      # hyphen
    "\u2011": "-",      # non-breaking hyphen
    "\u2012": "-",      # figure dash
    "\u2013": "-",      # en dash
    "\u2014": "-",      # em dash
    "\u2015": "-",      # horizontal bar
    "\u2212": "-",      # minus sign
    "\u2043": "-",      # hyphen bullet
    "\u2018": "'",      # left single quote
    "\u2019": "'",      # right single quote
    "\u201a": "'",      # single low quote
    "\u201b": "'",      # single high-reversed quote
    "\u2032": "'",      # prime
    "\u201c": '"',      # left double quote
    "\u201d": '"',      # right double quote
    "\u201e": '"',      # double low quote
    "\u201f": '"',      # double high-reversed quote
    "\u2033": '"',      # double prime
    "\u2026": "...",    # ellipsis
    "\u200b": "",       # zero-width space
    "\u200c": "",       # zero-width non-joiner
    "\u200d": "",       # zero-width joiner
    "\ufeff": "",       # byte-order mark
}


def _collapse(text: str) -> tuple[str, list[int]]:
    """Lowercase, fold typographic variants, drop formatting markers, collapse
    whitespace runs. Returns the normalized text and, for each of its
    characters, the index in `text` it came from."""
    chars: list[str] = []
    index_map: list[int] = []
    previous_space = True          # also suppresses leading whitespace
    for i, ch in enumerate(text):
        if ch in MARKDOWN_NOISE:
            continue
        folded = PUNCTUATION_FOLD.get(ch, ch)
        if not folded:
            continue
        if folded.isspace():
            if previous_space:
                continue
            chars.append(" ")
            index_map.append(i)
            previous_space = True
        else:
            # A fold may expand (… -> ...); every output character maps back to
            # the single source character it came from, keeping the map valid.
            for out in folded.lower():
                chars.append(out)
                index_map.append(i)
            previous_space = False
    while chars and chars[-1] == " ":
        chars.pop()
        index_map.pop()
    return "".join(chars), index_map


def locate(quote: str, card: CleanCard) -> tuple[int, int, str] | None:
    """Find a quote in the cleaned card, tolerant of whitespace and Markdown
    formatting differences. Returns (start, end, section_path) in clean-text
    coordinates.

    Tolerant is not lenient: every word of the quote must still occur, in order,
    contiguously in the versioned source.
    """
    needle, _ = _collapse(quote)
    if len(needle) < 12:
        return None
    haystack, index_map = _collapse(card.clean_text)
    position = haystack.find(needle)
    if position < 0:
        return None
    start = index_map[position]
    end = index_map[position + len(needle) - 1] + 1
    section = next((" > ".join(s.path) or "preamble" for s in card.sections
                    if s.clean_start <= start < s.clean_end), "preamble")
    return start, end, section


def validate(enrichment: dict[str, Any], card: CleanCard, item: dict[str, Any],
             taxonomy: Taxonomy) -> ValidatedEnrichment:
    dropped: list[dict[str, Any]] = []
    warnings: list[str] = []

    def check_claims(claims: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for claim in claims:
            quote = (claim.get("evidence_quote") or "").strip()
            found = locate(quote, card) if quote else None
            if found is None:
                dropped.append({"kind": kind, "label": claim.get("label"),
                                "reason": "quote_not_found_in_source",
                                "quote": quote[:160]})
                continue
            start, end, section = found
            evidence_type = claim.get("evidence_type", "merely_mentioned")
            if kind == "subject_area" and not taxonomy.establishes_specialty(evidence_type):
                dropped.append({
                    "kind": kind, "label": claim.get("label"),
                    "reason": f"evidence_type_{evidence_type}_does_not_establish_specialty"})
                continue
            ceiling = taxonomy.max_confidence(evidence_type)
            stated = claim.get("confidence", "low")
            confidence = stated
            if CONFIDENCE_ORDER.get(stated, 0) > CONFIDENCE_ORDER.get(ceiling, 0):
                confidence = ceiling
                warnings.append(
                    f"{kind}:{claim.get('label')} confidence lowered {stated}->{ceiling} "
                    f"by evidence type {evidence_type}")
            entry = {
                "label": str(claim.get("label", "")).strip(),
                "confidence": confidence,
                "evidence_type": evidence_type,
                "evidence_quote": card.clean_text[start:end],
                "source_section": section,
                "clean_offsets": [start, end],
            }
            if kind == "subject_area":
                term = taxonomy.resolve(entry["label"])
                entry["taxonomy_term"] = term["id"] if term else None
                entry["taxonomy_status"] = "mapped" if term else "proposed"
            kept.append(entry)
        return kept

    capabilities = check_claims(enrichment.get("capabilities", []), "capability")
    subject_areas = check_claims(enrichment.get("subject_areas", []), "subject_area")

    declared: set[str] = set()
    for lang in item.get("inLanguage", []):
        declared |= language_aliases(str(lang))
    card_text = _norm(card.clean_text)
    languages: list[str] = []
    for language in enrichment.get("languages", []):
        value = str(language).strip()
        if not value:
            continue
        if (language_aliases(value) & declared) or _norm(value) in card_text:
            languages.append(value)
        else:
            dropped.append({"kind": "language", "label": value,
                            "reason": "not_declared_and_not_present_in_card"})

    description = str(enrichment.get("short_description", "")).strip()
    if not description:
        warnings.append("empty_short_description")

    return ValidatedEnrichment(
        short_description=description,
        capabilities=capabilities,
        subject_areas=subject_areas,
        intended_uses=_clean_list(enrichment.get("intended_uses")),
        languages=languages,
        limitations=_clean_list(enrichment.get("limitations")),
        deployment_notes=_clean_list(enrichment.get("deployment_notes")),
        dropped=dropped,
        warnings=warnings,
    )
