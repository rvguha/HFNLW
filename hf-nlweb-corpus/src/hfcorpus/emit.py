"""Emit the compact schema.org item NLWeb ingests (design doc s8).

The published item carries only facts that improve discovery, filtering,
explanation, or provenance. Operational state stays in the internal record.

Facts are direct properties in the ``hf:`` namespace rather than
``PropertyValue`` wrappers, and subjects are bare terms rather than
``DefinedTerm`` nodes: the namespace defines what the terms mean, so a record
does not have to. That took one model from 1,084 lines to 118 with no loss of
content.

Evidence keeps the one distinction the corpus rests on -- a benchmark mention is
not a training claim -- by grouping the verbatim quotes under the relation they
support (``hf:evaluatedOn``, ``hf:trainedOn``, ...). Confidence and section are
dropped: confidence was the extractor grading its own inference, and the anchored
card URL is reconstructible from ``hf:modelCard`` plus the quote, so storing it
per claim was storing a derivation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote as urlquote

from .config import Taxonomy
from .evidence import LANGUAGE_CODES, LANGUAGE_NAMES, PUNCTUATION_FOLD
from .families import FamilyAssignment

HF_NAMESPACE = "https://huggingface.co/ns/1.0#"

# Evidence relation -> the property its verbatim quotes are grouped under. The
# distinction is load-bearing: "models that actually know molecular biology, not
# ones that merely cite a biology benchmark" is answerable only if trained_on and
# evaluated_on stay apart. `merely_mentioned` is deliberately absent -- a passing
# mention is not evidence and is dropped rather than published.
EVIDENCE_PROPERTY = {
    "trained_on": "hf:trainedOn",
    "fine_tuned_on": "hf:fineTunedOn",
    "evaluated_on": "hf:evaluatedOn",
    "intended_for": "hf:intendedFor",
    "architecture": "hf:architectureFor",
}

# Normalized propertyID -> emitted property name. Facts worth carrying into the
# index: used for ranking, filtering, or the explanation shown beside a result.
CARRIED_PROPERTIES = {
    "ml:parameterCount": "hf:parameters",
    "ml:contextLength": "hf:contextLength",
    "ml:architecture": "hf:architecture",
    "ml:modelFormat": "hf:format",
    "ml:mixtureOfExperts": "hf:mixtureOfExperts",
    "huggingface:pipelineTag": "hf:task",
    "huggingface:library": "hf:library",
    "huggingface:downloads": "hf:downloads",
    "huggingface:likes": "hf:likes",
    "huggingface:trendingScore": "hf:trendingScore",
    "huggingface:gatedStatus": "hf:gated",
    "corpus:licenseStatus": "hf:licenseStatus",
    "corpus:cardStatus": "hf:cardStatus",
    "corpus:declaredLicenseIdentifier": "hf:declaredLicense",
}


def subject_slug(label: str) -> str:
    """Stable code for a subject label outside the controlled vocabulary.

    Folds case, punctuation, and the typographic hyphens models emit, so
    'Vision-Language', 'Vision-language' and 'vision language' collapse to one term.
    """
    folded = "".join(PUNCTUATION_FOLD.get(ch, ch) for ch in label.strip().lower())
    return re.sub(r"[^a-z0-9]+", "-", folded).strip("-")


def card_anchor(item: dict[str, Any], quote: str) -> str:
    """URL of the card at the pinned revision, anchored at the quoted span."""
    base = f"{item['@id']}/blob/{item.get('softwareVersion') or 'main'}/README.md"
    snippet = " ".join(quote.split())[:60]
    return f"{base}#:~:text={urlquote(snippet)}" if snippet else base


ARXIV_URL = re.compile(r"arxiv\.org/abs/([\w.\-/]+?)(?:v\d+)?$", re.IGNORECASE)


def load_abstracts(path: Path | None) -> dict[str, dict[str, str]]:
    """Fetched arXiv abstracts, keyed by identifier. Empty when absent."""
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text())


def _cite(nodes: list[dict[str, Any]], abstracts: dict[str, dict[str, str]]
          ) -> list[dict[str, Any]]:
    """Citations, carrying the paper's abstract where one was fetched.

    The abstract is the model's own authors describing what it does, at a length
    the one-sentence enriched description cannot reach. Half the corpus cites a
    paper, and 2,241 distinct papers stand behind 4,452 citing records.
    """
    out = []
    for node in nodes:
        url = str(node.get("url") or node.get("@id") or "")
        match = ARXIV_URL.search(url.strip())
        paper = abstracts.get(match.group(1)) if match else None
        out.append({**node, "name": paper["title"], "abstract": paper["abstract"]}
                   if paper else node)
    return out


def context() -> dict[str, str]:
    """Two vocabularies and nothing else.

    A property's range and cardinality belong in the namespace definition, not
    repeated in every one of ten thousand records.
    """
    return {"@vocab": "https://schema.org/", "hf": HF_NAMESPACE}


def emit(normalized: dict[str, Any], enrichment: dict[str, Any] | None,
         family: FamilyAssignment | None, taxonomy: Taxonomy,
         variant_repo_ids: list[str] | None = None,
         abstracts: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    item = normalized["item"]
    props = {p["propertyID"]: p["value"] for p in item.get("additionalProperty", [])}
    identifiers = {i["propertyID"]: i["value"] for i in item.get("identifier", [])}

    out: dict[str, Any] = {
        "@context": context(),
        "@id": item["@id"],
        # Both types on purpose: a plain schema.org consumer still sees software
        # it can render, and anyone who reads the namespace sees a model. Neither
        # alone is honest -- weights are not an application, and SoftwareApplication
        # is the only thing schema.org offers.
        "@type": ["SoftwareApplication", "hf:Model"],
        "name": item["name"],
        "hf:repository": identifiers.get("huggingface:repo_id"),
        "hf:revision": identifiers.get("huggingface:sha"),
        "url": item["url"],
        "creator": item["creator"],
        "dateModified": item.get("dateModified"),
        "description": item.get("description", ""),
    }

    for source, target in CARRIED_PROPERTIES.items():
        if source in props:
            out[target] = props[source]

    out["inLanguage"] = _languages(item, enrichment)
    out["keywords"] = item.get("keywords", [])
    if item.get("applicationSubCategory"):
        out["hf:category"] = item["applicationSubCategory"]

    evidence: dict[str, list[str]] = {}
    if enrichment:
        out["description"] = enrichment.get("short_description") or out["description"]
        capabilities = enrichment.get("capabilities", [])
        subjects = enrichment.get("subject_areas", [])
        if capabilities:
            out["hf:capability"] = _unique(c["label"] for c in capabilities)
        if subjects:
            out["hf:subject"] = _unique(
                s.get("taxonomy_term") or subject_slug(s["label"]) for s in subjects
            )
        # One quote can support several subjects; it is one piece of evidence.
        for claim in list(subjects) + list(capabilities):
            target = EVIDENCE_PROPERTY.get(claim.get("evidence_type"))
            quote = " ".join((claim.get("evidence_quote") or "").split())
            if target and quote:
                evidence.setdefault(target, [])
                if quote not in evidence[target]:
                    evidence[target].append(quote)
        for key, values in (
            ("hf:intendedUse", enrichment.get("intended_uses", [])),
            ("hf:limitation", enrichment.get("limitations", [])),
            ("hf:deploymentNote", enrichment.get("deployment_notes", [])),
        ):
            if values:
                out[key] = values

    if family:
        out["hf:family"] = family.family_id
        out["hf:familyRelation"] = family.relation
        out["hf:discoveryScope"] = family.discovery_scope
        if family.canonical_repo_id and family.relation != "canonical":
            out["hf:canonical"] = f"https://huggingface.co/{family.canonical_repo_id}"
    if variant_repo_ids:
        out["hf:variant"] = [f"https://huggingface.co/{r}" for r in variant_repo_ids]

    if item.get("license"):
        out["license"] = item["license"]
    for based in item.get("isBasedOn", []):
        # `isBasedOn` was one property doing two jobs, which is why it needed
        # typed nodes to be readable at all.
        key = "hf:trainedOnDataset" if based.get("@type") == "Dataset" else "hf:baseModel"
        out.setdefault(key, []).append(based["@id"])
    if item.get("citation"):
        out["citation"] = _cite(item["citation"], abstracts or {})

    card = next((n for n in item.get("subjectOf", [])
                 if n.get("@type") == "TechArticle"), None)
    if card:
        out["hf:modelCard"] = card["@id"]
    out.update(evidence)

    return {k: v for k, v in out.items() if v not in (None, "", [], {})}


def _unique(values) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(str(value), None)
    return list(seen)


def _languages(item: dict[str, Any], enrichment: dict[str, Any] | None) -> list[str]:
    """Declared metadata first; validated card-derived languages fill gaps.

    Everything is folded to an ISO-639-1 code where one is known, because the
    extractor answers "English" for a card whose metadata says `en` and the two
    are the same fact. Emitting both made `inLanguage` unusable as a filter --
    a query for `zh` missed every record that only said "Chinese".
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in item.get("inLanguage", []):
        # Hub language tags are a curated namespace; take them as given.
        code = canonical_language(str(value))
        if code and code not in seen:
            out.append(code)
            seen.add(code)
    for value in (enrichment or {}).get("languages", []):
        # Card-derived values are free text and arrive as things like "92 coding
        # languages", "xml documents containing translation data", and "solidity".
        # They pass evidence validation honestly -- the phrase really is in the
        # card -- but they are not languages, and they made inLanguage unusable
        # as a filter.
        if not is_plausible_language(str(value)):
            continue
        code = canonical_language(str(value))
        if code and code not in seen:
            out.append(code)
            seen.add(code)
    return out


# Values that are languages in the programming sense. The extractor picks them
# up from cards for code models, where "languages supported" means Rust, not Urdu.
PROGRAMMING_LANGUAGES = frozenset({
    "c", "cpp", "c++", "c#", "csharp", "c-sharp", "java", "javascript", "typescript",
    "python", "ruby", "rust", "go", "golang", "php", "perl", "scala", "kotlin", "swift",
    "r", "matlab", "julia", "haskell", "lua", "sql", "bash", "shell", "powershell",
    "html", "css", "xml", "json", "yaml", "markdown", "latex", "solidity", "zig",
    "racket", "scheme", "clojure", "erlang", "elixir", "fortran", "cobol", "assembly",
    "verilog", "vhdl", "dart", "groovy", "objective-c", "visual basic", "common lisp",
    "lisp", "ocaml", "f#", "jupyter notebook", "jupyter-clean", "makefile", "dockerfile",
})


def is_plausible_language(value: str) -> bool:
    """Reject free-text descriptions and programming languages.

    Deliberately permissive about natural languages it has never heard of --
    'quechua' and 'latgalian' are exactly the long tail this corpus should carry
    -- and strict about the shapes that are never a language: anything with a
    digit, anything phrase-length, and the programming vocabulary.
    """
    key = " ".join(value.strip().lower().split())
    if not key or any(ch.isdigit() for ch in key):
        return False
    if key in PROGRAMMING_LANGUAGES:
        return False
    # "norwegian (bokmal/nynorsk)" is a language; "xml documents containing
    # translation data" is a sentence. Two words is the practical boundary.
    return len(key.split()) <= 2


def canonical_language(value: str) -> str:
    """ISO-639-1 code where recognised, else the cleaned original."""
    key = value.strip().lower()
    if not key:
        return ""
    if key in LANGUAGE_NAMES:          # already a code
        return key
    return LANGUAGE_CODES.get(key, key)
