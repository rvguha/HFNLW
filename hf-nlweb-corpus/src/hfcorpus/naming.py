"""Repository-name heuristics shared by selection and family resolution.

These are *signals*, never conclusions: a name that looks like a quantization is
a reason to check lineage, and a "7B" in a name is an inferred parameter count
that must be labelled as such (s5.2).
"""

from __future__ import annotations

import re
from typing import Any

SIZE_IN_NAME = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([bm])(?![a-z0-9])", re.I)
SEPARATORS = re.compile(r"[-_.]+")
# llama.cpp-style quantization suffixes: -Q4_K_M, -IQ3_XXS, -Q8_0.
QUANT_SUFFIX = re.compile(r"[-_.](?:i?q\d+(?:[_-][a-z0-9]+)*)$", re.I)


def repo_owner(repo_id: str) -> str:
    return repo_id.split("/")[0] if "/" in repo_id else ""


def repo_name(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def _compiled(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.I) for p in patterns]


def matches_any(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text, re.I):
            return pattern
    return None


def variant_kind(repo_id: str, patterns: dict[str, list[str]]) -> str | None:
    """'quantization', 'format_conversion', 'checkpoint', 'mirror', or None.

    Matched against tokens of the repository name only -- an owner called
    'onnx-community' publishes plenty of things that are not conversions.
    """
    name = repo_name(repo_id)
    if QUANT_SUFFIX.search(name):
        return "quantization"
    tokens = {t.lower() for t in SEPARATORS.split(name) if t}
    for kind in ("quantization", "format_conversion"):
        for pattern in patterns.get(kind, []):
            if pattern in tokens or (not pattern.isalnum() and re.search(pattern, name, re.I)):
                return kind
    for kind in ("checkpoint", "mirror"):
        if matches_any(name, patterns.get(kind, [])):
            return kind
    return None


def strip_variant_tokens(repo_id: str, patterns: dict[str, list[str]]) -> str:
    """Best-effort parent name: peel trailing packaging suffixes one at a time.

    Suffixes are removed from the end of the string rather than re-joining split
    tokens, so 'LFM2.5-2.6B-Heretic-GGUF' yields 'LFM2.5-2.6B-Heretic' and not a
    dot-mangled name that matches no real repository.
    """
    name = repo_name(repo_id)
    drop = [p.lower() for p in patterns.get("quantization", [])
            + patterns.get("format_conversion", []) if p.isalnum()]
    while True:
        match = QUANT_SUFFIX.search(name)
        if match and match.start() > 0:
            name = name[: match.start()]
            continue
        for token in drop:
            trimmed = re.sub(rf"[-_.]{re.escape(token)}$", "", name, flags=re.I)
            if trimmed != name and trimmed:
                name = trimmed
                break
        else:
            return name or repo_name(repo_id)


def provisional_family_key(repo_id: str, base_models: list[str],
                           patterns: dict[str, list[str]]) -> str:
    """Cheap family key for the selection-time family cap. families.py replaces
    this with lineage-resolved ids once cards and configs are available."""
    if base_models:
        root = sorted(base_models)[0]
        return strip_variant_tokens(root, patterns).lower()
    return strip_variant_tokens(repo_id, patterns).lower()


def inferred_parameters_from_name(repo_id: str) -> int | None:
    """Parse '7B'/'350M' out of a name. Fallback only, and every caller must
    label the result inferred (s5.2)."""
    matches = SIZE_IN_NAME.findall(repo_name(repo_id))
    if not matches:
        return None
    value, unit = matches[-1]
    scale = 1_000_000_000 if unit.lower() == "b" else 1_000_000
    return int(float(value) * scale)


def humanize(repo_id: str) -> str:
    """Display name for the record. Keeps the publisher's capitalisation and
    only replaces separators, so 'Qwen/Qwen3-8B' reads as 'Qwen3 8B'. Dots
    inside numbers survive: '0.6B' is a size, not two words."""
    name = re.sub(r"[-_]+", " ", repo_name(repo_id))
    name = re.sub(r"\.(?!\d)", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def declared_base_models(raw: dict[str, Any]) -> list[str]:
    """Union of API baseModels and card-YAML base_model, de-duplicated."""
    out: list[str] = []
    api = raw.get("baseModels") or {}
    if isinstance(api, dict):
        for entry in api.get("models") or []:
            if isinstance(entry, dict) and entry.get("id"):
                out.append(str(entry["id"]))
    card = (raw.get("cardData") or {}).get("base_model")
    if isinstance(card, str):
        out.append(card)
    elif isinstance(card, list):
        out.extend(str(c) for c in card if isinstance(c, str))
    seen: dict[str, None] = {}
    for item in out:
        seen.setdefault(item, None)
    return list(seen)
