"""Family, duplicate, and variant resolution (design doc s6).

Deduplication here is conservative on purpose. Merging is driven by lineage,
configuration, and naming; card similarity alone can only *flag* a pair for
review, because distinct fine-tunes routinely reuse an upstream card template.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from .naming import repo_name, repo_owner, strip_variant_tokens, variant_kind

SHINGLE_SIZE = 5
NEAR_DUPLICATE_JACCARD = 0.90
WORD = re.compile(r"[a-z0-9]+")


@dataclass
class FamilyAssignment:
    repo_id: str
    family_id: str
    canonical_repo_id: str
    relation: str            # canonical | derived_from | exact_duplicate_of |
                             # format_or_quantization_variant | mirror_of
    evidence: list[str] = field(default_factory=list)
    discovery_scope: str = "primary"   # primary | variant
    candidate_duplicates: list[str] = field(default_factory=list)


def family_id_for(root_repo_id: str, patterns: dict[str, list[str]]) -> str:
    stem = strip_variant_tokens(root_repo_id, patterns).lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    owner = re.sub(r"[^a-z0-9]+", "-", repo_owner(root_repo_id).lower()).strip("-")
    return f"{owner}--{stem}" if owner else stem


def resolve(records: list[dict[str, Any]], patterns: dict[str, list[str]],
            variants: list[dict[str, Any]] | None = None) -> dict[str, FamilyAssignment]:
    """`records` are normalized ``{item, internal}`` rows; `variants` are the
    suppressed packaging repositories from selection, which get attached to a
    family without becoming primary search results."""
    by_id = {r["internal"]["repo_id"]: r for r in records}
    assignments: dict[str, FamilyAssignment] = {}

    # -- stage 1: exact duplicates ------------------------------------------
    duplicate_of: dict[str, str] = {}
    by_card: dict[str, list[str]] = {}
    for repo_id, rec in by_id.items():
        card_sha = rec["internal"].get("clean_card_sha256")
        if card_sha:
            by_card.setdefault(card_sha, []).append(repo_id)
    for repo_ids in by_card.values():
        if len(repo_ids) < 2:
            continue
        groups: dict[tuple, list[str]] = {}
        for repo_id in repo_ids:
            groups.setdefault(_config_key(by_id[repo_id]), []).append(repo_id)
        for group in groups.values():
            if len(group) < 2:
                continue
            canonical = _canonical(group, by_id)
            for repo_id in group:
                if repo_id != canonical:
                    duplicate_of[repo_id] = canonical

    # -- stage 2: declared lineage ------------------------------------------
    roots: dict[str, str] = {}
    for repo_id in by_id:
        roots[repo_id] = _lineage_root(repo_id, by_id)

    # -- stage 3+5: family assignment and canonical choice -------------------
    families: dict[str, list[str]] = {}
    for repo_id in by_id:
        root = duplicate_of.get(repo_id) or repo_id
        fam = family_id_for(roots.get(root, root), patterns)
        families.setdefault(fam, []).append(repo_id)

    canonical_by_family = {fam: _canonical(members, by_id) for fam, members in families.items()}

    # -- stage 4: near-duplicate flags (review only, never a merge) ----------
    near = _near_duplicates(by_id)

    for fam, members in families.items():
        canonical = canonical_by_family[fam]
        for repo_id in members:
            if repo_id in duplicate_of:
                relation = "exact_duplicate_of"
                evidence = ["identical cleaned model card and config"]
                target = duplicate_of[repo_id]
            elif repo_id == canonical:
                relation, evidence, target = "canonical", [], repo_id
            else:
                kind = variant_kind(repo_id, patterns)
                if kind in ("quantization", "format_conversion"):
                    relation = "format_or_quantization_variant"
                    evidence = [f"repository name matches {kind} pattern"]
                elif kind == "mirror":
                    relation, evidence = "mirror_of", ["repository name matches mirror pattern"]
                else:
                    relation = "derived_from"
                    evidence = [f"declared base model {b}"
                                for b in by_id[repo_id]["internal"].get("declared_base_models", [])]
                target = canonical
            scope = "variant" if relation in (
                "exact_duplicate_of", "format_or_quantization_variant", "mirror_of") else "primary"
            assignments[repo_id] = FamilyAssignment(
                repo_id=repo_id,
                family_id=fam,
                canonical_repo_id=target,
                relation=relation,
                evidence=evidence,
                discovery_scope=scope,
                candidate_duplicates=sorted(near.get(repo_id, [])),
            )

    # -- suppressed packaging variants attach to a family, not to the index --
    for variant in variants or []:
        repo_id = variant["repo_id"]
        if repo_id in assignments:
            continue
        parent = strip_variant_tokens(repo_id, patterns)
        parent_id = f"{repo_owner(repo_id)}/{parent}"
        fam = family_id_for(parent_id, patterns)
        # Only claim a canonical repository we actually hold. A name-derived
        # parent that is not in the corpus is a guess, and a guessed repo id
        # would publish a link that may not resolve.
        canonical = canonical_by_family.get(fam, "")
        evidence = ["suppressed at selection as a packaging variant"]
        if not canonical:
            evidence.append(f"canonical parent {parent_id} not represented in this corpus")
        assignments[repo_id] = FamilyAssignment(
            repo_id=repo_id,
            family_id=fam,
            canonical_repo_id=canonical,
            relation=variant.get("relation", "format_or_quantization_variant"),
            evidence=evidence,
            discovery_scope="variant",
        )
    return assignments


def variants_by_family(assignments: dict[str, FamilyAssignment]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a in assignments.values():
        if a.discovery_scope == "variant":
            out.setdefault(a.family_id, []).append(a.repo_id)
    return {k: sorted(v) for k, v in out.items()}


def _lineage_root(repo_id: str, by_id: dict[str, dict[str, Any]], depth: int = 0) -> str:
    """Walk declared base models up to the root. Chains outside the corpus stop
    at the first external repository, which still yields a stable family key."""
    if depth > 6:
        return repo_id
    bases = by_id.get(repo_id, {}).get("internal", {}).get("declared_base_models") or []
    if not bases:
        return repo_id
    base = sorted(bases)[0]
    if base == repo_id:
        return repo_id
    if base in by_id:
        return _lineage_root(base, by_id, depth + 1)
    return base


def _config_key(record: dict[str, Any]) -> tuple:
    props = {p["propertyID"]: p["value"] for p in record["item"].get("additionalProperty", [])}
    return (props.get("ml:architecture"), props.get("ml:parameterCount"),
            props.get("ml:modelType"))


def _canonical(repo_ids: list[str], by_id: dict[str, dict[str, Any]]) -> str:
    """Prefer the original publisher: most downloads, then earliest creation,
    then the shortest name, then lexical order for determinism (s6)."""
    def key(repo_id: str):
        rec = by_id.get(repo_id)
        if rec is None:
            return (0, "9999", 999, repo_id)
        props = {p["propertyID"]: p["value"] for p in rec["item"].get("additionalProperty", [])}
        return (
            -int(props.get("huggingface:downloads") or 0),
            str(rec["item"].get("dateCreated") or "9999"),
            len(repo_name(repo_id)),
            repo_id,
        )
    return sorted(repo_ids, key=key)[0]


def _shingles(text: str) -> set[int]:
    words = WORD.findall(text.lower())
    if len(words) < SHINGLE_SIZE:
        return set()
    return {
        int.from_bytes(hashlib.blake2b(" ".join(words[i:i + SHINGLE_SIZE]).encode(),
                                       digest_size=8).digest(), "big")
        for i in range(len(words) - SHINGLE_SIZE + 1)
    }


def _near_duplicates(by_id: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Flag highly similar cards that share architecture and parameter count.
    Output is advisory: it feeds the review sample, never a merge."""
    buckets: dict[tuple, list[str]] = {}
    for repo_id, rec in by_id.items():
        buckets.setdefault(_config_key(rec), []).append(repo_id)

    descriptions = {rid: rec["item"].get("description", "") for rid, rec in by_id.items()}
    shingles = {rid: _shingles(text) for rid, text in descriptions.items()}
    out: dict[str, list[str]] = {}
    for key, members in buckets.items():
        if key == (None, None, None) or len(members) < 2:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                sa, sb = shingles.get(a) or set(), shingles.get(b) or set()
                if not sa or not sb:
                    continue
                jaccard = len(sa & sb) / len(sa | sb)
                if jaccard >= NEAR_DUPLICATE_JACCARD:
                    out.setdefault(a, []).append(b)
                    out.setdefault(b, []).append(a)
    return out
