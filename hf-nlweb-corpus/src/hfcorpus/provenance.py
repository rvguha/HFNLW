"""Snapshot manifests and run provenance (design doc s3.2, s9.2).

A snapshot names every version that could change the output: the policy, the
normalizer, the enrichment prompt, the taxonomy, and the pinned Hub client. A
rebuild from the manifest should produce equivalent NLWeb records (s14).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from . import __version__
from .config import Policy, Taxonomy
from .enrich import PROMPT_VERSION

NORMALIZER_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_snapshot(policy: Policy, taxonomy: Taxonomy, hub_version: str,
                    snapshot_id: str | None = None) -> dict[str, Any]:
    started = utc_now()
    return {
        "snapshot_id": snapshot_id or f"hf-{policy.profile}-{started[:10]}",
        "started_at": started,
        "pipeline_version": __version__,
        "huggingface_hub_version": hub_version,
        "selection_policy_version": policy.policy_version,
        "selection_profile": policy.profile,
        "random_seed": policy.seed,
        "normalizer_version": NORMALIZER_VERSION,
        "enrichment_prompt_version": PROMPT_VERSION,
        "enrichment_backend": policy.enrichment.get("backend", "anthropic"),
        "enrichment_model": (policy.enrichment.get("openrouter_model")
                             if policy.enrichment.get("backend") == "openrouter"
                             else policy.enrichment["model"]),
        "taxonomy_version": taxonomy.version,
        "taxonomy_id": taxonomy.id,
        "target_count": policy.target_count,
        "quotas": {s.name: s.quota for s in policy.strata},
        "source": "Hugging Face public Hub API and repository README.md files",
        "stages": {},
    }


def record_stage(snapshot: dict[str, Any], stage: str, **facts: Any) -> dict[str, Any]:
    snapshot.setdefault("stages", {})[stage] = {"completed_at": utc_now(), **facts}
    return snapshot
