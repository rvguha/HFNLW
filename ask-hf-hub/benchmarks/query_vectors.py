"""Query embeddings for the benchmark, cached on disk.

Embedding a hundred short queries costs a fraction of a cent, but a weight sweep
re-embeds them on every pass. Caching keyed on (model, query) makes the sweep
free after the first run and keeps repeated tuning off the bill.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / ".query-embeddings.json"


def _key(model: str, query: str) -> str:
    return hashlib.sha256(f"{model}\x00{query}".encode()).hexdigest()[:32]


def embed_queries(queries: list[str], config) -> dict[str, np.ndarray]:
    from askhub.providers import OpenRouterProvider

    store: dict[str, list[float]] = json.loads(CACHE.read_text()) if CACHE.is_file() else {}
    model = config.embedding_model
    missing = [q for q in queries if _key(model, q) not in store]

    if missing:
        provider = OpenRouterProvider(
            config.openrouter_api_key, config.llm_model, model,
            config.openrouter_base_url, config.app_url, config.app_title,
            config.llm_reasoning_effort,
        )
        print(f"embedding {len(missing)} uncached queries")
        matrix = asyncio.run(provider.embed(missing))
        for query, row in zip(missing, matrix, strict=True):
            store[_key(model, query)] = [float(x) for x in row]
        CACHE.write_text(json.dumps(store))

    return {q: np.asarray(store[_key(model, q)], dtype=np.float32) for q in queries}
