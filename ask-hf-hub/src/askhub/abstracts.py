"""Embeddings for the arXiv abstracts the corpus cites.

The abstract is the paper's own statement of what a model does, in the authors'
words and at a length the enriched one-sentence description cannot reach. It is
stored and embedded per *paper*, not per model: 2,241 papers stand behind 4,452
citing records, so keying on the paper avoids embedding the same text dozens of
times and keeps one abstract shared by every model that cites it.

Building the store is separate from using it. Nothing in retrieval reads these
vectors yet; wiring them into a scoring channel is a later decision, and this
module exists so that decision does not also have to pay for the embedding.

Vectors use the same model as the corpus fields, so they live in the same space
and can be combined with them without a projection.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from .config import Config
from .providers import Embeddings, OpenRouterProvider


def load(path: Path) -> tuple[list[str], np.ndarray, str]:
    """Return (arxiv ids, matrix, embedding model) from a built store."""
    with np.load(path, allow_pickle=False) as store:
        ids = [str(x) for x in store["ids"].tolist()]
        matrix = store["embeddings"].astype(np.float32, copy=False)
        model = str(store["model"])
    return ids, matrix, model


async def build(
    abstracts: dict[str, dict[str, str]],
    embedder: Embeddings,
    model: str,
    out: Path,
    batch: int = 64,
    log=print,
) -> np.ndarray:
    """Embed every abstract, checkpointing so an interrupted run resumes."""
    ids = sorted(abstracts)
    done: dict[str, np.ndarray] = {}
    # Filesystem work goes through a thread, as it does in the catalog: these
    # stores are tens of megabytes and blocking the loop on them is avoidable.
    if await asyncio.to_thread(out.is_file):
        try:
            cached_ids, cached, cached_model = await asyncio.to_thread(load, out)
            if cached_model == model:
                done = dict(zip(cached_ids, cached, strict=True))
                log(f"resuming: {len(done)} abstracts already embedded")
            else:
                # A different model means a different space; mixing them would
                # produce silently meaningless similarities.
                log(f"stored vectors are from {cached_model}, rebuilding for {model}")
        except (OSError, ValueError, KeyError):
            log("unreadable store; rebuilding")

    missing = [key for key in ids if key not in done]
    if missing:
        log(f"embedding {len(missing)} abstracts in {(len(missing) + batch - 1) // batch} batches")
        for start in range(0, len(missing), batch):
            chunk = missing[start : start + batch]
            vectors = await embedder.embed([abstracts[key]["abstract"] for key in chunk])
            done.update(zip(chunk, vectors, strict=True))
            await asyncio.to_thread(_write, out, ids, done, model)
            log(f"  {len(done)}/{len(ids)}")
    else:
        log("nothing to embed")

    _, matrix, _ = await asyncio.to_thread(load, out)
    return matrix


def _write(out: Path, ids: list[str], done: dict[str, np.ndarray], model: str) -> None:
    ready = [key for key in ids if key in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        ids=np.asarray(ready),
        embeddings=np.asarray([done[key] for key in ready], dtype=np.float32),
        model=np.asarray(model),
    )
    temporary.replace(out)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abstracts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    abstracts = json.loads(args.abstracts.read_text())
    config = Config.from_env()
    provider = OpenRouterProvider(
        config.openrouter_api_key, config.llm_model, config.embedding_model,
        config.openrouter_base_url, config.app_url, config.app_title,
        config.llm_reasoning_effort,
    )
    matrix = asyncio.run(
        build(abstracts, provider, config.embedding_model, args.out)
    )
    print(f"{matrix.shape[0]} abstract vectors of {matrix.shape[1]} dims -> {args.out}")


if __name__ == "__main__":
    main()
