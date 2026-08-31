from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from collections.abc import AsyncIterator, Collection
from dataclasses import dataclass, field

from .catalog import MemoryCatalog
from .config import Config
from .models import Candidate, Event, Mode, Result, SearchRequest
from .pre_retrieval import decontextualize
from .providers import (
    Embeddings,
    LanguageModel,
    MeteredLanguageModel,
    NullLanguageModel,
    embed_with_usage,
    similarity_score,
)
from .usage import UsageLedger

logger = logging.getLogger(__name__)

# Records per batch when asking for a verdict on every one. Ten is the largest
# size the judges were observed to label exhaustively; at thirty-three they
# shortlist regardless of the instruction.
FULL_MATRIX_BATCH_SIZE = 10


@dataclass(slots=True)
class Services:
    config: Config
    catalog: MemoryCatalog
    embedder: Embeddings
    llm: LanguageModel
    ranker: LanguageModel | None = None
    rankers: dict[str, LanguageModel] = field(default_factory=dict)
    ranking_max_tokens: dict[str, int] = field(default_factory=dict)
    default_ranking_model: str | None = None


async def search(request: SearchRequest, services: Services) -> AsyncIterator[Event]:
    query_id = str(uuid.uuid4())
    usage = UsageLedger(query_id)
    sequence = 0

    def event(kind: str, data=None) -> Event:
        nonlocal sequence
        sequence += 1
        return Event(kind, data, query_id, sequence)

    yield event("begin-nlweb-response")
    if not request.query.strip():
        yield event("error", "No query provided.")
        yield event("end-nlweb-response")
        return

    # Match NLWeb's reconciliation gate: fast track can run concurrently with
    # prechecks, but nothing from it is published until the checks finish. A
    # conversation follow-up is not fast-track eligible because it may need a
    # decontextualized query.
    fast_task = None
    fast_updates = None
    if not request.previous_queries:
        fast_task, fast_updates = _start_ranking(
            request.query, request, services, usage
        )
    replacement = None
    if request.previous_queries:
        replacement = await decontextualize(
            request,
            MeteredLanguageModel(services.llm, usage, "pre_retrieval"),
            services.config.decontextualize_timeout,
        )
    if replacement:
        request.canonical_query = replacement
        yield event("decontextualized_query", replacement)

    # Fast track speculated on the raw query. It is only valid if the query it
    # ran against is still the one being searched.
    if fast_task is None or replacement is not None:
        if fast_task is not None:
            fast_task.cancel()
            await asyncio.gather(fast_task, return_exceptions=True)
        ranking_task, updates = _start_ranking(request.effective_query, request, services, usage)
    else:
        ranking_task, updates = fast_task, fast_updates

    try:
        while (update := await updates.get()) is not None:
            # Candidates arrive batch-by-batch after the precheck gate opens.
            # The authoritative, sorted top-N follows as result events.
            yield event("candidate", [update.wire()])
        results, notices = await ranking_task
    except Exception:
        ranking_task.cancel()
        await asyncio.gather(ranking_task, return_exceptions=True)
        ranking_task, updates = _start_ranking(request.effective_query, request, services, usage)
        while (update := await updates.get()) is not None:
            yield event("candidate", [update.wire()])
        results, notices = await ranking_task

    for result in results:
        yield event("result", [result.wire()])
    for notice in notices:
        yield event("intermediate_message", notice)

    if not results:
        yield event("intermediate_message", "No matching results found.")
    elif request.mode is not Mode.LIST:
        try:
            answer = await _compose(
                request, results, MeteredLanguageModel(services.llm, usage, "generation")
            )
            yield event("nlws", {"answer": answer, "items": [result.wire() for result in results]})
        except Exception as exc:
            yield event("intermediate_message", f"Answer generation unavailable: {exc}")

    yield event("usage", usage.wire())
    yield event("complete")
    yield event("end-nlweb-response")


async def _retrieve_and_rank(
    query: str,
    request: SearchRequest,
    services: Services,
    usage: UsageLedger,
    updates: asyncio.Queue[Result | None] | None = None,
) -> tuple[list[Result], list[str]]:
    notices: list[str] = []
    ranking_id = request.ranking_model or services.default_ranking_model
    ranking_model = services.rankers.get(ranking_id) if ranking_id else None
    ranking_model = ranking_model or services.ranker or services.llm
    ranking_max_tokens = services.ranking_max_tokens.get(
        ranking_id, services.config.ranking_max_tokens
    )
    embedding_task = (
        asyncio.create_task(embed_with_usage(services.embedder, [query], usage, "query_embedding"))
        if request.retrieval in {"vector", "compare"}
        else None
    )

    async def rank(candidate: Candidate) -> Result | None:
        try:
            output = await MeteredLanguageModel(
                ranking_model, usage, "ranking"
            ).structured(
                "Act as a relevance filter, not a ranker. Classify the supplied record as "
                "strong, relevant, or exclude. Strong means it directly satisfies the user's "
                "intent; relevant means it is substantially useful but partial. Exclude weak, "
                "tangential, or merely keyword-overlapping records. Return only JSON as "
                "{\"m\":\"strong\",\"why\":\"...\"}. The why is one sentence of at most "
                "25 words naming the specific matching product, episode, eligibility rule, "
                "benefit, or topic. Start with a concrete fact or item name. Never refer to "
                "'the record', 'the query', 'relevant information', or say that something "
                "'provides information'. Use only supplied facts.",
                {"q": query, "r": _ranking_projection(candidate.document.schema_object,
                                                      services.config.ranking_fields)},
                max_tokens=(ranking_max_tokens * 8 if request.include_excluded
                            else ranking_max_tokens),
                temperature=0,
            )
            parsed = _parse_relevance(
                output, request.min_score, services.config.strong_score_threshold
            )
            if parsed is None:
                if not request.include_excluded:
                    return None
                score, relevance = 0, "exclude"
            else:
                score, relevance = parsed
            description = _clean_description(
                str(output.get("why", output.get("description", "")))
            )
            result = Result(
                candidate.document,
                score,
                description,
                candidate.vector_score,
                candidate.bm25_score,
                candidate.retrieval_source,
                relevance,
            )
            if updates is not None:
                updates.put_nowait(result)
            return result
        except Exception:
            return None

    async def rank_batch(
        indexed_candidates: list[tuple[int, Candidate]],
    ) -> list[Result]:
        # Serving stops after a few good records; a full matrix needs every one.
        # This break -- not the prompt -- was the real ceiling: three per batch
        # times three batches is nine verdicts however wide the retrieval.
        return_count = (
            len(indexed_candidates) if request.include_excluded
            else min(max(1, services.config.ranking_batch_return_count),
                     len(indexed_candidates))
        )
        # Serving asks for a short list; evaluation asks for a verdict on every
        # record. The difference is not a threshold -- "select at most 3 of these
        # 33" caps output at 9 per query however wide the retrieval, so agreement
        # between two judges measures overlap of two shortlists rather than
        # whether they assess the same record the same way.
        instruction = (
            "Act as a relevance filter, not a ranker. "
            f"Return a verdict for EVERY one of the {len(indexed_candidates)} records, "
            "in input order, omitting none. For each record return m as "
            "strong, relevant, or exclude. Strong means it directly satisfies the "
            "user's intent; relevant means it is substantially useful but partial; "
            "exclude means weak, tangential, or merely keyword-overlapping. "
            "Return only JSON as "
            "{\"results\":[{\"i\":0,\"m\":\"strong\",\"why\":\"...\"}]}. "
            "Each why is one sentence of at most 20 words naming the specific "
            "matching topic or capability. Use only supplied facts."
        ) if request.include_excluded else (
                "Act as a relevance filter, not a ranker. "
                f"Select at most {return_count} records. For each selected record return m as "
                "either strong or relevant. Strong means the record directly satisfies the "
                "user's intent; relevant means it is substantially useful but partial. Omit "
                "weak, tangential, or merely keyword-overlapping records, and do not fill the "
                "quota. Return only JSON as "
                "{\"results\":[{\"i\":0,\"m\":\"strong\",\"why\":\"...\"}]}. Group "
                "strong before relevant and preserve input order within each group. Each why is "
                "one sentence of at most 25 words naming the specific matching product, "
                "episode, eligibility rule, benefit, or topic. Never refer to 'the record', "
                "'the query', 'relevant information', or say that something 'provides "
                "information'. Use only supplied facts."
        )
        try:
            output = await MeteredLanguageModel(
                ranking_model, usage, "ranking"
            ).structured(
                instruction,
                {
                    "q": query,
                    "rs": [
                        {
                            "i": local,
                            "r": _ranking_projection(candidate.document.schema_object,
                                                     services.config.ranking_fields),
                        }
                        for local, (_index, candidate) in enumerate(indexed_candidates)
                    ],
                },
                max_tokens=(ranking_max_tokens * 8 if request.include_excluded
                            else ranking_max_tokens),
                temperature=0,
            )
            items = output.get("results", output.get("scores", []))
            if not isinstance(items, list):
                return []
            # The model echoes positions within the list it was shown, not the
            # global indices we supplied. Batches hold non-contiguous global
            # indices (round-robin), so a returned "3" matched only when global
            # index 3 happened to sit in that batch -- silently discarding most
            # verdicts. Number each batch locally and map back.
            candidates_by_index = {
                local: candidate for local, (_global, candidate) in enumerate(indexed_candidates)
            }
            results: list[Result] = []
            seen: set[int] = set()
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("i", item.get("index")))
                except (TypeError, ValueError):
                    continue
                candidate = candidates_by_index.get(index)
                if candidate is None or index in seen:
                    continue
                seen.add(index)
                parsed = _parse_relevance(
                    item, request.min_score, services.config.strong_score_threshold
                )
                if parsed is None:
                    if not request.include_excluded:
                        continue
                    parsed = (0, "exclude")
                score, relevance = parsed
                result = Result(
                    candidate.document,
                    score,
                    _clean_description(
                        str(item.get("why", item.get("description", "")))
                    ),
                    candidate.vector_score,
                    candidate.bm25_score,
                    candidate.retrieval_source,
                    relevance,
                )
                results.append(result)
                if updates is not None:
                    # Each completed batch independently feeds the SSE stream. The
                    # final authoritative threshold groups follow after all batches settle.
                    updates.put_nowait(result)
                if len(seen) == return_count:
                    break
            return results
        except Exception:
            return []

    # One query must see one catalog. A refresh rebinds services.catalog between
    # the two searches below - they are separated by an await on the embedding -
    # which would otherwise mix BM25 hits from the old generation with vector
    # hits from the new one in a single compare request.
    catalog = services.catalog

    # Per-request retrieval width, capped. Defaults to the configured counts, so
    # an ordinary query is unchanged; an evaluation asks for far more.
    bm25_width = _bounded(request.retrieval_count, services.config.bm25_rank_count,
                          services.config.max_retrieval_count)
    vector_width = _bounded(request.retrieval_count, services.config.vector_rank_count,
                            services.config.max_retrieval_count)

    bm25_candidates: list[Candidate] = []
    bm25_ids: set[str] = set()
    if request.retrieval in {"bm25", "compare"}:
        bm25_candidates = await catalog.search_bm25(query, request.site, bm25_width)
        bm25_ids = {candidate.document.id for candidate in bm25_candidates}

    vector_candidates: list[Candidate] = []
    if embedding_task is not None:
        try:
            vector = (await embedding_task)[0]
            vector_candidates = await catalog.search_vector(
                vector,
                request.site,
                vector_width,
                bm25_ids if request.retrieval == "compare" else None,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            notices.append("Vector retrieval unavailable; returning BM25 results.")

    candidates = bm25_candidates + vector_candidates
    if request.ranking_count is not None:
        # Judge only the top of what was retrieved. Separate from retrieval
        # width on purpose: retrieving 200 and ranking 50 is a different
        # experiment from retrieving 50 and ranking all of them.
        candidates = candidates[: _bounded(request.ranking_count, len(candidates),
                                           services.config.max_ranking_count)]

    if isinstance(ranking_model, NullLanguageModel):
        notices.append("No language model is configured; results use retrieval scores.")
        ranked = []
        for candidate in candidates:
            score = (
                similarity_score(candidate.vector_score)
                if candidate.vector_score is not None
                else min(100, int((candidate.bm25_score or 0) * 5))
            )
            result = Result(
                candidate.document,
                score,
                candidate.document.text[:240],
                candidate.vector_score,
                candidate.bm25_score,
                candidate.retrieval_source,
                "retrieval",
            )
            ranked.append(result)
            if updates is not None:
                updates.put_nowait(result)
    elif services.config.ranking_batch_count > 0:
        if request.include_excluded:
            # A verdict on every record only survives a batch the model will
            # actually work through. Given 33 records and told to label all of
            # them, it quietly returns its top few instead; given 10, it labels
            # 10. Batch to that size rather than to a fixed batch count.
            batch_count = max(1, math.ceil(len(candidates) / FULL_MATRIX_BATCH_SIZE))
        else:
            batch_count = min(services.config.ranking_batch_count, len(candidates))
        batches: list[list[tuple[int, Candidate]]] = [[] for _ in range(batch_count)]
        for index, candidate in enumerate(candidates):
            batches[index % batch_count].append((index, candidate))
        ranked = []
        for completed_batch in asyncio.as_completed(
            [rank_batch(batch) for batch in batches]
        ):
            ranked.extend(await completed_batch)
    else:
        ranked = [
            result for result in await asyncio.gather(*(rank(c) for c in candidates)) if result
        ]
    ranked = _group_by_relevance_band(ranked)
    if request.include_excluded:
        # Deduplication is a presentation concern: it collapses records that
        # share a display name so a result list does not show "Phi 4 mini
        # instruct" four times. 301 names in this corpus are shared by 649
        # records, and collapsing them silently dropped 45 of 1,000 verdicts the
        # judges had already produced. An evaluation needs every record.
        return ranked[: request.max_results], notices
    return _deduplicate_ranked_results(ranked)[: request.max_results], notices


def _start_ranking(
    query: str, request: SearchRequest, services: Services, usage: UsageLedger
):
    updates: asyncio.Queue[Result | None] = asyncio.Queue()

    async def run():
        try:
            return await _retrieve_and_rank(query, request, services, usage, updates)
        finally:
            updates.put_nowait(None)

    return asyncio.create_task(run()), updates


def _deduplicate_ranked_results(results: list[Result]) -> list[Result]:
    unique: list[Result] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        key = (result.document.site, " ".join(result.document.name.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _relevance_band(score: int, strong_threshold: int = 90) -> str:
    return "strong" if score >= strong_threshold else "relevant"


def _parse_relevance(
    output: dict, relevant_threshold: int = 70, strong_threshold: int = 90
) -> tuple[int, str] | None:
    """Accept categorical output, with numeric scores as a compatibility fallback."""
    match = str(output.get("m", output.get("match", ""))).strip().lower()
    if match == "strong":
        return strong_threshold, "strong"
    if match == "relevant":
        return relevant_threshold, "relevant"
    if match in {"exclude", "excluded", "none", "irrelevant"}:
        return None
    try:
        score = max(0, min(100, int(output.get("s", output.get("score", 0)))))
    except (TypeError, ValueError):
        return None
    if score < relevant_threshold:
        return None
    return score, _relevance_band(score, strong_threshold)


def _group_by_relevance_band(results: list[Result]) -> list[Result]:
    """Group strong matches first without ranking or tie-breaking inside a band."""
    return [result for result in results if result.relevance == "strong"] + [
        result for result in results if result.relevance != "strong"
    ]


def _bounded(requested: int | None, fallback: int, ceiling: int) -> int:
    """Honour a request's override within the server's ceiling."""
    if requested is None:
        return fallback
    return max(1, min(int(requested), ceiling))


def _ranking_projection(value, fields: Collection[str]):
    """Trim a schema.org object down to the keys worth spending ranking tokens on.

    `fields` is corpus-shaped and comes from config: a product catalog needs
    `offers` and `eligibleRegion`, a model catalog needs `hf:subject` and the
    evidence properties.
    Inheriting one corpus's list silently blinds the ranker on another -- with a
    product-catalog list in place, every evidence-bearing field on a model
    record was stripped before the relevance filter ever saw it.
    """
    if isinstance(value, dict):
        return {
            key: _ranking_projection(item, fields)
            for key, item in value.items()
            if key in fields
        }
    if isinstance(value, list):
        return [_ranking_projection(item, fields) for item in value[:20]]
    if isinstance(value, str):
        return value[:1000]
    return value


_DESCRIPTION_PREAMBLES = (
    "the record provides relevant information about ",
    "this record provides relevant information about ",
    "the record provides information about ",
    "this record provides information about ",
)


def _clean_description(value: str) -> str:
    description = value.strip().lstrip(". ")
    lowered = description.lower()
    for preamble in _DESCRIPTION_PREAMBLES:
        if lowered.startswith(preamble):
            description = description[len(preamble) :].lstrip()
            break
    if description:
        description = description[0].upper() + description[1:]
    return description


async def _compose(request: SearchRequest, results: list[Result], llm: LanguageModel) -> str:
    key = "summary" if request.mode is Mode.SUMMARIZE else "answer"
    instruction = (
        "Write the short overview shown above ranked search results. Use only the supplied "
        "evidence. Write 2-4 concise natural-language sentences that name the most useful "
        "results and explain concrete differences between them. The summary value must be "
        "reader-facing prose, never a list, table, object, code block, tool call, or JSON/Python "
        "serialization. Return exactly one JSON object with one string field named summary, "
        "for example {\"summary\":\"Several options fit...\"}."
        if request.mode is Mode.SUMMARIZE
        else "Answer the user's question using only the supplied search evidence. Write concise "
        "reader-facing prose and cite relevant supplied URLs. The answer value must never be a "
        "list, table, object, code block, tool call, or JSON/Python serialization. Return exactly "
        "one JSON object with one string field named answer."
    )
    output = await llm.structured(
        instruction,
        {
            "query": request.effective_query,
            "evidence": [
                {
                    "name": result.document.name,
                    "url": result.document.url,
                    "description": result.description,
                }
                for result in results
            ],
        },
        # A reasoning model spends its budget thinking before it writes. At 400
        # tokens gpt-oss-120b returned an empty completion for every summary and
        # answer -- the prose never got a turn. Composition is one call per
        # query, so the headroom is cheap.
        max_tokens=4000,
        temperature=0,
    )
    value = output.get(key)
    if isinstance(value, str):
        prose = value.strip()
        if prose.startswith("```") and prose.endswith("```"):
            prose = prose.strip("`").removeprefix("json").strip()
        if prose.startswith(("{", "[")):
            try:
                nested = json.loads(prose)
            except (TypeError, json.JSONDecodeError):
                nested = None
            if isinstance(nested, dict) and isinstance(nested.get(key), str):
                prose = nested[key].strip()
            else:
                prose = ""
        if prose:
            return prose
    return _fallback_composition(results)


def _fallback_composition(results: list[Result]) -> str:
    selected = results[:4]
    names = [result.document.name for result in selected]
    if not names:
        return "No matching results were available to summarize."
    name_list = (
        names[0]
        if len(names) == 1
        else f"{', '.join(names[:-1])}, and {names[-1]}"
    )
    details = []
    for result in selected[:3]:
        description = result.description.strip()
        if not description:
            continue
        details.append(description if description.endswith((".", "!", "?")) else f"{description}.")
    overview = f"The leading matches are {name_list}."
    return f"{overview} {' '.join(details)}".strip()


async def collect(request: SearchRequest, services: Services) -> list[Event]:
    return [event async for event in search(request, services)]
