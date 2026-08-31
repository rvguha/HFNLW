from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Keys the ranking projection keeps by default. Structural keys first, then the
# ones that carry meaning for this corpus: what a model is documented to do
# (`hf:capability`), what subjects it covers (`hf:subject`), and the evidence
# quotes behind them, grouped by the relation they support -- the judge needs to
# see that a benchmark mention is not a training claim.
DEFAULT_RANKING_FIELDS: tuple[str, ...] = (
    "@type",
    "name",
    "description",
    "url",
    "keywords",
    "license",
    "creator",
    "citation",
    "abstract",
    "inLanguage",
    "hf:repository",
    "hf:task",
    "hf:category",
    "hf:capability",
    "hf:subject",
    "hf:intendedUse",
    "hf:limitation",
    "hf:parameters",
    "hf:library",
    "hf:architecture",
    "hf:licenseStatus",
    "hf:downloads",
    "hf:baseModel",
    "hf:trainedOn",
    "hf:fineTunedOn",
    "hf:evaluatedOn",
    "hf:intendedFor",
)


# Set by benchmarks/run_retrieval.py: 100 queries whose relevance is stated as
# predicates over the corpus, so the gold sets are free and survive a rebuild.
# 40 are known-item lookups (gold is one named repository), 60 are descriptive;
# the 20 whose gold set exceeds 5% of the corpus are excluded from tuning
# because they cannot separate one ranking from another.
#
#                      known-item nDCG@10    descriptive nDCG@10
#   flat BM25                    0.701                  0.607
#   these weights                0.978                  0.627
#
# Two results here were counter to the design intuition and are worth keeping
# in view. Weighting `description` *up* hurts both categories -- the enriched
# one-sentence description is short and generic, while the specifics a
# descriptive query matches on live in `body` (keywords, subject terms,
# intended uses, evidence quotes), so promoting description suppresses the
# evidence. And `b` for `name` turned out not to matter at all: at this weight
# the name field saturates whatever the length normalisation, so all three
# fields keep the standard 0.75 rather than the damped value first proposed.
# The optimum is a broad plateau; these sit mid-plateau, not on a grid edge.
DEFAULT_BM25_FIELD_WEIGHTS: dict[str, float] = {"name": 32.0, "description": 1.0, "body": 0.5}
DEFAULT_BM25_FIELD_B: dict[str, float] = {"name": 0.75, "description": 0.75, "body": 0.75}
# Weights on the per-field embedding channels, tuned on the same benchmark
# (benchmarks/run_retrieval.py --vector).
#
# Splitting the embedding by field matters as much as splitting BM25 did. Scored
# alone, the three channels are good at completely different things:
#
#                    known-item nDCG@10    descriptive nDCG@10
#   name vector              0.885                  0.380
#   description vector       0.254                  0.612
#   body vector              0.231                  0.698
#
# A single vector over the whole record is dominated by `body`, which is 76% of
# the text -- which is why one flat embedding could not answer a query that
# names a model.
#
# These weights are tuned on the union's coverage, not on the vector channel's
# own ranking: the pipeline takes a head from each channel and hands the union
# to the ranking model, so retrieval's job is to get a relevant record in front
# of the judge, not to order it. On that measure, adding the vector channel
# lifts descriptive coverage from 0.923 to 0.974 and raises the mean number of
# relevant records reaching the ranker from 3.87 to 5.96, with known-item
# already at 1.000 from BM25F alone. Coverage is insensitive to the weights
# themselves -- every grid point scored 0.974 -- so these stay equal rather than
# pretending to a precision the measurement does not support.
DEFAULT_VECTOR_FIELD_WEIGHTS: dict[str, float] = {"name": 1.0, "description": 1.0, "body": 1.0}


@dataclass(frozen=True, slots=True)
class Config:
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-120b"
    ranking_model: str | None = None
    ranking_reasoning_effort: str | None = None
    llm_reasoning_effort: str | None = "low"
    ranking_provider_sort: str | None = None
    ranking_max_tokens: int = 100
    embedding_model: str = "openai/text-embedding-3-small"
    app_url: str | None = None
    app_title: str = "ask-hf-hub"
    data_dir: Path = Path("data")
    # Manifest naming where each collection is fetched from. When absent, the
    # data directory is used as-is, which is how tests and local checkouts run.
    sources_manifest: Path = Path("sources.yaml")
    # How often to recheck sources. A check that finds nothing changed costs one
    # conditional request per source and no embedding, so this can be frequent.
    refresh_interval_hours: float = 4.0
    cache_dir: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    bm25_rank_count: int = 10
    vector_rank_count: int = 10
    # Ceilings for the per-request overrides. A wide retrieval judged by an
    # expensive model is exactly what an evaluation wants and exactly what an
    # abusive request wants, so the endpoint caps what it will accept rather
    # than trusting the caller.
    max_retrieval_count: int = 200
    max_ranking_count: int = 200
    ranking_batch_count: int = 3
    ranking_batch_return_count: int = 3
    max_results: int = 10
    min_score: int = 70
    strong_score_threshold: int = 90
    # Decontextualization is one LLM call on the critical path; if it is slow
    # the original query is searched instead.
    decontextualize_timeout: float = 15.0
    rate_limit_per_client_per_minute: int = 10
    rate_limit_global_per_hour: int = 60
    rate_limit_global_per_day: int = 500
    max_concurrent_queries: int = 4
    cors_origins: tuple[str, ...] = ()
    mcp_allowed_hosts: tuple[str, ...] = ()
    trusted_proxy_ips: tuple[str, ...] = ("127.0.0.1", "::1")
    # Which schema.org keys survive the projection sent to the ranking model.
    # The list is corpus-shaped, not universal: a product catalog needs `offers`
    # and `eligibleRegion`, a model catalog needs `about` and `subjectOf`, and a
    # list inherited from one corpus silently blinds the ranker on another.
    # Keeping it in config means changing corpus does not mean changing code.
    ranking_fields: tuple[str, ...] = DEFAULT_RANKING_FIELDS
    # BM25F field weights and length normalisation. `name` is weighted heavily
    # because a query that types a model's name is asking for that model, not
    # for something that mentions it; `body` stays at 1.0 so it can corroborate
    # a match without winning one.
    bm25_field_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_BM25_FIELD_WEIGHTS)
    )
    bm25_field_b: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BM25_FIELD_B))
    vector_field_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VECTOR_FIELD_WEIGHTS)
    )

    @classmethod
    def from_env(cls) -> Config:
        load_dotenv()

        def integer(name: str, default: int) -> int:
            return int(os.getenv(name, str(default)))

        def values(name: str, default: str = "") -> tuple[str, ...]:
            return tuple(
                value.strip() for value in os.getenv(name, default).split(",") if value.strip()
            )

        def weights(name: str, default: dict[str, float]) -> dict[str, float]:
            """Per-field tuning as a JSON object, merged over the defaults.

            Merged rather than replaced so setting one field's weight does not
            silently drop the other fields to zero.
            """
            raw = os.getenv(name, "").strip()
            if not raw:
                return dict(default)
            return {**default, **{k: float(v) for k, v in json.loads(raw).items()}}

        host = os.getenv("ASKHUB_HOST", "127.0.0.1")
        port = integer("ASKHUB_PORT", 8000)
        local_mcp_hosts = f"{host}:{port},127.0.0.1:{port},localhost:{port}"

        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            llm_model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b"),
            ranking_model=os.getenv("OPENROUTER_RANKING_MODEL") or None,
            ranking_reasoning_effort=os.getenv("OPENROUTER_RANKING_REASONING_EFFORT") or None,
            llm_reasoning_effort=os.getenv("OPENROUTER_LLM_REASONING_EFFORT", "low") or None,
            ranking_provider_sort=os.getenv("OPENROUTER_RANKING_PROVIDER_SORT") or None,
            ranking_max_tokens=integer("ASKHUB_RANKING_MAX_TOKENS", 100),
            embedding_model=os.getenv(
                "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
            ),
            app_url=os.getenv("OPENROUTER_APP_URL") or None,
            app_title=os.getenv("OPENROUTER_APP_TITLE", "ask-hf-hub"),
            data_dir=Path(os.getenv("ASKHUB_DATA_DIR", "data")),
            sources_manifest=Path(os.getenv("ASKHUB_SOURCES_MANIFEST", "sources.yaml")),
            refresh_interval_hours=float(os.getenv("ASKHUB_REFRESH_INTERVAL_HOURS", "4")),
            cache_dir=(
                Path(value) if (value := os.getenv("ASKHUB_CACHE_DIR", "").strip()) else None
            ),
            host=host,
            port=port,
            bm25_rank_count=integer("ASKHUB_BM25_RANK_COUNT", 10),
            vector_rank_count=integer("ASKHUB_VECTOR_RANK_COUNT", 10),
            max_retrieval_count=integer("ASKHUB_MAX_RETRIEVAL_COUNT", 200),
            max_ranking_count=integer("ASKHUB_MAX_RANKING_COUNT", 200),
            ranking_batch_count=integer("ASKHUB_RANKING_BATCH_COUNT", 3),
            ranking_batch_return_count=integer("ASKHUB_RANKING_BATCH_RETURN_COUNT", 3),
            max_results=integer("ASKHUB_MAX_RESULTS", 10),
            min_score=integer("ASKHUB_MIN_SCORE", 70),
            strong_score_threshold=integer("ASKHUB_STRONG_SCORE_THRESHOLD", 90),
            decontextualize_timeout=float(
                os.getenv("ASKHUB_DECONTEXTUALIZE_TIMEOUT", "15")
            ),
            rate_limit_per_client_per_minute=integer("ASKHUB_RATE_LIMIT_PER_CLIENT_PER_MINUTE", 10),
            rate_limit_global_per_hour=integer("ASKHUB_RATE_LIMIT_GLOBAL_PER_HOUR", 60),
            rate_limit_global_per_day=integer("ASKHUB_RATE_LIMIT_GLOBAL_PER_DAY", 500),
            max_concurrent_queries=integer("ASKHUB_MAX_CONCURRENT_QUERIES", 4),
            cors_origins=values("ASKHUB_CORS_ORIGINS"),
            mcp_allowed_hosts=values("ASKHUB_MCP_ALLOWED_HOSTS", local_mcp_hosts),
            trusted_proxy_ips=values("ASKHUB_TRUSTED_PROXY_IPS", "127.0.0.1,::1"),
            ranking_fields=values("ASKHUB_RANKING_FIELDS", ",".join(DEFAULT_RANKING_FIELDS)),
            bm25_field_weights=weights("ASKHUB_BM25_FIELD_WEIGHTS", DEFAULT_BM25_FIELD_WEIGHTS),
            bm25_field_b=weights("ASKHUB_BM25_FIELD_B", DEFAULT_BM25_FIELD_B),
            vector_field_weights=weights(
                "ASKHUB_VECTOR_FIELD_WEIGHTS", DEFAULT_VECTOR_FIELD_WEIGHTS
            ),
        )
