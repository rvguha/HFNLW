from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .config import Config
from .mcp_api import create_mcp
from .models import Mode, SearchRequest
from .pipeline import Services, search
from .providers import HashEmbeddings, NullLanguageModel, OpenRouterProvider
from .ranking_models import (
    DEFAULT_RANKING_MODEL,
    RANKING_MODEL_OPTIONS,
    RANKING_MODELS,
    validate_ranking_model,
)
from .rate_limit import PaidQueryRateLimitMiddleware, RateLimiter
from .refresh import RefreshState, build_catalog, read_sources, refresh_once, run_refresh_loop
from .sources import Source

STATIC = Path(__file__).resolve().parent / "static"
# Dotted so the catalog loader skips it: every .json in the corpus directory is
# otherwise treated as a collection, and a manifest ingested as one adds a
# phantom record, a phantom site, and invalidates the embedding cache.
CORPUS_MANIFEST = Path(__file__).resolve().parent / "corpus" / ".manifest.json"


def corpus_identity() -> dict[str, Any]:
    """Which corpus is answering, for anything that saves a conversation.

    A thread saved against a 2,447-record corpus and one saved against 9,465
    are both faithful and not comparable. `items` alone cannot tell them apart,
    so the build's snapshot id is carried through to the client.
    """
    try:
        manifest = json.loads(CORPUS_MANIFEST.read_text())
    except (OSError, ValueError):
        return {}
    return {
        key: manifest[key]
        for key in ("snapshot_id", "profile", "built_at", "items")
        if key in manifest
    }


class RevalidatingStatic(StaticFiles):
    """Static files the browser must revalidate before reusing.

    The default headers let a browser serve `app.js` from cache without asking,
    so a UI change can be live on the server, visible to curl, and still absent
    from a freshly opened tab. `no-cache` does not disable caching -- the ETag
    still avoids re-sending an unchanged file -- it just forbids using a cached
    copy without checking.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        response_headers["cache-control"] = "no-cache"
        return super().is_not_modified(response_headers, request_headers)

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response


logger = logging.getLogger(__name__)


def _request(body: dict[str, Any], config: Config) -> SearchRequest:
    query_value = body.get("query", "")
    if isinstance(query_value, dict):
        query = str(query_value.get("text", ""))
        site = query_value.get("site")
    else:
        query = str(query_value)
        site = body.get("site")
    context = body.get("context") if isinstance(body.get("context"), dict) else {}
    prefer = body.get("prefer") if isinstance(body.get("prefer"), dict) else {}
    previous = body.get("previous_queries", context.get("prev", body.get("prev", [])))
    if isinstance(previous, str):
        try:
            previous = json.loads(previous)
        except json.JSONDecodeError:
            previous = [previous]
    mode = prefer.get("mode", body.get("mode", body.get("generate_mode", "summarize")))
    retrieval = str(prefer.get("retrieval", body.get("retrieval", "compare")))
    if retrieval not in {"vector", "bm25", "compare"}:
        raise ValueError("retrieval must be vector, bm25, or compare")
    ranking_model = validate_ranking_model(
        str(prefer.get("ranking_model", body.get("ranking_model", ""))) or None
    )
    return SearchRequest(
        query=query.strip(),
        site=None if site in (None, "", "all") else str(site),
        mode=Mode(mode),
        previous_queries=tuple(str(value) for value in previous or ()),
        max_results=int(prefer.get("max_results", body.get("max_results", config.max_results))),
        min_score=int(prefer.get("min_score", body.get("min_score", config.min_score))),
        retrieval=retrieval,
        ranking_model=ranking_model,
        retrieval_count=_optional_int(prefer, body, "retrieval_count"),
        ranking_count=_optional_int(prefer, body, "ranking_count"),
        include_excluded=bool(prefer.get("include_excluded",
                                         body.get("include_excluded", False))),
    )


def _optional_int(prefer: dict[str, Any], body: dict[str, Any], key: str) -> int | None:
    """Absent means "use the server's default"; present must be a positive int.

    A bad value is rejected rather than silently falling back, because a typo in
    an evaluation harness should not quietly produce ordinary-width results that
    look like a finished run.
    """
    raw = prefer.get(key, body.get(key))
    if raw in (None, ""):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{key} must be at least 1")
    return value


async def ask_route(request: Request):
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        return JSONResponse({"error": "Service is starting"}, status_code=503)
    try:
        body = dict(request.query_params)
        if request.method == "POST":
            incoming = await request.json()
            if isinstance(incoming, dict):
                body.update(incoming)
        query = _request(body, services.config)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    async def stream_events():
        async for event in search(query, services):
            yield f"event: {event.type}\nid: {event.event_id}\ndata: {event.json()}\n\n"

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def health(request: Request):
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return JSONResponse(
        {
            "status": "ok",
            "items": len(services.catalog.documents),
            "sites": services.catalog.sites,
            "embedding_cache_hit": services.catalog.cache_hit,
            "corpus": corpus_identity(),
            "sources": getattr(request.app.state, "refresh_state", RefreshState()).snapshot(),
        }
    )


def create_app(config: Config | None = None, services: Services | None = None) -> Starlette:
    config = config or Config.from_env()
    holder: dict[str, Services | None] = {"services": services}
    rate_limiter = RateLimiter(
        config.rate_limit_per_client_per_minute,
        config.rate_limit_global_per_hour,
        config.rate_limit_global_per_day,
        config.max_concurrent_queries,
    )

    refresh_state = RefreshState()
    sources_for_refresh: list[Source] = []

    def require_services() -> Services:
        if holder["services"] is None:
            raise RuntimeError("Service is starting")
        return holder["services"]

    mcp = create_mcp(require_services, config.mcp_allowed_hosts)
    mcp_app = mcp.streamable_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        if holder["services"] is None:
            if config.openrouter_api_key:
                provider = OpenRouterProvider(
                    config.openrouter_api_key,
                    config.llm_model,
                    config.embedding_model,
                    config.openrouter_base_url,
                    config.app_url,
                    config.app_title,
                    # Decontextualisation and composition are short, bounded
                    # tasks; the utility model should not spend its output
                    # budget reasoning about them.
                    config.llm_reasoning_effort,
                )
                embedder, llm = provider, provider
                default_ranking_model = config.ranking_model or DEFAULT_RANKING_MODEL
                if default_ranking_model not in RANKING_MODELS:
                    logger.warning(
                        "Unsupported OPENROUTER_RANKING_MODEL %s; using %s",
                        default_ranking_model,
                        DEFAULT_RANKING_MODEL,
                    )
                    default_ranking_model = DEFAULT_RANKING_MODEL
                rankers = {
                    option.id: OpenRouterProvider(
                        config.openrouter_api_key,
                        option.id,
                        config.embedding_model,
                        config.openrouter_base_url,
                        config.app_url,
                        config.app_title,
                        option.reasoning_effort,
                        config.ranking_provider_sort or "throughput",
                    )
                    for option in RANKING_MODEL_OPTIONS
                }
                ranking_max_tokens = {
                    option.id: option.max_tokens for option in RANKING_MODEL_OPTIONS
                }
                ranker = rankers[default_ranking_model]
            else:
                embedder, llm = HashEmbeddings(), NullLanguageModel()
                ranker = llm
                rankers = {}
                ranking_max_tokens = {}
                default_ranking_model = None
            # Fetch the corpus before building the index, so a fresh machine
            # needs only the manifest rather than a data directory baked into
            # the image.
            sources = read_sources(config)
            sources_for_refresh.extend(sources)
            await refresh_once(config, sources, refresh_state)
            catalog = await build_catalog(config, sources, embedder)
            holder["services"] = Services(
                config,
                catalog,
                embedder,
                llm,
                ranker,
                rankers,
                ranking_max_tokens,
                default_ranking_model,
            )
        app.state.services = holder["services"]
        app.state.refresh_state = refresh_state

        refresher: asyncio.Task[None] | None = None
        if sources_for_refresh and config.refresh_interval_hours > 0:
            refresher = asyncio.create_task(
                run_refresh_loop(
                    holder["services"],
                    sources_for_refresh,
                    refresh_state,
                    config.refresh_interval_hours * 3600,
                ),
                name="source-refresh",
            )
            logger.info(
                "rechecking %d sources every %.1f hours",
                len(sources_for_refresh),
                config.refresh_interval_hours,
            )

        try:
            async with mcp.session_manager.run():
                yield
        finally:
            if refresher is not None:
                refresher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresher

    routes = [
        Route("/ask", ask_route, methods=["GET", "POST"]),
        Route("/health", health),
        Mount("/mcp", mcp_app),
        Mount("/", RevalidatingStatic(directory=STATIC, html=True), name="static"),
    ]
    middleware = []
    if config.cors_origins:
        middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=list(config.cors_origins),
                allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                allow_headers=["content-type", "accept", "mcp-session-id", "mcp-protocol-version"],
                expose_headers=["mcp-session-id"],
            )
        )
    middleware.append(
        Middleware(
            PaidQueryRateLimitMiddleware,
            limiter=rate_limiter,
            trusted_proxy_ips=config.trusted_proxy_ips,
        )
    )
    app = Starlette(routes=routes, lifespan=lifespan, middleware=middleware)
    app.state.services = services
    app.state.mcp = mcp
    app.state.rate_limiter = rate_limiter
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config = Config.from_env()
    uvicorn.run(create_app(config), host=args.host or config.host, port=args.port or config.port)
