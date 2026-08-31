from __future__ import annotations

import contextlib
import json
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .models import Mode, SearchRequest
from .pipeline import Services, search
from .ranking_models import RANKING_MODEL_OPTIONS, validate_ranking_model


def create_mcp(get_services, allowed_hosts: tuple[str, ...] = ()) -> FastMCP:
    security = TransportSecuritySettings()
    if allowed_hosts:
        hosts = [*security.allowed_hosts, *allowed_hosts]
        origins = [*security.allowed_origins]
        for host in allowed_hosts:
            origins.extend([f"http://{host}", f"https://{host}"])
        security = TransportSecuritySettings(allowed_hosts=hosts, allowed_origins=origins)
    mcp = FastMCP(
        "ask-hf-hub",
        instructions="Search the in-memory Schema.org catalog with the ask tool.",
        streamable_http_path="/",
        transport_security=security,
    )

    @mcp.tool(description="List the indexed content collections.")
    async def list_sites() -> dict[str, Any]:
        services: Services = get_services()
        catalog = services.catalog          # one generation, not two
        return {
            "sites": catalog.sites,
            "items": len(catalog.documents),
            "ranking_models": [option.wire() for option in RANKING_MODEL_OPTIONS],
            "default_ranking_model": services.default_ranking_model,
        }

    @mcp.tool(description="Search indexed content using natural language.")
    async def ask(
        query: str,
        ctx: Context,
        site: str | None = None,
        mode: str = "summarize",
        previous_queries: list[str] | None = None,
        max_results: int | None = None,
        retrieval: str = "compare",
        ranking_model: str | None = None,
    ) -> dict[str, Any]:
        services: Services = get_services()
        if retrieval not in {"vector", "bm25", "compare"}:
            raise ValueError("retrieval must be vector, bm25, or compare")
        ranking_model = validate_ranking_model(ranking_model)
        try:
            parsed_mode = Mode(mode)
        except ValueError as exc:
            raise ValueError("mode must be list, summarize, or generate") from exc
        request = SearchRequest(
            query=query,
            site=None if site in (None, "", "all") else site,
            mode=parsed_mode,
            previous_queries=tuple(previous_queries or ()),
            max_results=max_results or services.config.max_results,
            min_score=services.config.min_score,
            retrieval=retrieval,
            ranking_model=ranking_model,
        )
        items: list[dict[str, Any]] = []
        notices: list[str] = []
        answer: str | None = None
        usage: dict[str, Any] | None = None
        effective_query = query
        async for event in search(request, services):
            if event.type == "result":
                items.extend(event.data)
            elif event.type == "nlws":
                answer = event.data.get("answer")
            elif event.type == "decontextualized_query":
                effective_query = str(event.data)
            elif event.type == "usage":
                usage = event.data
            elif event.type in ("intermediate_message", "error") and event.data:
                notices.append(str(event.data))
            if event.type in ("candidate", "result", "decontextualized_query"):
                with contextlib.suppress(Exception):
                    await ctx.info(
                        json.dumps(
                            {"type": event.type, "data": event.data},
                            separators=(",", ":"),
                        )
                    )
        result: dict[str, Any] = {
            "query": query,
            "effective_query": effective_query,
            "items": items,
            "notices": notices,
        }
        if answer:
            result["answer"] = answer
        if usage:
            result["usage"] = usage
        return result

    return mcp
