"""Decontextualization: rewriting a conversational follow-up to stand alone.

This was a plugin registry with a Protocol, a decision record and a concurrent
executor, serving one live operation. Reading it meant reconstructing a general
handler architecture in order to understand a single branch, so it is now one
function. If a second pre-retrieval operation arrives, how it orders against
this one should decide the control flow then, from two concrete cases rather
than one speculative abstraction.
"""

from __future__ import annotations

import asyncio
import logging

from .models import SearchRequest
from .providers import LanguageModel

logger = logging.getLogger(__name__)


async def decontextualize(
    request: SearchRequest, llm: LanguageModel, timeout_seconds: float
) -> str | None:
    """Rewrite a follow-up as a standalone query, or None if it already stands.

    Fails open: a timeout or a provider error returns None, so the original
    query is searched rather than the request failing. A degraded rewrite is
    much better than a dead search.
    """
    try:
        output = await asyncio.wait_for(
            llm.structured(
                "Rewrite the current query as a standalone search query. "
                "Preserve every explicit constraint. "
                "Return JSON with standalone_query and changed.",
                {
                    "current_query": request.query,
                    "previous_queries": request.previous_queries[-5:],
                },
            ),
            timeout_seconds,
        )
        # Interpreting the response is inside the try on purpose: a provider that
        # returns a list or a string instead of a mapping is a provider failure,
        # and should degrade the same way a timeout does rather than fail the
        # request. CancelledError is a BaseException, so cancellation still
        # propagates and does not become "search the original query".
        replacement = str(output.get("standalone_query") or request.query).strip()
        changed = bool(output.get("changed")) and replacement != request.query
    except Exception as error:
        # No query content in the log; this line goes to shared operator output.
        logger.warning("decontextualization failed (%s); searching the original query",
                       type(error).__name__)
        return None
    return replacement if changed else None
