from __future__ import annotations

import asyncio
import json
import math
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RateLimitExceeded(Exception):
    reason: str
    retry_after: int

    def __str__(self) -> str:
        return self.reason


class RateLimitLease:
    def __init__(self, limiter: RateLimiter):
        self._limiter = limiter
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._limiter.release()


class RateLimiter:
    """Process-local sliding-window and concurrency limits for paid queries."""

    def __init__(
        self,
        per_client_per_minute: int,
        global_per_hour: int,
        global_per_day: int,
        max_concurrent: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.per_client_per_minute = per_client_per_minute
        self.global_per_hour = global_per_hour
        self.global_per_day = global_per_day
        self.max_concurrent = max_concurrent
        self._clock = clock
        self._lock = asyncio.Lock()
        self._client_events: dict[str, deque[float]] = defaultdict(deque)
        self._hour_events: deque[float] = deque()
        self._day_events: deque[float] = deque()
        self._active = 0

    async def acquire(self, client: str) -> RateLimitLease:
        now = self._clock()
        async with self._lock:
            if self.max_concurrent > 0 and self._active >= self.max_concurrent:
                raise RateLimitExceeded("Too many queries are already running.", 1)
            global_checks = (
                (self._hour_events, 3600, self.global_per_hour, "Hourly query budget reached."),
                (self._day_events, 86400, self.global_per_day, "Daily query budget reached."),
            )
            for events, window, limit, reason in global_checks:
                self._prune(events, now - window)
                if limit > 0 and len(events) >= limit:
                    retry_after = max(1, math.ceil(events[0] + window - now))
                    raise RateLimitExceeded(reason, retry_after)

            client_events = self._client_events.get(client)
            if client_events is None:
                client_events = deque()
            self._prune(client_events, now - 60)
            if (
                self.per_client_per_minute > 0
                and len(client_events) >= self.per_client_per_minute
            ):
                retry_after = max(1, math.ceil(client_events[0] + 60 - now))
                raise RateLimitExceeded("Client query limit reached.", retry_after)

            if len(self._client_events) > 1000:
                self._client_events = defaultdict(
                    deque,
                    {
                        key: events
                        for key, events in self._client_events.items()
                        if events and events[-1] > now - 60
                    },
                )
            self._client_events[client] = client_events
            client_events.append(now)
            self._hour_events.append(now)
            self._day_events.append(now)
            self._active += 1
        return RateLimitLease(self)

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)

    @staticmethod
    def _prune(events: deque[float], cutoff: float) -> None:
        while events and events[0] <= cutoff:
            events.popleft()


class PaidQueryRateLimitMiddleware:
    """Limit direct and MCP ask calls without charging handshakes or tool discovery."""

    def __init__(
        self,
        app,
        limiter: RateLimiter,
        trusted_proxy_ips: tuple[str, ...] = (),
    ):
        self.app = app
        self.limiter = limiter
        self.trusted_proxy_ips = frozenset(trusted_proxy_ips)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        body: bytes | None = None
        if scope.get("path", "").rstrip("/") == "/mcp" and scope.get("method") == "POST":
            original_receive = receive
            body = await _read_body(receive)
            receive = _replay_body(body, original_receive)

        if not _is_paid_query(scope, body):
            await self.app(scope, receive, send)
            return

        client = _client_key(scope, self.trusted_proxy_ips)
        try:
            lease = await self.limiter.acquire(client)
        except RateLimitExceeded as exc:
            await _send_rate_limit(send, exc)
            return

        async def limited_send(message: dict[str, Any]) -> None:
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                await lease.release()

        try:
            await self.app(scope, receive, limited_send)
        finally:
            await lease.release()


def _is_paid_query(scope: dict[str, Any], body: bytes | None) -> bool:
    path = scope.get("path", "").rstrip("/")
    if path == "/ask":
        return True
    if path != "/mcp" or body is None:
        return False
    try:
        message = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    messages = message if isinstance(message, list) else [message]
    return any(
        isinstance(item, dict)
        and item.get("method") == "tools/call"
        and isinstance(item.get("params"), dict)
        and item["params"].get("name") == "ask"
        for item in messages
    )


def _client_key(scope: dict[str, Any], trusted_proxy_ips: frozenset[str]) -> str:
    peer = scope.get("client")
    peer_ip = str(peer[0]) if peer else "unknown"
    if peer_ip not in trusted_proxy_ips:
        return f"ip:{peer_ip}"
    headers = {key.lower(): value for key, value in scope.get("headers", ())}
    forwarded = headers.get(b"x-forwarded-for", b"").decode("latin-1").strip()
    return f"ip:{forwarded or peer_ip}"


async def _read_body(receive) -> bytes:
    chunks = []
    more = True
    while more:
        message = await receive()
        if message["type"] != "http.request":
            continue
        chunks.append(message.get("body", b""))
        more = message.get("more_body", False)
    return b"".join(chunks)


def _replay_body(
    body: bytes,
    original_receive: Callable[[], Awaitable[dict[str, Any]]],
) -> Callable[[], Awaitable[dict[str, Any]]]:
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await original_receive()

    return receive


async def _send_rate_limit(send, error: RateLimitExceeded) -> None:
    body = json.dumps(
        {
            "error": "Rate limit exceeded",
            "detail": error.reason,
            "retry_after": error.retry_after,
        },
        separators=(",", ":"),
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"retry-after", str(error.retry_after).encode()),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
