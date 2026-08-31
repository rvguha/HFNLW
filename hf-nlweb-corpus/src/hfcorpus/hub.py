"""Hugging Face Hub access.

Two access paths, deliberately:

* discovery and listing go through the official ``HfApi`` client (its version is
  pinned into the snapshot manifest); and
* the per-repository record is re-fetched as **verbatim JSON** through the
  client's own HTTP session, because "preserve these fields verbatim" (s3.1)
  means the bytes the Hub returned, not a re-serialisation of a typed object
  whose field set drifts with the client version.

Nothing here writes to the Hub, follows instructions found in card text, or
attempts to reach a gated or private repository.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

from huggingface_hub import HfApi, constants, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

# "Retry after 93 seconds" in a 429 body, when the header is absent.
RETRY_AFTER_TEXT = re.compile(r"retry after (\d+(?:\.\d+)?) seconds?", re.I)
# A rate-limit window can be minutes; anything past this is a stuck run.
MAX_BACKOFF_SECONDS = 180.0

# Validated against the Hub's own accepted-option list. The API rejects unknown
# expansions outright, which is the behaviour we want: a silently ignored field
# would mean silently missing provenance.
EXPAND_FIELDS = [
    "author",
    "baseModels",
    "cardData",
    "config",
    "createdAt",
    "disabled",
    "downloads",
    "downloadsAllTime",
    "evalResults",
    "gated",
    "inference",
    "lastModified",
    "library_name",
    "likes",
    "model-index",
    "pipeline_tag",
    "private",
    "safetensors",
    "sha",
    "tags",
    "trendingScore",
    "transformersInfo",
]

# Fields cheap enough to request for every candidate in an oversized pool.
# `evalResults` is deliberately absent: the client parses it eagerly during
# pagination and raises on repositories whose eval entries omit a dataset id,
# which would abort a whole discovery query. Evaluation metadata is read from
# the verbatim per-repository JSON instead, where nothing is parsed.
DISCOVERY_EXPAND = [
    "author",
    "baseModels",
    "cardData",
    "createdAt",
    "downloads",
    "downloadsAllTime",
    "gated",
    "lastModified",
    "library_name",
    "likes",
    "pipeline_tag",
    "private",
    "safetensors",
    "sha",
    "tags",
    "trendingScore",
]


@dataclass
class CardFetch:
    repo_id: str
    sha: str
    status: str          # ok | missing_readme | gated | not_found | too_large | error
    attempts: int
    bytes: int = 0
    content_sha256: str = ""
    text: str = ""
    retrieved_at: str = ""
    error: str = ""


@dataclass
class Hub:
    token: str | None = None
    max_attempts: int = 4
    api: HfApi = field(init=False)

    def __post_init__(self) -> None:
        self.api = HfApi(token=self.token)

    @property
    def client_version(self) -> str:
        from huggingface_hub import __version__

        return __version__

    # -- discovery -----------------------------------------------------------
    def list_models(self, query: dict[str, Any], limit: int) -> Iterator[Any]:
        """Run one discovery query. `query` mirrors the YAML query spec."""
        # `filter` carries Hub tags: library names, language codes, and free
        # tags all live in the same tag namespace.
        tags = list(query.get("tags") or [])
        if "library" in query:
            tags.append(str(query["library"]))
        if "language" in query:
            tags.append(str(query["language"]))

        kwargs: dict[str, Any] = {
            "limit": limit,
            "expand": DISCOVERY_EXPAND,
            "sort": query.get("sort", "downloads"),
        }
        if "pipeline_tag" in query:
            kwargs["pipeline_tag"] = query["pipeline_tag"]
        if "search" in query:
            kwargs["search"] = query["search"]
        if tags:
            kwargs["filter"] = tags
        return self._retry(lambda: iter(list(self.api.list_models(**kwargs))))

    # -- verbatim metadata ---------------------------------------------------
    def fetch_raw_model(self, repo_id: str) -> dict[str, Any]:
        """The Hub's own JSON for one repository, unmodified."""
        url = f"{constants.ENDPOINT}/api/models/{repo_id}"
        params = [("expand[]", f) for f in EXPAND_FIELDS]
        headers = build_hf_headers(token=self.token)

        def call() -> dict[str, Any]:
            response = get_session().get(url, params=params, headers=headers, timeout=30)
            hf_raise_for_status(response)
            return response.json()

        return self._retry(call)

    # -- model cards ---------------------------------------------------------
    def fetch_card(self, repo_id: str, sha: str, max_bytes: int) -> CardFetch:
        """Download README.md at a pinned revision. Never raises for an expected
        repository state -- missing, gated, and oversized cards are outcomes the
        corpus records, not failures that abort a run (s4.1, s9.3)."""
        started = _utc_now()
        attempts = 0
        last_error = ""
        while attempts < self.max_attempts:
            attempts += 1
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename="README.md",
                    repo_type="model",
                    revision=sha,
                    token=self.token,
                )
            except EntryNotFoundError:
                return CardFetch(repo_id, sha, "missing_readme", attempts, retrieved_at=started)
            except GatedRepoError:
                return CardFetch(repo_id, sha, "gated", attempts, retrieved_at=started)
            except RepositoryNotFoundError:
                return CardFetch(repo_id, sha, "not_found", attempts, retrieved_at=started)
            except (HfHubHTTPError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempts >= self.max_attempts:
                    break
                _sleep_backoff(attempts, retry_after(exc))
                continue

            with open(path, "rb") as fh:
                data = fh.read()
            if len(data) > max_bytes:
                return CardFetch(
                    repo_id, sha, "too_large", attempts, bytes=len(data),
                    content_sha256=hashlib.sha256(data).hexdigest(), retrieved_at=started,
                )
            return CardFetch(
                repo_id=repo_id,
                sha=sha,
                status="ok",
                attempts=attempts,
                bytes=len(data),
                content_sha256=hashlib.sha256(data).hexdigest(),
                # Retain the byte hash above; decode lossily for working text so a
                # single bad byte never costs us the whole card (s4.1).
                text=data.decode("utf-8", errors="replace"),
                retrieved_at=started,
            )
        return CardFetch(repo_id, sha, "error", attempts, retrieved_at=started, error=last_error)

    # -- internals -----------------------------------------------------------
    def _retry(self, call):
        attempts = 0
        while True:
            attempts += 1
            try:
                return call()
            except (HfHubHTTPError, OSError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable or attempts >= self.max_attempts:
                    raise
                _sleep_backoff(attempts, retry_after(exc))


def retry_after(exc: Exception) -> float | None:
    """Seconds the Hub asked us to wait, from the header or the message body.

    A 429 is not a guess to be backed off blindly: the Hub states the window it
    wants. Anonymous access allows 500 requests per 300s, so a rate-limited
    fetch is routinely told to wait ~90s -- far past any exponential schedule
    that starts at two seconds, which is why these were being abandoned.
    """
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    value = header.get("retry-after") or header.get("Retry-After")
    if value:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    match = RETRY_AFTER_TEXT.search(str(exc))
    return float(match.group(1)) if match else None


def _sleep_backoff(attempt: int, retry_after_seconds: float | None = None) -> None:
    if retry_after_seconds is not None:
        # Honour what the server asked for, plus jitter so a burst of workers
        # does not resume in lockstep and immediately re-trip the limit.
        time.sleep(min(MAX_BACKOFF_SECONDS, retry_after_seconds + random.random() * 5))
        return
    time.sleep(min(30.0, 2.0 ** attempt) * (0.5 + random.random()))


def _utc_now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
