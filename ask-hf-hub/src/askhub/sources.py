"""Remote corpus sources.

The corpus used to live in the repository, so updating a feed meant shipping a
release. Instead the repository carries a manifest naming where each collection
is fetched from, and the server pulls them at startup and on a timer.

Freshness is checked with conditional requests: the ETag and Last-Modified of
each fetch are recorded, and a later check that returns 304 costs one round trip
and no download. A source is re-read, and re-embedded, only when its bytes
actually change.

The manifest also declares which kind each collection is, which is what the web
UI's scope filter uses. Declaring it beats inferring it: adding a second
collection of the same kind should not mean editing code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import yaml

from .adapters import SUPPORTED_SUFFIXES

logger = logging.getLogger(__name__)

MANIFEST_NAME = "sources.yaml"
# Section names in the manifest. Each is also an aggregate scope in the UI.
KINDS = ("models",)
STATE_NAME = ".sources-state.json"
MAX_BYTES = 128 * 1024 * 1024
TIMEOUT = 120.0


class ManifestError(Exception):
    """The manifest is missing, malformed, or names nothing usable."""


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    url: str
    filename: str
    kind: str

    @property
    def is_remote(self) -> bool:
        """False for a collection bundled into the installed application.

        A local entry is never fetched. It appears in the manifest so that it
        still declares its kind, and so prune() knows the file belongs there.
        """
        return bool(self.url)


@dataclass(slots=True)
class FetchOutcome:
    source: Source
    path: Path
    changed: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class RefreshReport:
    outcomes: list[FetchOutcome] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(o.changed for o in self.outcomes)

    @property
    def failures(self) -> list[FetchOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def summary(self) -> str:
        changed = sum(1 for o in self.outcomes if o.changed)
        failed = len(self.failures)
        return (
            f"{len(self.outcomes)} sources, {changed} changed, {failed} failed"
        )


def load_manifest(path: Path) -> list[Source]:
    """Read the manifest naming where each collection comes from.

    Shape:

        models:
          - name: huggingface
            file: huggingface.jsonl      # bundled with the app; never fetched
          - name: huggingface_nightly
            url: https://example.org/models.jsonl

    An entry states either a `url`, in which case it is fetched and refreshed on
    the timer, or a `file` alone, in which case it is read from the installed
    application's bundled corpus. `file` may also accompany a `url` to override
    the download name, which otherwise defaults to the name plus the URL's suffix.
    """
    try:
        payload = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ManifestError(f"no manifest at {path}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(payload, dict):
        raise ManifestError(f"{path} must be a mapping of {' and '.join(KINDS)}")

    unknown = sorted(set(payload) - set(KINDS))
    if unknown:
        raise ManifestError(
            f"{path} has unknown section(s): {', '.join(unknown)}. "
            f"Expected {' and '.join(KINDS)}"
        )

    sources: list[Source] = []
    seen: set[str] = set()
    filenames: dict[str, str] = {}
    for kind in KINDS:
        entries = payload.get(kind)
        if entries is None:
            entries = []
        # `or []` would quietly accept a mapping here and report the confusing
        # "lists no sources" instead of naming the actual mistake.
        if not isinstance(entries, list):
            raise ManifestError(
                f"{path}: section {kind!r} must be a list, got {type(entries).__name__}"
            )
        for index, entry in enumerate(entries, 1):
            if not isinstance(entry, dict):
                raise ManifestError(f"{path}: {kind} entry {index} is not a mapping")
            url = str(entry.get("url") or "").strip()
            name = str(entry.get("name") or "").strip()
            # `filename` is the older spelling of `file`; both are accepted.
            explicit = str(entry.get("file") or entry.get("filename") or "").strip()
            if not name:
                raise ManifestError(f"{path}: {kind} entry {index} has no name")
            if not url and not explicit:
                raise ManifestError(
                    f"{path}: {kind} entry {name!r} has neither a url to fetch "
                    f"nor a file to read from the data directory"
                )
            if name in seen:
                raise ManifestError(f"{path} names {name!r} more than once")
            seen.add(name)
            filename = explicit or _default_filename(name, url)
            _check_filename(path, kind, name, filename)
            if filename in filenames:
                raise ManifestError(
                    f"{path}: {name!r} and {filenames[filename]!r} both write "
                    f"{filename!r}. Distinct sources need distinct files - they would "
                    f"race on the same download and overwrite each other."
                )
            filenames[filename] = name
            sources.append(Source(name=name, url=url, filename=filename, kind=kind))

    if not sources:
        raise ManifestError(f"{path} lists no sources")
    return sources


def kinds_by_collection(sources: list[Source]) -> dict[str, str]:
    """Map each collection name to its manifest section."""
    return {source.name: source.kind for source in sources}


def _check_filename(path: Path, kind: str, name: str, filename: str) -> None:
    """Reject a filename that would write outside the data directory.

    The manifest is trusted-ish, but `file: ../../etc/thing` or an absolute path
    silently escapes the corpus, and a dotfile is skipped by the directory scan
    so the collection would go missing rather than fail loudly.
    """
    bad = None
    if PurePosixPath(filename).is_absolute() or (len(filename) > 1 and filename[1] == ":"):
        bad = "is an absolute path"
    elif "/" in filename or "\\" in filename:
        bad = "contains a path separator"
    elif filename in (".", "..") or filename.startswith("."):
        bad = "is a dot name"
    elif Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
        bad = f"has suffix {Path(filename).suffix!r}, which is not one of " \
              f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
    if bad:
        raise ManifestError(f"{path}: {kind} entry {name!r} file {filename!r} {bad}")


def _default_filename(name: str, url: str) -> str:
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    suffix = ""
    for candidate in (".xml", ".rss", ".atom", ".json", ".jsonl"):
        if tail.lower().endswith(candidate):
            suffix = candidate
            break
    return f"{name}{suffix or '.xml'}"


# --------------------------------------------------------------------------
# fetch state, so a check can be conditional
# --------------------------------------------------------------------------


def _read_state(path: Path) -> dict[str, dict[str, str]]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(path: Path, state: dict[str, dict[str, str]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=1, sort_keys=True))
        temporary.replace(path)
    except OSError as exc:
        logger.warning("could not record source state: %s", exc)


async def fetch_all(
    sources: list[Source],
    directory: Path,
    *,
    client: httpx.AsyncClient | None = None,
    concurrency: int = 6,
) -> RefreshReport:
    """Fetch every source into `directory`, skipping those that have not changed.

    The file operations here are small, local, and run once per refresh; moving
    them to a thread would cost more than it saves.
    """
    directory.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240
    state_path = directory / STATE_NAME
    state = _read_state(state_path)
    limit = asyncio.Semaphore(concurrency)
    remote = [source for source in sources if source.is_remote]

    owned = client is None
    client = client or httpx.AsyncClient(
        follow_redirects=True, timeout=TIMEOUT, headers={"User-Agent": "ask-hf-hub/1.0"}
    )
    try:
        outcomes = await asyncio.gather(
            *(_fetch_one(source, directory, state, client, limit) for source in remote)
        )
    finally:
        if owned:
            await client.aclose()

    _write_state(state_path, state)
    return RefreshReport(list(outcomes))


async def _fetch_one(
    source: Source,
    directory: Path,
    state: dict[str, dict[str, str]],
    client: httpx.AsyncClient,
    limit: asyncio.Semaphore,
) -> FetchOutcome:
    path = directory / source.filename
    previous = state.get(source.name, {})
    headers: dict[str, str] = {}
    # Only ask for the body if it has changed since we last looked.
    if path.is_file():
        if etag := previous.get("etag"):
            headers["If-None-Match"] = etag
        if modified := previous.get("last_modified"):
            headers["If-Modified-Since"] = modified

    async with limit:
        try:
            response = await client.get(source.url, headers=headers)
        except httpx.HTTPError as exc:
            return FetchOutcome(source, path, False, f"{type(exc).__name__}: {exc}")

    if response.status_code == 304 and path.is_file():
        logger.debug("%s unchanged (304)", source.name)
        return FetchOutcome(source, path, False)
    if response.status_code >= 400:
        return FetchOutcome(source, path, False, f"HTTP {response.status_code}")

    body = response.content
    if len(body) > MAX_BYTES:
        return FetchOutcome(source, path, False, f"{len(body)} bytes exceeds the limit")
    if not body.strip():
        return FetchOutcome(source, path, False, "empty response")

    digest = hashlib.sha256(body).hexdigest()
    # A server may ignore the conditional headers and send 200 anyway, so
    # compare content before declaring a change and paying to re-embed.
    if previous.get("sha256") == digest and path.is_file():
        _remember(state, source, response, digest)
        return FetchOutcome(source, path, False)

    try:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(body)
        temporary.replace(path)
    except OSError as exc:
        return FetchOutcome(source, path, False, f"could not write {path.name}: {exc}")

    _remember(state, source, response, digest)
    logger.info("%s updated (%.0f KB)", source.name, len(body) / 1024)
    return FetchOutcome(source, path, True)


def _remember(
    state: dict[str, dict[str, str]],
    source: Source,
    response: httpx.Response,
    digest: str,
) -> None:
    entry: dict[str, Any] = {"sha256": digest, "url": source.url}
    if etag := response.headers.get("etag"):
        entry["etag"] = etag
    if modified := response.headers.get("last-modified"):
        entry["last_modified"] = modified
    state[source.name] = entry


def prune(directory: Path, sources: list[Source]) -> list[str]:
    """Delete downloaded files no longer named by the manifest."""
    keep = {source.filename for source in sources} | {STATE_NAME}
    removed: list[str] = []
    if not directory.is_dir():
        return removed
    for path in directory.iterdir():
        if path.is_file() and path.name not in keep and not path.name.startswith("."):
            try:
                path.unlink()
                removed.append(path.name)
            except OSError:
                pass
    return removed
