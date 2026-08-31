"""Fetch abstracts for the arXiv papers this corpus cites.

Model cards cite the paper behind the model, and the corpus already carries
those citations as `ScholarlyArticle` nodes -- but only the URL. The abstract is
the one paragraph that states what the model does in the authors' own words,
which is exactly what a descriptive query is trying to match, and the enriched
one-sentence description cannot carry.

Fetching is separated from using: this module produces a keyed store of
abstracts and nothing else. Wiring them into retrieval is a later, separate
decision.

The store is written after every batch and reread on startup, so an interrupted
run resumes instead of re-fetching what it already paid for in wall time. arXiv
asks for no more than one request every three seconds and that is honoured.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
BATCH = 100
# arXiv's stated courtesy limit for the legacy API.
DELAY_SECONDS = 3.0
# A version suffix identifies a revision of the same paper; the abstract is
# keyed on the paper.
ARXIV_URL = re.compile(r"arxiv\.org/abs/([\w.\-/]+?)(?:v\d+)?$", re.IGNORECASE)
# Old-style (hep-th/9901001) and new-style (2212.04356) identifiers.
WELL_FORMED = re.compile(r"\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7}")


def extract_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    """arXiv identifiers cited by the given emitted records."""
    found: set[str] = set()
    for record in records:
        for citation in record.get("citation") or []:
            if not isinstance(citation, dict):
                continue
            url = str(citation.get("url") or citation.get("@id") or "").strip()
            match = ARXIV_URL.search(url)
            if match and WELL_FORMED.fullmatch(match.group(1)):
                found.add(match.group(1))
    return found


def _parse(payload: bytes) -> dict[str, dict[str, str]]:
    root = ET.fromstring(payload)
    out: dict[str, dict[str, str]] = {}
    for entry in root.findall(f"{ATOM}entry"):
        raw = (entry.findtext(f"{ATOM}id") or "").strip()
        match = ARXIV_URL.search(raw)
        if not match:
            continue
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        abstract = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
        if not abstract:
            # A withdrawn or malformed entry comes back with a title and no
            # summary. Recording it as empty would look like a successful fetch.
            continue
        out[match.group(1)] = {
            "title": title,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{match.group(1)}",
        }
    return out


def _request(ids: list[str], timeout: float) -> dict[str, dict[str, str]]:
    query = urllib.parse.urlencode({"id_list": ",".join(ids), "max_results": len(ids)})
    request = urllib.request.Request(
        f"{API}?{query}", headers={"User-Agent": "hf-nlweb-corpus/1.0 (abstract fetch)"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return _parse(response.read())


def fetch(
    ids: Iterable[str],
    store_path: Path,
    timeout: float = 60.0,
    log=print,
) -> dict[str, dict[str, str]]:
    """Fetch every missing abstract, checkpointing after each batch."""
    store: dict[str, dict[str, str]] = {}
    if store_path.is_file():
        store = json.loads(store_path.read_text())
        log(f"resuming: {len(store)} abstracts already stored")

    missing = sorted(set(ids) - set(store))
    if not missing:
        log("nothing to fetch")
        return store

    batches = [missing[start : start + BATCH] for start in range(0, len(missing), BATCH)]
    log(f"fetching {len(missing)} abstracts in {len(batches)} batches")
    store_path.parent.mkdir(parents=True, exist_ok=True)

    for number, batch in enumerate(batches, 1):
        try:
            found = _request(batch, timeout)
        except (urllib.error.URLError, ET.ParseError, TimeoutError) as error:
            # One bad batch must not discard the run's earlier work.
            log(f"  batch {number}/{len(batches)}: FAILED ({error}); continuing")
            continue
        store.update(found)
        store_path.write_text(json.dumps(store, indent=2, sort_keys=True))
        log(f"  batch {number}/{len(batches)}: +{len(found)}/{len(batch)} (store {len(store)})")
        if number < len(batches):
            time.sleep(DELAY_SECONDS)

    unresolved = sorted(set(missing) - set(store))
    if unresolved:
        log(f"{len(unresolved)} identifiers returned no abstract, e.g. {unresolved[:5]}")
    return store


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, action="append", required=True,
                        help="emitted .jsonl corpus to read citations from (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("corpus/external/arxiv-abstracts.json"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for path in args.corpus:
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ids = extract_ids(records)
    print(f"{len(records)} records, {len(ids)} distinct well-formed arXiv identifiers")
    store = fetch(ids, args.out)
    print(f"{len(store)} abstracts in {args.out}")


if __name__ == "__main__":
    main()
