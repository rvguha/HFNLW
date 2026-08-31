"""Corpus directory layout and small I/O helpers.

Every stage reads and writes through this module so the on-disk contract in the
design doc (s9.1) is stated in exactly one place.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def repo_slug(repo_id: str) -> str:
    """Filesystem-safe name for a repo id. Reversible: '/' is never legal in a
    repo namespace or name, so '__' can only have come from the separator."""
    return repo_id.replace("/", "__")


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------
    @property
    def snapshot(self) -> Path:
        return self.root / "manifests" / "snapshot.json"

    @property
    def selection(self) -> Path:
        return self.root / "manifests" / "selection.jsonl"

    @property
    def candidates(self) -> Path:
        return self.root / "manifests" / "candidates.jsonl"

    def raw_api(self, repo_id: str) -> Path:
        return self.root / "raw" / "api" / f"{repo_slug(repo_id)}.json"

    def raw_card(self, repo_id: str) -> Path:
        return self.root / "raw" / "cards" / f"{repo_slug(repo_id)}.md"

    def clean_card(self, repo_id: str) -> Path:
        return self.root / "processed" / "clean_cards" / f"{repo_slug(repo_id)}.json"

    @property
    def unusable(self) -> Path:
        """Repositories with no usable card, remembered across runs.

        A repository with no README does not acquire one because we selected it
        again. Recording the terminal states lets the next selection pass over
        them and fill the quota with candidates that can actually be used.
        """
        return self.root / "manifests" / "unusable.jsonl"

    @property
    def fetch_log(self) -> Path:
        return self.root / "manifests" / "fetch-log.jsonl"

    @property
    def normalized(self) -> Path:
        return self.root / "normalized" / "models.jsonl"

    @property
    def families(self) -> Path:
        return self.root / "normalized" / "families.json"

    @property
    def enriched(self) -> Path:
        return self.root / "enriched" / "models.jsonl"

    @property
    def quarantine(self) -> Path:
        return self.root / "enriched" / "quarantine.jsonl"

    def enrichment_cache(self, key: str) -> Path:
        return self.root / "cache" / "enrichment" / f"{key}.json"

    @property
    def nlweb(self) -> Path:
        return self.root / "nlweb" / "models.jsonl"

    @property
    def validation_report(self) -> Path:
        return self.root / "reports" / "validation.json"

    @property
    def quality_report(self) -> Path:
        return self.root / "reports" / "quality-report.json"

    @property
    def review_sample(self) -> Path:
        return self.root / "reports" / "review-sample.csv"

    def judgement(self, key: str) -> Path:
        """One judge's verdict on one query, content-addressed.

        Judgements are bought, not computed, so they are cached like the
        enrichment they resemble. Overwriting a reference set cost this project
        $31.73 of paid Opus and Sol verdicts; nothing here should be payable
        twice for the same question.
        """
        return self.root / "cache" / "judgements" / f"{key}.json"

    @property
    def golden_set(self) -> Path:
        return self.root / "reports" / "golden-set.json"

    @property
    def retrieval_report(self) -> Path:
        return self.root / "reports" / "retrieval-eval.json"

    # -- io ------------------------------------------------------------------
    def write_json(self, path: Path, obj: Any) -> Path:
        return write_json(path, obj)

    def read_json(self, path: Path) -> Any:
        return json.loads(path.read_text())

    def write_jsonl(self, path: Path, rows: Iterable[Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _atomic(path) as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def append_jsonl(self, path: Path, rows: Iterable[Any]) -> Path:
        """Decision logs are append-only: an exclusion is never erased (s1.2)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def read_jsonl(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return iter(())
        return _iter_jsonl(path)


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _atomic(path) as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return path


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class _atomic:
    """Write to a temp file in the target directory, then rename. A killed run
    leaves either the previous file or the new one, never a half-written stage
    output that the next stage would happily read."""

    def __init__(self, path: Path):
        self.path = path
        self.tmp: Any = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        self.name = name
        self.tmp = os.fdopen(fd, "w", encoding="utf-8")
        return self.tmp

    def __exit__(self, exc_type, exc, tb):
        self.tmp.close()
        if exc_type is None:
            os.replace(self.name, self.path)
        else:
            os.unlink(self.name)
        return False
