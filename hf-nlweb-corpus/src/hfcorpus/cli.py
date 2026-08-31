"""Command line entry point.

Stages run in the order of design doc s9 and can be run individually or as a
whole build. Every stage is resumable: outputs are content-addressed or keyed by
(repo_id, SHA), so re-running skips work whose inputs have not changed.

    hfcorpus build --profile pilot          # everything
    hfcorpus select --profile pilot         # one stage
    hfcorpus evaluate                       # golden queries against the emitted corpus
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import emit as emit_mod
from . import evaluate as evaluate_mod
from . import families as families_mod
from . import goldenset as goldenset_mod
from . import normalize as normalize_mod
from . import report as report_mod
from . import select as select_mod
from . import validate as validate_mod
from .cards import CleanCard, clean_card, is_boilerplate, placeholder_count, select_sections
from .config import load_policy, load_taxonomy
from .enrich import Enricher, effective_prompt_version, enrichment_key, make_backend
from .evidence import validate as validate_evidence_claims
from .hub import Hub
from .provenance import create_snapshot, record_stage, utc_now
from .store import Store

STAGES = ["select", "fetch", "normalize", "families", "enrich", "emit", "validate",
          "report", "evaluate"]


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file without overriding the real
    environment, so an exported credential always wins over a checked-out file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------- helpers ---
def _context(args: argparse.Namespace):
    policy = load_policy(args.config, args.profile)
    taxonomy = load_taxonomy(args.taxonomy)
    store = Store(args.corpus)
    return policy, taxonomy, store


def _load_snapshot(store: Store) -> dict[str, Any]:
    if not store.snapshot.exists():
        raise SystemExit(f"no snapshot at {store.snapshot}; run `hfcorpus select` first")
    return store.read_json(store.snapshot)


def _included(store: Store) -> list[dict[str, Any]]:
    """Latest decision per repository, filtered to the set still in play.

    A repository quarantined for a transient fetch failure stays in play: the
    quarantine records that we could not retrieve it, not that it does not
    belong. Re-running fetch retries exactly those, and everything already held
    is skipped from cache.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in store.read_jsonl(store.selection):
        latest[row["repo_id"]] = row
    return [r for r in latest.values()
            if r["decision"] == "included"
            or (r["decision"] == "quarantined" and r["reason_code"] == "quarantined_fetch")]


def _suppressed(store: Store) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in store.read_jsonl(store.selection):
        latest[row["repo_id"]] = row
    return [r for r in latest.values() if r["decision"] == "suppressed"]


def _clean_card(store: Store, repo_id: str) -> CleanCard | None:
    path = store.clean_card(repo_id)
    if not path.exists():
        return None
    return CleanCard.from_json(store.read_json(path))


# ----------------------------------------------------------------- stages ---
def stage_select(args: argparse.Namespace) -> None:
    policy, taxonomy, store = _context(args)
    hub = Hub(token=os.environ.get("HF_TOKEN"), max_attempts=policy.card["max_attempts"])
    snapshot = create_snapshot(policy, taxonomy, hub.client_version, args.snapshot_id)
    log(f"snapshot {snapshot['snapshot_id']}: discovering candidates")

    candidates = select_mod.discover(policy, hub, log=log)
    unusable = {row["repo_id"] for row in store.read_jsonl(store.unusable)}
    if unusable:
        before = len(candidates)
        candidates = [c for c in candidates if c.repo_id not in unusable]
        log(f"  skipped {before - len(candidates)} candidates already known to have no card")
    log(f"  {len(candidates)} distinct candidates")
    store.write_jsonl(store.candidates, [c.to_json() for c in candidates])

    features = select_mod.deterministic_features(policy, candidates)
    scores = select_mod.score(policy, candidates, features)
    decisions = select_mod.select(policy, candidates, features, scores)

    rows = [{
        "stage": "selection",
        "decided_at": utc_now(),
        "policy_version": policy.policy_version,
        "snapshot_id": snapshot["snapshot_id"],
        **asdict(d),
    } for d in decisions]
    if store.selection.exists() and not args.append:
        store.selection.unlink()
    store.append_jsonl(store.selection, rows)

    included = sum(1 for d in decisions if d.decision == "included")
    log(f"  included {included}, suppressed "
        f"{sum(1 for d in decisions if d.decision == 'suppressed')}, excluded "
        f"{sum(1 for d in decisions if d.decision == 'excluded')}")
    record_stage(snapshot, "select", candidates=len(candidates), included=included)
    store.write_json(store.snapshot, snapshot)


def stage_fetch(args: argparse.Namespace) -> None:
    policy, _taxonomy, store = _context(args)
    snapshot = _load_snapshot(store)
    hub = Hub(token=os.environ.get("HF_TOKEN"), max_attempts=policy.card["max_attempts"])
    targets = _included(store)
    if args.limit:
        targets = targets[: args.limit]
    log(f"fetching metadata and cards for {len(targets)} repositories")

    max_bytes = int(policy.card["max_download_bytes"])
    min_chars = int(policy.card["min_clean_chars"])
    max_placeholders = int(policy.card.get("max_placeholders", 5))

    def work(row: dict[str, Any]) -> dict[str, Any]:
        repo_id = row["repo_id"]
        api_path = store.raw_api(repo_id)
        clean_path = store.clean_card(repo_id)

        # Already held at a known revision: skip without touching the API. This
        # is what makes a larger profile, or a retry after a rate limit, cost
        # only the repositories it does not already have -- at 500 anonymous
        # requests per 300s, re-fetching what we hold is what trips the limit.
        # The metadata is then as of the earlier run; `--force` refreshes it.
        if api_path.exists() and clean_path.exists() and not args.force:
            existing = store.read_json(api_path)
            if store.read_json(clean_path).get("sha") == (existing.get("sha") or ""):
                return {"repo_id": repo_id, "sha": existing.get("sha", ""), "status": "cached",
                        "retrieved_at": utc_now()}

        try:
            raw = hub.fetch_raw_model(repo_id)
        except Exception as exc:
            return {"repo_id": repo_id, "status": "api_error", "error": str(exc)[:300],
                    "retrieved_at": utc_now()}
        store.write_json(api_path, raw)
        sha = raw.get("sha") or ""

        card_path = store.raw_card(repo_id)
        clean_path = store.clean_card(repo_id)
        if (card_path.exists() and clean_path.exists()
                and store.read_json(clean_path).get("sha") == sha and not args.force):
            return {"repo_id": repo_id, "sha": sha, "status": "cached",
                    "retrieved_at": utc_now()}

        fetched = hub.fetch_card(repo_id, sha, max_bytes)
        entry = {
            "repo_id": repo_id, "sha": sha, "status": fetched.status,
            "attempts": fetched.attempts, "bytes": fetched.bytes,
            "content_sha256": fetched.content_sha256, "retrieved_at": fetched.retrieved_at,
            "error": fetched.error,
        }
        if fetched.status != "ok":
            return entry
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(fetched.text, encoding="utf-8")
        cleaned = clean_card(repo_id, fetched.text)
        payload = cleaned.to_json()
        payload["sha"] = sha
        store.write_json(clean_path, payload)
        entry["clean_chars"] = len(cleaned.clean_text)
        entry["roles"] = sorted(cleaned.roles())
        entry["placeholders"] = placeholder_count(cleaned)
        if len(cleaned.clean_text) < min_chars or is_boilerplate(cleaned, max_placeholders):
            # Length alone does not catch the generated template: it runs to
            # ~2,900 characters of "[More Information Needed]" (s1.2).
            entry["status"] = "empty_card"
        return entry

    with ThreadPoolExecutor(max_workers=int(policy.card["concurrency"])) as pool:
        results = list(pool.map(work, targets))

    store.append_jsonl(store.fetch_log, results)

    # A transient failure is not a property of the repository. Rate limits and
    # network errors are quarantined so a re-run retries them; only a genuine
    # repository state becomes an exclusion (s9.3).
    TRANSIENT = ("api_error", "error")
    EMPTY = ("empty_card", "missing_readme")

    def demotion(status: str) -> tuple[str, str]:
        if status in TRANSIENT:
            return "quarantined", "quarantined_fetch"
        if status in EMPTY:
            return "excluded", "excluded_empty_card"
        return "excluded", "excluded_inaccessible"

    demotions = []
    for r in results:
        if r["status"] not in (*EMPTY, *TRANSIENT, "gated", "not_found", "too_large"):
            continue
        decision, reason = demotion(r["status"])
        demotions.append({
            "stage": "fetch",
            "decided_at": utc_now(),
            "snapshot_id": snapshot["snapshot_id"],
            "policy_version": policy.policy_version,
            "repo_id": r["repo_id"],
            "stratum": next((t["stratum"] for t in targets if t["repo_id"] == r["repo_id"]), ""),
            "decision": decision,
            "reason_code": reason,
            "score": 0.0,
            "features": {"card_status": r["status"], "error": r.get("error", "")[:200]},
        })
    if demotions:
        store.append_jsonl(store.selection, demotions)

    # Terminal card states are a property of the repository, not of this run.
    # Remembering them lets the next selection fill the quota with candidates
    # that have a card, instead of re-picking the same empty ones every time.
    known = {row["repo_id"] for row in store.read_jsonl(store.unusable)}
    terminal = [{"repo_id": r["repo_id"], "status": r["status"], "noted_at": utc_now()}
                for r in results
                if r["status"] in ("empty_card", "missing_readme", "gated", "not_found",
                                   "too_large") and r["repo_id"] not in known]
    if terminal:
        store.append_jsonl(store.unusable, terminal)
        log(f"  {len(terminal)} repositories recorded as having no usable card")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    log(f"  {counts}")
    record_stage(snapshot, "fetch", **counts)
    store.write_json(store.snapshot, snapshot)


def stage_normalize(args: argparse.Namespace) -> None:
    _policy, _taxonomy, store = _context(args)
    snapshot = _load_snapshot(store)
    fetch_status = {r["repo_id"]: r for r in store.read_jsonl(store.fetch_log)}
    rows: list[dict[str, Any]] = []
    for row in _included(store):
        repo_id = row["repo_id"]
        api_path = store.raw_api(repo_id)
        if not api_path.exists():
            continue
        raw = store.read_json(api_path)
        card = _clean_card(store, repo_id)
        status = fetch_status.get(repo_id, {}).get("status", "unknown")
        record = normalize_mod.normalize(
            raw, card, retrieved_at=fetch_status.get(repo_id, {}).get("retrieved_at", utc_now()),
            card_status="ok" if card else status,
        )
        record["internal"]["stratum"] = row.get("stratum", "")
        record["internal"]["selection_reason"] = row.get("reason_code", "")
        record["internal"]["selection_score"] = row.get("score", 0.0)
        rows.append(record)
    store.write_jsonl(store.normalized, rows)
    log(f"normalized {len(rows)} records")
    record_stage(snapshot, "normalize", records=len(rows))
    store.write_json(store.snapshot, snapshot)


def stage_families(args: argparse.Namespace) -> None:
    policy, _taxonomy, store = _context(args)
    snapshot = _load_snapshot(store)
    records = list(store.read_jsonl(store.normalized))
    variants = [{"repo_id": r["repo_id"]} for r in _suppressed(store)]
    assignments = families_mod.resolve(records, policy.patterns, variants)
    store.write_json(store.families, {k: asdict(v) for k, v in assignments.items()})
    primaries = sum(1 for a in assignments.values() if a.discovery_scope == "primary")
    log(f"resolved {len(assignments)} repositories into "
        f"{len({a.family_id for a in assignments.values()})} families "
        f"({primaries} primary, {len(assignments) - primaries} variants)")
    record_stage(snapshot, "families", repositories=len(assignments),
                 families=len({a.family_id for a in assignments.values()}))
    store.write_json(store.snapshot, snapshot)


def stage_enrich(args: argparse.Namespace) -> None:
    policy, taxonomy, store = _context(args)
    snapshot = _load_snapshot(store)
    settings = policy.enrichment
    backend = make_backend(settings, backend=args.backend, model=args.model)
    enricher = Enricher(backend)
    model = enricher.model
    log(f"enriching via {backend.name} ({backend.model})")
    max_chars = int(policy.card["max_clean_chars_for_enrichment"])

    records = list(store.read_jsonl(store.normalized))
    if args.limit:
        records = records[: args.limit]
    vocabulary = [str(term["name"]) for term in taxonomy.terms]

    def work(record: dict[str, Any]) -> dict[str, Any] | None:
        repo_id = record["internal"]["repo_id"]
        card = _clean_card(store, repo_id)
        if card is None:
            return None
        text, sections = select_sections(card, max_chars)
        key = enrichment_key(text, record["item"], effective_prompt_version(vocabulary), model)
        cache_path = store.enrichment_cache(key)
        if cache_path.exists() and not args.force:
            cached = store.read_json(cache_path)
            raw_enrichment, provenance = cached["enrichment"], cached["provenance"]
            provenance = {**provenance, "cache_hit": True}
        else:
            result = enricher.enrich(repo_id, record["item"], text, sections, vocabulary)
            if result.status != "ok":
                return {"quarantined": True, "repo_id": repo_id, "error": result.error,
                        "enrichment_key": result.key, "reason": "quarantined_enrichment"}
            raw_enrichment, provenance = result.enrichment, result.provenance
            store.write_json(cache_path, {"repo_id": repo_id, "enrichment": raw_enrichment,
                                          "provenance": provenance})
        validated = validate_evidence_claims(raw_enrichment, card, record["item"], taxonomy)
        internal = dict(record["internal"])
        internal["enrichment"] = provenance
        internal["dropped_claims"] = validated.dropped
        internal["enrichment_warnings"] = validated.warnings
        internal["raw_enrichment"] = raw_enrichment
        return {"item": record["item"], "enrichment": validated.to_json(), "internal": internal}

    with ThreadPoolExecutor(max_workers=int(settings.get("concurrency", 4))) as pool:
        results = [r for r in pool.map(work, records) if r is not None]

    enriched = [r for r in results if not r.get("quarantined")]
    quarantined = [r for r in results if r.get("quarantined")]
    store.write_jsonl(store.enriched, enriched)
    if quarantined:
        store.append_jsonl(store.quarantine, quarantined)
        store.append_jsonl(store.selection, [{
            "stage": "enrich", "decided_at": utc_now(), "snapshot_id": snapshot["snapshot_id"],
            "policy_version": policy.policy_version, "repo_id": q["repo_id"],
            "stratum": "", "decision": "quarantined",
            "reason_code": "quarantined_enrichment", "score": 0.0,
            "features": {"error": q["error"][:200]},
        } for q in quarantined])

    dropped = sum(len(r["internal"]["dropped_claims"]) for r in enriched)
    log(f"enriched {len(enriched)} records ({len(quarantined)} quarantined, "
        f"{dropped} unsupported claims dropped)")
    record_stage(snapshot, "enrich", enriched=len(enriched), quarantined=len(quarantined),
                 dropped_claims=dropped, model=model, backend=backend.name)
    store.write_json(store.snapshot, snapshot)


def stage_emit(args: argparse.Namespace) -> None:
    _policy, taxonomy, store = _context(args)
    snapshot = _load_snapshot(store)
    families = store.read_json(store.families) if store.families.exists() else {}
    assignments = {k: families_mod.FamilyAssignment(**v) for k, v in families.items()}
    variants = families_mod.variants_by_family(assignments)

    enriched = {r["internal"]["repo_id"]: r for r in store.read_jsonl(store.enriched)}
    abstracts = emit_mod.load_abstracts(store.root / "external" / "arxiv-abstracts.json")
    if abstracts:
        log(f"carrying {len(abstracts)} arXiv abstracts into citations")
    items: list[dict[str, Any]] = []
    for record in store.read_jsonl(store.normalized):
        repo_id = record["internal"]["repo_id"]
        assignment = assignments.get(repo_id)
        enrichment = (enriched.get(repo_id) or {}).get("enrichment")
        family_variants = variants.get(assignment.family_id, []) if assignment else []
        items.append(emit_mod.emit(record, enrichment, assignment, taxonomy,
                                   [v for v in family_variants if v != repo_id],
                                   abstracts))
    store.write_jsonl(store.nlweb, items)

    # A manifest beside the items, so whatever serves this corpus can say which
    # corpus it is. The build has always known; nothing downstream could ask.
    # Two answers from the same question are only comparable if you can tell
    # whether the corpus moved between them.
    store.write_json(store.nlweb.parent / "manifest.json", {
        "snapshot_id": snapshot["snapshot_id"],
        "items": len(items),
        "built_at": utc_now(),
        "source": snapshot.get("source"),
        "profile": snapshot.get("selection_profile"),
        "versions": {
            "pipeline": snapshot.get("pipeline_version"),
            "selection_policy": snapshot.get("selection_policy_version"),
            "normalizer": snapshot.get("normalizer_version"),
            "enrichment_prompt": snapshot.get("enrichment_prompt_version"),
            "enrichment_model": snapshot.get("enrichment_model"),
            "taxonomy": snapshot.get("taxonomy_version"),
            "huggingface_hub": snapshot.get("huggingface_hub_version"),
        },
    })
    log(f"emitted {len(items)} NLWeb items -> {store.nlweb}")
    record_stage(snapshot, "emit", items=len(items))
    store.write_json(store.snapshot, snapshot)


def stage_validate(args: argparse.Namespace) -> None:
    _policy, _taxonomy, store = _context(args)
    snapshot = _load_snapshot(store)
    schema_errors: dict[str, list[str]] = {}
    evidence_errors: dict[str, list[str]] = {}
    items = list(store.read_jsonl(store.nlweb))
    claims = 0
    for item in items:
        repo_id = item.get("hf:repository", "")
        errors = validate_mod.validate_item(item)
        if errors:
            schema_errors[repo_id] = errors
        claims += sum(len(item.get(key, [])) for key in validate_mod.EVIDENCE_PROPERTIES)
        card = _clean_card(store, repo_id)
        failures = validate_mod.validate_evidence(item, card)
        if failures:
            evidence_errors[repo_id] = failures

    total = max(1, len(items))
    report = {
        "items": len(items),
        "claims": claims,
        "schema_valid": len(items) - len(schema_errors),
        "schema_valid_fraction": round((len(items) - len(schema_errors)) / total, 4),
        "evidence_valid": len(items) - len(evidence_errors),
        "evidence_valid_fraction": round((len(items) - len(evidence_errors)) / total, 4),
        "schema_errors": schema_errors,
        "evidence_errors": evidence_errors,
    }
    store.write_json(store.validation_report, report)
    log(f"validation: {report['schema_valid']}/{len(items)} schema-valid, "
        f"{report['evidence_valid']}/{len(items)} evidence-valid")
    record_stage(snapshot, "validate", **{k: report[k] for k in
                                          ("items", "claims", "schema_valid", "evidence_valid")})
    store.write_json(store.snapshot, snapshot)


def stage_report(args: argparse.Namespace) -> None:
    policy, _taxonomy, store = _context(args)
    snapshot = _load_snapshot(store)
    items = list(store.read_jsonl(store.nlweb))
    internals: dict[str, dict[str, Any]] = {}
    for record in store.read_jsonl(store.normalized):
        internals[record["item"]["@id"]] = record["internal"]
    for record in store.read_jsonl(store.enriched):
        internals.setdefault(record["item"]["@id"], {}).update(record["internal"])
    provenance = [r["internal"]["enrichment"] for r in store.read_jsonl(store.enriched)
                  if r["internal"].get("enrichment")]
    decisions = list(store.read_jsonl(store.selection))
    families = store.read_json(store.families) if store.families.exists() else {}
    validation = store.read_json(store.validation_report) \
        if store.validation_report.exists() else {}

    quality = report_mod.build(items, internals, decisions, families, provenance,
                               {k: v for k, v in validation.items()
                                if not k.endswith("_errors")}, snapshot)
    store.write_json(store.quality_report, quality)
    n = report_mod.review_sample(items, decisions, policy.review_sample, store.review_sample)
    log(f"quality report -> {store.quality_report}; {n}-record review sample -> "
        f"{store.review_sample}")
    record_stage(snapshot, "report", review_sample=n)
    store.write_json(store.snapshot, snapshot)


def stage_evaluate(args: argparse.Namespace) -> None:
    _policy, _taxonomy, store = _context(args)
    items = list(store.read_jsonl(store.nlweb))
    if not items:
        raise SystemExit("no emitted items; run `hfcorpus emit` first")
    queries = evaluate_mod.load_queries(args.queries)
    result = evaluate_mod.evaluate(items, queries, k=args.k)
    store.write_json(store.retrieval_report, result)
    full = result["summary"]["full"]
    baseline = result["summary"]["metadata_only"]
    log(f"golden queries: {result['queries']} at k={args.k}")
    log(f"  full record     precision@k {full['precision_at_k']:.3f}  "
        f"recall@k {full['recall_at_k']:.3f}  traps {full['trap_rate']:.3f}")
    log(f"  metadata only   precision@k {baseline['precision_at_k']:.3f}  "
        f"recall@k {baseline['recall_at_k']:.3f}  traps {baseline['trap_rate']:.3f}")
    if result["queries_with_no_qualifying_record"]:
        log(f"  no qualifying record in corpus for: "
            f"{', '.join(result['queries_with_no_qualifying_record'])}")


def stage_goldenset(args: argparse.Namespace) -> None:
    _policy, _taxonomy, store = _context(args)
    queries = evaluate_mod.load_queries(args.queries)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if args.only:
        wanted = {q.strip() for q in args.only.split(",") if q.strip()}
        queries = [q for q in queries if q["id"] in wanted]
    if args.limit:
        queries = queries[: args.limit]
    log(f"judging {len(queries)} queries at width {args.width} with: {', '.join(judges)}")
    log("  a wide retrieval read by strong models, so relevance is judged from "
        "the record rather than from the field that retrieved it")

    snapshot = store.read_json(store.snapshot).get("snapshot_id", "") \
        if store.snapshot.exists() else ""
    result = goldenset_mod.build(args.endpoint, queries, judges, args.width,
                                 store=store, corpus=snapshot,
                                 full_matrix=args.full_matrix, log=log)
    # Never overwrite a reference set: each is written under its own name, and
    # golden-set.json points at the most recent. Overwriting one cost $31.73 of
    # paid judgements.
    stamp = "-".join(sorted(j.split("/")[-1] for j in judges))
    mode = "full" if args.full_matrix else "shortlist"
    store.write_json(
        store.golden_set.with_name(f"golden-set-{stamp}-w{args.width}-{mode}.json"), result)
    store.write_json(store.golden_set, result)
    log(f"golden set -> {store.golden_set}")
    log(f"  {result['queries']} queries | mean agreement {result['mean_agreement']} | "
        f"{result['mean_agreed_per_query']} agreed records per query | "
        f"${result['total_cost_usd']:.2f}")
    if result["incomplete"]:
        log(f"  INCOMPLETE -- a judge failed on {len(result['incomplete'])} queries: "
            f"{', '.join(result['incomplete'][:8])}"
            f"{' ...' if len(result['incomplete']) > 8 else ''}")
        log("  this set is not usable as a reference until those are re-run")
    if result["queries_with_no_relevant_record"]:
        log(f"  judged, but nothing relevant found: "
            f"{', '.join(result['queries_with_no_relevant_record'])}")


def stage_build(args: argparse.Namespace) -> None:
    for name in STAGES:
        if name == "enrich" and args.no_enrich:
            log("skipping enrich (--no-enrich)")
            continue
        log(f"== {name}")
        HANDLERS[name](args)


HANDLERS = {
    "select": stage_select,
    "fetch": stage_fetch,
    "normalize": stage_normalize,
    "families": stage_families,
    "enrich": stage_enrich,
    "emit": stage_emit,
    "validate": stage_validate,
    "report": stage_report,
    "evaluate": stage_evaluate,
    "goldenset": stage_goldenset,
    "build": stage_build,
}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(prog="hfcorpus", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="corpus", help="output directory (default: corpus)")
    parser.add_argument("--config", default=str(root / "config" / "selection.yaml"))
    parser.add_argument("--taxonomy", default=str(root / "config" / "taxonomy.yaml"))
    parser.add_argument("--profile", default="pilot",
                        help="pilot | corpus_v1 | demonstration | expansion")
    parser.add_argument("--queries", default=str(root / "tests" / "golden-queries.yaml"))
    parser.add_argument("--limit", type=int, default=0, help="cap records for a smoke run")
    parser.add_argument("--force", action="store_true", help="ignore caches for this stage")
    parser.add_argument("--append", action="store_true",
                        help="append to an existing selection log instead of starting a new one")
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--backend", default=None, choices=["anthropic", "openrouter"],
                        help="override the enrichment backend")
    parser.add_argument("--model", default=None, help="override the enrichment model")
    parser.add_argument("--no-enrich", action="store_true",
                        help="build without the LLM stage (deterministic records only)")
    parser.add_argument("--k", type=int, default=10, help="cutoff for golden-query metrics")
    parser.add_argument("--endpoint", default="http://localhost:8000/ask",
                        help="ask endpoint to judge against (goldenset)")
    parser.add_argument("--judges", default="openai/gpt-5.6-sol,anthropic/claude-opus-5",
                        help="comma-separated reference models (goldenset)")
    parser.add_argument("--full-matrix", action="store_true",
                        help="ask each judge for a verdict on every retrieved record, "
                             "not a shortlist (goldenset)")
    parser.add_argument("--only", default="",
                        help="comma-separated query ids to judge (goldenset)")
    parser.add_argument("--width", type=int, default=200,
                        help="how many records to retrieve and judge per query (goldenset)")
    parser.add_argument("stage", choices=sorted(HANDLERS), help="stage to run ('build' = all)")

    args = parser.parse_args(argv)
    load_env_file(Path.cwd() / ".env")
    load_env_file(root / ".env")
    HANDLERS[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
