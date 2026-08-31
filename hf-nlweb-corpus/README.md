# hf-nlweb-corpus

A reproducible pipeline that builds a grounded, provenance-preserving
[schema.org](https://schema.org) corpus of Hugging Face model repositories for an
NLWeb **ask** interface — "Ask the Hugging Face Hub".

The point is not to mirror the Hub. It is to assemble a small, representative,
traceable catalog that lets a conversational interface answer questions like
*"which models appear to know a lot about molecular biology?"* **and show the
evidence behind each recommendation**.

Implements the design in
[`docs/design/`](docs/design) — see *Building a Demonstration Corpus for Ask the
Hugging Face Hub*.

## What it produces

One JSON-LD `SoftwareApplication` item per model, where every non-trivial subject
claim is backed by a verbatim quote from the model card at a pinned repository
revision:

```jsonc
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "@id": "https://huggingface.co/acme/demo-1b",
  "name": "Demo 1B",
  "description": "Demo 1B is a 1.1B parameter biomedical language model.",
  "applicationSubCategory": ["Text Generation"],
  "about": [{"@type": "DefinedTerm", "name": "Biomedicine", "termCode": "biomedicine"}],
  "featureList": ["biomedical entity extraction"],
  "subjectOf": [{
    "@type": "Claim",
    "text": "Documented as trained on biomedicine data.",
    "appearance": {"url": "https://huggingface.co/acme/demo-1b/blob/<sha>/README.md#:~:text=trained%20on%20PubMed%20abstracts"},
    "additionalProperty": [
      {"@type": "PropertyValue", "propertyID": "corpus:confidence", "value": "high"},
      {"@type": "PropertyValue", "propertyID": "corpus:evidenceType", "value": "trained_on"},
      {"@type": "PropertyValue", "propertyID": "corpus:evidenceQuote", "value": "trained on PubMed abstracts"}
    ]
  }],
  "additionalProperty": [
    {"@type": "PropertyValue", "propertyID": "ml:parameterCount", "value": 1100000000,
     "valueReference": [{"@type": "PropertyValue", "propertyID": "corpus:certainty", "value": "measured"}]},
    {"@type": "PropertyValue", "propertyID": "corpus:familyId", "value": "acme--demo-1b"}
  ]
}
```

The extension vocabulary (`huggingface:`, `ml:`, `corpus:`) is documented and
versioned in [`docs/extension-vocabulary.md`](docs/extension-vocabulary.md).

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Credentials go in `.env` (gitignored) or the environment — an exported variable
always wins over the file:

```bash
HF_TOKEN=...              # optional: higher Hub rate limits
OPENROUTER_API_KEY=...    # enrichment via OpenRouter (the default backend)
ANTHROPIC_API_KEY=...     # enrichment via the first-party API instead
```

```bash
.venv/bin/hfcorpus build --profile pilot --corpus corpus
```

That runs the full pipeline for the 100-model pilot. Individual stages:

```bash
.venv/bin/hfcorpus select --profile pilot --corpus corpus
```

Add `--no-enrich` to `build` for a deterministic-only corpus (no LLM calls, no
subject claims). `--limit N` caps a stage for a smoke run; `--force` ignores
caches.

## Pipeline

| Stage | Reads | Writes | What it does |
|---|---|---|---|
| `select` | `config/selection.yaml` | `manifests/{snapshot,selection,candidates}` | Stratified discovery, in-stratum ranking, publisher and family caps |
| `fetch` | selection | `raw/api/*.json`, `raw/cards/*.md`, `processed/clean_cards/*.json` | Verbatim API JSON, README at the pinned SHA, cleaned text + section tree |
| `normalize` | raw + clean cards | `normalized/models.jsonl` | Deterministic JSON-LD; conflicts flagged, nothing inferred |
| `families` | normalized | `normalized/families.json` | Lineage, exact duplicates, packaging variants, canonical choice |
| `enrich` | normalized + clean cards | `enriched/models.jsonl` | Schema-constrained extraction, then deterministic evidence validation |
| `emit` | normalized + enriched + families | `nlweb/models.jsonl` | The compact item NLWeb ingests |
| `validate` | emitted | `reports/validation.json` | Schema-shape validator **and** evidence validator |
| `report` | everything | `reports/quality-report.json`, `reports/review-sample.csv` | Composition, duplication, provenance, enrichment metrics |
| `evaluate` | emitted | `reports/retrieval-eval.json` | 45 golden queries, full record vs metadata-only baseline |

Profiles scale the same pipeline: `pilot` (100) → `corpus_v1` (1,000) →
`demonstration` (2,500) → `expansion` (5,000) → `wide` (10,000).

## What scale buys

Measured on the 45 golden queries, same code and same validator at every rung.
`builds/` holds each rung's manifests, reports, and emitted JSONL.

| records | precision@10 | recall@10 | traps | baseline precision | baseline traps | unanswerable |
|--------:|---:|---:|---:|---:|---:|---:|
| 100 | 0.260 | 0.535 | 0.0067 | 0.256 | 0.0089 | 5/45 |
| 985 | 0.518 | 0.555 | 0.000 | 0.380 | 0.0044 | 0/45 |
| 2,447 | 0.529 | 0.532 | 0.000 | 0.420 | 0.0067 | 0/45 |
| 4,821 | 0.558 | 0.558 | 0.000 | 0.398 | 0.0133 | 0/45 |

Two things worth reading here. At 100 records the corpus barely beats a
metadata-only keyword index — the pilot's job is to expose schema and prompt
faults cheaply, not to look good. And from 2,447 to 4,821 the two curves
diverge: the corpus improves on every metric while the baseline gets worse on
every metric, because more records give a keyword index more ways to be wrong
while evidence-bearing records get better at separating a documented specialty
from a passing mention.

Recall dipping at 2,447 is not a regression. Once a query's qualifying pool
exceeds `k`, the top ten face more competition rather than less.

## Enrichment backends

The same prompt and the same pydantic schema run over either route:

| Backend | Credential | Default model | Notes |
|---|---|---|---|
| `openrouter` (default) | `OPENROUTER_API_KEY` | `openai/gpt-oss-20b` at `reasoning_effort: medium` | OpenAI-compatible; strict JSON schema, `temperature: 0`, per-request cost reported by the gateway |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5` | First-party API; `messages.parse` with a schema-constrained output, `effort: low` |

The default is an open-weights model, per §11's "capable small model for routine
extraction". Whether it is good enough is measured, not assumed — the evidence
validator's drop rate, the taxonomy mapping rate, and the golden-query scores
all report it directly.

**Reasoning effort matters more than the model choice did.** Measured over 50
records, same cards, same prompt, same validator:

| | provider default | medium | low |
|---|---:|---:|---:|
| median latency | 32.2s | 41.0s | 13.1s |
| median output tokens | 3,692 | 3,332 | 906 |
| subject areas kept | 58 | 65 | 51 |
| claims dropped | 122 | 109 | 847 |
| throughput | 6.2/min | 14.2/min | 48.7/min |

Medium dominates the default outright. Low is another 3.4× faster and tempting
for large builds, but it proposes 2.6× as many claims and 847 of them fail quote
validation — the corpus stays sound because the validator drops them, which is
exactly why the drop rate is the number to watch rather than the kept-claim
counts.

A model comparison at 100 records, for reference: `gpt-4.1-mini` mapped more
subjects (119/127 vs 106/119) and dropped half as many claims (118 vs 235), at
4× the cost.

```bash
.venv/bin/hfcorpus enrich --backend anthropic --model claude-sonnet-5
```

The backend, model, **and reasoning effort** are all part of the enrichment cache
key (`openrouter:openai/gpt-oss-20b@medium`) and of every record's provenance, so
changing any of them forks a new derived version rather than reusing the old one,
and no corpus silently mixes them. OpenRouter output is re-validated against the
schema locally — a gateway is not a guarantee of schema compliance, so loose JSON
quarantines the record instead of entering the corpus.

## Design commitments

**Evidence before inference.** A subject specialty needs support from training
data, intended use, evaluation, architecture, or an explicit card claim. The
extractor must return a verbatim quote and a section path for every claim;
`hfcorpus.evidence` then re-locates that quote in the cleaned card and **drops**
claims it cannot find. Confidence is capped by evidence type, so a benchmark
mention cannot be published as `high`. A topic that only appears in a citation or
a limitation is not a specialty at all.

**Provenance is not optional.** Every record carries the repository ID, canonical
URL, SHA, and retrieval time. Cards are fetched at a pinned SHA, never from a
moving `main`. Raw API responses and raw cards are stored separately from
derived text.

**Nothing is silently erased.** `manifests/selection.jsonl` is an append-only
decision log: every candidate ever considered has a reason code
(`included_ranked`, `suppressed_variant`, `excluded_empty_card`, …) and the
policy version that produced it.

**Internal state stays internal.** Retry counts, scores, prompt hashes, parser
warnings, and quarantine states live in the internal record keyed by the same
`@id`. The published item carries only facts that improve discovery, filtering,
explanation, or provenance — and the validator fails a record that leaks.

**Family-aware, not deduplicated to death.** Twenty GGUF quantizations do not
crowd out twenty distinct models: packaging variants are suppressed from broad
results, attached to their family, and still listed as
`corpus:deploymentVariants` so a hardware-constrained query can reach them.
Merges require lineage, config, or naming evidence — card similarity alone only
raises a review flag.

**Model cards are untrusted input.** They are fenced as data in the prompt, the
system prompt says so explicitly, and nothing in the pipeline lets card text
change control flow, call a tool, or reach a secret.

**Reproducible.** Enrichment is content-addressed by
`SHA-256(clean_card + normalized_metadata + prompt_version + model_version)`, so
re-runs are free, failures resume, and a prompt change creates a new derived
version instead of overwriting the old one. `prompt_version` folds in the
controlled vocabulary and `model_version` folds in the backend and reasoning
effort, so nothing that changes the output can leave the key unchanged.

**A transient failure is not a fact about a repository.** Rate limits and network
errors quarantine as `quarantined_fetch` and stay in play for a later run;
`excluded_inaccessible` is reserved for repositories that really are private,
disabled, or gated. In an append-only decision log the difference is permanent,
so it has to be right the first time.

## Evaluation

`tests/golden-queries.yaml` holds 45 questions written before any retrieval
tuning, covering domain knowledge, task, language coverage, deployability,
licensing, and — deliberately — misleading near-matches ("a legal domain model,
not a general chat model that mentions legal disclaimers").

Queries state the *properties* an acceptable result must have rather than a
hand-labelled result list, so the suite survives a rebuild. Each is scored
against two indexes built from the same corpus: `metadata_only` (name, tags,
task, publisher — roughly what Hub keyword search sees) and `full` (plus the
grounded description, capabilities, subject terms, and evidence quotes). The gap
between them is the measurement that justifies the pipeline.

```bash
.venv/bin/hfcorpus evaluate --corpus corpus --k 10
```

## Deviations from the design document

Stated explicitly, because each was a judgement call:

1. **`temperature` depends on the route.** §7.4 asks for temperature near zero.
   The OpenRouter backend sets `temperature: 0` as asked; the first-party Claude
   API rejects the parameter outright, so on that route determinism comes from
   schema-constrained output and low effort instead. Content-addressed caching
   makes either route reproducible in practice.
2. **Beyond 5,000 records is unsanctioned territory.** §13 tops out at 5,000; the
   `wide` profile builds 10,000. The pipeline scales, but two things tuned for
   the smaller ceiling should be revisited there — a 10% publisher cap admits
   1,000 records from one publisher, and near-duplicate detection is quadratic
   within architecture buckets.
3. **Evidence is per claim, not a separate `evidence` field.** §7.1 lists
   `evidence` as its own field; carrying the quote, section, and offsets *inside*
   each claim makes an unsupported claim impossible to express, rather than
   merely detectable.
4. **`evalResults` is not requested during discovery.** The pinned Hub client
   parses it eagerly and raises on repositories whose eval entries omit a dataset
   id, aborting a whole discovery query. Evaluation metadata is read from the
   verbatim per-repository JSON instead.
5. **Discovery-time documentation quality is a metadata proxy.** §2.2 wants
   cleaned README length and section presence, which do not exist before the card
   is fetched. Metadata completeness ranks the pool; the card-derived score is
   computed after fetch and recorded alongside it.
6. **Packaging variants are attached, not independently enriched.** They are
   discoverable through their family's `corpus:deploymentVariants` and their own
   selection rows, but the pilot does not fetch and enrich a card for each
   quantization. Expanding that is a scope decision, not a design change.
7. **Retrieval evaluation runs against a local BM25 harness**, not a live NLWeb
   deployment. It exists so golden queries can be scored the moment the JSONL is
   written and so NLWeb numbers have a baseline to beat.

## Known limitations

- Broken-canonical-link checking (§10.1) is not implemented; it needs a network
  pass over every record at snapshot time.
- The `Person` vs `Organization` distinction for publishers is not derivable from
  Hub metadata; every publisher is emitted as an `Organization`.
- Active parameter counts for mixture-of-experts models are not derivable from
  Hub metadata; the record says so rather than guessing.
- Near-duplicate detection is O(n²) within architecture buckets. Fine to 5,000
  records; beyond that it wants MinHash.

## Responsible presentation

The corpus records what publishers *documented*, not what a model *knows*. An
unknown license is reported as unknown and never described as open source or
commercially usable. Model-card assertions are publisher-provided and may be
incomplete or inaccurate — the anchored evidence link exists so a reader can
check. Only public content is stored; gating and repository terms are respected,
never bypassed.

## Development

```bash
.venv/bin/python -m pytest tests -q
```

Tests are offline: the Hub and the model API are stubbed, so the suite runs
without a token or a key.

## License

MIT.
