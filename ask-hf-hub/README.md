---
title: Ask the Hugging Face Hub
emoji: 🤗
colorFrom: yellow
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Natural-language search over a grounded catalog of Hub models
---

# Ask the Hugging Face Hub

An NLWeb **ask** interface over a grounded, evidence-bearing catalog of Hugging
Face model repositories. Ask *"which models appear to know a lot about molecular
biology?"* and get back models whose cards actually document training or
evaluation on that subject — each with the quote that supports the claim.

The corpus is built by
[hf-nlweb-corpus](https://github.com/rvguha/hf-nlweb-corpus): stratified
selection, provenance-preserving acquisition at a pinned repository revision,
deterministic schema.org normalization, family-aware deduplication, and
evidence-validated enrichment. This repo is the serving half.

## What makes the answers different from keyword search

Every subject claim carries an evidence type (`trained_on`, `fine_tuned_on`,
`evaluated_on`, `intended_for`, `architecture`), a confidence, and a verbatim
quote from the model card at the SHA it was read from. A model that merely
*mentions* a legal benchmark is not a legal model, and the catalog says so.
Packaging variants — GGUF, AWQ, ONNX conversions — collapse under their family
so twenty quantizations cannot crowd out twenty distinct models.

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
cp .env.example .env    # add OPENROUTER_API_KEY for ranked answers
.venv/bin/ask-hf-hub
```

Open <http://127.0.0.1:8000>. Without an API key the server still starts and
serves deterministic offline embeddings plus BM25 — enough to exercise
retrieval, but not the ranked, explained answers.

## Run as a container

```bash
docker build -t ask-hf-hub . && docker run -p 7860:7860 -e OPENROUTER_API_KEY=... ask-hf-hub
```

## Deploying as a Hugging Face Space

The frontmatter above configures a Docker Space on port 7860. Set
`OPENROUTER_API_KEY` as a Space **secret**, never in the repo.

## Evaluation width

Retrieval width, ranking width, and the ranking model are per-request, so the
same corpus and the same pipeline can be run at serving settings or at
evaluation settings:

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' -d '{
  "query": "models for low-resource African languages",
  "retrieval_count": 200,
  "ranking_count": 60,
  "ranking_model": "anthropic/claude-opus-5"
}'
```

Measured on the running server, same question:

| | results | latency | tokens | cost |
|---|---|---|---|---|
| default (10 retrieved, GPT-OSS 20B) | 9 | 2.3s | 41,996 | $0.0033 |
| 200 retrieved, 60 ranked by Claude Opus 5 | 3 | 6.9s | 165,128 | $0.8456 |

The reference model is the stricter judge, which is the point: a wide retrieval
scored by a model too slow and expensive to serve gives relevance opinions that
do not come from the retrieval being measured. That is what a self-scored
precision number cannot give you.

At roughly $0.85 a query, a 45-query suite costs about $38 — affordable to run
occasionally, not on every commit. `ASKHUB_MAX_RETRIEVAL_COUNT` and
`ASKHUB_MAX_RANKING_COUNT` cap what a request may ask for.

`ranking_models.py` separates **serving models** (cheap, fast enough for a live
query) from **reference models** (Claude Opus 5, Claude Sonnet 5, GPT-5.4,
Gemini 3.1 Pro). Both are selectable per request; only a serving model is ever
the default.

## Endpoints

| Path | What it is |
|---|---|
| `/` | The ask UI |
| `/ask` | SSE search endpoint; accepts `GET` or `POST` |
| `/health` | Item counts, collections, refresh state |
| `/mcp` | MCP server exposing the same search |

## Corpus

`src/askhub/corpus/huggingface.jsonl` is one schema.org `SoftwareApplication` item
per line, as emitted by `hfcorpus emit`. To refresh it, rebuild the corpus and
copy the file in, or point `sources.yaml` at a hosted URL and let the refresh
timer pick it up.

`ASKHUB_RANKING_FIELDS` controls which schema.org keys reach the relevance
filter. The list is corpus-shaped -- a product catalog needs `offers` and
`eligibleRegion`, a model catalog needs `about`, `featureList`, and `subjectOf`
-- and inheriting one corpus's list silently blinds the ranker on another
without any error to notice.

`src/askhub/adapters.py` turns each item into retrieval text from the fields a
person actually searches on — description, capabilities, subject areas, evidence
quotes, and a spelled-out parameter size so "7B" and "small" match at all.

## Provenance

Derived from the [nlw-codex](https://github.com/TechSoup/tsnlweb) NLWeb server
(MIT). The retrieval pipeline, ranking, rate limiting, and MCP surface are that
project's work; the corpus adapter, catalog scopes, UI copy, and packaging are
specific to this catalog. See `LICENSE`.
