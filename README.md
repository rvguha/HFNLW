# HFNLW

An NLWeb prototype for the Hugging Face Hub: a schema.org corpus describing
Hub models, and a natural-language search service over it.

The two halves are separate because they run at different times and carry
different risks. Building the corpus is slow, costs money, and talks to the
Hub and to an LLM. Serving it is neither of those things.

## `hf-nlweb-corpus` — the builder

Nine resumable stages, `select → fetch → normalize → families → enrich → emit
→ validate → report → evaluate`, run as `hfcorpus <stage>`. Enrichment reads a
cleaned model card and returns subjects, capabilities, intended uses and
limitations, each citing a verbatim span the pipeline then verifies against the
card. Output is one file of 9,977 schema.org records.

## `ask-hf-hub` — the server

An ASGI service that answers questions over an emitted corpus: fielded BM25F
and per-field vector retrieval, LLM relevance ranking, SSE streaming, and an
MCP endpoint. Carries its own copy of the corpus and a 100-query retrieval
benchmark whose relevance is stated as predicates, so gold sets are free to
compute and survive a corpus rebuild.

## The seam

The only contract between them is the emitted record shape: schema.org terms
plus an `hf:` namespace for what schema.org has no vocabulary for. Facts are
direct properties, and evidence is verbatim card quotes grouped under the
relation they support — `hf:trainedOn`, `hf:evaluatedOn` — because "models that
know molecular biology, not ones that cite a biology benchmark" is only
answerable if those stay apart.

Note that the served corpus is a copy, not a view: it holds 9,465 records
against the builder's 9,977, and about a thousand of them come from an earlier
selection run and have no counterpart in the current wide build.
