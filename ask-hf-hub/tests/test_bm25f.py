"""Fielded BM25 scoring.

The defect these cover: with one flat retrieval blob, a record whose `name`
matched the query byte-for-byte ranked 20th, because its length -- 100 language
codes and a dozen enriched bullets -- was penalised while a terse derivative
repo scored higher on the same terms.
"""

import json

import numpy as np
import pytest

from askhub.adapters import load_file
from askhub.catalog import MemoryCatalog, _query_tokens
from askhub.models import FIELD_BODY, FIELD_DESCRIPTION, FIELD_NAME, Document

FLAT = {FIELD_NAME: 1.0, FIELD_DESCRIPTION: 1.0, FIELD_BODY: 1.0}
EVEN_B = {FIELD_NAME: 0.75, FIELD_DESCRIPTION: 0.75, FIELD_BODY: 0.75}


def model(repo, name, description, body=""):
    return {
        "@type": ["SoftwareApplication", "hf:Model"],
        "@id": f"https://huggingface.co/{repo}",
        "url": f"https://huggingface.co/{repo}",
        "name": name,
        "hf:repository": repo,
        "description": description,
        "hf:deploymentNote": [body],
    }


def catalog(items, weights=None, b=None):
    documents = tuple(
        Document(i["@id"], i["url"], i["name"], "hf", "", i, _fields(i)) for i in items
    )
    return MemoryCatalog(
        documents,
        np.empty((len(documents), 1), dtype=np.float32),
        field_weights=weights,
        field_b=b or EVEN_B,
    )


def _fields(item):
    from askhub.adapters import _fields as build

    return build(item)


def ranked(cat, query):
    scores = cat._bm25(query)
    order = sorted(range(len(cat.documents)), key=lambda i: -float(scores[i]))
    return [cat.documents[i].name for i in order]


def test_a_name_match_outranks_a_body_mention():
    items = [
        model("acme/target", "Whisper Large V3", "Speech recognition."),
        # Same terms, but only in the corroborating detail.
        model("other/decoy", "Something Else", "Unrelated.", "whisper large v3 " * 8),
    ]
    assert ranked(catalog(items), "whisper large v3")[0] == "Whisper Large V3"


def test_flat_weights_let_the_body_mention_win():
    """The baseline this replaces, pinned so the fix cannot silently regress."""
    items = [
        model("acme/target", "Whisper Large V3", "Speech recognition."),
        model("other/decoy", "Something Else", "Unrelated.", "whisper large v3 " * 8),
    ]
    assert ranked(catalog(items, weights=FLAT), "whisper large v3")[0] == "Something Else"


def test_a_long_record_is_not_penalised_for_being_well_documented():
    """Field-local normalisation: body length must not dilute a name match."""
    items = [
        model("acme/target", "Whisper Large V3", "Speech recognition.", "detail " * 500),
        model("other/near", "Distil Whisper Large V3", "Speech recognition."),
    ]
    assert ranked(catalog(items), "whisper large v3")[0] == "Whisper Large V3"


def test_saturation_applies_once_across_fields_not_per_field():
    """BM25F, not a sum of per-field BM25 scores.

    Summing would saturate per field, so one occurrence in each of three fields
    would beat three occurrences in the highest-weighted one.
    """
    spread = model("a/spread", "Alpha ranker", "A ranker.", "ranker")
    concentrated = model("b/focus", "Ranker ranker ranker", "Unrelated.")
    order = ranked(catalog([spread, concentrated]), "ranker")
    assert order[0] == "Ranker ranker ranker"


def test_documents_without_fields_score_as_plain_bm25():
    """Adapters predating fielded retrieval must keep working."""
    documents = (
        Document("a", "http://a", "A", "s", "alpha beta gamma", {}),
        Document("b", "http://b", "B", "s", "delta epsilon", {}),
    )
    cat = MemoryCatalog(documents, np.empty((2, 1), dtype=np.float32))
    assert float(cat._bm25("alpha")[0]) > 0
    assert float(cat._bm25("alpha")[1]) == 0


def test_document_frequency_is_counted_once_per_document():
    """Not once per field -- per-field IDF is the classic fielded-scoring bug."""
    items = [model(f"o/m{i}", f"Model {i}", "shared shared", "shared") for i in range(5)]
    cat = catalog(items)
    assert cat._document_frequency["shared"] == 5


def test_an_unweighted_field_still_scores_at_the_neutral_weight():
    items = [model("a/one", "Alpha", "beta")]
    cat = catalog(items, weights={FIELD_NAME: 32.0})
    assert float(cat._bm25("beta")[0]) > 0


@pytest.fixture
def corpus_documents():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "src/askhub/corpus/huggingface.jsonl"
    if not path.is_file():
        pytest.skip("bundled corpus not present")
    return tuple(load_file(path))


def test_the_known_item_case_that_motivated_this(corpus_documents):
    cat = MemoryCatalog(corpus_documents, np.empty((len(corpus_documents), 1), dtype=np.float32))
    scores = cat._bm25("whisper large v3")
    index = next(
        i for i, d in enumerate(corpus_documents)
        if d.id == "https://huggingface.co/openai/whisper-large-v3"
    )
    rank = int((scores > float(scores[index])).sum()) + 1
    assert rank == 1, f"openai/whisper-large-v3 ranked {rank}"


def test_fields_are_split_not_duplicated(tmp_path):
    path = tmp_path / "hf.jsonl"
    path.write_text(json.dumps(model("acme/demo-1b", "Demo 1B", "A biomedical model.")))
    fields = dict(load_file(path)[0].fields)
    assert fields[FIELD_NAME] == "Demo 1B acme/demo-1b"
    assert "biomedical" in fields[FIELD_DESCRIPTION]
    assert "biomedical" not in fields[FIELD_NAME]


def test_function_words_are_dropped_from_queries():
    assert _query_tokens("A small Japanese language model I can run locally") == [
        "small", "japanese", "language", "model", "run", "locally"
    ]


def test_a_query_of_only_function_words_keeps_them():
    """Some models really are named `it` or `A`; an empty query returns nothing."""
    assert _query_tokens("it") == ["it"]
    assert _query_tokens("the who") == ["the", "who"]


def test_a_stopword_in_the_query_cannot_carry_the_match():
    """The q093 failure: `i` supplied 68% of the winning score."""
    items = [
        model("acme/target", "Japanese Reranker Small", "A Japanese reranking model."),
        # `I` is rare in the corpus, so it earns high IDF and used to dominate.
        model("other/decoy", "IF-I-M-v1.0", "An image generation model."),
    ]
    order = ranked(catalog(items), "A small Japanese language model I can run locally")
    assert order[0] == "Japanese Reranker Small"


def test_language_codes_survive_in_the_index(corpus_documents):
    """Stopword removal is query-side only.

    Nineteen stopwords are also language codes on records in this corpus, so
    stripping documents would delete the only marker Italian or Icelandic
    records carry.
    """
    catalogue = MemoryCatalog(
        corpus_documents, np.empty((len(corpus_documents), 1), dtype=np.float32)
    )
    for code in ("it", "is", "my"):
        assert catalogue._document_frequency.get(code, 0) > 0, code


def test_vector_scores_combine_the_field_channels():
    documents = tuple(
        Document(f"id{i}", f"http://{i}", f"N{i}", "s", "", {}) for i in range(2)
    )
    matrices = {
        FIELD_NAME: np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        FIELD_BODY: np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    }
    query = np.array([1.0, 0.0], dtype=np.float32)

    cat = MemoryCatalog(
        documents, np.empty((2, 2), dtype=np.float32), field_matrices=matrices,
        vector_weights={FIELD_NAME: 3.0, FIELD_BODY: 1.0},
    )
    assert list(cat._vector_scores(query)) == [3.0, 1.0]

    # A zero weight silences a channel rather than merely shrinking it.
    cat.vector_weights = {FIELD_NAME: 0.0, FIELD_BODY: 1.0}
    assert list(cat._vector_scores(query)) == [0.0, 1.0]


def test_vector_scoring_falls_back_to_the_single_matrix():
    documents = tuple(Document(f"id{i}", f"u{i}", f"N{i}", "s", "", {}) for i in range(2))
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cat = MemoryCatalog(documents, matrix)
    assert list(cat._vector_scores(np.array([1.0, 0.0], dtype=np.float32))) == [1.0, 0.0]


def test_an_empty_field_embeds_to_zero_and_scores_zero():
    """A record with no description should not be dragged toward the origin."""
    import asyncio

    from askhub.catalog import _embed_field

    class Stub:
        dimensions = 3

        async def embed(self, texts):
            return np.ones((len(texts), 3), dtype=np.float32)

    matrix = asyncio.run(_embed_field(Stub(), ["something", "", "  ", "more"]))
    assert matrix.shape == (4, 3)
    assert list(matrix[:, 0]) == [1.0, 0.0, 0.0, 1.0]
