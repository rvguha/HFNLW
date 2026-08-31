"""Retrieval width, ranking width, and ranking model are per-request.

The point is offline evaluation: retrieve far wider than a user would see and
judge it with a model too slow to serve, so the relevance opinion does not come
from the retrieval being measured.
"""

import pytest

from askhub.config import Config
from askhub.models import SearchRequest
from askhub.pipeline import _bounded
from askhub.ranking_models import (
    DEFAULT_RANKING_MODEL,
    REFERENCE_MODEL_IDS,
    SERVING_MODELS,
    validate_ranking_model,
)


def test_defaults_are_unchanged_when_nothing_is_requested():
    request = SearchRequest(query="q")
    assert request.retrieval_count is None and request.ranking_count is None
    config = Config()
    assert _bounded(None, config.bm25_rank_count, config.max_retrieval_count) == 10


def test_a_request_can_widen_retrieval_and_ranking():
    config = Config()
    assert _bounded(200, config.bm25_rank_count, config.max_retrieval_count) == 200
    assert _bounded(50, config.vector_rank_count, config.max_retrieval_count) == 50


def test_the_ceiling_bounds_what_a_caller_can_ask_for():
    """A wide retrieval judged by an expensive model is what an evaluation wants
    and what an abusive request wants; the server decides the limit."""
    config = Config(max_retrieval_count=200)
    assert _bounded(10_000, config.bm25_rank_count, config.max_retrieval_count) == 200
    assert _bounded(0, config.bm25_rank_count, config.max_retrieval_count) == 1


def test_the_serving_default_is_not_a_reference_model():
    """Reference models are selectable but must never become the default: they
    are chosen for judgement quality, not for latency on a live query."""
    assert SERVING_MODELS[0].id == DEFAULT_RANKING_MODEL
    assert DEFAULT_RANKING_MODEL not in REFERENCE_MODEL_IDS


def test_reference_models_are_selectable_per_request():
    for model in REFERENCE_MODEL_IDS:
        assert validate_ranking_model(model) == model


def test_an_unknown_ranking_model_is_refused():
    with pytest.raises(ValueError, match="ranking_model must be one of"):
        validate_ranking_model("some/model-that-does-not-exist")
