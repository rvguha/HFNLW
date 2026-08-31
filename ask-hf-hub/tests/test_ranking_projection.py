"""The projection decides what the relevance filter is allowed to see.

Getting it wrong is silent: the ranker still returns confident verdicts, just
without the fields that distinguish one record from another.
"""

import json

from askhub.config import DEFAULT_RANKING_FIELDS, Config
from askhub.pipeline import _ranking_projection

RECORD = {
    "@type": ["SoftwareApplication", "hf:Model"],
    "@id": "https://huggingface.co/acme/demo",
    "name": "Demo 1B",
    "description": "A biomedical language model.",
    "keywords": ["transformers"],
    "hf:category": ["Token Classification"],
    "hf:capability": ["biomedical entity extraction"],
    "hf:subject": ["biomedicine"],
    "inLanguage": ["en"],
    "license": "https://opensource.org/license/mit",
    "creator": {"@type": "Organization", "name": "acme"},
    "hf:trainedOn": ["trained on PubMed abstracts"],
    "hf:evaluatedOn": ["BLURB benchmark"],
    "hf:repository": "acme/demo",
    "hf:revision": "abc123",
    "hf:modelCard": "https://huggingface.co/acme/demo/blob/abc123/README.md",
}


def test_the_evidence_bearing_fields_survive_the_projection():
    kept = _ranking_projection(RECORD, DEFAULT_RANKING_FIELDS)
    for field in ("hf:subject", "hf:capability", "hf:category", "inLanguage", "license"):
        assert field in kept, f"{field} must reach the ranking model"
    assert kept["hf:subject"] == ["biomedicine"]


def test_the_judge_can_tell_training_from_evaluation():
    """The distinction the whole corpus rests on has to survive the projection,
    or the ranker cannot answer "knows biology" versus "cites a biology
    benchmark"."""
    kept = _ranking_projection(RECORD, DEFAULT_RANKING_FIELDS)
    assert kept["hf:trainedOn"] == ["trained on PubMed abstracts"]
    assert kept["hf:evaluatedOn"] == ["BLURB benchmark"]


def test_plumbing_is_still_trimmed():
    """The projection exists to save ranking tokens, so it must still drop the
    keys that carry no signal."""
    kept = _ranking_projection(RECORD, DEFAULT_RANKING_FIELDS)
    for field in ("@id", "hf:revision", "hf:modelCard"):
        assert field not in kept
    assert len(json.dumps(kept)) < len(json.dumps(RECORD))


def test_a_list_from_another_corpus_blinds_the_ranker():
    """Regression: the inherited product-catalog allowlist stripped every field
    that distinguishes one model record from another."""
    product_catalog = {"@type", "name", "description", "category", "audience",
                       "eligibleRegion", "offers", "price", "brand", "url",
                       "keywords", "additionalProperty", "value", "valueReference"}
    blinded = _ranking_projection(RECORD, product_catalog)
    assert "about" not in blinded and "subjectOf" not in blinded and "featureList" not in blinded


def test_the_list_is_configurable_not_hardcoded():
    assert Config.from_env().ranking_fields == DEFAULT_RANKING_FIELDS
    narrow = Config(ranking_fields=("name",))
    assert _ranking_projection(RECORD, narrow.ranking_fields) == {"name": "Demo 1B"}
