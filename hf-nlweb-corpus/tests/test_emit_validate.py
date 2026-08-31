from hfcorpus.cards import clean_card
from hfcorpus.emit import emit
from hfcorpus.evidence import validate as validate_claims
from hfcorpus.families import FamilyAssignment
from hfcorpus.normalize import normalize
from hfcorpus.validate import validate_evidence, validate_item
from tests.conftest import raw_model

ENRICHMENT = {
    "short_description": "Demo 1B is a 1.1B parameter biomedical language model.",
    "capabilities": [{"label": "biomedical entity extraction", "confidence": "high",
                      "evidence_type": "intended_for",
                      "evidence_quote": "intended for entity extraction",
                      "source_section": "preamble"}],
    "subject_areas": [{"label": "Biomedicine", "confidence": "high",
                       "evidence_type": "trained_on",
                       "evidence_quote": "trained on PubMed abstracts",
                       "source_section": "Training Data"}],
    "intended_uses": ["entity extraction"],
    "languages": ["en"],
    "limitations": ["not evaluated on legal text"],
    "deployment_notes": [],
}


def build(card_text, taxonomy, family=None, variants=None):
    card = clean_card("acme/demo-1b", card_text)
    record = normalize(raw_model(), card, retrieved_at="x", card_status="ok")
    validated = validate_claims(ENRICHMENT, card, record["item"], taxonomy)
    item = emit(record, validated.to_json(), family, taxonomy, variants)
    return card, item


def test_emitted_item_is_schema_valid_and_evidence_backed(card_text, taxonomy):
    card, item = build(card_text, taxonomy)
    assert validate_item(item) == []
    assert validate_evidence(item, card) == []


def test_evidence_is_grouped_by_the_relation_it_supports(card_text, taxonomy):
    """A benchmark mention is not a training claim, and the shape must keep them
    apart: the golden queries turn on exactly that distinction."""
    _card, item = build(card_text, taxonomy)
    assert item["hf:trainedOn"] == ["trained on PubMed abstracts"]
    assert "hf:evaluatedOn" not in item
    # Quotes are verbatim spans, not restatements.
    assert all(q in " ".join(card_text.split()) for q in item["hf:trainedOn"])


def test_one_quote_supporting_several_subjects_is_one_piece_of_evidence(card_text, taxonomy):
    """The old shape emitted a Claim per subject, so a single sentence looked
    like several independent corroborations."""
    enrichment = dict(ENRICHMENT, subject_areas=[
        ENRICHMENT["subject_areas"][0],
        {**ENRICHMENT["subject_areas"][0], "label": "Molecular biology"},
    ])
    card = clean_card("acme/demo-1b", card_text)
    record = normalize(raw_model(), card, retrieved_at="x", card_status="ok")
    validated = validate_claims(enrichment, card, record["item"], taxonomy)
    item = emit(record, validated.to_json(), None, taxonomy)
    assert len(item["hf:trainedOn"]) == 1
    assert len(item["hf:subject"]) == 2


def test_the_model_card_is_linked_so_a_quote_can_be_located(card_text, taxonomy):
    _card, item = build(card_text, taxonomy)
    assert item["hf:modelCard"].startswith("https://huggingface.co/acme/demo-1b/blob/")


def test_subjects_are_bare_taxonomy_terms(card_text, taxonomy):
    """The namespace defines what a term means; the record only names it."""
    _card, item = build(card_text, taxonomy)
    assert item["hf:subject"] == ["biomedicine"]


def test_family_and_variants_are_direct_properties(card_text, taxonomy):
    family = FamilyAssignment("acme/demo-1b", "acme--demo-1b", "acme/demo-1b", "canonical")
    _card, item = build(card_text, taxonomy, family, ["acme/demo-1b-GGUF"])
    assert item["hf:family"] == "acme--demo-1b"
    assert item["hf:discoveryScope"] == "primary"
    assert item["hf:variant"] == ["https://huggingface.co/acme/demo-1b-GGUF"]


def test_internal_state_never_reaches_the_published_item(card_text, taxonomy):
    _card, item = build(card_text, taxonomy)
    assert "internal" not in item
    assert validate_item({**item, "internal": {"score": 1}}) != []


def test_the_validator_rejects_a_quote_that_is_not_in_the_card(card_text, taxonomy):
    card, item = build(card_text, taxonomy)
    item["hf:trainedOn"] = ["trained on a corpus that is not in this card"]
    assert validate_evidence(item, card) != []


def test_languages_fold_to_codes_and_reject_non_languages(card_text, taxonomy):
    """A card saying "English" and metadata saying `en` are one fact; "solidity"
    and "92 coding languages" are not languages at all."""
    from hfcorpus.emit import _languages

    item = {"inLanguage": ["en", "zh"]}
    enrichment = {"languages": ["English", "Chinese", "quechua", "solidity",
                                "92 coding languages", "xml documents with translations"]}
    assert _languages(item, enrichment) == ["en", "zh", "quechua"]


def test_the_validator_rejects_yaml_coerced_language_codes():
    """`no` is Norwegian; YAML 1.1 resolves it to False.

    Fifty records shipped a language called "false" before this was caught, and
    no query for Norwegian could reach any of them. cards.py stops it at the
    parser, but the invariant belongs at the door too: the coercion can arrive
    from any future metadata source, not only from card front matter.
    """
    item = {
        "@context": {"@vocab": "https://schema.org/",
                     "hf": "https://huggingface.co/ns/1.0#"},
        "@type": ["SoftwareApplication", "hf:Model"],
        "@id": "https://huggingface.co/acme/demo", "name": "Demo",
        "url": "https://huggingface.co/acme/demo", "description": "A model.",
        "dateModified": "2024-01-01T00:00:00.000Z",
        "hf:repository": "acme/demo", "hf:revision": "abc123",
        "creator": {"@type": "Organization", "name": "acme"},
    }
    assert not [e for e in validate_item({**item, "inLanguage": ["en", "no"]})
                if "inLanguage" in e]
    for bad in ("false", "true", False, True):
        assert [e for e in validate_item({**item, "inLanguage": ["en", bad]})
                if "inLanguage" in e], bad


def test_proposed_subjects_get_a_stable_code(card_text, taxonomy):
    """Otherwise "Computer Vision" and "Computer vision" split their records."""
    card = clean_card("acme/demo-1b", card_text)
    record = normalize(raw_model(), card, retrieved_at="x", card_status="ok")

    def build(label):
        enrichment = dict(ENRICHMENT, subject_areas=[
            {**ENRICHMENT["subject_areas"][0], "label": label}])
        validated = validate_claims(enrichment, card, record["item"], taxonomy)
        return emit(record, validated.to_json(), None, taxonomy)["hf:subject"]

    assert build("Computer Vision") == build("Computer vision") == ["computer-vision"]


def test_a_record_without_enrichment_still_validates(card_text, taxonomy):
    card = clean_card("acme/demo-1b", card_text)
    record = normalize(raw_model(), card, retrieved_at="x", card_status="ok")
    item = emit(record, None, None, taxonomy)
    assert validate_item(item) == []
    assert "hf:subject" not in item
