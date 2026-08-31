from hfcorpus.cards import clean_card
from hfcorpus.normalize import normalize
from tests.conftest import raw_model


def props(item):
    return {p["propertyID"]: p for p in item["additionalProperty"]}


def test_record_carries_identity_provenance_and_lineage(card_text):
    card = clean_card("acme/demo-1b", card_text)
    record = normalize(raw_model(), card, retrieved_at="2026-08-28T00:00:00Z", card_status="ok")
    item = record["item"]
    ids = {p["propertyID"]: p["value"] for p in item["identifier"]}
    assert ids["huggingface:repo_id"] == "acme/demo-1b"
    assert ids["huggingface:sha"] == "0" * 40
    assert item["@id"] == item["url"] == "https://huggingface.co/acme/demo-1b"
    assert item["softwareVersion"] == "0" * 40
    assert {b["@id"] for b in item["isBasedOn"]} == {
        "https://huggingface.co/acme/demo-1b-base",
        "https://huggingface.co/datasets/acme/pubmed-mini",
    }


def test_parameter_count_records_its_source_and_certainty(card_text):
    card = clean_card("acme/demo-1b", card_text)
    record = normalize(raw_model(), card, retrieved_at="x", card_status="ok")
    parameters = props(record["item"])["ml:parameterCount"]
    assert parameters["value"] == 1_100_000_000
    reference = {p["propertyID"]: p["value"] for p in parameters["valueReference"]}
    assert reference == {"corpus:source": "safetensors", "corpus:certainty": "measured"}


def test_a_name_derived_parameter_count_is_labelled_inferred():
    raw = raw_model("acme/demo-7b", safetensors={})
    record = normalize(raw, None, retrieved_at="x", card_status="missing_readme")
    parameters = props(record["item"])["ml:parameterCount"]
    reference = {p["propertyID"]: p["value"] for p in parameters["valueReference"]}
    assert reference["corpus:certainty"] == "inferred"
    assert reference["corpus:source"] == "repository_name"


def test_conflicting_sources_are_flagged_and_both_values_kept():
    raw = raw_model("acme/demo-70b")     # name says 70B, safetensors says 1.1B
    record = normalize(raw, None, retrieved_at="x", card_status="missing_readme")
    conflict = record["internal"]["conflicts"][0]
    assert conflict["field"] == "parameter_count"
    assert conflict["safetensors"] == 1_100_000_000
    assert conflict["repository_name"] == 70_000_000_000
    assert "parameter_count" in props(record["item"])["corpus:conflict"]["value"]


def test_unknown_license_is_explicit_and_never_implied_permissive():
    raw = raw_model(cardData={})
    record = normalize(raw, None, retrieved_at="x", card_status="missing_readme")
    item = record["item"]
    assert "license" not in item
    assert props(item)["corpus:licenseStatus"]["value"] == "unknown"


def test_recognised_license_resolves_to_a_url(card_text):
    card = clean_card("acme/demo-1b", card_text)
    record = normalize(raw_model(), card, retrieved_at="x", card_status="ok")
    assert record["item"]["license"] == "https://www.apache.org/licenses/LICENSE-2.0"
    assert props(record["item"])["corpus:licenseStatus"]["value"] == "recognized_identifier"


def test_description_comes_from_card_prose_when_available(card_text):
    card = clean_card("acme/demo-1b", card_text)
    record = normalize(raw_model(), card, retrieved_at="x", card_status="ok")
    assert record["internal"]["description_source"] == "model_card_prose"
    assert "biomedical text" in record["item"]["description"]


def test_metadata_only_description_when_there_is_no_card():
    record = normalize(raw_model(), None, retrieved_at="x", card_status="missing_readme")
    assert record["internal"]["description_source"] == "metadata_only"
    assert record["item"]["description"]


def test_plumbing_tags_are_not_published_as_keywords(card_text):
    card = clean_card("acme/demo-1b", card_text)
    item = normalize(raw_model(), card, retrieved_at="x", card_status="ok")["item"]
    assert not any(k.startswith(("license:", "arxiv:", "region:")) for k in item["keywords"])
    assert any("arxiv.org" in c["url"] for c in item["citation"])
