"""Coverage for the corpus-specific parts of the fork: how a schema.org model
item becomes retrieval text, and how collections resolve to scopes."""

import json

from askhub.adapters import load_file

ITEM = {
    "@context": {"@vocab": "https://schema.org/", "hf": "https://huggingface.co/ns/1.0#"},
    "@type": ["SoftwareApplication", "hf:Model"],
    "@id": "https://huggingface.co/acme/demo-1b",
    "name": "Demo 1B",
    "hf:repository": "acme/demo-1b",
    "url": "https://huggingface.co/acme/demo-1b",
    "description": "Demo 1B is a biomedical language model.",
    "hf:category": ["Token Classification"],
    "hf:capability": ["biomedical entity extraction"],
    "hf:subject": ["biomedicine"],
    "keywords": ["transformers"],
    "inLanguage": ["en"],
    "creator": {"@type": "Organization", "name": "acme"},
    "hf:parameters": 1_100_000_000,
    "hf:task": "token-classification",
    "hf:intendedUse": ["clinical note tagging"],
    "hf:trainedOn": ["trained on PubMed abstracts"],
}


def write(tmp_path, *items):
    path = tmp_path / "huggingface.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in items))
    return path


def test_a_model_item_becomes_searchable_prose(tmp_path):
    document = load_file(write(tmp_path, ITEM))[0]
    assert document.name == "Demo 1B"
    assert document.url == "https://huggingface.co/acme/demo-1b"
    text = document.text
    for expected in ("Demo 1B", "acme/demo-1b", "biomedical language model",
                     "Token Classification", "biomedical entity extraction", "biomedicine",
                     "clinical note tagging", "trained on PubMed abstracts"):
        assert expected in text, expected


def test_retrieval_text_is_prose_not_a_json_dump(tmp_path):
    """propertyID plumbing in the text lets vocabulary tokens dominate scoring."""
    text = load_file(write(tmp_path, ITEM))[0].text
    assert "propertyID" not in text
    assert "@type" not in text
    assert "hf:" not in text


def test_parameter_counts_are_spelled_out_for_matching(tmp_path):
    text = load_file(write(tmp_path, ITEM))[0].text
    assert "1.1B parameters" in text
    # 1.1B is past "small" but still runs locally, so the deployability terms
    # have to reach further than the size terms.
    assert "small" not in text
    assert "laptop" in text


def test_size_and_deployability_bands_overlap(tmp_path):
    def phrase(parameters):
        item = dict(ITEM, **{"hf:parameters": parameters})
        return load_file(write(tmp_path, item))[0].text

    tiny = phrase(300_000_000)
    assert "300M parameters" in tiny and "small" in tiny and "laptop" in tiny

    mid = phrase(8_000_000_000)
    assert "8B parameters" in mid and "laptop" in mid and "small" not in mid

    large = phrase(70_000_000_000)
    assert "70B parameters" in large and "large" in large and "laptop" not in large


def test_non_model_items_still_fall_back_to_the_generic_dump(tmp_path):
    other = {"@type": "CreativeWork", "name": "Something else", "url": "https://example.org/x"}
    document = load_file(write(tmp_path, other))[0]
    assert "Something else" in document.text
