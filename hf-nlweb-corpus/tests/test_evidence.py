from hfcorpus.cards import clean_card
from hfcorpus.evidence import locate, validate

CARD = """# Demo

Demo was trained on PubMed abstracts   and MIMIC-III notes.

## Limitations

It has not been evaluated on legal text.
"""


def enrichment(**overrides):
    base = {
        "short_description": "A biomedical model.",
        "capabilities": [],
        "subject_areas": [],
        "intended_uses": [],
        "languages": [],
        "limitations": [],
        "deployment_notes": [],
    }
    base.update(overrides)
    return base


def claim(label, evidence_type, quote, confidence="high"):
    return {"label": label, "confidence": confidence, "evidence_type": evidence_type,
            "evidence_quote": quote, "source_section": "preamble"}


def test_quotes_are_located_across_whitespace_differences():
    card = clean_card("a/b", CARD)
    found = locate("trained on PubMed abstracts and MIMIC-III notes", card)
    assert found is not None
    start, end, _section = found
    assert "PubMed" in card.clean_text[start:end]


def test_a_quote_stripped_of_markdown_emphasis_still_matches():
    """Models quote the rendered text, not the asterisks. Dropping the claim
    would discard real evidence as if it were a fabrication."""
    source = "# T\n\n- **Significantly better** at reasoning, surpassing prior models.\n"
    card = clean_card("a/b", source)
    found = locate("Significantly better at reasoning, surpassing prior models.", card)
    assert found is not None
    start, end, _ = found
    # The span runs first matched character to last, so markers inside it are
    # preserved -- the stored evidence is the source as written, not the
    # model's cleaned-up rendering of it.
    assert card.clean_text[start:end] == \
        "Significantly better** at reasoning, surpassing prior models."


def test_typographic_variants_fold_to_the_same_character():
    """Models 'improve' ASCII punctuation on the way out. A byte comparison
    reads the result as a fabrication; the quote is faithful."""
    source = '# T\n\nGPT-2 is a transformer-based model with a 1 M context and "quotes".\n'
    card = clean_card("a/b", source)
    assert locate("GPT‑2 is a transformer‑based model", card) is not None  # U+2011
    assert locate("a 1 M context", card) is not None                           # U+202F
    assert locate("and “quotes”", card) is not None                       # curly
    assert locate("a transformer—based model", card) is not None               # em dash
    # Folding punctuation does not fold away content.
    assert locate("trained on the ChEMBL corpus of molecules", card) is None


def test_tolerant_matching_is_not_lenient_matching():
    source = "# T\n\n- **Significantly better** at reasoning, surpassing prior models.\n"
    card = clean_card("a/b", source)
    # Word dropped from the middle: no longer a contiguous quote of the source.
    assert locate("Significantly better at reasoning, surpassing models.", card) is None
    # Words reordered.
    assert locate("at reasoning Significantly better surpassing prior", card) is None


def test_a_supported_claim_survives_with_offsets(taxonomy):
    card = clean_card("a/b", CARD)
    result = validate(enrichment(subject_areas=[
        claim("Biomedicine", "trained_on", "trained on PubMed abstracts")]),
        card, {"inLanguage": []}, taxonomy)
    assert len(result.subject_areas) == 1
    entry = result.subject_areas[0]
    assert entry["taxonomy_term"] == "biomedicine"
    assert entry["taxonomy_status"] == "mapped"
    assert card.clean_text[slice(*entry["clean_offsets"])] == entry["evidence_quote"]


def test_a_fabricated_quote_is_dropped(taxonomy):
    card = clean_card("a/b", CARD)
    result = validate(enrichment(subject_areas=[
        claim("Chemistry", "trained_on", "trained on the ChEMBL molecule corpus")]),
        card, {"inLanguage": []}, taxonomy)
    assert result.subject_areas == []
    assert result.dropped[0]["reason"] == "quote_not_found_in_source"


def test_a_topic_only_mentioned_is_not_a_specialty(taxonomy):
    card = clean_card("a/b", CARD)
    result = validate(enrichment(subject_areas=[
        claim("Law", "merely_mentioned", "has not been evaluated on legal text")]),
        card, {"inLanguage": []}, taxonomy)
    assert result.subject_areas == []
    assert "does_not_establish_specialty" in result.dropped[0]["reason"]


def test_confidence_is_capped_by_evidence_type(taxonomy):
    card = clean_card("a/b", CARD)
    result = validate(enrichment(subject_areas=[
        claim("Biomedicine", "evaluated_on", "trained on PubMed abstracts", confidence="high")]),
        card, {"inLanguage": []}, taxonomy)
    assert result.subject_areas[0]["confidence"] == "medium"
    assert result.warnings


def test_an_unrecognised_subject_is_kept_but_marked_proposed(taxonomy):
    card = clean_card("a/b", CARD)
    result = validate(enrichment(subject_areas=[
        claim("Underwater basket weaving", "trained_on", "trained on PubMed abstracts")]),
        card, {"inLanguage": []}, taxonomy)
    assert result.subject_areas[0]["taxonomy_status"] == "proposed"
    assert result.subject_areas[0]["taxonomy_term"] is None


def test_languages_are_reconciled_with_declared_metadata(taxonomy):
    card = clean_card("a/b", CARD)
    result = validate(enrichment(languages=["English", "Klingon"]), card,
                      {"inLanguage": ["en"]}, taxonomy)
    assert result.languages == ["English"]
    assert result.dropped[0]["label"] == "Klingon"
