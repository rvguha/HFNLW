from hfcorpus.cards import clean_card, documentation_quality, select_sections


def test_front_matter_is_split_out_and_parsed(card_text):
    card = clean_card("acme/demo-1b", card_text)
    assert card.front_matter["license"] == "apache-2.0"
    assert "license: apache-2.0" not in card.clean_text
    assert card.front_matter_error == ""


def test_badges_comments_and_bibtex_are_dropped_but_headings_survive(card_text):
    card = clean_card("acme/demo-1b", card_text)
    assert "img.shields.io" not in card.clean_text
    assert "a comment that should not survive" not in card.clean_text
    assert "@article{demo2026" not in card.clean_text
    assert "## Training Data" in card.clean_text
    assert card.stats["dropped_lines"]["badge"] >= 1
    assert card.stats["dropped_lines"]["citation"] == 1


def test_sections_are_classified_by_role(card_text):
    card = clean_card("acme/demo-1b", card_text)
    roles = {s.heading: s.role for s in card.sections}
    assert roles["Training Data"] == "training"
    assert roles["Evaluation"] == "evaluation"
    assert roles["Limitations"] == "limitations"
    assert roles["Citation"] == "citation"


def test_section_offsets_address_the_clean_text(card_text):
    card = clean_card("acme/demo-1b", card_text)
    training = next(s for s in card.sections if s.heading == "Training Data")
    assert "PubMed abstracts" in training.text(card.clean_text)


def test_long_cards_select_whole_sections_not_a_prefix(card_text):
    card = clean_card("acme/demo-1b", card_text)
    text, paths = select_sections(card, 400)
    assert len(text) <= 400 + 200          # whole sections, so a small overshoot is fine
    assert "Citation" not in " ".join(paths)
    assert text.strip()


def test_documentation_quality_rewards_evidence_sections(card_text):
    rich = documentation_quality(clean_card("a/b", card_text))
    poor = documentation_quality(clean_card("a/c", "# Title\n\nA model.\n"))
    assert rich > poor
    assert 0.0 <= poor <= rich <= 1.0


def test_unparsable_front_matter_is_recorded_not_raised():
    card = clean_card("a/b", "---\nlicense: [unclosed\n---\n\n# T\n\nBody.\n")
    assert card.front_matter == {}
    assert card.front_matter_error
    assert any(w.startswith("front_matter") for w in card.warnings)
