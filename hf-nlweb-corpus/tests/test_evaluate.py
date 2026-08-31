from pathlib import Path

from hfcorpus.evaluate import evaluate, load_queries, matches
from hfcorpus.retrieve import Index

QUERIES = Path(__file__).resolve().parent / "golden-queries.yaml"


# Normalized propertyID -> the property name the emitter uses, so fixtures can
# keep speaking in the pipeline's internal vocabulary.
EMITTED = {
    "huggingface:pipelineTag": "hf:task",
    "huggingface:library": "hf:library",
    "huggingface:downloads": "hf:downloads",
    "ml:parameterCount": "hf:parameters",
    "corpus:licenseStatus": "hf:licenseStatus",
    "corpus:discoveryScope": "hf:discoveryScope",
}


def item(repo_id, **kwargs):
    base = {
        "@context": {"@vocab": "https://schema.org/",
                     "hf": "https://huggingface.co/ns/1.0#"},
        "@type": ["SoftwareApplication", "hf:Model"],
        "@id": f"https://huggingface.co/{repo_id}",
        "hf:repository": repo_id,
        "name": repo_id.split("/")[-1],
        "description": kwargs.pop("description", ""),
        "keywords": kwargs.pop("keywords", []),
        "hf:category": kwargs.pop("subcategories", []),
        "inLanguage": kwargs.pop("languages", []),
        "hf:subject": list(kwargs.pop("subjects", [])),
        "creator": {"@type": "Organization", "name": repo_id.split("/")[0]},
    }
    for key, value in kwargs.pop("props", {}).items():
        base[EMITTED.get(key, key)] = value
    for entry in kwargs.pop("claims", []):
        for key, quotes in entry.items():
            base.setdefault(key, []).extend(quotes)
    base.update(kwargs)
    return base


def claim(evidence_type="trained_on", confidence="high", quote="trained on PubMed"):
    """Evidence in the emitted shape: quotes grouped by the relation they support.

    `confidence` is accepted and ignored -- it is no longer published, because it
    was the extractor grading its own inference.
    """
    del confidence
    key = {"trained_on": "hf:trainedOn", "fine_tuned_on": "hf:fineTunedOn",
           "evaluated_on": "hf:evaluatedOn", "intended_for": "hf:intendedFor",
           "architecture": "hf:architectureFor"}.get(evidence_type)
    # `merely_mentioned` is no longer emitted at all: a passing mention is not
    # evidence, so it produces no property.
    return {key: [quote]} if key else {}


def test_requirement_blocks_combine_conjunctively():
    biomedical = item("a/bio-ner", subjects=["biomedicine"],
                      props={"huggingface:pipelineTag": "token-classification",
                             "ml:parameterCount": 110_000_000},
                      claims=[claim()])
    spec = {"task_any": ["token-classification"], "subject_any": ["biomedicine"],
            "max_parameters": 1_000_000_000}
    assert matches(biomedical, spec)
    assert not matches(biomedical, {**spec, "max_parameters": 10_000_000})
    assert not matches(biomedical, {**spec, "subject_any": ["law"]})


def test_evidence_type_and_confidence_filters_separate_specialists_from_mentions():
    trained = item("a/x", subjects=["law"], claims=[claim("trained_on", "high")])
    mentioned = item("b/x", subjects=["law"], claims=[claim("merely_mentioned", "low")])
    spec = {"evidence_type_any": ["trained_on", "fine_tuned_on"], "min_confidence": "medium"}
    assert matches(trained, spec)
    assert not matches(mentioned, spec)


def test_the_full_record_finds_what_metadata_alone_cannot():
    """The differentiator lives in the grounded description and subject terms,
    not in the repository name -- which is exactly the case the Hub's own
    keyword search handles badly."""
    items = [
        item("a/model-one", keywords=["transformers"],
             description="A biomedical entity recognition model trained on PubMed abstracts.",
             subjects=["biomedicine"], props={"huggingface:pipelineTag": "token-classification"},
             claims=[claim()]),
        item("b/generic", keywords=["transformers"],
             description="A general purpose chat model.",
             props={"huggingface:pipelineTag": "text-generation"}),
    ]
    query = "biomedical entity recognition"
    full = [i["@id"] for i, _ in Index.build(items, "full").search(query, 2)]
    metadata = [i["@id"] for i, _ in Index.build(items, "metadata_only").search(query, 2)]
    assert full and full[0].endswith("a/model-one")
    assert metadata == []


def test_evaluation_reports_both_profiles_and_flags_unanswerable_queries():
    items = [item("a/x", description="A text generation model.",
                  props={"huggingface:pipelineTag": "text-generation",
                         "ml:parameterCount": 1_000_000_000})]
    result = evaluate(items, load_queries(QUERIES), k=5)
    assert set(result["summary"]) == {"full", "metadata_only"}
    assert result["queries"] == 45
    assert "q001" in result["queries_with_no_qualifying_record"]


def test_the_golden_suite_is_well_formed():
    queries = load_queries(QUERIES)
    assert 40 <= len(queries) <= 60
    assert len({q["id"] for q in queries}) == len(queries)
    allowed = {"task_any", "subject_any", "language_any", "library_any", "license_status_any",
               "max_parameters", "min_parameters", "discovery_scope", "evidence_type_any",
               "min_confidence", "keyword_any"}
    for query in queries:
        assert query["question"].strip()
        assert set(query.get("requires", {})) <= allowed
        assert set(query.get("avoid", {})) <= allowed
