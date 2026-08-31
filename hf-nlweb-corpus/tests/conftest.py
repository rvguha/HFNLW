from pathlib import Path

import pytest

from hfcorpus.config import load_policy, load_taxonomy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def policy():
    return load_policy(ROOT / "config" / "selection.yaml", "pilot")


@pytest.fixture(scope="session")
def taxonomy():
    return load_taxonomy(ROOT / "config" / "taxonomy.yaml")


@pytest.fixture(scope="session")
def patterns(policy):
    return policy.patterns


def raw_model(repo_id="acme/demo-1b", **overrides):
    """A Hub API response shaped like the real thing, minus the noise."""
    raw = {
        "id": repo_id,
        "author": repo_id.split("/")[0],
        "sha": "0" * 40,
        "pipeline_tag": "text-generation",
        "library_name": "transformers",
        "tags": ["transformers", "safetensors", "text-generation", "arxiv:2401.00001",
                 "license:apache-2.0", "region:us"],
        "downloads": 1234,
        "downloadsAllTime": 45678,
        "likes": 42,
        "trendingScore": 1.5,
        "createdAt": "2025-01-02T00:00:00.000Z",
        "lastModified": "2026-06-01T00:00:00.000Z",
        "gated": False,
        "private": False,
        "disabled": False,
        "safetensors": {"total": 1_100_000_000, "parameters": {"BF16": 1_100_000_000}},
        "config": {"architectures": ["DemoForCausalLM"], "model_type": "demo",
                   "max_position_embeddings": 8192},
        "cardData": {"license": "apache-2.0", "language": ["en"], "datasets": ["acme/pubmed-mini"],
                     "base_model": "acme/demo-1b-base"},
        "baseModels": {"relation": "finetune", "models": [{"id": "acme/demo-1b-base"}]},
    }
    raw.update(overrides)
    return raw


CARD = """---
license: apache-2.0
language:
- en
datasets:
- acme/pubmed-mini
base_model: acme/demo-1b-base
---

# Demo 1B

[![badge](https://img.shields.io/badge/demo-1.0-blue)](https://example.com)
<!-- a comment that should not survive -->

Demo 1B is a 1.1 billion parameter language model for biomedical text. It is the
smallest member of the Demo family and is intended for entity extraction.

## Training Data

The model was trained on PubMed abstracts and MIMIC-III discharge summaries.

## Evaluation

Evaluated on BC5CDR and NCBI-disease, reaching 88.2 F1 on BC5CDR.

## Limitations

The model has not been evaluated on legal text and should not be used for
clinical decision making.

## Citation

```bibtex
@article{demo2026,
  title={Demo},
  author={Anon}
}
```
"""


@pytest.fixture
def card_text():
    return CARD
