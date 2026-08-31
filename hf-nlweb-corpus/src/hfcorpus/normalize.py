"""Deterministic normalization into schema.org JSON-LD (design doc s5).

Nothing here infers, summarises, or guesses. It reconciles the Hub API record,
the card's YAML front matter, and the model config into a
``SoftwareApplication`` node, keeping ML-specific facts as typed
``PropertyValue`` nodes under ``additionalProperty`` rather than inventing
top-level keys.

Two hard rules:

* when sources disagree, both values survive and a conflict flag is raised
  (s5.2) -- normalization never silently picks a winner without saying so; and
* operational state (paths, warnings, scores) stays in the ``internal`` record,
  never in the published item (s8.2).
"""

from __future__ import annotations

from typing import Any

from .cards import CleanCard, documentation_quality
from .naming import declared_base_models, humanize, inferred_parameters_from_name, repo_owner

HF = "https://huggingface.co"
CONTEXT = "https://schema.org"

# Extension vocabulary. Compact identifiers, documented and versioned in
# docs/extension-vocabulary.md -- not a claim that schema.org defines them.
VOCAB_VERSION = "1.0"

LICENSE_URLS = {
    "apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
    "mit": "https://opensource.org/license/mit",
    "bsd-3-clause": "https://opensource.org/license/bsd-3-clause",
    "bsd-2-clause": "https://opensource.org/license/bsd-2-clause",
    "gpl-3.0": "https://www.gnu.org/licenses/gpl-3.0.html",
    "agpl-3.0": "https://www.gnu.org/licenses/agpl-3.0.html",
    "lgpl-3.0": "https://www.gnu.org/licenses/lgpl-3.0.html",
    "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc-4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nc-sa-4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "cc0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "openrail": "https://www.licenses.ai/ai-licenses",
    "creativeml-openrail-m": "https://huggingface.co/spaces/CompVis/stable-diffusion-license",
    "bigscience-openrail-m": "https://www.licenses.ai/ai-licenses",
    "artistic-2.0": "https://opensource.org/license/artistic-2-0",
    "mpl-2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
}

# Licenses whose terms restrict fields of use or commercial use. The corpus does
# not decide whether a license is "open"; it records what the publisher declared
# and whether that declaration is a recognised identifier (s12).
PIPELINE_LABELS = {
    "text-generation": ["Text Generation"],
    "text2text-generation": ["Text Generation"],
    "conversational": ["Conversational AI"],
    "fill-mask": ["Masked Language Modeling"],
    "sentence-similarity": ["Sentence Embedding", "Semantic Similarity"],
    "feature-extraction": ["Feature Extraction", "Embedding"],
    "token-classification": ["Token Classification", "Named Entity Recognition"],
    "text-classification": ["Text Classification"],
    "zero-shot-classification": ["Zero-Shot Classification"],
    "question-answering": ["Question Answering"],
    "summarization": ["Summarization"],
    "translation": ["Machine Translation"],
    "table-question-answering": ["Table Question Answering"],
    "automatic-speech-recognition": ["Speech Recognition"],
    "text-to-speech": ["Speech Synthesis"],
    "text-to-audio": ["Audio Generation"],
    "audio-classification": ["Audio Classification"],
    "audio-to-audio": ["Audio Enhancement"],
    "voice-activity-detection": ["Voice Activity Detection"],
    "image-classification": ["Image Classification"],
    "object-detection": ["Object Detection"],
    "image-segmentation": ["Image Segmentation"],
    "image-to-text": ["Image Captioning"],
    "image-text-to-text": ["Vision Language Modeling"],
    "text-to-image": ["Image Generation"],
    "image-to-image": ["Image To Image"],
    "text-to-video": ["Video Generation"],
    "video-text-to-text": ["Video Understanding"],
    "video-classification": ["Video Classification"],
    "depth-estimation": ["Depth Estimation"],
    "document-question-answering": ["Document Question Answering"],
    "visual-question-answering": ["Visual Question Answering"],
    "reinforcement-learning": ["Reinforcement Learning"],
    "robotics": ["Robotics"],
    "time-series-forecasting": ["Time Series Forecasting"],
    "tabular-classification": ["Tabular Classification"],
    "mask-generation": ["Mask Generation"],
    "any-to-any": ["Multimodal"],
}

# Hub tags that are machine plumbing rather than descriptive keywords.
TAG_PREFIX_DROP = ("license:", "arxiv:", "base_model:", "dataset:", "region:", "doi:",
                   "modality:", "size_categories:", "language:", "endpoints_compatible",
                   "autotrain_compatible", "has_space", "co2_eq_emissions", "model-index")


def prop(property_id: str, name: str, value: Any, **extra: Any) -> dict[str, Any]:
    node: dict[str, Any] = {
        "@type": "PropertyValue",
        "propertyID": property_id,
        "name": name,
        "value": value,
    }
    node.update(extra)
    return node


def normalize(raw: dict[str, Any], card: CleanCard | None, *,
              retrieved_at: str, card_status: str) -> dict[str, Any]:
    """Return ``{"item": <JSON-LD>, "internal": {...}}`` for one repository."""
    repo_id = raw.get("id") or raw.get("modelId") or ""
    url = f"{HF}/{repo_id}"
    front = (card.front_matter if card else {}) or {}
    api_card_data = raw.get("cardData") or {}
    config = raw.get("config") or {}
    conflicts: list[dict[str, Any]] = []
    warnings: list[str] = []

    # -- licence ------------------------------------------------------------
    license_id = _first_str(front.get("license"), api_card_data.get("license"))
    if front.get("license") and api_card_data.get("license") and \
            str(front["license"]).lower() != str(api_card_data["license"]).lower():
        conflicts.append({"field": "license", "card_yaml": front["license"],
                          "api_card_data": api_card_data["license"]})
    license_key = (license_id or "").strip().lower()
    license_url = LICENSE_URLS.get(license_key)
    license_link = _first_str(front.get("license_link"), api_card_data.get("license_link"))
    if license_url:
        license_value: Any = license_url
        license_status = "recognized_identifier"
    elif license_link and str(license_link).startswith("http"):
        license_value = license_link
        license_status = "declared_link"
    elif license_id:
        license_value = str(license_id)
        license_status = "declared_text"
    else:
        license_value = None
        license_status = "unknown"

    # -- languages ----------------------------------------------------------
    languages = _as_list(front.get("language")) or _as_list(api_card_data.get("language"))
    languages = [str(x).strip().lower() for x in languages if str(x).strip()]

    # -- parameters ---------------------------------------------------------
    safetensors = raw.get("safetensors") or {}
    total_params = safetensors.get("total")
    param_source, param_certainty = "safetensors", "measured"
    if not total_params:
        total_params = inferred_parameters_from_name(repo_id)
        param_source, param_certainty = "repository_name", "inferred"
    else:
        named = inferred_parameters_from_name(repo_id)
        if named and total_params and abs(named - total_params) / max(named, total_params) > 0.35:
            conflicts.append({"field": "parameter_count", "safetensors": total_params,
                              "repository_name": named})

    # -- identity and classification ----------------------------------------
    pipeline_tag = raw.get("pipeline_tag") or front.get("pipeline_tag")
    subcategories = list(PIPELINE_LABELS.get(pipeline_tag or "", []))
    tags = [t for t in (raw.get("tags") or []) if isinstance(t, str)]
    if "conversational" in tags and "Conversational AI" not in subcategories:
        subcategories.append("Conversational AI")

    base_models = declared_base_models(raw)
    datasets = _as_list(front.get("datasets")) or _as_list(api_card_data.get("datasets"))
    architectures = config.get("architectures") or []
    description, description_source = _deterministic_description(card, repo_id, pipeline_tag)

    identifiers = [
        prop("huggingface:repo_id", "Hugging Face repository ID", repo_id),
        prop("huggingface:sha", "Repository revision", raw.get("sha") or ""),
    ]

    is_based_on: list[dict[str, Any]] = [
        {"@type": "SoftwareApplication", "@id": f"{HF}/{b}", "name": humanize(b)}
        for b in base_models
    ]
    is_based_on += [
        {"@type": "Dataset", "@id": f"{HF}/datasets/{d}", "name": str(d)}
        for d in datasets if isinstance(d, str)
    ]

    subject_of: list[dict[str, Any]] = [{
        "@type": "TechArticle",
        "@id": f"{url}/blob/{raw.get('sha') or 'main'}/README.md",
        "url": url,
        "name": "Model card",
        "dateModified": raw.get("lastModified"),
    }] if card_status == "ok" else []

    citations = [
        {"@type": "ScholarlyArticle", "@id": f"https://arxiv.org/abs/{t.split(':', 1)[1]}",
         "url": f"https://arxiv.org/abs/{t.split(':', 1)[1]}"}
        for t in tags if t.startswith("arxiv:")
    ]

    additional: list[dict[str, Any]] = [
        prop("huggingface:pipelineTag", "Pipeline task", pipeline_tag),
        prop("huggingface:library", "Library", raw.get("library_name")),
        prop("huggingface:downloads", "Downloads (last 30 days)", int(raw.get("downloads") or 0)),
        prop("huggingface:downloadsAllTime", "Downloads (all time)",
             int(raw.get("downloadsAllTime") or 0)),
        prop("huggingface:likes", "Likes", int(raw.get("likes") or 0)),
        prop("huggingface:trendingScore", "Trending score", float(raw.get("trendingScore") or 0.0)),
        prop("huggingface:gatedStatus", "Gated status", _gated_status(raw.get("gated"))),
        prop("corpus:licenseStatus", "License status", license_status,
             description="How the declared license was resolved. 'unknown' is not a claim of "
                         "permissive terms."),
        prop("corpus:cardStatus", "Model card status", card_status),
        prop("corpus:vocabularyVersion", "Extension vocabulary version", VOCAB_VERSION),
    ]
    if license_id:
        additional.append(prop("corpus:declaredLicenseIdentifier", "Declared license identifier",
                               str(license_id)))
    if total_params:
        additional.append(prop(
            "ml:parameterCount", "Parameter count", int(total_params), unitText="parameters",
            description=f"Source: {param_source}; certainty: {param_certainty}.",
            valueReference=[
                prop("corpus:source", "Source", param_source),
                prop("corpus:certainty", "Certainty", param_certainty),
            ],
        ))
    if safetensors.get("parameters"):
        weights = safetensors["parameters"]
        dtypes = sorted(weights) if isinstance(weights, dict) else []
        if dtypes:
            additional.append(prop("ml:weightDataTypes", "Weight data types", dtypes))
    experts = config.get("num_experts") or config.get("num_local_experts")
    if experts:
        additional.append(prop(
            "ml:mixtureOfExperts", "Mixture of experts", True,
            description="Total parameters are reported above; active parameters per token are "
                        "not derivable from Hub metadata alone.",
        ))
        additional.append(prop("ml:expertCount", "Expert count", int(experts)))
    if architectures:
        additional.append(prop("ml:architecture", "Architecture", architectures[0]))
    if config.get("model_type"):
        additional.append(prop("ml:modelType", "Model type", config["model_type"]))
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        if isinstance(config.get(key), int):
            additional.append(prop("ml:contextLength", "Context length", config[key],
                                   unitText="tokens"))
            break
    formats = _model_formats(raw)
    if formats:
        additional.append(prop("ml:modelFormat", "Model format", formats))
    eval_entries = raw.get("model-index") or front.get("model-index") or []
    if eval_entries:
        additional.append(prop("corpus:evaluationEntryCount", "Structured evaluation entries",
                               _count_eval_results(eval_entries)))
    if conflicts:
        additional.append(prop(
            "corpus:conflict", "Conflicting source values",
            [c["field"] for c in conflicts],
            description="Both source values are retained in the pipeline's internal record.",
        ))

    item: dict[str, Any] = {
        "@context": CONTEXT,
        "@type": "SoftwareApplication",
        "@id": url,
        "identifier": identifiers,
        "name": humanize(repo_id),
        "alternateName": [repo_id],
        "url": url,
        "mainEntityOfPage": url,
        "applicationCategory": "Machine Learning Model",
        "applicationSubCategory": subcategories,
        "creator": {
            "@type": "Organization",
            "@id": f"{HF}/{raw.get('author') or repo_owner(repo_id)}",
            "name": raw.get("author") or repo_owner(repo_id),
            "url": f"{HF}/{raw.get('author') or repo_owner(repo_id)}",
        },
        "description": description,
        "softwareVersion": raw.get("sha") or "",
        "dateCreated": raw.get("createdAt"),
        "dateModified": raw.get("lastModified"),
        "inLanguage": languages,
        "keywords": _keywords(tags, pipeline_tag),
        "isBasedOn": is_based_on,
        "additionalProperty": [a for a in additional if a["value"] not in (None, "", [])],
    }
    if license_value:
        item["license"] = license_value
    if subject_of:
        item["subjectOf"] = subject_of
    if citations:
        item["citation"] = citations

    internal = {
        "repo_id": repo_id,
        "sha": raw.get("sha") or "",
        "retrieved_at": retrieved_at,
        "card_status": card_status,
        "clean_card_sha256": card.to_json()["clean_sha256"] if card else "",
        "raw_card_sha256": card.raw_sha256 if card else "",
        "documentation_quality": documentation_quality(card) if card else 0.0,
        "card_sections": [{"path": " > ".join(s.path), "role": s.role} for s in card.sections]
        if card else [],
        "card_warnings": card.warnings if card else [],
        "conflicts": conflicts,
        "warnings": warnings,
        "description_source": description_source,
        "declared_base_models": base_models,
        "declared_datasets": [str(d) for d in datasets],
        "parameter_source": param_source if total_params else "none",
        "normalizer_version": "1.0",
    }
    return {"item": item, "internal": internal}


def _deterministic_description(card: CleanCard | None, repo_id: str,
                               pipeline_tag: str | None) -> tuple[str, str]:
    """First substantive prose paragraph of the card, else a metadata sentence.
    Replaced by the grounded short_description during enrichment."""
    if card:
        for section in card.sections:
            if section.role == "citation":
                continue
            for para in section.text(card.clean_text).split("\n\n"):
                text = " ".join(
                    line.strip() for line in para.splitlines()
                    if line.strip() and not line.strip().startswith(("#", "|", ">", "```", "["))
                ).strip()
                if len(text) >= 80:
                    return (text[:497] + "...") if len(text) > 500 else text, "model_card_prose"
    task = (pipeline_tag or "model").replace("-", " ")
    return f"{humanize(repo_id)} is a {task} repository published on the Hugging Face Hub.", \
        "metadata_only"


def _keywords(tags: list[str], pipeline_tag: str | None) -> list[str]:
    out: list[str] = []
    for tag in tags:
        if tag.startswith(TAG_PREFIX_DROP) or ":" in tag:
            continue
        out.append(tag)
    if pipeline_tag and pipeline_tag not in out:
        out.insert(0, pipeline_tag)
    seen: dict[str, None] = {}
    for tag in out:
        seen.setdefault(tag.replace("-", " ") if " " not in tag and "-" in tag else tag, None)
    return list(seen)[:40]


def _model_formats(raw: dict[str, Any]) -> list[str]:
    formats: list[str] = []
    if raw.get("safetensors"):
        formats.append("safetensors")
    if raw.get("gguf"):
        formats.append("gguf")
    for tag in raw.get("tags") or []:
        known = ("onnx", "openvino", "coreml", "tflite", "gguf", "pytorch", "jax",
                 "tensorboard", "tensorflow")
        if tag in known and tag not in formats:
            formats.append(tag)
    return formats


def _count_eval_results(model_index: Any) -> int:
    count = 0
    if isinstance(model_index, list):
        for entry in model_index:
            for result in (entry or {}).get("results", []) if isinstance(entry, dict) else []:
                count += len(result.get("metrics", []) or [])
    return count


def _gated_status(value: Any) -> str:
    if value in (False, None, ""):
        return "open"
    return f"gated:{value}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return None
