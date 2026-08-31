from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .models import FIELD_BODY, FIELD_DESCRIPTION, FIELD_NAME, Document

SUPPORTED_SUFFIXES = {".atom", ".json", ".jsonl", ".rss", ".xml"}


def load_directory(directory: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.iterdir()):
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.suffix.lower() in SUPPORTED_SUFFIXES
        ):
            documents.extend(load_file(path))
    # Source adapters assign stable identities. Distinct offers can legitimately
    # share a vendor URL, so URL-based deduplication would lose records.
    return list({document.id: document for document in documents}.values())


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return _json_documents(values, path.stem)
    if suffix == ".json":
        return _json_documents(json.loads(path.read_text()), path.stem)
    return _xml_documents(path)


def _json_documents(value: Any, site: str) -> list[Document]:
    if isinstance(value, dict) and isinstance(value.get("@graph"), list):
        values = value["@graph"]
    elif isinstance(value, dict) and isinstance(value.get("items"), list):
        values = value["items"]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]

    documents: list[Document] = []
    for number, raw in enumerate(values, 1):
        if not isinstance(raw, dict):
            continue
        schema = raw.get("schema_object") if isinstance(raw.get("schema_object"), dict) else raw
        url = str(raw.get("url") or schema.get("url") or schema.get("@id") or "").strip()
        name = str(raw.get("name") or schema.get("name") or schema.get("headline") or "").strip()
        if not url:
            url = f"urn:ask-hf-hub:{site}:{number}"
        if not name:
            name = url
        source_site = str(raw.get("site") or site)
        text = str(raw.get("text") or _document_text(schema))
        documents.append(
            Document(str(raw.get("id") or url), url, name, source_site, text, schema,
                     _fields(schema))
        )
    return documents


def _xml_documents(path: Path) -> list[Document]:
    root = ET.parse(path).getroot()
    name = _local(root.tag).lower()
    if name == "rss" or _child(root, "channel") is not None:
        return _rss_documents(root, path.stem)
    if name == "feed":
        return _atom_documents(root, path.stem)
    return _generic_xml_documents(root, path.stem)


def _rss_documents(root: ET.Element, site: str) -> list[Document]:
    found_channel = _child(root, "channel")
    channel = found_channel if found_channel is not None else root
    series_name = _text(channel, "title")
    series_description = _clean(_text(channel, "description"))
    series_url = _text(channel, "link")
    series_image = _attribute(channel, "image", "href") or _nested_text(channel, "image", "url")
    series: dict[str, Any] = {
        "@type": "PodcastSeries",
        "name": series_name,
        "description": series_description,
    }
    if series_url:
        series["url"] = series_url
    if series_image:
        series["image"] = series_image

    documents: list[Document] = []
    for number, item in enumerate(_children(channel, "item"), 1):
        title = _clean(_text(item, "title"))
        description = _clean(_text(item, "description") or _text(item, "encoded"))
        link = _text(item, "link")
        guid = _text(item, "guid")
        enclosure = _element(item, "enclosure")
        audio_url = enclosure.get("url", "") if enclosure is not None else ""
        url = link or (guid if guid.startswith(("http://", "https://")) else "") or audio_url
        if not url:
            url = f"urn:ask-hf-hub:{site}:episode:{number}"
        schema: dict[str, Any] = {
            "@context": "https://schema.org",
            "@type": "PodcastEpisode",
            "name": title or f"Episode {number}",
            "description": description,
            "url": url,
            "partOfSeries": series,
        }
        published = _text(item, "pubDate") or _text(item, "published")
        if published:
            schema["datePublished"] = published
        duration = _duration(_text(item, "duration"))
        if duration:
            schema["duration"] = duration
        image = _attribute(item, "image", "href") or series_image
        if image:
            schema["image"] = image
        if guid:
            schema["identifier"] = guid
        if audio_url:
            media: dict[str, Any] = {"@type": "AudioObject", "contentUrl": audio_url}
            if enclosure is not None and enclosure.get("type"):
                media["encodingFormat"] = enclosure.get("type")
            schema["associatedMedia"] = media
        documents.append(
            Document(guid or url, url, schema["name"], site, _search_text(schema), schema,
                     _fields(schema))
        )
    return documents


def _atom_documents(root: ET.Element, site: str) -> list[Document]:
    series = _clean(_text(root, "title"))
    documents: list[Document] = []
    for number, entry in enumerate(_children(root, "entry"), 1):
        name = _clean(_text(entry, "title")) or f"Entry {number}"
        link = _attribute(entry, "link", "href")
        identifier = _text(entry, "id") or link or f"urn:ask-hf-hub:{site}:{number}"
        schema = {
            "@context": "https://schema.org",
            "@type": "CreativeWork",
            "name": name,
            "description": _clean(_text(entry, "summary") or _text(entry, "content")),
            "url": link or identifier,
            "isPartOf": {"@type": "CreativeWorkSeries", "name": series},
        }
        published = _text(entry, "published") or _text(entry, "updated")
        if published:
            schema["datePublished"] = published
        documents.append(
            Document(identifier, schema["url"], name, site, _search_text(schema), schema,
                     _fields(schema))
        )
    return documents


def _generic_xml_documents(root: ET.Element, site: str) -> list[Document]:
    nodes = list(root) or [root]
    documents: list[Document] = []
    for number, node in enumerate(nodes, 1):
        schema = {"@type": _local(node.tag), **_element_dict(node)}
        name = str(schema.get("name") or schema.get("title") or f"{_local(node.tag)} {number}")
        url = str(
            schema.get("url")
            or schema.get("link")
            or schema.get("id")
            or f"urn:ask-hf-hub:{site}:{number}"
        )
        schema.setdefault("name", name)
        schema.setdefault("url", url)
        documents.append(
            Document(url, url, name, site, _search_text(schema), schema, _fields(schema))
        )
    return documents


def _element_dict(element: ET.Element) -> dict[str, Any]:
    result: dict[str, Any] = {_local(key): value for key, value in element.attrib.items()}
    for child in element:
        key = _local(child.tag)
        value: Any = (
            _element_dict(child) if list(child) or child.attrib else _clean(child.text or "")
        )
        if key in result:
            result[key] = result[key] if isinstance(result[key], list) else [result[key]]
            result[key].append(value)
        else:
            result[key] = value
    return result


def _fields(schema: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Field pairs for Document, dropping fields with nothing in them."""
    return tuple((key, value) for key, value in _document_fields(schema).items() if value)


def _is_model(schema: dict[str, Any]) -> bool:
    types = schema.get("@type")
    types = types if isinstance(types, list) else [types]
    return "hf:Model" in types or "SoftwareApplication" in types


# Evidence properties, most specific first. A model trained on a subject is a
# stronger match than one merely evaluated on it, and the split exists so a
# query can tell them apart.
EVIDENCE_KEYS = (
    ("trained on", "hf:trainedOn"),
    ("fine-tuned on", "hf:fineTunedOn"),
    ("evaluated on", "hf:evaluatedOn"),
    ("intended for", "hf:intendedFor"),
    ("architecture for", "hf:architectureFor"),
)


def _document_fields(schema: dict[str, Any]) -> dict[str, str]:
    """Retrieval text for one item, split into the fields scoring weights apply to.

    A model record gets prose assembled from the fields a person would actually
    search on -- the grounded description, capabilities, subject areas, and the
    evidence quotes behind them. Dumping the raw JSON instead (the fallback for
    other item shapes) buries that under namespace plumbing and lets vocabulary
    tokens dominate the match.

    The split matters as much as the content. `name` holds the identifiers a
    known-item query types verbatim; `description` holds what the model is and
    does; `body` holds the corroborating detail that should support a match but
    never win one on its own.
    """
    if not _is_model(schema):
        dump = _search_text(schema)
        return {
            FIELD_NAME: str(schema.get("name") or schema.get("headline") or ""),
            FIELD_DESCRIPTION: str(schema.get("description") or ""),
            FIELD_BODY: dump,
        }

    def values(key: str) -> list[str]:
        value = schema.get(key)
        if value is None:
            return []
        return [str(v) for v in (value if isinstance(value, list) else [value])]

    name = [
        str(schema.get("name") or ""),
        str(schema.get("hf:repository") or ""),
        str((schema.get("creator") or {}).get("name") or ""),
    ]

    description = [
        str(schema.get("description") or ""),
        " ".join(values("hf:category")),
        " ".join(values("hf:capability")),
    ]
    if schema.get("hf:task"):
        # The task is what the model is *for*, so it belongs with the
        # description rather than buried in the body with the plumbing.
        description.append(f"task {schema['hf:task']}")

    body: list[str] = [
        # Subjects are slugs in the record; retrieval wants the words. Without
        # this, `multilingual-nlp` is one opaque token and a query saying
        # "language processing" has nothing to match.
        " ".join(v.replace("-", " ") for v in values("hf:subject")),
        " ".join(values("keywords")),
        " ".join(values("inLanguage")),
        *values("hf:intendedUse"),
        *values("hf:limitation"),
        *values("hf:deploymentNote"),
    ]

    # Evidence quotes carry the relation that justifies them, so a record
    # trained on a subject reads differently from one merely evaluated on it.
    for phrase, key in EVIDENCE_KEYS:
        quotes = values(key)
        if quotes:
            body.append(f"{phrase} {' '.join(quotes)}")

    for label, key in (("library", "hf:library"), ("architecture", "hf:architecture"),
                       ("format", "hf:format")):
        if schema.get(key):
            body.append(f"{label} {' '.join(values(key))}")

    # The paper's own abstract, where the corpus carries one.
    for node in schema.get("citation", []) or []:
        if isinstance(node, dict) and node.get("abstract"):
            body.append(str(node.get("name") or ""))
            body.append(str(node["abstract"]))

    parameters = schema.get("hf:parameters")
    if isinstance(parameters, int):
        body.append(_parameter_phrase(parameters))

    return {
        FIELD_NAME: _clean(" ".join(part for part in name if part)),
        FIELD_DESCRIPTION: _clean(" ".join(part for part in description if part)),
        FIELD_BODY: _clean(" ".join(part for part in body if part)),
    }


def _document_text(schema: dict[str, Any]) -> str:
    """Flat retrieval text: every field, in order. Used for embedding."""
    if not _is_model(schema):
        return _search_text(schema)
    return " ".join(v for v in _document_fields(schema).values() if v)


def _parameter_phrase(parameters: int) -> str:
    """Size as a person would type it, so '7B', 'small', and 'runs on a laptop'
    have something to match.

    The deployability bands overlap the size bands deliberately: an 8B model
    quantized to 4 bits is a laptop model, so 'local' and 'laptop' have to reach
    past the point where 'small' stops applying.
    """
    if parameters >= 1_000_000_000:
        size = f"{parameters / 1_000_000_000:.6g}B"
    else:
        size = f"{parameters / 1_000_000:.6g}M"

    terms: list[str] = []
    if parameters <= 1_000_000_000:
        terms += ["small", "tiny", "lightweight"]
    elif parameters <= 15_000_000_000:
        terms += ["mid-sized", "single GPU"]
    else:
        terms += ["large", "frontier scale", "datacenter"]
    if parameters <= 9_000_000_000:
        terms += ["local", "laptop", "on-device"]
    return f"{size} parameters {' '.join(terms)}"


def _search_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, separators=(",", ":"))


def _clean(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", " ", value or ""))
    return re.sub(r"\s+", " ", value).strip()


def _duration(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith("PT"):
        return value
    parts = value.split(":")
    if all(part.isdigit() for part in parts):
        seconds = sum(int(part) * (60**power) for power, part in enumerate(reversed(parts)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return (
            "PT"
            + (f"{hours}H" if hours else "")
            + (f"{minutes}M" if minutes else "")
            + f"{seconds}S"
        )
    return value


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local(child.tag) == name]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next(iter(_children(element, name)), None)


def _element(element: ET.Element, name: str) -> ET.Element | None:
    return next((node for node in element.iter() if _local(node.tag) == name), None)


def _text(element: ET.Element, name: str) -> str:
    child = _element(element, name)
    return (child.text or "").strip() if child is not None else ""


def _attribute(element: ET.Element, name: str, attribute: str) -> str:
    child = _element(element, name)
    return child.get(attribute, "").strip() if child is not None else ""


def _nested_text(element: ET.Element, parent: str, child: str) -> str:
    parent_element = _element(element, parent)
    return _text(parent_element, child) if parent_element is not None else ""
