"""Model-card preprocessing.

Raw Markdown is preserved untouched on disk; this module derives a *clean text*
view plus a section tree with offsets back into both (s4.2). The clean view is
what the enrichment stage reads and what evidence quotes are validated against,
so every transformation here is deterministic and recorded.

Model cards are untrusted input (s12). Nothing in this module executes card
content, follows links, or treats card text as instructions.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

FRONT_MATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)


class CardLoader(yaml.SafeLoader):
    """YAML for card front matter, with implicit booleans switched off.

    Card metadata is identifiers, not typed data, and YAML 1.1 resolves `no`,
    `off`, `n`, `yes`, `on` and `y` to booleans. `language: [en, no]` therefore
    made Norwegian arrive as False and ship as the string "false" -- 51 records
    in this corpus carried a phantom language code, and no query for Norwegian
    could reach them.
    """


CardLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
HTML_COMMENT_OPEN = re.compile(r"<!--")
HTML_COMMENT_CLOSE = re.compile(r"-->")
ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_+-]*)")
IMAGE_ONLY = re.compile(
    r"^\s*(!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>|\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\))+\s*$")
BADGE_HOST = re.compile(
    r"(shields\.io|badgen\.net|badge\.fury\.io|img\.shields|colab\.research\.google\.com/assets)")
HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*\b[^>]*>")
BIBTEX = re.compile(r"^\s*@(article|inproceedings|misc|book|techreport|inbook|phdthesis)\s*\{")
WEIGHT_FILE = re.compile(r"\.(gguf|safetensors|bin|onnx|pt|pth|ckpt|npz|tflite|mlmodel)\b", re.I)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")

# Long code blocks are dropped down to a marker; short ones are kept because a
# four-line usage snippet is often the clearest statement of intended use.
MAX_KEPT_CODE_LINES = 18

SECTION_ROLES: list[tuple[str, tuple[str, ...]]] = [
    ("intended_use", ("intended use", "intended uses", "direct use", "downstream use",
                      "use cases", "usage", "how to use", "get started", "applications")),
    ("training", ("training", "training data", "training details", "training procedure",
                  "dataset", "datasets", "data", "pretraining", "fine-tuning", "finetuning")),
    ("evaluation", ("evaluation", "results", "benchmark", "benchmarks", "performance",
                    "metrics", "testing", "accuracy", "leaderboard")),
    ("limitations", ("limitation", "limitations", "bias", "risks", "caveats", "ethical",
                     "out-of-scope", "out of scope", "known issues", "safety")),
    ("languages", ("language", "languages", "supported languages", "multilingual")),
    ("deployment", ("hardware", "requirements", "inference", "deployment", "quantization",
                    "quantisation", "system requirements", "memory", "vram", "runtime",
                    "installation", "install", "serving")),
    ("citation", ("citation", "citing", "bibtex", "references", "reference", "acknowledg")),
    ("license", ("license", "licence", "terms", "usage policy")),
    ("model_details", ("model details", "model description", "model card", "overview",
                       "introduction", "abstract", "summary", "about")),
]


@dataclass
class Section:
    level: int
    heading: str
    path: list[str]
    role: str
    clean_start: int
    clean_end: int
    raw_start: int
    raw_end: int

    def text(self, clean: str) -> str:
        return clean[self.clean_start : self.clean_end]


@dataclass
class CleanCard:
    repo_id: str
    raw_sha256: str
    front_matter: dict[str, Any]
    front_matter_error: str
    clean_text: str
    sections: list[Section]
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "raw_sha256": self.raw_sha256,
            "front_matter": self.front_matter,
            "front_matter_error": self.front_matter_error,
            "clean_text": self.clean_text,
            "clean_sha256": hashlib.sha256(self.clean_text.encode("utf-8")).hexdigest(),
            "sections": [asdict(s) for s in self.sections],
            "warnings": self.warnings,
            "stats": self.stats,
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> CleanCard:
        card = cls(
            repo_id=obj["repo_id"],
            raw_sha256=obj["raw_sha256"],
            front_matter=obj["front_matter"],
            front_matter_error=obj.get("front_matter_error", ""),
            clean_text=obj["clean_text"],
            sections=[Section(**s) for s in obj["sections"]],
            warnings=obj.get("warnings", []),
            stats=obj.get("stats", {}),
        )
        return card

    def roles(self) -> set[str]:
        return {s.role for s in self.sections}


def clean_card(repo_id: str, raw: str) -> CleanCard:
    raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    warnings: list[str] = []

    front_matter, fm_error, body, body_offset = _split_front_matter(raw)
    if fm_error:
        warnings.append(f"front_matter: {fm_error}")

    kept: list[tuple[str, int, int]] = []  # (line, raw_start, raw_end)
    dropped = {"badge": 0, "image": 0, "comment": 0, "code": 0, "citation": 0, "inventory": 0}

    lines = body.splitlines(keepends=True)
    offset = body_offset
    in_comment = False
    i = 0
    while i < len(lines):
        line = lines[i]
        start, end = offset, offset + len(line)
        offset = end
        stripped = line.strip()

        if in_comment:
            dropped["comment"] += 1
            if HTML_COMMENT_CLOSE.search(line):
                in_comment = False
            i += 1
            continue
        if HTML_COMMENT_OPEN.search(line) and not HTML_COMMENT_CLOSE.search(line):
            in_comment = True
            dropped["comment"] += 1
            i += 1
            continue
        if HTML_COMMENT_OPEN.search(line):
            line = re.sub(r"<!--.*?-->", "", line)
            stripped = line.strip()

        fence = FENCE.match(line)
        if fence:
            marker, lang = fence.group(1)[:3], fence.group(2)
            block: list[str] = []
            block_start = start
            i += 1
            while i < len(lines):
                nxt = lines[i]
                nend = offset + len(lines[i])
                offset = nend
                i += 1
                if nxt.strip().startswith(marker):
                    block_end = nend
                    break
                block.append(nxt)
            else:
                block_end = offset
                warnings.append("unterminated_code_fence")
            body_text = "".join(block)
            is_citation = bool(BIBTEX.search(body_text)) or lang.lower() in {"bibtex", "bib"}
            if is_citation:
                dropped["citation"] += 1
            elif len(block) > MAX_KEPT_CODE_LINES:
                dropped["code"] += 1
                kept.append((f"[code block omitted: {lang or 'text'}, {len(block)} lines]\n",
                             block_start, block_end))
            else:
                kept.append((f"```{lang}\n", block_start, block_start))
                kept.append((body_text, block_start, block_end))
                kept.append(("```\n", block_end, block_end))
            continue

        if ATX_HEADING.match(stripped):
            kept.append((line, start, end))
            i += 1
            continue
        if stripped and IMAGE_ONLY.match(stripped):
            dropped["badge" if BADGE_HOST.search(stripped) else "image"] += 1
            i += 1
            continue
        if BADGE_HOST.search(stripped):
            dropped["badge"] += 1
            i += 1
            continue
        if TABLE_ROW.match(stripped) and _is_file_inventory_row(stripped):
            dropped["inventory"] += 1
            i += 1
            continue

        cleaned = HTML_TAG.sub("", line)
        kept.append((cleaned, start, end))
        i += 1

    clean_text, spans = _assemble(kept)
    sections = _sections(clean_text, spans)
    stats = {
        "placeholders": len(PLACEHOLDER.findall(clean_text)),
        "raw_chars": len(raw),
        "clean_chars": len(clean_text),
        "dropped_lines": dropped,
        "section_count": len(sections),
        "roles": sorted({s.role for s in sections}),
    }
    return CleanCard(repo_id, raw_sha, front_matter, fm_error, clean_text, sections,
                     warnings, stats)


def _split_front_matter(raw: str) -> tuple[dict[str, Any], str, str, int]:
    match = FRONT_MATTER.match(raw)
    if not match:
        return {}, "", raw, 0
    text = match.group(1)
    try:
        parsed = yaml.load(text, Loader=CardLoader)
        if not isinstance(parsed, dict):
            return {}, "front matter is not a mapping", raw[match.end():], match.end()
        return parsed, "", raw[match.end():], match.end()
    except yaml.YAMLError as exc:
        return {}, str(exc).splitlines()[0], raw[match.end():], match.end()


def _is_file_inventory_row(row: str) -> bool:
    cells = [c.strip() for c in row.strip("|").split("|")]
    if len(cells) < 2:
        return False
    filey = sum(1 for c in cells if WEIGHT_FILE.search(c))
    return filey >= 1 and filey >= len(cells) / 3


def _assemble(kept: list[tuple[str, int, int]]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate kept lines and return (clean_text, spans) where each span is
    (clean_start, raw_start, raw_end) -- the map back to the original card."""
    out: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0
    blank_run = 0
    for line, raw_start, raw_end in kept:
        if not line.strip():
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        out.append(line)
        spans.append((pos, raw_start, raw_end))
        pos += len(line)
    return "".join(out), spans


def _sections(clean: str, spans: list[tuple[int, int, int]]) -> list[Section]:
    heads: list[tuple[int, int, str, int, int]] = []  # clean_pos, level, title, raw_s, raw_e
    for clean_pos, raw_start, raw_end in spans:
        line_end = clean.find("\n", clean_pos)
        line = clean[clean_pos : line_end if line_end != -1 else len(clean)]
        m = ATX_HEADING.match(line.strip())
        if m:
            heads.append((clean_pos, len(m.group(1)), m.group(2).strip(), raw_start, raw_end))

    sections: list[Section] = []
    if not heads or heads[0][0] > 0:
        end = heads[0][0] if heads else len(clean)
        raw_end = heads[0][3] if heads else (spans[-1][2] if spans else 0)
        sections.append(Section(0, "", [], "preamble", 0, end,
                                spans[0][1] if spans else 0, raw_end))

    stack: list[tuple[int, str]] = []
    for idx, (clean_pos, level, title, raw_start, _raw_end) in enumerate(heads):
        while stack and stack[-1][0] >= level:
            stack.pop()
        path = [t for _, t in stack] + [title]
        stack.append((level, title))
        next_pos = heads[idx + 1][0] if idx + 1 < len(heads) else len(clean)
        next_raw = heads[idx + 1][3] if idx + 1 < len(heads) else (spans[-1][2] if spans else 0)
        sections.append(Section(level, title, path, classify_heading(title),
                                clean_pos, next_pos, raw_start, next_raw))
    return sections


def classify_heading(title: str) -> str:
    low = title.strip().lower()
    if not low:
        return "preamble"
    for role, keywords in SECTION_ROLES:
        if any(k in low for k in keywords):
            return role
    return "other"


# Priority for section selection when a card is too long to send whole (s7.5).
ENRICHMENT_PRIORITY = [
    "preamble", "model_details", "intended_use", "training", "evaluation",
    "limitations", "languages", "deployment", "license", "other",
]


def select_sections(card: CleanCard, max_chars: int) -> tuple[str, list[str]]:
    """Return (text, section_paths) for enrichment. Selects whole sections in
    evidence-value order rather than truncating the head of the card, so quotes
    stay verifiable against a contiguous source span."""
    if len(card.clean_text) <= max_chars:
        return card.clean_text, [" > ".join(s.path) for s in card.sections]

    chosen: list[Section] = []
    budget = max_chars
    for role in ENRICHMENT_PRIORITY:
        for section in card.sections:
            if section.role != role or section in chosen:
                continue
            if section.role == "citation":
                continue
            length = section.clean_end - section.clean_start
            if length > budget:
                continue
            chosen.append(section)
            budget -= length
    chosen.sort(key=lambda s: s.clean_start)
    text = "\n".join(s.text(card.clean_text).rstrip() for s in chosen)
    return text, [" > ".join(s.path) for s in chosen]


# The unfilled Hub card template. Every field reads "[More Information Needed]",
# and the result runs to ~2,900 characters -- far past any length threshold --
# while saying nothing. Cards like this are what s1.2 means by "boilerplate-only
# cards with no usable description".
PLACEHOLDER = re.compile(r"\[?More Information Needed\]?", re.I)


def placeholder_count(card: CleanCard) -> int:
    return len(PLACEHOLDER.findall(card.clean_text))


def is_boilerplate(card: CleanCard, max_placeholders: int = 5) -> bool:
    """True when the card is the generated template with the blanks still in it.

    Measured over 4,000 cards: real cards that leave a field or two unfilled
    carry one or two placeholders, while untouched templates carry six to forty.
    The gap between those populations is wide enough that a count separates them
    without needing to weigh it against length.
    """
    return placeholder_count(card) > max_placeholders


def documentation_quality(card: CleanCard) -> float:
    """Deterministic card-derived quality in [0, 1] (s2.2). Computed after the
    card is fetched; the discovery-time proxy in select.py uses metadata only."""
    roles = card.roles()
    score = 0.0
    length = len(card.clean_text)
    score += 0.25 * min(1.0, length / 4000.0)
    for role, weight in (("intended_use", 0.15), ("training", 0.15),
                         ("evaluation", 0.15), ("limitations", 0.10)):
        if role in roles:
            score += weight
    fm = card.front_matter
    if fm.get("license"):
        score += 0.05
    if fm.get("language"):
        score += 0.05
    if fm.get("datasets"):
        score += 0.05
    if fm.get("model-index") or fm.get("model_index"):
        score += 0.05
    return round(min(1.0, score), 4)
