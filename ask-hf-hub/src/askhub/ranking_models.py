from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankingModelOption:
    id: str
    label: str
    reasoning_effort: str | None
    max_tokens: int

    def wire(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


# Serving models: cheap and fast enough to rank a handful of candidates on the
# critical path of a live query.
SERVING_MODELS = (
    RankingModelOption("openai/gpt-oss-20b", "GPT-OSS 20B", "low", 400),
    RankingModelOption(
        "google/gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite", None, 160
    ),
    RankingModelOption("microsoft/phi-4", "Phi-4", None, 160),
    RankingModelOption("openai/gpt-oss-120b", "GPT-OSS 120B", "low", 400),
)

# Reference models: too slow and too expensive to serve, which is exactly why
# they are useful for judging. A wide retrieval ranked by one of these produces
# relevance opinions that do not come from the retrieval being measured -- the
# circularity that makes a self-scored precision number worth so little.
REFERENCE_MODELS = (
    RankingModelOption("anthropic/claude-opus-5", "Claude Opus 5", None, 1200),
    RankingModelOption("anthropic/claude-sonnet-5", "Claude Sonnet 5", None, 1200),
    RankingModelOption("openai/gpt-5.6-sol", "GPT-5.6 Sol", None, 1200),
    RankingModelOption("openai/gpt-5.6-terra", "GPT-5.6 Terra", None, 1200),
    # Candidate cheap judges, on trial. Whether a model can judge relevance is
    # measured against the expensive judges, not assumed from its price.
    # Cheapest capable judge in each family, for full-matrix evaluation runs.
    RankingModelOption("openai/gpt-5-nano", "GPT-5 nano", None, 1200),
    RankingModelOption("google/gemma-4-26b-a4b-it", "Gemma 4 26B", None, 1200),
    RankingModelOption("openai/gpt-5.6-luna", "GPT-5.6 Luna", None, 1200),
    RankingModelOption("openai/gpt-5-mini", "GPT-5 mini", None, 1200),
    RankingModelOption("google/gemini-3.7-flash", "Gemini 3.7 Flash", None, 1200),
)

RANKING_MODEL_OPTIONS = SERVING_MODELS + REFERENCE_MODELS
RANKING_MODELS = {option.id: option for option in RANKING_MODEL_OPTIONS}
DEFAULT_RANKING_MODEL = SERVING_MODELS[0].id
REFERENCE_MODEL_IDS = frozenset(option.id for option in REFERENCE_MODELS)


def validate_ranking_model(model: str | None) -> str | None:
    if model in (None, ""):
        return None
    if model not in RANKING_MODELS:
        choices = ", ".join(RANKING_MODELS)
        raise ValueError(f"ranking_model must be one of: {choices}")
    return model
