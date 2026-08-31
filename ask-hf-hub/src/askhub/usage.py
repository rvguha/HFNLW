from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {
        name: getattr(value, name)
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "cost",
            "prompt_tokens_details",
            "completion_tokens_details",
            "input_tokens_details",
            "output_tokens_details",
        )
        if hasattr(value, name)
    }


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class ModelUsage:
    model: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    unpriced_calls: int = 0
    phases: dict[str, int] = field(default_factory=dict)

    def wire(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "cost": round(self.cost, 12),
            "unpriced_calls": self.unpriced_calls,
            "phases": dict(sorted(self.phases.items())),
        }


@dataclass(slots=True)
class UsageLedger:
    query_id: str
    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    def record(self, model: str, phase: str, usage: Any) -> None:
        data = _mapping(usage)
        prompt = _integer(data.get("prompt_tokens", data.get("input_tokens")))
        completion = _integer(data.get("completion_tokens", data.get("output_tokens")))
        total = _integer(data.get("total_tokens")) or prompt + completion
        prompt_details = _mapping(
            data.get("prompt_tokens_details", data.get("input_tokens_details"))
        )
        completion_details = _mapping(
            data.get("completion_tokens_details", data.get("output_tokens_details"))
        )
        cached = _integer(prompt_details.get("cached_tokens"))
        reasoning = _integer(completion_details.get("reasoning_tokens"))
        if not reasoning:
            reasoning = _integer(data.get("reasoning_tokens"))
        cost = _number(data.get("cost"))

        item = self.by_model.setdefault(model, ModelUsage(model))
        item.calls += 1
        item.prompt_tokens += prompt
        item.completion_tokens += completion
        item.total_tokens += total
        item.cached_tokens += cached
        item.reasoning_tokens += reasoning
        item.phases[phase] = item.phases.get(phase, 0) + 1
        if cost is None:
            item.unpriced_calls += 1
        else:
            item.cost += cost

    def wire(self) -> dict[str, Any]:
        models = [item.wire() for item in sorted(self.by_model.values(), key=lambda x: x.model)]
        return {
            "query_id": self.query_id,
            "models": models,
            "calls": sum(item.calls for item in self.by_model.values()),
            "prompt_tokens": sum(item.prompt_tokens for item in self.by_model.values()),
            "completion_tokens": sum(item.completion_tokens for item in self.by_model.values()),
            "total_tokens": sum(item.total_tokens for item in self.by_model.values()),
            "reasoning_tokens": sum(item.reasoning_tokens for item in self.by_model.values()),
            "cached_tokens": sum(item.cached_tokens for item in self.by_model.values()),
            "cost": round(sum(item.cost for item in self.by_model.values()), 12),
            "unpriced_calls": sum(item.unpriced_calls for item in self.by_model.values()),
        }
