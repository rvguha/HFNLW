from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Protocol

import numpy as np
from openai import AsyncOpenAI

from .usage import UsageLedger


class Embeddings(Protocol):
    dimensions: int
    cache_key: str

    async def embed(self, texts: list[str]) -> np.ndarray: ...


class LanguageModel(Protocol):
    async def structured(
        self,
        instruction: str,
        payload: dict[str, Any],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]: ...


class HashEmbeddings:
    """Deterministic local embedding for demos and tests."""

    dimensions = 384
    cache_key = "hash-embeddings-v1-384"

    async def embed(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                number = int.from_bytes(digest, "little")
                output[row, number % self.dimensions] += 1 if number & 1 else -1
            norm = float(np.linalg.norm(output[row]))
            if norm:
                output[row] /= norm
        return output


class NullLanguageModel:
    async def structured(
        self, instruction: str, payload: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        raise RuntimeError("No language model is configured")


class OpenRouterProvider:
    dimensions = 1536

    def __init__(
        self,
        key: str,
        llm_model: str,
        embedding_model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        app_url: str | None = None,
        app_title: str = "ask-hf-hub",
        reasoning_effort: str | None = None,
        provider_sort: str | None = None,
    ):
        headers = {"X-OpenRouter-Title": app_title}
        if app_url:
            headers["HTTP-Referer"] = app_url
        self.client = AsyncOpenAI(api_key=key, base_url=base_url, default_headers=headers)
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.reasoning_effort = reasoning_effort
        self.provider_sort = provider_sort
        self.cache_key = f"openrouter:{embedding_model}"

    async def embed(
        self,
        texts: list[str],
        *,
        usage: UsageLedger | None = None,
        usage_phase: str = "embedding",
    ) -> np.ndarray:
        options = {"user": usage.query_id} if usage is not None else {}
        response = await self.client.embeddings.create(
            model=self.embedding_model, input=texts, **options
        )
        if usage is not None:
            usage.record(
                response.model or self.embedding_model,
                usage_phase,
                getattr(response, "usage", None),
            )
        matrix = np.asarray([item.embedding for item in response.data], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        self.dimensions = matrix.shape[1]
        return matrix

    async def structured(
        self,
        instruction: str,
        payload: dict[str, Any],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        usage: UsageLedger | None = None,
        usage_phase: str = "language_model",
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if usage is not None:
            options["user"] = usage.query_id
        extra_body: dict[str, Any] = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        if self.provider_sort:
            extra_body["provider"] = {"sort": self.provider_sort}
        if extra_body:
            options["extra_body"] = extra_body
        response = await self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
            **options,
        )
        if usage is not None:
            usage.record(
                response.model or self.llm_model,
                usage_phase,
                getattr(response, "usage", None),
            )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("OpenRouter returned an empty response")
        result = json.loads(content)
        # Some otherwise-compatible models return a one-element JSON array even
        # when response_format requests an object. Accept that harmless wrapper.
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        if not isinstance(result, dict):
            raise RuntimeError("OpenRouter returned JSON with an unsupported shape")
        return result


class MeteredLanguageModel:
    def __init__(self, model: LanguageModel, usage: UsageLedger, phase: str):
        self.model = model
        self.usage = usage
        self.phase = phase

    async def structured(
        self,
        instruction: str,
        payload: dict[str, Any],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        options = {"max_tokens": max_tokens, "temperature": temperature}
        options = {key: value for key, value in options.items() if value is not None}
        if isinstance(self.model, OpenRouterProvider):
            return await self.model.structured(
                instruction,
                payload,
                usage=self.usage,
                usage_phase=self.phase,
                **options,
            )
        return await self.model.structured(instruction, payload, **options)


async def embed_with_usage(
    embedder: Embeddings,
    texts: list[str],
    usage: UsageLedger,
    phase: str,
) -> np.ndarray:
    if isinstance(embedder, OpenRouterProvider):
        return await embedder.embed(texts, usage=usage, usage_phase=phase)
    return await embedder.embed(texts)


def similarity_score(value: float) -> int:
    return max(0, min(100, math.floor((value + 1) * 50)))
