#!/usr/bin/env python3
"""OpenRouter provider patch for Synalinks.

Synalinks doesn't recognize the `openrouter/` prefix. The patch routes
OpenRouter through the OpenAI structured-output path (OpenRouter is
OpenAI-compatible at the wire level).

Also exposes `OpenRouterEmbeddingModel` for embeddings, since LiteLLM does NOT
support OpenRouter embeddings — it uses direct API calls instead. The wrapper
also coerces non-string inputs to strings, which makes it compatible with
OMEGA's `tree.flatten()` over trainable variables.

Usage:
    from openrouter_patch import (
        create_openrouter_language_model,
        OpenRouterEmbeddingModel,
    )

    lm = create_openrouter_language_model("meta-llama/llama-3.1-8b-instruct")

    em = OpenRouterEmbeddingModel(
        "qwen/qwen3-embedding-8b",
        provider={"only": ["nebius"], "allow_fallbacks": False},
    )
"""

import asyncio
import copy
import json
import os
import warnings
from typing import Any, Dict, List, Optional

import httpx
import litellm
import synalinks
from synalinks.src.backend import ChatRole
from synalinks.src.modules.language_models.language_model import LanguageModel
from synalinks.src.utils.nlp_utils import shorten_text


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

_provider_configs: Dict[int, Dict[str, Any]] = {}
_original_call = None


async def _patched_call_openrouter(self, messages, schema=None, streaming=False, **kwargs):
    formatted_messages = messages.get_json().get("messages", [])
    input_kwargs = copy.deepcopy(kwargs)
    schema = copy.deepcopy(schema)

    if schema:
        if self.model.startswith("openrouter"):
            if "properties" in schema:
                for prop_value in schema["properties"].values():
                    if "$ref" in prop_value and "description" in prop_value:
                        del prop_value["description"]
            kwargs.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "structured_output", "strict": True, "schema": schema},
                }
            })
        elif self.model.startswith("anthropic"):
            kwargs.update({
                "tools": [{
                    "name": "structured_output",
                    "description": "Generate a valid JSON output",
                    "input_schema": {
                        "type": "object",
                        "properties": schema.get("properties"),
                        "required": schema.get("required"),
                    },
                }],
                "tool_choice": {"type": "tool", "name": "structured_output"},
            })
        elif self.model.startswith("ollama") or self.model.startswith("mistral"):
            kwargs.update({
                "response_format": {"type": "json_schema", "json_schema": {"schema": schema}, "strict": True}
            })
        elif self.model.startswith("openai") or self.model.startswith("azure"):
            if "properties" in schema:
                for prop_value in schema["properties"].values():
                    if "$ref" in prop_value and "description" in prop_value:
                        del prop_value["description"]
            kwargs.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "structured_output", "strict": True, "schema": schema},
                }
            })
        elif (
            self.model.startswith("gemini")
            or self.model.startswith("xai")
            or self.model.startswith("hosted_vllm")
        ):
            kwargs.update({
                "response_format": {"type": "json_schema", "json_schema": {"schema": schema}, "strict": True}
            })
        else:
            raise ValueError(f"LM provider '{self.model.split('/')[0]}' not supported")

    if self.api_base:
        kwargs.update({"api_base": self.api_base})
    if streaming and schema:
        streaming = False
    if streaming:
        kwargs.update({"stream": True})

    instance_id = id(self)
    if instance_id in _provider_configs:
        kwargs.setdefault("extra_body", {})
        kwargs["extra_body"]["provider"] = _provider_configs[instance_id]

    for _ in range(self.retry):
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=formatted_messages,
                timeout=self.timeout,
                caching=self.caching,
                **kwargs,
            )
            if hasattr(response, "_hidden_params") and "response_cost" in response._hidden_params:
                self.last_call_cost = response._hidden_params["response_cost"]
                if self.last_call_cost is not None:
                    self.cumulated_cost += self.last_call_cost

            if self.model.startswith("anthropic") and schema:
                response_str = response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            else:
                response_str = response["choices"][0]["message"]["content"].strip()

            if schema:
                return json.loads(response_str)
            return {"role": ChatRole.ASSISTANT, "content": response_str, "tool_call_id": None, "tool_calls": []}
        except Exception as e:
            warnings.warn(f"Error calling {self}: {shorten_text(str(e))}")
        await asyncio.sleep(1)

    return self.fallback(messages, schema=schema, streaming=streaming, **input_kwargs) if self.fallback else None


def patch_synalinks_for_openrouter():
    """Patch Synalinks to support OpenRouter. Idempotent."""
    global _original_call
    if _original_call is None:
        _original_call = LanguageModel.__call__
        LanguageModel.__call__ = _patched_call_openrouter


def create_openrouter_language_model(
    model_name: str,
    provider: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> synalinks.LanguageModel:
    """Create a Synalinks LanguageModel configured for OpenRouter.

    Args:
        model_name: OpenRouter model name (e.g. "meta-llama/llama-3.1-8b-instruct")
        provider: Optional provider routing (e.g. {"only": ["DeepInfra"], "allow_fallbacks": False})
    """
    patch_synalinks_for_openrouter()
    full_model_name = f"openrouter/{model_name}"
    lm = synalinks.LanguageModel(model=full_model_name, **kwargs)
    if provider:
        _provider_configs[id(lm)] = provider
    return lm


class OpenRouterEmbeddingModel:
    """OpenRouter embeddings via direct API calls (LiteLLM doesn't support them).

    OMEGA-compatible: coerces non-string leaves to strings so tree.flatten()
    output from trainable variables works without surprises.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[Dict[str, Any]] = None,
        retry: int = 5,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.provider = provider
        self.retry = retry
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")

    async def __call__(self, texts: List[str]) -> Optional[Dict[str, List[List[float]]]]:
        texts = [str(t) for t in texts]

        for _ in range(self.retry):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    payload: Dict[str, Any] = {"model": self.model, "input": texts}
                    if self.provider:
                        payload["provider"] = self.provider

                    response = await client.post(
                        f"{OPENROUTER_API_BASE}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        vectors = [item.get("embedding", []) for item in data.get("data", [])]
                        return {"embeddings": vectors}
                    warnings.warn(f"OpenRouter embedding error ({response.status_code}): {response.text}")
            except Exception as e:
                warnings.warn(f"Error calling OpenRouter embeddings: {e}")
            await asyncio.sleep(1)
        return None


if __name__ == "__main__":
    async def main():
        lm = create_openrouter_language_model("meta-llama/llama-3.1-8b-instruct")
        em = OpenRouterEmbeddingModel(
            "qwen/qwen3-embedding-8b",
            provider={"only": ["nebius"], "allow_fallbacks": False},
        )

        class Q(synalinks.DataModel):
            query: str = synalinks.Field(description="The user query")

        class A(synalinks.DataModel):
            answer: str = synalinks.Field(description="The answer")

        inputs = synalinks.Input(data_model=Q)
        outputs = await synalinks.Generator(data_model=A, language_model=lm)(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)
        result = await program(Q(query="Capital of France?"))
        print(result.prettify_json())

        embeddings = await em(["Hello, world!"])
        print(f"Got {len(embeddings['embeddings'][0])}-dim embedding")

    asyncio.run(main())
