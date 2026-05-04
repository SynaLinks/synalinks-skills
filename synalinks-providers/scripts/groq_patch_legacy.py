#!/usr/bin/env python3
"""Groq provider patch for Synalinks.

Synalinks defaults to tool-calling for structured output, but Groq's tool
calling is unreliable for that purpose, and Groq rejects the `tool_calls` field
in `ChatMessage` serialization.

This patch:
1. Strips `tool_calls` / `tool_call_id` from messages.
2. For regular Groq models — switches to `json_schema` response format.
3. For compound models (compound-beta, compound-beta-mini) — uses
   `json_object` mode (compound models don't support json_schema).

Usage:
    from groq_patch import create_groq_language_model

    lm = create_groq_language_model("llama-3.1-8b-instant")
    lm = create_groq_language_model("compound-beta")
"""

import asyncio
import copy
import json
import warnings

import litellm
import synalinks
from synalinks.src.backend import ChatRole
from synalinks.src.modules.language_models.language_model import LanguageModel
from synalinks.src.utils.nlp_utils import shorten_text


def _clean_messages_for_groq(messages: list) -> list:
    """Remove tool_calls / tool_call_id fields Groq rejects."""
    cleaned = []
    for msg in messages:
        clean_msg = {"role": msg.get("role"), "content": msg.get("content", "")}
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            clean_msg["tool_call_id"] = msg["tool_call_id"]
        cleaned.append(clean_msg)
    return cleaned


_original_call = None


async def _patched_call(self, messages, schema=None, streaming=False, **kwargs):
    formatted_messages = messages.get_json().get("messages", [])
    input_kwargs = copy.deepcopy(kwargs)
    schema = copy.deepcopy(schema)

    if self.model.startswith("groq"):
        formatted_messages = _clean_messages_for_groq(formatted_messages)

    if schema:
        if self.model.startswith("groq"):
            kwargs.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "structured_output", "schema": schema},
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
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": schema},
                    "strict": True,
                }
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
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"schema": schema},
                    "strict": True,
                }
            })
        else:
            raise ValueError(f"LM provider '{self.model.split('/')[0]}' not supported")

    if self.api_base:
        kwargs.update({"api_base": self.api_base})
    if streaming and schema:
        streaming = False
    if streaming:
        kwargs.update({"stream": True})

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


def patch_synalinks_for_groq():
    """Patch Synalinks to support Groq structured output. Idempotent."""
    global _original_call
    if _original_call is None:
        _original_call = LanguageModel.__call__
        LanguageModel.__call__ = _patched_call


def create_groq_language_model(model_name: str, **kwargs) -> synalinks.LanguageModel:
    """Create a Groq LanguageModel with the patch applied."""
    patch_synalinks_for_groq()
    return synalinks.LanguageModel(model=f"groq/{model_name}", **kwargs)


if __name__ == "__main__":
    # Demo
    async def main():
        lm = create_groq_language_model("llama-3.1-8b-instant")

        class Q(synalinks.DataModel):
            query: str = synalinks.Field(description="The user query")

        class A(synalinks.DataModel):
            answer: str = synalinks.Field(description="The answer")

        inputs = synalinks.Input(data_model=Q)
        outputs = await synalinks.Generator(data_model=A, language_model=lm)(inputs)
        program = synalinks.Program(inputs=inputs, outputs=outputs)
        result = await program(Q(query="Capital of France?"))
        print(result.prettify_json())

    asyncio.run(main())
