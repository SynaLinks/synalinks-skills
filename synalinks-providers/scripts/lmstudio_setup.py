#!/usr/bin/env python3
"""LMStudio / vLLM setup helper for Synalinks.

Local OpenAI-compatible servers need:
1. A dummy OPENAI_API_KEY (any non-empty string)
2. A LiteLLM model registration (so cost tracking returns 0.0 instead of None)
3. An api_base pointing to the local server

Usage:
    from lmstudio_setup import create_lmstudio_language_model

    lm = create_lmstudio_language_model("ibm/granite-4-h-tiny")
"""

import os

import litellm
import synalinks


def create_lmstudio_language_model(
    model_name: str,
    api_base: str = "http://localhost:1234/v1",
    max_tokens: int = 4096,
    api_key: str = "lm-studio",
    **kwargs,
) -> synalinks.LanguageModel:
    """Create a Synalinks LanguageModel configured for an LMStudio-compatible server.

    Args:
        model_name: Model name (without provider prefix). Will be prefixed with "openai/".
        api_base: Base URL of the local OpenAI-compatible server.
        max_tokens: Max output tokens (used for LiteLLM registration).
        api_key: Dummy API key. Any non-empty string works.
        **kwargs: Forwarded to synalinks.LanguageModel.
    """
    os.environ["OPENAI_API_KEY"] = api_key
    full_model_name = f"openai/{model_name}"

    litellm.register_model({
        full_model_name: {
            "max_tokens": max_tokens,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    })

    return synalinks.LanguageModel(
        model=full_model_name,
        api_base=api_base,
        **kwargs,
    )


# Same helper, friendlier name for vLLM
def create_vllm_language_model(
    model_name: str,
    api_base: str = "http://localhost:8000/v1",
    **kwargs,
) -> synalinks.LanguageModel:
    """Create a Synalinks LanguageModel configured for a vLLM server."""
    return create_lmstudio_language_model(model_name, api_base=api_base, **kwargs)


if __name__ == "__main__":
    import asyncio

    async def main():
        lm = create_lmstudio_language_model("ibm/granite-4-h-tiny")

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
