#!/usr/bin/env python3
"""Minimal RecursiveLanguageModelAgent (RLM) example.

Usage:
    PYTHONPATH=<repo-root> uv run python scripts/rlm_agent.py

Run log:
    references/rlm_agent.log

This script demonstrates:
1. synalinks.RLM over a small text input.
2. recursive=False — purely computational, no separate sub-LM needed.
3. The agent writes Python that runs in a persistent Mirage sandbox REPL
   and terminates by calling the in-sandbox `submit(result=...)`.
"""

import asyncio

import synalinks

synalinks.enable_logging()


class Doc(synalinks.DataModel):
    """Input document."""

    text: str = synalinks.Field(description="The text to analyze")


class Answer(synalinks.DataModel):
    """Final answer."""

    word_count: int = synalinks.Field(description="Number of words in the text")
    answer: str = synalinks.Field(description="A short summary of the result")


async def main():
    lm = synalinks.LanguageModel(model="ollama/qwen3:8b")

    inputs = synalinks.Input(data_model=Doc)
    outputs = await synalinks.RLM(
        data_model=Answer,
        language_model=lm,
        recursive=False,  # purely computational; no sub_language_model required
        max_iterations=6,
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="rlm_word_counter",
        description="Counts words in a text using code execution",
    )

    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Synalinks builds neuro-symbolic language model programs. "
        "Recursive language models reason by running code in a sandbox."
    )

    print("=== RLM word count ===")
    result = await agent(Doc(text=text))
    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
