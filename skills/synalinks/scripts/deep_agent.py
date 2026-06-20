#!/usr/bin/env python3
"""Minimal DeepAgent example.

Usage:
    PYTHONPATH=<repo-root> uv run python scripts/deep_agent.py

Run log:
    references/deep_agent.log

This script demonstrates:
1. synalinks.DeepAgent mounting a temp workdir in a Mirage sandbox.
2. Filesystem + shell tools (list_files / read_file / ...) over a seeded dir.
3. Inspecting sandbox changes via agent.sandbox.diff() at the end.

The workdir is seeded into a copy-on-write sandbox; the real directory on
disk is never modified.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import synalinks

synalinks.enable_logging()

UTILS_SOURCE = '''\
"""Small string utilities."""


def slugify(text):
    """Lowercase a string and replace spaces with hyphens."""
    return text.strip().lower().replace(" ", "-")


def shout(text):
    """Return the text uppercased with an exclamation mark."""
    return text.upper() + "!"
'''


async def main():
    lm = synalinks.LanguageModel(model="ollama/qwen3:8b")

    workdir = tempfile.mkdtemp(prefix="deep_agent_")
    try:
        (Path(workdir) / "utils.py").write_text(UTILS_SOURCE)

        # Keep a reference to the DeepAgent module itself — the sandbox lives
        # on the module instance, not on the wrapping Program.
        deep_agent = synalinks.DeepAgent(
            workdir=workdir,
            language_model=lm,
            max_iterations=8,
            timeout=30,
        )

        inputs = synalinks.Input(data_model=synalinks.ChatMessages)
        outputs = await deep_agent(inputs)

        agent = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="deep_agent_demo",
            description="Inspects a small workdir",
        )

        messages = synalinks.ChatMessages(
            messages=[
                synalinks.ChatMessage(
                    role="user",
                    content="List the files and summarize what utils.py does.",
                ),
            ]
        )

        print("=== DeepAgent ===")
        result = await agent(messages)
        print(result.get("messages")[-1].get("content"))

        print("\n=== Sandbox diff ===")
        print(deep_agent.sandbox.diff())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
