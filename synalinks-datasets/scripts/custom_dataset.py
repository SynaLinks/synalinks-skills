#!/usr/bin/env python3
"""Custom iterable dataset (Synalinks v0.8.004+).

Usage:
    uv run -- python scripts/custom_dataset.py

Demonstrates passing a custom iterable (with optional __len__) directly to
program.fit(). Useful for streaming from files, databases, or APIs without
materializing the entire dataset in memory.
"""

import asyncio
import json
from pathlib import Path

import synalinks


class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="A question")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="Final answer")


class JSONLDataset:
    """Yield (Question, Answer) tuples from a JSONL file."""

    def __init__(self, path: str):
        self.path = Path(path)

    def __iter__(self):
        with self.path.open() as f:
            for line in f:
                row = json.loads(line)
                yield (
                    Question(question=row["q"]),
                    Answer(answer=row["a"]),
                )

    def __len__(self):
        # __len__ is optional but enables progress bars during fit/evaluate.
        # It does NOT enable validation_split — that arg only works with NumPy arrays.
        with self.path.open() as f:
            return sum(1 for _ in f)


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="custom_data")

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )

    # Write a tiny JSONL file for the demo
    demo = Path("demo.jsonl")
    demo.write_text(
        "\n".join([
            json.dumps({"q": "What is 2+2?", "a": "4"}),
            json.dumps({"q": "Capital of France?", "a": "Paris"}),
            json.dumps({"q": "Who wrote Hamlet?", "a": "William Shakespeare"}),
        ])
    )

    dataset = JSONLDataset("demo.jsonl")
    print(f"Dataset has {len(dataset)} examples")

    # NOTE: validation_split is only supported for NumPy arrays. With a custom
    # iterable, build a separate validation dataset and pass it via
    # `validation_data=...` if you need validation during training.
    history = await program.fit(
        x=dataset,           # custom iterable yielding (Question, Answer) tuples
        epochs=2,
        batch_size=1,
    )
    synalinks.utils.plot_history(history, to_folder=".")


if __name__ == "__main__":
    asyncio.run(main())
