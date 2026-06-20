#!/usr/bin/env python3
"""Custom iterable dataset (Synalinks v0.8.004+).

Usage:
    uv run -- python scripts/custom_dataset.py

Run log:
    references/custom_dataset.log

Demonstrates passing a custom iterable (with optional __len__) directly to
program.fit(). Useful for streaming from files, databases, or APIs without
materializing the entire dataset in memory.

Key detail (Keras-style semantics): a custom iterable yields *batches*, not
individual examples. Each yield is a `(x_batch, y_batch)` tuple whose leaves are
arrays/lists of DataModel instances — the iterable controls its own batching, so
`fit(batch_size=...)` is ignored for iterables. `__len__` returns the number of
*batches* (enables the progress bar). Yielding a single DataModel per step fails
with "iteration over a 0-d array".
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import synalinks


class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="A question")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="Final answer")


class JSONLDataset:
    """Stream batches of (Question, Answer) from a JSONL file.

    Each iteration yields a `(x_batch, y_batch)` tuple of NumPy object arrays of
    DataModels — i.e. the dataset batches itself.
    """

    def __init__(self, path: str, batch_size: int = 1):
        self.path = Path(path)
        self.batch_size = batch_size

    def _rows(self):
        with self.path.open() as f:
            for line in f:
                row = json.loads(line)
                yield Question(question=row["q"]), Answer(answer=row["a"])

    def __iter__(self):
        xs, ys = [], []
        for q, a in self._rows():
            xs.append(q)
            ys.append(a)
            if len(xs) == self.batch_size:
                yield np.array(xs, dtype="object"), np.array(ys, dtype="object")
                xs, ys = [], []
        if xs:  # trailing partial batch
            yield np.array(xs, dtype="object"), np.array(ys, dtype="object")

    def __len__(self):
        # Optional, but enables the fit/evaluate progress bar. Returns the number
        # of *batches* (not examples). It does NOT enable validation_split —
        # that arg only works with NumPy arrays.
        n = sum(1 for _ in self._rows())
        return (n + self.batch_size - 1) // self.batch_size


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

    dataset = JSONLDataset("demo.jsonl", batch_size=1)
    print(f"Dataset has {len(dataset)} batches")

    # NOTE: `validation_split` (which defaults to 0.1) is only supported for
    # NumPy arrays, so with a custom iterable you MUST instead pass an explicit
    # `validation_data=(x_val, y_val)`. fit() always runs validation, and the
    # validation set is fancy-indexed, so x_val/y_val must be NumPy arrays of
    # DataModels (the training data can still be a streaming iterable).
    x_val = np.array([Question(question="What is 3 + 1?")], dtype="object")
    y_val = np.array([Answer(answer="4")], dtype="object")

    # epochs=1: synalinks consumes the iterable once (it calls iter(x) a single
    # time), so a streaming dataset is a single-pass-per-fit source. For multiple
    # epochs over a finite set, materialize it into NumPy arrays instead.
    history = await program.fit(
        x=dataset,           # custom iterable yielding (x_batch, y_batch) tuples
        validation_data=(x_val, y_val),
        epochs=1,            # batch_size is ignored for iterables (dataset batches itself)
    )
    synalinks.utils.plot_history(history, to_folder=".")


if __name__ == "__main__":
    asyncio.run(main())
