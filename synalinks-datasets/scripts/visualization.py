#!/usr/bin/env python3
"""Visualization utilities demo.

Usage:
    uv run -- python scripts/visualization.py

Demonstrates:
1. plot_program     — render the DAG
2. plot_history     — training curves
3. plot_metrics_with_mean_and_std       — eval stability
4. plot_metrics_comparison_with_mean_and_std — before/after
"""

import asyncio
import numpy as np
import synalinks


class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="A question")


class Answer(synalinks.DataModel):
    # ChainOfThought prepends its own `thinking` field automatically — don't add one here.
    answer: str = synalinks.Field(description="Final answer")


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=lm,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="vis_demo")

    # 1) Plot the program DAG
    synalinks.utils.plot_program(
        program,
        to_folder=".",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    # 2) Train briefly and plot history
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )

    x = np.array([
        Question(question="What is 2 + 2?"),
        Question(question="Capital of France?"),
        Question(question="Who wrote Hamlet?"),
    ], dtype="object")
    y = np.array([
        Answer(answer="4"),
        Answer(answer="Paris"),
        Answer(answer="William Shakespeare"),
    ], dtype="object")

    history = await program.fit(x=x, y=y, epochs=2, batch_size=1, validation_split=0.34)
    synalinks.utils.plot_history(history, to_folder=".")

    # 3) Multiple evaluations -> mean+std plot
    runs = [await program.evaluate(x=x, y=y, batch_size=1) for _ in range(3)]
    synalinks.utils.plot_metrics_with_mean_and_std(
        runs,
        to_folder=".",
        title="Eval stability",
    )

    # 4) Comparison plot (baseline vs trained)
    # In a real workflow you'd save baseline runs before training.
    synalinks.utils.plot_metrics_comparison_with_mean_and_std(
        {"trained": runs, "trained_again": runs},
        to_folder=".",
        title="Comparison",
    )


if __name__ == "__main__":
    asyncio.run(main())
