#!/usr/bin/env python3
"""OMEGA optimizer example.

Usage:
    uv run -- python scripts/omega_example.py

Demonstrates:
1. Configuring OMEGA with both language_model and embedding_model
2. Tuning population_size, k_nearest_fitter, mutation_temperature
3. Comparing before/after training metrics
"""

import asyncio
import numpy as np
import synalinks


class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="A math word problem")


class Answer(synalinks.DataModel):
    thinking: str = synalinks.Field(description="Step-by-step reasoning")
    answer: int = synalinks.Field(description="Numerical answer")


async def main():
    # Use cheap models for both program and optimizer
    program_lm = synalinks.LanguageModel(model="openai/gpt-4o-mini")
    optimizer_lm = synalinks.LanguageModel(model="openai/gpt-4o-mini")
    em = synalinks.EmbeddingModel(model="openai/text-embedding-3-small")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=program_lm,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="math_omega")

    # Alternative: register defaults and use the Keras-style string identifier:
    #     synalinks.set_default_language_model(optimizer_lm)
    #     synalinks.set_default_embedding_model(em)
    #     program.compile(
    #         reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    #         optimizer="omega",   # case-insensitive: "OMEGA", "randomfewshot", ...
    #     )
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.OMEGA(
            language_model=optimizer_lm,
            embedding_model=em,
            population_size=10,
            k_nearest_fitter=5,
            mutation_temperature=0.4,
            crossover_temperature=0.3,
            selection_temperature=0.3,
            merging_rate=0.02,
            algorithm="dns",
            selection="softmax",
            instructions="Improve clarity of step-by-step reasoning and arithmetic accuracy.",
        ),
    )

    x_train = np.array([
        Question(question="What is 5 + 3?"),
        Question(question="What is 10 - 4?"),
        Question(question="What is 6 * 2?"),
        Question(question="What is 20 / 4?"),
    ], dtype="object")

    y_train = np.array([
        Answer(thinking="5 + 3 = 8", answer=8),
        Answer(thinking="10 - 4 = 6", answer=6),
        Answer(thinking="6 * 2 = 12", answer=12),
        Answer(thinking="20 / 4 = 5", answer=5),
    ], dtype="object")

    x_test = np.array([
        Question(question="What is 9 + 2?"),
        Question(question="What is 16 - 7?"),
    ], dtype="object")

    y_test = np.array([
        Answer(thinking="9 + 2 = 11", answer=11),
        Answer(thinking="16 - 7 = 9", answer=9),
    ], dtype="object")

    print("=== Before training ===")
    before = await program.evaluate(x=x_test, y=y_test, batch_size=2)
    print(before)

    history = await program.fit(
        x=x_train, y=y_train,
        validation_split=0.25,
        epochs=3,
        batch_size=2,
    )
    synalinks.utils.plot_history(history, to_folder=".")

    print("\n=== After training ===")
    after = await program.evaluate(x=x_test, y=y_test, batch_size=2)
    print(after)


if __name__ == "__main__":
    asyncio.run(main())
