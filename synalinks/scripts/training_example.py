#!/usr/bin/env python3
"""Training a Synalinks program with rewards and optimizers.

Usage:
    uv run -- python scripts/training_example.py

This script demonstrates:
1. Creating training data as numpy arrays
2. Compiling with reward and optimizer
3. Training with fit()
4. Evaluating with evaluate()
5. Saving/loading programs
"""

import asyncio
import numpy as np
import synalinks

synalinks.enable_logging()


class MathQuestion(synalinks.DataModel):
    """Input data model."""
    question: str = synalinks.Field(description="A math word problem")


class MathAnswer(synalinks.DataModel):
    """Output data model."""
    thinking: str = synalinks.Field(description="Step-by-step solution")
    answer: int = synalinks.Field(description="The numerical answer")


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    # Create training data
    x_train = np.array([
        MathQuestion(question="What is 5 + 3?"),
        MathQuestion(question="What is 10 - 4?"),
        MathQuestion(question="What is 6 * 2?"),
        MathQuestion(question="What is 20 / 4?"),
        MathQuestion(question="What is 7 + 8?"),
        MathQuestion(question="What is 15 - 9?"),
    ], dtype="object")

    y_train = np.array([
        MathAnswer(thinking="5 + 3 = 8", answer=8),
        MathAnswer(thinking="10 - 4 = 6", answer=6),
        MathAnswer(thinking="6 * 2 = 12", answer=12),
        MathAnswer(thinking="20 / 4 = 5", answer=5),
        MathAnswer(thinking="7 + 8 = 15", answer=15),
        MathAnswer(thinking="15 - 9 = 6", answer=6),
    ], dtype="object")

    # Test data
    x_test = np.array([
        MathQuestion(question="What is 9 + 2?"),
        MathQuestion(question="What is 16 - 7?"),
    ], dtype="object")

    y_test = np.array([
        MathAnswer(thinking="9 + 2 = 11", answer=11),
        MathAnswer(thinking="16 - 7 = 9", answer=9),
    ], dtype="object")

    # Build program
    inputs = synalinks.Input(data_model=MathQuestion)
    outputs = await synalinks.Generator(
        data_model=MathAnswer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_solver",
        description="Solves basic math problems",
    )

    # Compile with reward and optimizer
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
        metrics=[],
    )

    # Evaluate before training
    print("=== Before Training ===")
    metrics_before = await program.evaluate(x=x_test, y=y_test, batch_size=2)
    print(f"Metrics: {metrics_before}")

    # Train
    print("\n=== Training ===")
    checkpoint = synalinks.callbacks.ProgramCheckpoint(
        filepath="best_math_solver.json",
        monitor="val_reward",
        mode="max",
        save_best_only=True,
    )

    history = await program.fit(
        x=x_train,
        y=y_train,
        validation_split=0.33,
        epochs=3,
        batch_size=2,
        callbacks=[checkpoint],
    )

    # Visualize training
    synalinks.utils.plot_history(history, to_folder=".")

    # Evaluate after training
    print("\n=== After Training ===")
    metrics_after = await program.evaluate(x=x_test, y=y_test, batch_size=2)
    print(f"Metrics: {metrics_after}")

    # Save and load
    program.save("math_solver.json")
    print("\nProgram saved to math_solver.json")

    # Load and test
    loaded_program = synalinks.Program.load("math_solver.json")
    result = await loaded_program(MathQuestion(question="What is 4 * 5?"))
    print(f"\nTest: 4 * 5 = {result.get('answer')}")


if __name__ == "__main__":
    asyncio.run(main())
