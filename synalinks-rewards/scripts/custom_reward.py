#!/usr/bin/env python3
"""Custom reward function example.

Usage:
    uv run -- python scripts/custom_reward.py

Demonstrates:
1. Writing an async reward with @register_synalinks_serializable
2. Wrapping with RewardFunctionWrapper
3. Combining multiple criteria into one reward
4. Mixing built-in rewards with custom metrics
"""

import asyncio
import numpy as np
import synalinks


class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="A question")


class Answer(synalinks.DataModel):
    # ChainOfThought prepends its own `thinking` field, so we only
    # declare the final answer here.
    answer: str = synalinks.Field(description="Final answer")


@synalinks.saving.register_synalinks_serializable()
async def correctness_with_reasoning(y_true, y_pred):
    """Combine exact answer match with a small bonus for showing reasoning."""
    if not y_true or not y_pred:
        return 0.0

    correct = float(y_true.get("answer") == y_pred.get("answer"))
    has_reasoning = float(bool((y_pred.get("thinking") or "").strip()))

    return 0.8 * correct + 0.2 * has_reasoning


@synalinks.saving.register_synalinks_serializable()
async def thinking_length(y_true, y_pred):
    """Metric: average thinking length normalized to [0, 1] using a 200-char ceiling."""
    if not y_pred:
        return 0.0
    thinking = y_pred.get("thinking") or ""
    return min(len(thinking) / 200.0, 1.0)


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.ChainOfThought(
        data_model=Answer,
        language_model=lm,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="reward_demo")

    program.compile(
        reward=synalinks.rewards.RewardFunctionWrapper(fn=correctness_with_reasoning),
        optimizer=synalinks.optimizers.RandomFewShot(),
        metrics=[
            synalinks.metrics.MeanMetricWrapper(fn=thinking_length),
            synalinks.metrics.F1Score(in_mask=["answer"]),
        ],
    )

    x = np.array([
        Question(question="What is 5 + 3?"),
        Question(question="Capital of France?"),
    ], dtype="object")

    y = np.array([
        Answer(answer="8"),
        Answer(answer="Paris"),
    ], dtype="object")

    metrics = await program.evaluate(x=x, y=y, batch_size=2)
    print("Metrics:", metrics)


if __name__ == "__main__":
    asyncio.run(main())
