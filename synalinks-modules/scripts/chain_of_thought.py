#!/usr/bin/env python3
"""Chain-of-thought reasoning program.

Usage:
    uv run -- python scripts/chain_of_thought.py

This script demonstrates:
1. Using ChainOfThought module for step-by-step reasoning
2. Custom DataModel with thinking field
3. return_inputs parameter
"""

import asyncio
import synalinks

synalinks.enable_logging()


class Query(synalinks.DataModel):
    """Input data model."""
    query: str = synalinks.Field(description="The user query")


class ReasonedAnswer(synalinks.DataModel):
    """Output with reasoning."""
    answer: str = synalinks.Field(description="The final answer")


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    # Method 1: Using ChainOfThought module (auto-adds thinking)
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.ChainOfThought(
        data_model=ReasonedAnswer,
        language_model=lm,
        return_inputs=True,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="chain_of_thought",
        description="Answers questions with step-by-step reasoning",
    )

    # Visualize
    synalinks.utils.plot_program(program, to_folder=".", show_schemas=True)

    # Execute
    result = await program(
        Query(query="If a train travels 120 km in 2 hours, what is its average speed?")
    )
    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
