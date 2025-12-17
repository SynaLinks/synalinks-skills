#!/usr/bin/env python3
"""Conditional branching with Decision and Branch modules.

Usage:
    uv run -- python scripts/conditional_branches.py

This script demonstrates:
1. Decision module for classification
2. Branch module for conditional routing
3. JSON operators (|) for merging branches
"""

import asyncio
import synalinks

synalinks.enable_logging()


class Query(synalinks.DataModel):
    """Input data model."""
    query: str = synalinks.Field(description="The user query")


class SimpleAnswer(synalinks.DataModel):
    """Short answer for easy questions."""
    answer: str = synalinks.Field(description="A brief, direct answer")


class DetailedAnswer(synalinks.DataModel):
    """Detailed answer with reasoning for hard questions."""
    thinking: str = synalinks.Field(description="Step-by-step reasoning")
    answer: str = synalinks.Field(description="The detailed answer")


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    inputs = synalinks.Input(data_model=Query)

    # Conditional branching based on question difficulty
    (easy_output, hard_output) = await synalinks.Branch(
        question="Evaluate the difficulty of this query",
        labels=["easy", "difficult"],
        branches=[
            synalinks.Generator(
                data_model=SimpleAnswer,
                language_model=lm,
            ),
            synalinks.Generator(
                data_model=DetailedAnswer,
                language_model=lm,
            ),
        ],
        language_model=lm,
        return_decision=False,
    )(inputs)

    # Merge branches using logical OR
    # Returns whichever branch was taken (other is None)
    outputs = easy_output | hard_output

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="adaptive_qa",
        description="Adapts response complexity to question difficulty",
    )

    # Visualize
    synalinks.utils.plot_program(program, to_folder=".", show_schemas=True)

    # Test with easy question
    print("=== Easy Question ===")
    result1 = await program(Query(query="What color is the sky?"))
    print(result1.prettify_json())

    # Test with hard question
    print("\n=== Hard Question ===")
    result2 = await program(
        Query(query="Explain the implications of quantum entanglement for cryptography")
    )
    print(result2.prettify_json())

if __name__ == "__main__":
    asyncio.run(main())