#!/usr/bin/env python3
"""Simple Q&A program using Synalinks.

Usage:
    uv run -- python scripts/simple_qa.py

This script demonstrates the basic Synalinks pattern:
1. Define DataModels for input/output
2. Create a Program using the Functional API
3. Execute the program
"""

import asyncio
import synalinks

synalinks.enable_logging()


class Query(synalinks.DataModel):
    """Input data model."""
    query: str = synalinks.Field(description="The user query")


class Answer(synalinks.DataModel):
    """Output data model."""
    answer: str = synalinks.Field(description="The answer to the query")


async def main():
    # Initialize language model (adjust model name as needed)
    lm = synalinks.LanguageModel(model="ollama/mistral")

    # Build program using Functional API
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_qa",
        description="A simple question-answering program",
    )

    # Visualize the program
    synalinks.utils.plot_program(
        program,
        to_folder=".",
        show_module_names=True,
        show_schemas=True,
    )

    # Execute the program
    result = await program(Query(query="What is the capital of France?"))
    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
