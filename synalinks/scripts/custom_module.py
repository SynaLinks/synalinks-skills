#!/usr/bin/env python3
"""Creating a custom module via subclassing.

Usage:
    uv run -- python scripts/custom_module.py

This script demonstrates:
1. Subclassing synalinks.Module
2. Implementing call() and compute_output_spec()
3. Serialization with get_config() and from_config()
4. Composing custom modules in programs
"""

import asyncio
import synalinks

synalinks.enable_logging()


class Query(synalinks.DataModel):
    """Input data model."""
    query: str = synalinks.Field(description="The user query")


class Analysis(synalinks.DataModel):
    """Analysis output."""
    category: str = synalinks.Field(description="Query category")
    complexity: str = synalinks.Field(description="Complexity level")
    suggested_approach: str = synalinks.Field(description="Recommended approach")


class FinalAnswer(synalinks.DataModel):
    """Final answer."""
    answer: str = synalinks.Field(description="The answer")


class QueryAnalyzer(synalinks.Module):
    """Custom module that analyzes queries before answering.

    This module:
    1. Analyzes the query to determine category and complexity
    2. Generates a tailored response based on the analysis
    """

    def __init__(
        self,
        language_model=None,
        return_inputs=True,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name or "query_analyzer",
            description=description or "Analyzes queries and generates tailored responses",
            trainable=trainable,
        )
        self.language_model = language_model
        self.return_inputs = return_inputs

        # Sub-modules
        # Note: instructions must be a string, not a list
        self.analyzer = synalinks.Generator(
            data_model=Analysis,
            language_model=language_model,
            instructions="Analyze the query to determine its category (factual, opinion, creative, technical). Assess complexity (simple, moderate, complex). Suggest the best approach to answer.",
            return_inputs=True,
        )

        self.responder = synalinks.Generator(
            data_model=FinalAnswer,
            language_model=language_model,
            instructions="Generate an answer based on the analysis. Adapt response style to the identified category and complexity.",
            return_inputs=return_inputs,
        )

    async def call(self, inputs, training=False):
        """Core computation: analyze then respond."""
        if not inputs:
            return None  # Support logical flows

        # Step 1: Analyze the query
        analysis = await self.analyzer(inputs, training=training)

        # Step 2: Generate response based on analysis
        response = await self.responder(analysis, training=training)

        return response

    async def compute_output_spec(self, inputs, training=False):
        """Define output schema."""
        analysis = await self.analyzer(inputs)
        return await self.responder(analysis)

    def to_config(self):
        """Serialization config for Module subclasses."""
        return {
            "return_inputs": self.return_inputs,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
            "language_model": synalinks.saving.serialize_synalinks_object(
                self.language_model
            ),
        }

    @classmethod
    def from_config(cls, config):
        """Deserialization."""
        lm = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=lm, **config)


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    # Use custom module in a program
    inputs = synalinks.Input(data_model=Query)
    outputs = await QueryAnalyzer(
        language_model=lm,
        return_inputs=True,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="smart_qa",
        description="Analyzes queries before answering",
    )

    # Visualize
    synalinks.utils.plot_program(
        program,
        to_folder=".",
        show_module_names=True,
        show_schemas=True,
        show_trainable=True,
    )

    # Test
    print("=== Factual Question ===")
    result1 = await program(Query(query="What is the capital of Japan?"))
    print(result1.prettify_json())

    print("\n=== Complex Question ===")
    result2 = await program(
        Query(query="How does machine learning differ from traditional programming?")
    )
    print(result2.prettify_json())

    # Save and reload
    program.save("smart_qa.json")
    loaded = synalinks.Program.load("smart_qa.json")
    print("\nProgram saved and reloaded successfully!")


if __name__ == "__main__":
    asyncio.run(main())
