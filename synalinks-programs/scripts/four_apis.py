#!/usr/bin/env python3
"""Four ways to build a Synalinks Program.

Usage:
    uv run -- python scripts/four_apis.py

Demonstrates:
1. Functional API
2. Sequential API
3. Subclassing (with explicit get_config/from_config)
4. Mixed (Subclassing + Functional via build())

All four programs answer the same Q&A task so the differences are purely
structural.
"""

import asyncio

import synalinks


class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer to the query")


# --- 1. Functional API -------------------------------------------------------


def build_functional(language_model):
    inputs = synalinks.Input(data_model=Query)
    outputs = synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
    )
    # In a real async context you'd `await` the call. For construction-only
    # examples we build the program inside an async wrapper below.
    return inputs, outputs


async def make_functional(language_model):
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer, language_model=language_model,
    )(inputs)
    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="functional_qa",
        description="Q&A built with the Functional API",
    )


# --- 2. Sequential API -------------------------------------------------------


def make_sequential(language_model):
    return synalinks.Sequential(
        [
            synalinks.Input(data_model=Query),
            synalinks.Generator(
                data_model=Answer, language_model=language_model,
            ),
        ],
        name="sequential_qa",
        description="Q&A built with the Sequential API",
    )


# --- 3. Subclassing ----------------------------------------------------------


@synalinks.saving.register_synalinks_serializable()
class SubclassedQA(synalinks.Program):
    """Q&A built by subclassing Program with a custom call()."""

    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(name=name, description=description, trainable=trainable)
        self.language_model = language_model
        self.gen = synalinks.Generator(
            data_model=Answer, language_model=language_model,
        )

    async def call(self, inputs, training=False):
        return await self.gen(inputs, training=training)

    def get_config(self):
        return {
            "language_model": synalinks.saving.serialize_synalinks_object(
                self.language_model
            ),
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        lm = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=lm, **config)


# --- 4. Mixed (Subclassing + Functional) -------------------------------------


@synalinks.saving.register_synalinks_serializable()
class FunctionalQA(synalinks.Program):
    """Q&A built with the Mixed pattern — Functional graph behind a class API."""

    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=True,
    ):
        # First super().__init__() — basic init only.
        super().__init__(name=name, description=description, trainable=trainable)
        self.language_model = language_model

    async def build(self, inputs):
        outputs = await synalinks.Generator(
            data_model=Answer, language_model=self.language_model,
        )(inputs)
        # Second super().__init__() — re-init with the graph.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    programs = {
        "functional": await make_functional(lm),
        "sequential": make_sequential(lm),
        "subclassed": SubclassedQA(
            language_model=lm,
            name="subclassed_qa",
            description="Q&A built by subclassing Program",
        ),
        "mixed": FunctionalQA(
            language_model=lm,
            name="mixed_qa",
            description="Q&A built with the Mixed pattern",
        ),
    }

    q = Query(query="What is the capital of France?")
    for label, program in programs.items():
        result = await program(q)
        print(f"[{label}] {result.prettify_json() if result else 'No result'}")


if __name__ == "__main__":
    asyncio.run(main())
