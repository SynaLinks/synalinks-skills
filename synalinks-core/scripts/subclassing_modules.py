#!/usr/bin/env python3
"""Subclassing patterns for Synalinks Module and Program.

Usage:
    uv run -- python scripts/subclassing_modules.py

Demonstrates:
1. Subclassing `Module` with a custom `call()` (keyword-only `__init__`).
2. Subclassing `Program` with a `call()` override.
3. The mixed Functional + Subclassing pattern (two `super().__init__()` calls).

Note: `Module.__init__` is keyword-only (`*,` after `self`), so you must pass
`name`/`description`/`trainable` (and any other knobs) by keyword. The only
exception in core is `Tool(func, *, ...)` which keeps `func` positional.
"""

import asyncio

import synalinks


class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer to the query")


# 1. Custom Module subclass — wraps a Generator with extra logic.
@synalinks.saving.register_synalinks_serializable()
class GeneratorWithRetry(synalinks.Module):
    """Calls a Generator and retries once on a None result."""

    def __init__(
        self,
        *,
        data_model=None,
        language_model=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(name=name, description=description, trainable=trainable)
        self.data_model = data_model
        self.language_model = language_model
        self.generator = synalinks.Generator(
            data_model=data_model,
            language_model=language_model,
        )

    async def call(self, inputs, training=False):
        result = await self.generator(inputs)
        if result is None:
            result = await self.generator(inputs)
        return result

    def get_config(self):
        return {
            "data_model": synalinks.saving.serialize_synalinks_object(self.data_model),
            "language_model": synalinks.saving.serialize_synalinks_object(
                self.language_model
            ),
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        data_model = synalinks.saving.deserialize_synalinks_object(
            config.pop("data_model")
        )
        language_model = synalinks.saving.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(data_model=data_model, language_model=language_model, **config)


# 2. Subclassed Program with a `call()` override.
@synalinks.saving.register_synalinks_serializable()
class SubclassedQA(synalinks.Program):
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
        return await self.gen(inputs)

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
        lm = synalinks.saving.deserialize_synalinks_object(config.pop("language_model"))
        return cls(language_model=lm, **config)


# 3. Mixed: Functional graph built lazily inside `build()`.
@synalinks.saving.register_synalinks_serializable()
class FunctionalQA(synalinks.Program):
    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=True,
    ):
        # First `super().__init__()` — basic init only.
        super().__init__(name=name, description=description, trainable=trainable)
        self.language_model = language_model

    async def build(self, inputs):
        outputs = await synalinks.Generator(
            data_model=Answer, language_model=self.language_model,
        )(inputs)
        # Second `super().__init__()` — re-init with the graph.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )

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
        lm = synalinks.saving.deserialize_synalinks_object(config.pop("language_model"))
        return cls(language_model=lm, **config)


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    program = SubclassedQA(
        language_model=lm,
        name="subclassed_qa",
        description="Subclassed Q&A program",
    )
    result = await program(Query(query="What is the capital of France?"))
    print(result.prettify_json() if result is not None else "No result")


if __name__ == "__main__":
    asyncio.run(main())
