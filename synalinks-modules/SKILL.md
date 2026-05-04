---
name: synalinks-modules
description: Use when working with Synalinks generation modules — Generator, ChainOfThought, SelfCritique, Identity, PythonSynthesis, SequentialPlanSynthesis — or building custom modules via subclassing (call, compute_output_spec, get_config, add_variable). For Decision/Branch/guards see synalinks-control-flow; for FunctionCallingAgent see synalinks-agents.
---

# Synalinks Modules

Built-in generation/synthesis modules and how to subclass `Module` for custom logic.

## Quick Start

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")

async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
        instructions="Be concise and accurate.",
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="qa")
    result = await program(Query(query="What is the capital of France?"))
    print(result.prettify_json())

asyncio.run(main())
```

## Keyword-only constructor args

Most Module subclasses (`Generator`, `ChainOfThought`, `SelfCritique`,
`PythonSynthesis`, `SequentialPlanSynthesis`, etc.) declare `*,` immediately
after `self`, so **all constructor arguments must be passed by keyword**.

```python
# CORRECT
synalinks.Generator(data_model=Answer, language_model=lm)

# WRONG — TypeError: takes 1 positional argument
synalinks.Generator(Answer, lm)
```

The exception is `synalinks.Tool(func, *, name=None, description=None, trainable=False)`,
which keeps `func` positional.

## Default language / embedding model

`Generator(language_model=None)` no longer raises at construction time.
Resolution happens inside `ops.predict` at call time: if no model is passed
and no default is configured, the call raises a clear `ValueError`. Set a
process-wide default with:

```python
synalinks.set_default_language_model("ollama/mistral")
synalinks.set_default_embedding_model("ollama/mxbai-embed-large")
```

The fallback identifier accepts a string, a config dict, or a
`LanguageModel`/`EmbeddingModel` instance. Both classes are now Modules and
accept `**default_kwargs` (`temperature`, `top_p`, `top_k`, `max_tokens`,
`reasoning_effort`, `dimensions`, ...) that are forwarded to every call and
overridable per-call.

## Generator

Core LLM generation module with structured output.

```python
outputs = await synalinks.Generator(
    data_model=Answer,          # Required: output schema (or pass `schema=...`)
    language_model=lm,          # Optional: falls back to set_default_language_model
    prompt_template=None,       # Custom Jinja2 template
    instructions=None,          # MUST be a string (not a list!)
    seed_instructions=None,     # Optional: list of strings used as optimization seeds
    examples=None,              # Few-shot examples (filled in by training)
    return_inputs=False,        # Include inputs in output
    use_inputs_schema=False,    # Show input schema in prompt
    use_outputs_schema=False,   # Show output schema in prompt
    temperature=0.0,
    reasoning_effort=None,      # 'minimal'|'low'|'medium'|'high'|'disable'|'none'|None
    streaming=False,            # Disabled automatically when `schema` is set
    name=None,
    description=None,
    trainable=True,
)(inputs)
```

**Trainable variables:** `instructions` and `examples` — optimized during training.

**`instructions` MUST be a string:**

```python
# CORRECT
synalinks.Generator(data_model=Answer, language_model=lm,
                    instructions="Be concise. Focus on key points.")

# WRONG — raises ValidationError
synalinks.Generator(data_model=Answer, language_model=lm,
                    instructions=["Be concise", "Focus on key points"])
```

## ChainOfThought

Generator that prepends a `thinking` field to the output schema. Defaults to
`reasoning_effort="low"`, which delegates the thinking to the model's
extended-thinking output when the provider supports it; otherwise the
thinking field is generated as part of the JSON.

```python
outputs = await synalinks.ChainOfThought(
    data_model=Answer,
    language_model=lm,
    return_inputs=True,
)(inputs)
# Output schema = inputs + thinking + answer
```

## SelfCritique

Self-evaluation module that produces a critique and (optionally) a reward score.

```python
outputs = await synalinks.SelfCritique(
    language_model=lm,
    return_reward=True,   # Include reward 0.0-1.0 (default True)
    return_inputs=True,   # Forward inputs (default True)
)(previous_output)
# Output: inputs + critique + reward
```

## Identity

Pass-through (no-op). Useful as a placeholder branch or to forward inputs without modification.

```python
outputs = await synalinks.Identity()(inputs)
```

## Synthesis Modules (Test-time adaptation)

### PythonSynthesis

Generate and execute a Python script inside a `MontySandbox` (no LLM call —
the script itself is the trainable variable, evolved by advanced optimizers).
Has **no `language_model` parameter**.

```python
default_python_script = """
def transform(inputs):
    return {"answer": inputs.get("query")}

result = transform(inputs)
"""

outputs = await synalinks.PythonSynthesis(
    data_model=Result,                          # Required
    python_script=default_python_script,        # Required
    default_return_value={"answer": ""},        # Required, must satisfy schema
    seed_scripts=None,                          # Optional list[str] of seeds
    return_python_script=False,
    timeout=5,
    tools=None,                                 # Optional: list[Tool] exposed in sandbox
)(inputs)
```

Works **only with advanced optimizers** (not `RandomFewShot`).

### SequentialPlanSynthesis

Generate a step-by-step execution plan and run each step through a `runner`
(any `Module`/`Program`, typically a `Generator`, `ChainOfThought`, or
`FunctionCallingAgent`). The `runner` is required.

```python
outputs = await synalinks.SequentialPlanSynthesis(
    data_model=FinalReport,            # Required: final output schema
    language_model=lm,                 # Used by the internal final ChainOfThought
    runner=synalinks.FunctionCallingAgent(  # Required
        data_model=TaskSummary,
        language_model=lm,
        tools=tools,
        return_inputs_with_trajectory=False,
    ),
    steps=None,                        # Optional default plan
    seed_steps=None,                   # Optional seed plans for the optimizer
    return_inputs=True,
    reasoning_effort=None,
)(inputs)
```

The runner must not return its inputs (use `return_inputs=False` /
`return_inputs_with_trajectory=False`); inputs are concatenated by the module.

## Creating Custom Modules

### Subclass Template

```python
class MyModule(synalinks.Module):
    def __init__(
        self,
        language_model=None,
        return_inputs=False,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(name=name, description=description, trainable=trainable)
        self.language_model = language_model
        self.return_inputs = return_inputs

        # instructions must be a string, not a list
        # Generator uses keyword-only args
        self.generator = synalinks.Generator(
            data_model=OutputSchema,
            language_model=language_model,
            instructions="Your instructions here as a single string.",
            return_inputs=return_inputs,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None  # Support logical flows (XOR guards)
        return await self.generator(inputs, training=training)

    async def compute_output_spec(self, inputs, training=False):
        """Define output schema (required for custom modules)."""
        return await self.generator(inputs)

    def get_config(self):
        """Module subclasses use get_config() (same as Program)."""
        return {
            "return_inputs": self.return_inputs,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
            "language_model": synalinks.saving.serialize_synalinks_object(self.language_model),
        }

    @classmethod
    def from_config(cls, config):
        lm = synalinks.saving.deserialize_synalinks_object(config.pop("language_model"))
        return cls(language_model=lm, **config)
```

### Adding Variables

`add_variable` takes an `initializer` (an `Initializer` instance or a JSON
dict) and a `data_model` describing the variable's schema. There is no
`Constant` initializer in the public API; pass a `data_model` instance's
`get_json()` (or any plain dict) as the initializer to seed a value.

```python
class MyTrainable(synalinks.DataModel):
    notes: str = synalinks.Field(description="Trainable notes")

class MyModule(synalinks.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = self.add_variable(
            initializer=MyTrainable(notes="initial").get_json(),
            data_model=MyTrainable,
            name="state_" + self.name,
        )
```

### Returning `None` for Logical Flows

A custom module can return `None` to short-circuit downstream computation when combined with `&`, `^`, or `|` operators. This is the foundation of guard patterns — see **synalinks-control-flow**.

## References

- **references/modules-catalog.md** — Complete catalog of built-in modules with all parameters

## See Also

- **synalinks-core** — DataModel, Program, operators
- **synalinks-control-flow** — Decision, Branch, parallel patterns, XOR guards
- **synalinks-agents** — FunctionCallingAgent, Tool, MCP
- **synalinks-training** — How trainable modules are optimized
