---
name: synalinks-programs
description: Use when building or composing a Synalinks Program — the four building APIs (Functional, Sequential, Subclassing, Mixed), Input nodes, multi-input/multi-output graphs, the call/build lifecycle, training=True/False semantics, summary, get_module, plot_program, save/load, get_state_tree/set_state_tree, get_config/from_config and custom serialization. For DataModel/Field, JSON operators (+ & | ^ ~), and LanguageModel/EmbeddingModel basics see synalinks-core. For inner modules see synalinks-modules; for compile/fit/evaluate/predict see synalinks-training.
---

# Synalinks Programs

`Program` groups modules into a trainable / serialisable / deployable object.
It inherits from both `Trainer` and `Module`, so a `Program` is itself a
module — you can nest one inside another.

## When this skill activates

- You're constructing a `synalinks.Program`, `synalinks.Sequential`, or a
  `synalinks.Program` subclass.
- You're wiring `synalinks.Input(...)` to outputs and calling
  `synalinks.Program(inputs=..., outputs=...)`.
- You're saving/loading a program, inspecting it (`summary`, `get_module`,
  `plot_program`), or implementing `get_config` / `from_config` for a custom
  Program.

For DataModel/Field, JSON operators, and LanguageModel/EmbeddingModel basics,
go to **synalinks-core** instead.

## Four ways to build a Program

| API | When to pick it |
|-----|-----------------|
| **Functional** | Default. Multi-input / multi-output graphs, branching, parallelism. |
| **Sequential** | Strictly linear chain of single-input/single-output modules. |
| **Subclassing** | Custom Python control flow inside `call()`. Loses graph introspection. |
| **Mixed (Subclassing + Functional)** | Encapsulate a graph behind a class-shaped public API. Best of both worlds. |

### 1. Functional API (recommended)

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
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_qa",
        description="A simple Q&A program",
    )

    result = await program(Query(query="What is the capital of France?"))
    print(result.prettify_json())


asyncio.run(main())
```

Note: Only `dict`, `list`, and `tuple` of input data models are supported.
Nested containers (lists of lists, dicts of dicts) are **not** supported.

### 2. Sequential API

`Sequential` is strictly single-input / single-output and **requires a
`description`** (raises `ValueError` otherwise).

```python
program = synalinks.Sequential(
    [
        synalinks.Input(data_model=Query),
        synalinks.Generator(data_model=Answer, language_model=lm),
    ],
    name="chain_of_thought",
    description="A simple Q&A chain",
)
```

### 3. Subclassing

`Module.__init__` is keyword-only (`*,` after `self`); always pass
`name` / `description` / `trainable` by keyword. Subclasses **must** implement
`get_config` / `from_config` to be JSON-serialisable.

```python
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
            data_model=Answer,
            language_model=language_model,
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
```

If the class docstring's first line is set, it is used as the program
`description` when `super().__init__()` is called without one. The class name
becomes the default `name`.

### 4. Mixed: Subclassing + Functional

This is the recommended pattern for most reusable programs — you get a clean
class-shaped API and keep the introspectable Functional graph (so
`summary`, `plot_program`, and the default `from_config` all work). Call
`super().__init__()` **twice**:

```python
@synalinks.saving.register_synalinks_serializable()
class FunctionalQA(synalinks.Program):
    def __init__(
        self,
        language_model=None,
        name=None,
        description=None,
        trainable=True,
    ):
        # First call — basic init, no graph yet.
        super().__init__(name=name, description=description, trainable=trainable)
        self.language_model = language_model

    async def build(self, inputs):
        outputs = await synalinks.Generator(
            data_model=Answer, language_model=self.language_model,
        )(inputs)
        # Second call — re-init with the graph.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            name=self.name,
            description=self.description,
            trainable=self.trainable,
        )
```

The graph is built lazily on the first call; you don't have to implement
`call`, `get_config`, or `from_config`.

## Lifecycle: build / call / training flag

- **Functional / Sequential**: graph is built immediately at construction time.
- **Subclassing**: `__init__` runs first, then `call(inputs, training=False)`
  runs on every invocation. Use the `training` flag to switch behaviour
  (e.g. enable self-critique only at train time).
- **Mixed**: `__init__` runs first, then `build(inputs)` runs once on the
  first call (or on `from_config` deserialisation), then subsequent calls go
  through the materialised graph.

```python
async def call(self, inputs, training=False):
    if training:
        x = await self.with_critique(inputs)
    else:
        x = await self.fast_path(inputs)
    return x
```

## Multi-input / multi-output

Pass `dict`, `list`, or `tuple` of `Input` nodes. The program's call signature
mirrors the structure you used at construction time.

```python
inputs = {
    "query": synalinks.Input(data_model=Query),
    "context": synalinks.Input(data_model=Context),
}
merged = inputs["query"] + inputs["context"]
answer = await synalinks.Generator(data_model=Answer, language_model=lm)(merged)
critique = await synalinks.SelfCritique(language_model=lm)(answer)

program = synalinks.Program(
    inputs=inputs,
    outputs={"answer": answer, "critique": critique},
)
```

## Inspection

```python
program.summary()                       # printed table of modules + variables
program.summary(expand_nested=True)     # recurse into sub-programs
program.summary(show_trainable=True)

len(program.modules)                    # all child modules (non-recursive)
program.get_module(index=0)
program.get_module(name="generator")

program.trainable_variables             # list of Variable
program.non_trainable_variables
program.variables
```

For graph rendering see `synalinks.utils.plot_program`:

```python
synalinks.utils.plot_program(
    program,
    to_folder=".",
    show_module_names=True,
    show_schemas=True,
)
```

## Saving and loading

A saved `.json` contains: program config (architecture), variables, optimizer
state (if any), and reward state (if any).

```python
# Whole program
program.save("my_program.json")
program = synalinks.Program.load("my_program.json")

# JSON string (no variables)
config_str = program.to_json()
program = synalinks.programs.program_from_json(config_str)

# Variables only
program.save_variables("my_program.variables.json")
program.load_variables("my_program.variables.json")
```

For custom subclassed programs, register them so `load` can resolve the class:

```python
@synalinks.saving.register_synalinks_serializable()
class MyProgram(synalinks.Program):
    ...
```

If `register_synalinks_serializable` was not used at save time, you can pass
`custom_objects={"MyProgram": MyProgram}` to `load` / `from_config`.

## State tree

Use `get_state_tree` / `set_state_tree` to copy variables across instances
without round-tripping through disk:

```python
state = source_program.get_state_tree()
target_program.set_state_tree(state)
```

This is the same nested-dict structure that `save` writes under the
`"variables"` key.

## Gotchas

### Subclassing without `get_config`/`from_config` is not serialisable

Pure subclassed programs (option 3) **must** implement both methods. The
default `from_config` only works for Functional configs (it looks for keys
`name`, `modules`, `input_modules`, `output_modules`). Skip option 3 if you
want save/load and don't need custom Python control flow — use option 4
(Mixed) instead.

### `Sequential` requires `description`

Omitting it raises `ValueError`. The class doc says it isn't optional:
the description is used downstream by callers (Decision/Branch routing,
agents) to pick which sub-program to run.

### Program is itself a Module

You can use a `Program` anywhere a `Module` is expected — including inside
another `Program`. Nest with care: `summary(expand_nested=True)` is your
friend.

### `Module.__init__` is keyword-only

```python
# CORRECT
class MyProgram(synalinks.Program):
    def __init__(self, language_model=None, name=None, description=None, trainable=True):
        super().__init__(name=name, description=description, trainable=trainable)

# WRONG — positional `name` will raise TypeError
super().__init__(name)
```

### `program()` vs `program.predict()`

`program(inputs)` is the async forward pass on a single input — same as
`call`. `program.predict(x, batch_size=...)` runs many inputs in batches with
progress bars and applies callbacks. See **synalinks-training** for the full
training/inference workflow.

## References

- **references/programs.md** — multi-IO patterns, deserialisation internals,
  state-tree shape, `build_from_config`, custom-class checklist.

## See Also

- **synalinks-core** — DataModel, JSON operators, LanguageModel/EmbeddingModel basics
- **synalinks-modules** — Generator, ChainOfThought, custom Module subclassing
- **synalinks-control-flow** — Decision, Branch, parallel/self-consistency patterns
- **synalinks-training** — `compile` / `fit` / `evaluate` / `predict`, callbacks
- **synalinks-agents** — FunctionCallingAgent (built on top of Program)
