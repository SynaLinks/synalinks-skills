# Synalinks Programs — Reference

Deeper coverage of `synalinks.Program`, `synalinks.Sequential`, the Functional
API, lifecycle, persistence format, and the custom-class checklist.

## `synalinks.Input`

```python
synalinks.Input(
    data_model,        # DataModel class (or schema=Answer.get_schema())
    *,
    name=None,         # Optional symbolic name (used in summary / plots)
)
```

Returns a `SymbolicDataModel` you can compose with operators (`+ & | ^ ~`)
or pass through modules. `Input` is a thin marker — it carries the schema but
no runtime data; values flow through it when the program is invoked.

## `synalinks.Program`

```python
class Program(Trainer, Module):
    # Functional construction
    Program(inputs, outputs, name=None, description=None, trainable=True)

    # Subclass construction (no positional inputs/outputs)
    Program(name=None, description=None, trainable=True)
```

`Program.__new__` inspects its arguments: if it sees Functional-style
`inputs=`/`outputs=`, it returns a `Functional` instance instead. Subclasses
that pass Functional kwargs to `super().__init__()` get the same dispatch
(via `inject_functional_program_class`) — this is what makes the **Mixed**
pattern work.

### Public methods worth knowing

| Method | Purpose |
|--------|---------|
| `summary(...)` | Print a table of modules, output schemas, variable counts. |
| `get_module(name=..., index=...)` | Look up a child module. Indices are bottom-up graph traversal. |
| `save(filepath)` | Persist config + variables to a `.json` file. |
| `to_json(**kwargs)` | Return config as a JSON string (no variables). |
| `Program.load(filepath, custom_objects=None)` | Classmethod — reload a saved program. |
| `save_variables(filepath)` / `load_variables(filepath)` | Variables only. |
| `get_state_tree()` / `set_state_tree(tree)` | In-memory variable dict. |
| `compile(...)` / `fit(...)` / `evaluate(...)` / `predict(...)` | Training APIs — see **synalinks-training**. |

### Properties

| Property | Returns |
|----------|---------|
| `program.modules` | Direct child modules (not recursive). |
| `program.trainable_variables` | All trainable variables across the graph. |
| `program.non_trainable_variables` | All non-trainable variables. |
| `program.variables` | Both, concatenated. |
| `program.metrics` | Compiled metrics (after `compile`). |
| `program.optimizer` | Compiled optimizer (after `compile`). |

## `synalinks.Sequential`

Strictly single-input / single-output stack of modules. Convenience over the
Functional API for purely linear graphs.

```python
program = synalinks.Sequential(
    [
        synalinks.Input(data_model=Query),
        synalinks.ChainOfThought(data_model=Answer, language_model=lm),
    ],
    name="cot_qa",
    description="Chain-of-thought Q&A",   # REQUIRED
)

# Add modules incrementally
program.add(synalinks.SelfCritique(language_model=lm))
program.pop()                              # remove the last module
```

`Sequential` will lazily build on the first input it sees if you skip the
leading `Input(...)` — but providing `Input` is the safer pattern.

## Multi-input / multi-output graphs

Inputs and outputs can be a single tensor, a list, a tuple, or a dict. The
program's call signature mirrors the structure.

```python
q = synalinks.Input(data_model=Query, name="q")
ctx = synalinks.Input(data_model=Context, name="ctx")
merged = q + ctx                            # JSON concat operator
answer = await synalinks.Generator(
    data_model=Answer, language_model=lm,
)(merged)
critique = await synalinks.SelfCritique(language_model=lm)(answer)

# Dict-shaped — keys must match call kwargs
program = synalinks.Program(
    inputs={"q": q, "ctx": ctx},
    outputs={"answer": answer, "critique": critique},
)
result = await program(q=Query(...), ctx=Context(...))
result["answer"]
result["critique"]

# Or list-shaped — positional
program = synalinks.Program(inputs=[q, ctx], outputs=[answer, critique])
answer_out, critique_out = await program([Query(...), Context(...)])
```

**Not supported:** nested containers (lists of lists, dicts of dicts).
Flatten or merge with operators before constructing the program.

## Lifecycle in detail

### Functional / Sequential

```
Program(inputs=..., outputs=...)
  -> Functional.__init__
       -> records the graph (input/output schemas, module DAG)
       -> creates trainable variables eagerly
```

Calling `await program(x)` walks the recorded DAG and forwards the data
through each module's `call`.

### Subclassing (option 3)

```
MyProgram(...)
  -> super().__init__(name=, description=, trainable=)
  -> __init__ stores child modules as attributes
await program(x)
  -> Module.__call__ -> self.call(inputs, training=False)
```

Variables are registered when child modules are first invoked (or
constructed, depending on the module). There is no DAG to introspect.

### Mixed (option 4)

```
MyProgram(...)
  -> super().__init__(name=, description=, trainable=)   # 1st call
await program(x)
  -> first call: self.build(x)
       -> graph constructed
       -> super().__init__(inputs=, outputs=, ...)         # 2nd call
       -> Functional state replaces Module state
  -> later calls: forward through the graph
```

`build` is `async` and may itself call modules. Anything created inside
`build` becomes a tracked child of the program.

### `training=True/False`

If your subclassed `call` accepts `training`, the trainer passes
`training=True` during `fit` and `training=False` during `evaluate` /
`predict`. Use it to gate noisy modules (self-critique, exploration):

```python
async def call(self, inputs, training=False):
    x = await self.gen(inputs, training=training)
    if training:
        x = await self.critique(x, training=training)
    return x
```

## Persistence format

`program.save("p.json")` writes a JSON document of this shape:

```json
{
  "module": "synalinks.programs.functional",
  "class_name": "Functional",
  "config": {
    "name": "...",
    "description": "...",
    "trainable": true,
    "modules": [...],
    "input_modules": [...],
    "output_modules": [...]
  },
  "registered_name": "...",
  "variables": {
    "<module_name>": {
      "<variable_name>": { ... }
    }
  }
}
```

- The top-level config is what `to_json()` returns.
- The `"variables"` block is what `get_state_tree()` returns.
- `Program.load` reads the file, calls `from_config` on the resolved class,
  then `set_state_tree(...)` to restore variable values.

### `from_config` resolution

```
Program.from_config(config, custom_objects)
  if config has functional keys (name, modules, input_modules, output_modules)
     and the class is revivable as Functional:
       -> functional_from_config(...)
  else:
       -> cls(**config)              # generic fallback
```

A class is "revivable as Functional" if its `__init__` signature matches
`Functional.__init__`, **or** if it accepts `*args, **kwargs`. Most Mixed
programs satisfy the latter.

If your Subclassed program's `__init__` signature does not match `**config`,
implement `from_config` yourself — see the SKILL.md template.

### `build_from_config`

For Mixed programs, `Program.load` calls `build_from_config(config)` to
re-run `build(...)` on the recorded input schemas. If your `build` doesn't
match the recorded shape, override `build_from_config` and (optionally)
`get_build_config` to control what's recorded.

## State tree shape

```python
{
  "trainable_variables":   {<scope>: {<name>: <value>}},
  "non_trainable_variables": {<scope>: {<name>: <value>}},
  "optimizer_variables":   {...},
  "metrics_variables":     {...},
}
```

`set_state_tree` writes values back into existing variables — it does **not**
create variables. The target program must already have the same structure
(typically because it was constructed from the same config).

## Custom-class checklist

For a serialisable custom Program subclass:

- [ ] Decorate the class with `@synalinks.saving.register_synalinks_serializable()`
- [ ] Make `__init__` keyword-only and call `super().__init__(name=..., description=..., trainable=...)`
- [ ] If using **Subclassing**: implement `call` + `get_config` + `from_config`
- [ ] If using **Mixed**: implement `__init__` + `async build` (no `call`,
      `get_config`, or `from_config` needed unless you want them)
- [ ] In `get_config`, serialise nested objects with
      `synalinks.saving.serialize_synalinks_object(...)`
- [ ] In `from_config`, deserialise them with
      `synalinks.saving.deserialize_synalinks_object(...)`
- [ ] Test the round-trip: `save` -> `load` -> compare `summary()` output

## See also

- `synalinks/src/programs/program.py` — `Program` source
- `synalinks/src/programs/functional.py` — Functional construction logic
- `synalinks/src/programs/sequential.py` — `Sequential` source
- `synalinks-training/SKILL.md` — `compile` / `fit` / `evaluate` / `predict`
