# Synalinks Control Flow Reference

## Operator Truth Tables

### Concatenation (`+`)

| x1 | x2 | Result |
|----|----|--------|
| set | set | x1 + x2 (merged fields) |
| set | None | Exception |
| None | set | Exception |
| None | None | Exception |

### Logical And (`&`)

| x1 | x2 | Result |
|----|----|--------|
| set | set | x1 + x2 |
| set | None | None |
| None | set | None |
| None | None | None |

### Logical Or (`\|`)

| x1 | x2 | Result |
|----|----|--------|
| set | set | x1 + x2 |
| set | None | x1 |
| None | set | x2 |
| None | None | None |

### Logical Xor (`^`)

| x1 | x2 | Result |
|----|----|--------|
| set | set | None |
| set | None | x1 |
| None | set | x2 |
| None | None | None |

### Logical Not (`~` / `synalinks.ops.logical_not`)

`~` **always** returns `None`, for any input. It never raises — even on a
`None` operand. (Contrast with `+`, which raises if either operand is
`None`.) Use it as a placeholder / stop-condition that forces a path to
`None` so downstream `^` and `|` short-circuit.

| x | Result |
|---|--------|
| set | None |
| None | None |

The `synalinks.Not` module is the module form of this operator (a
placeholder that maps any input to `None`).

## Decision

```python
synalinks.Decision(
    question="Plain-English routing question shown to the LLM",
    labels=["label1", "label2", ...],
    language_model=lm,
)
```

`labels` are converted to a constrained-output schema (a dynamic Enum on
the `choice` field). The output has two fields:

- `thinking` (str) — step-by-step reasoning produced by the LLM
- `choice` (str) — exactly one of the provided labels

`Decision` does not accept a `return_inputs` argument; concatenate the
inputs manually with `inputs & decision` if you need both.

`temperature` defaults to `0.0`. Other accepted (keyword-only) args:
`prompt_template`, `examples`, `instructions`, `seed_instructions`,
`reasoning_effort`, `use_inputs_schema`, `use_outputs_schema`, `name`,
`description`, `trainable`.

All module constructors are keyword-only (`*` after `self`) — pass every
argument by name.

## MultiDecision

`synalinks.MultiDecision` mirrors `Decision` but allows **multiple**
labels to be chosen at once (multi-label classification). The output
field is `choices` (a `List[str]`), not `choice`, alongside `thinking`.

```python
x = await synalinks.MultiDecision(
    question="Which topics does this article cover?",
    labels=["science", "finance", "sports"],
    language_model=lm,
)(inputs)
# Output: { "thinking": "...", "choices": ["science", "finance"] }
```

Extra arg: `inline` (bool, default `True`) — places the enum directly in
the array items (no `$defs`/`$ref` indirection) for simpler schemas.
Use `MultiDecision` as `Branch`'s `decision_type` for multi-label routing
(see below).

## Branch

```python
synalinks.Branch(
    question="Routing question",
    labels=["a", "b"],
    branches=[module_a, module_b],
    language_model=lm,
    return_decision=True,   # default True
    inject_decision=True,   # default True
    decision_type=synalinks.Decision,  # default; or synalinks.MultiDecision
)
```

`labels` and `branches` must be the same length (one branch per label);
the branch outputs are returned in label order. The selector class is
controlled by `decision_type`:

- `decision_type=synalinks.Decision` (default) — exactly **one** branch
  is selected per call; all other slots are `None`.
- `decision_type=synalinks.MultiDecision` — **one or more** branches may
  fire per call (multi-label routing); non-selected slots stay `None`.

Returns a tuple of length `len(labels)`. With `return_decision=True` the
decision is concatenated *into* each selected branch's output (tuple
length is unchanged); with `inject_decision=True` each branch sees
`inputs + decision` rather than `inputs`. Both default to `True`.

### Branch Output Patterns

```python
# Two branches, merge with or
(easy, hard) = await synalinks.Branch(
    question="Difficulty?",
    labels=["easy", "hard"],
    branches=[easy_module, hard_module],
    language_model=lm,
)(inputs)
final = easy | hard

# Three branches
(a, b, c) = await synalinks.Branch(
    question="Category?",
    labels=["a", "b", "c"],
    branches=[mod_a, mod_b, mod_c],
    language_model=lm,
)(inputs)
final = await synalinks.Or()([a, b, c])

# Get the decision concatenated with the selected module output
(easy, hard) = await synalinks.Branch(
    ...,
    return_decision=True,
)(inputs)

# Inject the decision before the selected module computation
(easy, hard) = await synalinks.Branch(
    ...,
    inject_decision=True,
)(inputs)
```

## Self-Consistency

The standard self-consistency recipe:

```python
inputs = synalinks.Input(data_model=Query)

# 1) Generate N parallel hypotheses with high temperature
b0 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)
b1 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)
b2 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)

# 2) Concatenate (auto-renames colliding fields: answer, answer_1, answer_2)
merged = b0 & b1 & b2

# 3) Final aggregator at low temperature
outputs = await synalinks.Generator(
    data_model=AnswerWithRationale,
    language_model=lm,
    instructions="Critically analyze the candidate answers and produce the final answer.",
)(inputs & merged)

program = synalinks.Program(inputs=inputs, outputs=outputs)
```

Variants:
- Use `synalinks.ops.factorize` to group `answer`, `answer_1`, `answer_2` into a single `answers: [...]` list before aggregation.
- Replace the aggregator with `SelfCritique` for a learned reweighting step.

## Guard Patterns

### Input Guard

A custom module returns either a `ChatMessage` (block) or `None` (allow).

```python
class InputGuard(synalinks.Module):
    async def call(self, inputs, training=False):
        if self._should_block(inputs):
            return synalinks.ChatMessage(role="assistant", content="Cannot process this request.")
        return None
```

Compose:

```python
warning = await InputGuard()(inputs)
guarded_inputs = warning ^ inputs       # bypass downstream if blocked
answer = await synalinks.Generator(language_model=lm)(guarded_inputs)
outputs = warning | answer              # warning OR generated answer
```

### Output Guard

```python
answer = await synalinks.Generator(language_model=lm)(inputs)
warning = await OutputGuard()(answer)
outputs = (answer ^ warning) | warning  # replace answer with warning if needed
```

### Multi-Stage Pipeline Guard

```python
input_warning = await InputGuard()(inputs)
guarded = input_warning ^ inputs

raw_answer = await synalinks.Generator(language_model=lm)(guarded)
output_warning = await OutputGuard()(raw_answer)
safe_answer = (raw_answer ^ output_warning) | output_warning

outputs = input_warning | safe_answer
```

## Common Mistakes

### Using `+` after a Branch

```python
# WRONG — non-activated branches are None, + raises
(easy, hard) = await synalinks.Branch(...)(inputs)
result = easy + hard  # Exception at runtime

# CORRECT
result = easy | hard
```

### Forgetting to return None in custom guards

A guard module that returns an empty DataModel instead of `None` will not be bypassed by `^`.

```python
# WRONG — empty Warning is still a value
async def call(self, inputs, training=False):
    if not self._should_block(inputs):
        return Warning()  # still truthy!
    return Warning(content="Blocked")

# CORRECT
async def call(self, inputs, training=False):
    if not self._should_block(inputs):
        return None
    return Warning(content="Blocked")
```

### Misreading `return_decision` semantics

`return_decision=True` does **not** add an extra tuple element. It concatenates the decision into each selected branch's output. The tuple length always matches `len(labels)`.

```python
# Same destructuring with or without return_decision
(easy, hard) = await synalinks.Branch(..., return_decision=True)(inputs)
# easy / hard now contain decision fields alongside the branch outputs
```

If you need the decision *separately*, run `Decision` upstream and pass it through, or use `inject_decision=True` to make each branch see the decision in its inputs.

## Merging Modules (`And` / `Or` / `Xor`)

The `&`, `|`, `^` operators have module forms that take a **list** of N
inputs and fold the operator across them left-to-right:

```python
merged = await synalinks.And()([b0, b1, b2])   # all required -> None if any is None
picked = await synalinks.Or()([b0, b1, b2])     # pick non-None (concat if several set)
xored  = await synalinks.Xor()([b0, b1, b2])    # exactly-one non-None, else None
```

There is no `Concat` / `+` module form; use `synalinks.ops.concat` or `+`
directly. `synalinks.Not` is the module form of `~` (always outputs
`None`, see the Logical Not truth table above).

## See Also

- the **Core** section — operators, DataModel, Program
- the **Modules** section — building blocks used inside branches
- the **Training** section — branches are optimized as separate specialists
