---
name: synalinks-control-flow
description: Use when routing or composing Synalinks programs — Decision, Branch (return_decision, inject_decision), parallel branches via asyncio, self-consistency with multiple Generators + temperature, XOR input/output guard patterns, And/Or modules, merging branches with `|`, or anywhere you need conditional execution paths in a Program graph.
---

# Synalinks Control Flow

Patterns for routing, parallelism, conditional execution, and guards in Synalinks programs.

## Overview

Control flow in Synalinks is implemented through:

1. **Operators** (`+ & | ^`) on DataModels — see **synalinks-core**
2. **Decision** / **Branch** modules — LLM-driven routing
3. **Parallel branches** — auto-detected by the graph; run concurrently via asyncio
4. **None propagation** — modules returning `None` short-circuit downstream `&`/`+` operations

> **API conventions**
> - All module constructors take **keyword-only** arguments — call as
>   `Decision(question=..., labels=..., language_model=lm)`, never positionally.
> - `language_model` may be omitted if a process-wide default is set via
>   `synalinks.set_default_language_model(lm)`; `ops.predict` resolves it at
>   call time.
> - `LanguageModel` / `EmbeddingModel` accept `**default_kwargs` (e.g.
>   `temperature`, `top_p`, `max_tokens`) forwarded to every call.

## Decision

Single-label classification for routing. Output includes a `choice` field constrained to the provided labels (plus a `thinking` field with step-by-step reasoning).

```python
decision = await synalinks.Decision(
    question="What type of query is this?",
    labels=["factual", "opinion", "creative"],
    language_model=lm,
)(inputs)
# Output: { "thinking": "...", "choice": "factual" | "opinion" | "creative" }
```

`labels` constrain the LLM output (a dynamic Enum on the `choice` field),
preventing hallucination of new categories. Note: `Decision` does not accept
`return_inputs`; concatenate manually with `inputs & decision` if you need
the original input alongside the choice.

All Synalinks module constructors are keyword-only (`*` after `self`), so
always pass arguments by name.

## Branch

Conditional routing to one of N modules based on a Decision.

```python
(easy_answer, hard_answer) = await synalinks.Branch(
    question="Evaluate query difficulty",
    labels=["easy", "difficult"],
    branches=[
        synalinks.Generator(data_model=SimpleAnswer, language_model=lm),
        synalinks.Generator(data_model=DetailedAnswer, language_model=lm),
    ],
    language_model=lm,
    return_decision=True,   # Default True — decision is concatenated into each branch output
    inject_decision=True,   # Default True — decision is injected into each branch's inputs
)(inputs)

# Merge with logical or — non-activated branches are None
final = easy_answer | hard_answer
```

**Key Branch behaviors:**

- Non-activated branches return `None` (not just empty — they don't run at all)
- Each branch module is optimized **separately** during training, becoming a specialized expert
- `labels` constrain LLM output to valid choices

### Returning the Decision

`return_decision=True` does **not** add an extra tuple element — it concatenates the decision into each selected branch's output, so destructuring stays the same:

```python
(easy, hard) = await synalinks.Branch(
    ...,
    return_decision=True,
)(inputs)
# easy / hard now include the decision fields
```

### Injecting the Decision Into Branch Inputs

```python
(easy, hard) = await synalinks.Branch(
    ...,
    inject_decision=True,  # Each branch receives inputs + decision before computation
)(inputs)
```

## Parallel Branches (Auto-Detected)

Multiple modules consuming the same input run **concurrently** via asyncio:

```python
x1 = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
x2 = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
# Both run in parallel — combine afterwards
combined = x1 + x2
```

## Self-Consistency Pattern

Generate multiple answers with `temperature > 0`, then merge them:

```python
async def build_self_consistency_program(lm):
    inputs = synalinks.Input(data_model=Query)

    b0 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)
    b1 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)
    b2 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)

    merged = b0 & b1 & b2  # Concatenate (auto-renames colliding fields)

    outputs = await synalinks.Generator(
        data_model=AnswerWithRationale,
        language_model=lm,
        instructions="Critically analyze the given answers to produce the final answer.",
    )(inputs & merged)

    return synalinks.Program(inputs=inputs, outputs=outputs)
```

## XOR Guard Patterns

Use `^` to bypass computation conditionally. The truth table for `x1 ^ x2`:

| x1 | x2 | Result |
|----|----|--------|
| set | set | None |
| set | None | x1 |
| None | set | x2 |
| None | None | None |

### Input Guard — Block Processing on Invalid Input

```python
class InputGuard(synalinks.Module):
    """Block invalid inputs."""
    async def call(self, inputs, training=False):
        if self._is_blocked(inputs):
            return synalinks.ChatMessage(role="assistant", content="Cannot process this request")
        return None  # Allow through

async def build_guarded_program(lm):
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    warning = await InputGuard()(inputs)

    # XOR: if warning exists, inputs becomes None (bypassing generator)
    guarded_inputs = warning ^ inputs

    # Generator only runs if guarded_inputs is not None
    answer = await synalinks.Generator(language_model=lm)(guarded_inputs)

    # OR: return warning if it exists, otherwise return answer
    outputs = warning | answer

    return synalinks.Program(inputs=inputs, outputs=outputs)
```

### Output Guard — Replace Invalid Output

```python
async def build_output_guarded_program(lm):
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    answer = await synalinks.Generator(language_model=lm)(inputs)

    warning = await OutputGuard()(answer)

    # XOR + OR: if warning exists, replace answer with warning
    outputs = (answer ^ warning) | warning

    return synalinks.Program(inputs=inputs, outputs=outputs)
```

## And / Or as Modules

Equivalent to `&` and `|` operators, but accept N inputs at once:

```python
merged = await synalinks.And()([b0, b1, b2])
result = await synalinks.Or()([b0, b1, b2])
```

## Merging Branches

| Pattern | When to use |
|---------|-------------|
| `b1 \| b2` | One of two branches activates — pick whichever is non-None |
| `b1 & b2` | Both required — None if either is None |
| `b1 + b2` | Both required, strict — raises if either is None |
| `await synalinks.Or()([b0, b1, b2, ...])` | N branches, pick non-None |
| `await synalinks.And()([b0, b1, b2, ...])` | All N required |

## Common Gotchas

1. **`return_decision=True` does NOT change tuple length** — it concatenates the decision into each branch output. Destructure as usual.
2. **Inside a custom Module, return `None` for skipped paths** so `^` and `&` operators short-circuit correctly.
3. **`Branch` trains each branch independently** — small specialist modules can outperform a single big module on routed sub-tasks.
4. **Don't merge with `+` after a Branch** — non-activated branches are `None`, which makes `+` raise. Use `|` instead.

## References

- **references/control-flow.md** — Truth tables, advanced merging, guard recipes

## See Also

- **synalinks-core** — Operators, DataModel, Program
- **synalinks-modules** — Generator, ChainOfThought, custom modules
- **synalinks-training** — Specialized branches optimized separately
