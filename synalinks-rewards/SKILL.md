---
name: synalinks-rewards
description: Use when configuring or writing Synalinks reward functions and metrics — ExactMatch, CosineSimilarity, LMAsJudge, ProgramAsJudge, RewardFunctionWrapper, custom reward functions (async, register_synalinks_serializable), in_mask / out_mask filtering, F1Score / FBetaScore / BinaryF1Score / ListF1Score metrics, MeanMetricWrapper, or whenever you're shaping the signal that drives optimization.
---

# Synalinks Rewards & Metrics

Rewards drive optimization (the optimizer maximizes them). Metrics are passive — monitored but not used to update variables.

Both return a float, typically in `[0.0, 1.0]`. Both support field masking via `in_mask` and `out_mask` so you can score only the fields you care about.

## Quick Start

```python
program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.RandomFewShot(),
    metrics=[
        synalinks.metrics.F1Score(in_mask=["answer"]),
    ],
)
```

`compile()` also accepts string identifiers, Keras-style (case-insensitive lookup against `cls.__name__.lower()`):

```python
program.compile(
    reward="exactmatch",        # or "ExactMatch", "EXACTMATCH"
    optimizer=synalinks.optimizers.RandomFewShot(),
    metrics=["mean"],           # wraps as MeanMetricWrapper around the reward
)
```

## Built-in Rewards

### ExactMatch

String equality on the JSON of selected fields. Simple, deterministic, free.

```python
synalinks.rewards.ExactMatch(
    in_mask=["answer"],   # Compare only these fields
    out_mask=None,        # Or: exclude these fields
)
```

**Use when:** Discrete answers (numbers, multiple choice, named entities).

### CosineSimilarity

Semantic similarity via embeddings. Returns `(cosine + 1) / 2`, mapping `[-1, 1] -> [0, 1]`.

```python
synalinks.rewards.CosineSimilarity(
    embedding_model=em,   # Optional: falls back to synalinks.default_embedding_model() at call time
    in_mask=["answer"],
)
```

**Use when:** Free-form text where exact wording varies but meaning matters (summaries, paraphrases).

**Cost:** Two embedding calls per (y_true, y_pred) pair per evaluation.

### LMAsJudge

Use an LLM as judge. Internally builds a `SelfCritique` module that returns a `reward` field in `[0, 1]`.

```python
synalinks.rewards.LMAsJudge(
    language_model=lm,            # Optional: falls back to synalinks.default_language_model() at call time
    instructions=None,            # Optional. Default instructions string for the judge generator.
    prompt_template=None,         # Optional. Jinja2 prompt template (see Generator).
    examples=None,                # Optional. Few-shot examples for the judge.
    in_mask=None,
    out_mask=None,
)
```

**Use when:** Subjective criteria — helpfulness, safety, formatting quality. Steer the judge via `instructions=...` (there is no `criteria=` parameter).

**Cost:** One LM call per evaluation. Use a small/cheap model — judging is easier than generating.

### ProgramAsJudge

Use another Synalinks program as judge. Lets you build complex multi-criterion graders.

The judge program is called with `[y_true, y_pred]` and **its output schema must include a field named `reward`** (a float in `[0, 1]`).

```python
judge = synalinks.Program(...)  # outputs include a "reward" field
synalinks.rewards.ProgramAsJudge(program=judge)
```

**Use when:** You need a custom judge with structured intermediate steps (rubric scoring, multi-aspect evaluation, retrieval-grounded judging).

## Custom Rewards

A custom reward is an async function `(y_true, y_pred) -> float`, decorated with `register_synalinks_serializable` and wrapped in `RewardFunctionWrapper` (or passed as a bare function/string to `compile`, which auto-wraps it):

```python
@synalinks.saving.register_synalinks_serializable()
async def word_overlap(y_true, y_pred):
    """Fraction of ground-truth words that appear in the prediction."""
    true_val = y_true.get("answer") if y_true else None
    pred_val = y_pred.get("answer") if y_pred else None

    if true_val is None or pred_val is None:
        return 0.0

    true_words = set(true_val.lower().split())
    pred_words = set(pred_val.lower().split())
    overlap = len(true_words & pred_words)
    return overlap / max(len(true_words), 1)


program.compile(
    reward=synalinks.rewards.RewardFunctionWrapper(fn=word_overlap),
    optimizer=...,
)
```

**Always handle `None`** — `y_true` or `y_pred` may be `None` (especially in branched programs where some outputs aren't activated).

## Field Masking

`in_mask` and `out_mask` on rewards and metrics accept a `list[str]` of field names. They are passed through to the underlying `JsonDataModel.in_mask(mask=..., pattern=...)` / `out_mask(mask=..., pattern=...)` methods.

```python
# Compare only "answer"
synalinks.rewards.ExactMatch(in_mask=["answer"])

# Compare everything except "thinking"
synalinks.rewards.ExactMatch(out_mask=["thinking"])
```

The `pattern=` regex argument lives on the data-model methods, not on `Reward`/`Metric` constructors. To filter by regex inside a custom reward, call the method directly:

```python
@synalinks.saving.register_synalinks_serializable()
async def answer_only(y_true, y_pred):
    if not y_true or not y_pred:
        return 0.0
    yt = y_true.in_mask(pattern="^answer")
    yp = y_pred.in_mask(pattern="^answer")
    return float(yt.get_json() == yp.get_json())
```

**Why mask:** ChainOfThought outputs include `thinking` which is rarely worth scoring directly. Masking focuses the optimizer on the field that drives task success.

## Built-in Metrics

```python
synalinks.metrics.F1Score(in_mask=["answer"])         # token-set F1 over flattened fields
synalinks.metrics.FBetaScore(beta=0.5, in_mask=["answer"])
synalinks.metrics.BinaryF1Score(in_mask=["label"])    # for boolean / Score classification
synalinks.metrics.ListF1Score(in_mask=["sources"])    # for list/retrieval fields
```

There are no `Precision` or `Recall` classes — F1Score is implemented at the token level over flattened fields. For precision-only / recall-only signals, write a custom metric or use `FBetaScore` with a high/low `beta`.

Metrics are tracked alongside reward during `fit()` and `evaluate()` and appear in the `History` object.

## Custom Metrics

```python
@synalinks.saving.register_synalinks_serializable()
async def length_match(y_true, y_pred):
    if not y_true or not y_pred:
        return 0.0
    return float(len(y_true.get("answer")) == len(y_pred.get("answer")))

program.compile(
    metrics=[synalinks.metrics.MeanMetricWrapper(fn=length_match)],
    ...
)
```

## Choosing a Reward

| Task signal | Recommended reward |
|-------------|--------------------|
| Discrete / categorical answer | `ExactMatch` |
| Numeric answer | `ExactMatch` (cast to string) or custom (within tolerance) |
| Free-form text | `CosineSimilarity` |
| Subjective quality | `LMAsJudge` |
| Multi-criterion / structured | `ProgramAsJudge` or custom |
| Trajectory / agent | Custom reward inspecting `trajectory` field |

## Common Pitfalls

1. **Forgetting `register_synalinks_serializable`** on a custom reward — the program won't save/load correctly.
2. **Returning Python `bool` instead of `float`** — works but loses gradient signal in optimizers like OMEGA that use softmax.
3. **Ignoring `None`** — leads to `AttributeError: 'NoneType' object has no attribute 'get'` on branched outputs.
4. **Scoring `thinking` fields** — usually noise; mask them out.
5. **Mixing reward and metric** — metric changes don't drive optimization; only `reward` does.
6. **`ProgramAsJudge` output without a `reward` field** — the judge program must expose a float field literally named `reward`; otherwise `ProgramAsJudge` defaults to `0.0`.

## References

- **references/rewards-metrics.md** — All built-ins, masking patterns, advanced custom rewards

## See Also

- **synalinks-training** — How rewards plug into compile/fit
- **synalinks-optimizers** — How rewards are consumed (especially OMEGA's variable selection)
- **synalinks-modules** — `SelfCritique` outputs a reward field that can feed a `ProgramAsJudge`
