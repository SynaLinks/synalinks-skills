# Synalinks Rewards & Metrics Reference

## Reward vs Metric

| | Reward | Metric |
|-|--------|--------|
| Drives optimization | Yes | No |
| Required by `compile()` | Yes | No |
| Returns float | Yes | Yes |
| Per-example or batch | Per-example, averaged | Per-example, averaged |
| Multiple allowed | One | Many |

A reward IS a metric — but only one signal drives optimization. Other signals you want to track go in `metrics=[...]`.

## Reward Function Signature

```python
@synalinks.saving.register_synalinks_serializable()
async def my_reward(y_true, y_pred):
    """
    Args:
        y_true: Ground truth JsonDataModel (or None)
        y_pred: Prediction JsonDataModel (or None)

    Returns:
        float in [0.0, 1.0] (by convention)
    """
    ...
```

`y_true` is the ground truth from your `y_train` / `y_test` array. `y_pred` is whatever the program returned for the corresponding `x`. **Either may be `None`** in branched programs.

Wrap the function in `RewardFunctionWrapper` when passing to `compile`:

```python
program.compile(reward=synalinks.rewards.RewardFunctionWrapper(fn=my_reward))
```

You can also pass the bare async function or a string identifier — `compile()` auto-wraps via `RewardFunctionWrapper` internally:

```python
program.compile(reward=my_reward)            # auto-wrapped
program.compile(reward="exactmatch")         # case-insensitive class name lookup
```

## Metric Function Signature

Identical to reward — wrap with `MeanMetricWrapper`:

```python
program.compile(metrics=[synalinks.metrics.MeanMetricWrapper(fn=my_metric)])
```

`metrics.get(identifier)` and `metrics.deserialize(...)` are also case-insensitive
(class name matched against `cls.__name__.lower()`).

## Built-in Rewards

### ExactMatch

```python
synalinks.rewards.ExactMatch(
    name="exact_match",
    in_mask=None,    # list[str] or None — fields to keep
    out_mask=None,   # list[str] or None — fields to drop
)
```

Compares `y_true.get_json() == y_pred.get_json()` after masking. Returns 1.0 on full match, 0.0 otherwise. There is no `pattern=` kwarg on the constructor — pattern-based masking happens via the data model's own `in_mask(pattern=...)` / `out_mask(pattern=...)` methods, which you can call inside a custom reward.

### CosineSimilarity

```python
synalinks.rewards.CosineSimilarity(
    embedding_model=None,   # Optional: resolved at call time via synalinks.default_embedding_model()
    axis=-1,
    name="cosine_similarity",
    in_mask=None,
    out_mask=None,
)
```

Embeds the masked `y_true` and `y_pred` (one embedding call each), then returns
`(sum(l2_norm(y_true) * l2_norm(y_pred)) + 1) / 2` — the classic cosine similarity
remapped from `[-1, 1]` to `[0, 1]` so larger is better.

### LMAsJudge

```python
synalinks.rewards.LMAsJudge(
    language_model=None,     # Optional: resolved at call time via synalinks.default_language_model()
    prompt_template=None,    # Jinja2 prompt template (see Generator)
    examples=None,           # Few-shot examples for the judge
    instructions=None,       # Default instructions string for the judge generator
    name="lm_as_judge",
    in_mask=None,
    out_mask=None,
)
```

Internally constructs an `LMAsJudgeProgram` that wraps a `SelfCritique` module.
`SelfCritique` produces a `CritiqueWithReward` data model (`critique: str`,
`reward: Score`); `LMAsJudge` extracts the `reward` field. Steer the judge via
`instructions=...` — there is no `criteria=` parameter.

**Tip:** Use a smaller/cheaper model for the judge than for the program being trained.

### ProgramAsJudge

```python
judge_program = synalinks.Program(...)  # output schema MUST include a 'reward' float field
synalinks.rewards.ProgramAsJudge(
    program=judge_program,
    name=None,
    in_mask=None,
    out_mask=None,
)
```

Calls `judge_program([y_true, y_pred])` and returns `float(result.get("reward", 0.0))`.
Use for multi-stage / rubric judging. If the underlying call fails and the program
returns `None`, `ProgramAsJudge` warns and returns `0.0`.

## Built-in Metrics

```python
synalinks.metrics.F1Score(in_mask=["answer"])                # token-set F1 over flattened fields
synalinks.metrics.FBetaScore(beta=0.5, in_mask=["answer"])
synalinks.metrics.BinaryF1Score(in_mask=["label"])           # bool / Score fields
synalinks.metrics.BinaryFBetaScore(beta=2.0, in_mask=["label"])
synalinks.metrics.ListF1Score(in_mask=["sources"])           # list / retrieval fields
synalinks.metrics.ListFBetaScore(beta=1.0, in_mask=["sources"])
synalinks.metrics.Mean()                                     # generic averaging
synalinks.metrics.MeanMetricWrapper(fn=my_metric)
synalinks.metrics.Sum()
```

There are no standalone `Precision` or `Recall` classes. `F1Score` is token-set
based: it tokenizes the masked fields with `nlp_utils.normalize_and_tokenize` and
compares token overlap. Useful for short-form answers. For multi-class / multi-label
classification with boolean or `synalinks.Score` fields, use `BinaryF1Score`.

## Custom Reward Patterns

### Numeric tolerance

```python
@synalinks.saving.register_synalinks_serializable()
async def numeric_within_tolerance(y_true, y_pred, tol=0.01):
    if not y_true or not y_pred:
        return 0.0
    try:
        true_v = float(y_true.get("answer"))
        pred_v = float(y_pred.get("answer"))
    except (TypeError, ValueError):
        return 0.0
    return float(abs(true_v - pred_v) <= tol * max(abs(true_v), 1))
```

### Trajectory length penalty (agents)

```python
@synalinks.saving.register_synalinks_serializable()
async def trajectory_efficient(y_true, y_pred):
    """Reward correct answers, with a small penalty for long trajectories."""
    if not y_true or not y_pred:
        return 0.0
    correct = float(y_true.get("answer") == y_pred.get("answer"))
    if not correct:
        return 0.0
    n_steps = len(y_pred.get("trajectory") or [])
    return correct * max(0.0, 1.0 - 0.1 * n_steps)
```

### Multi-aspect

```python
@synalinks.saving.register_synalinks_serializable()
async def multi_aspect(y_true, y_pred):
    if not y_true or not y_pred:
        return 0.0
    correctness = float(y_true.get("answer") == y_pred.get("answer"))
    has_reasoning = float(bool(y_pred.get("thinking")))
    return 0.7 * correctness + 0.3 * has_reasoning
```

## Combining Rewards

There's no built-in reward composition — combine inside a single custom reward:

```python
@synalinks.saving.register_synalinks_serializable()
async def combined(y_true, y_pred):
    em_score = ...   # exact match logic
    cos_score = ...  # cosine logic
    return 0.5 * em_score + 0.5 * cos_score
```

## Pitfalls

1. **Missing decorator** — `register_synalinks_serializable` is required for save/load.
2. **Sync function** — must be `async def`.
3. **Wrong wrapper** — wrap functions in `RewardFunctionWrapper` / `MeanMetricWrapper`. Built-in classes (`ExactMatch`, `CosineSimilarity`, ...) are passed directly. (`MeanRewardWrapper` does not exist — that name was an alias mistake.)
4. **Returning `None`** — breaks the optimizer. Always return a float, even on failure (use 0.0).
5. **Reward leakage** — using `y_true` as part of a custom reward that's later applied at inference (without ground truth) silently breaks evaluation.
6. **`ProgramAsJudge` field name** — the judge program's output schema must use `reward` (not `score`); the wrapper reads `result.get("reward", 0.0)`.

## See Also

- **synalinks-training** — `compile()` API
- **synalinks-optimizers** — How OMEGA uses reward variance for variable selection
