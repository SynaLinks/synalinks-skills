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

## Reduction

Every `Reward` (and `RewardFunctionWrapper`, `ExactMatch`, `CosineSimilarity`,
`LMAsJudge`, `ProgramAsJudge`) constructor also takes `reduction` (default
`"mean"`). Allowed: `"mean"`, `"sum"`, `"min"`, `"max"`, `"none"`/`None`. It controls
how per-sample rewards collapse into the scalar shown in progress logs and used for
candidate scoring (`"min"` = pessimistic/worst-sample, `"max"` = best-of-N).
`"none"`/`None` falls back to `"mean"` for scalar consumers while preserving
per-sample values for the optimizer's RL bookkeeping.

## Built-in Rewards

### ExactMatch

```python
synalinks.rewards.ExactMatch(
    name="exact_match",
    in_mask=None,            # list[str] or None — exact field names to keep
    out_mask=None,           # list[str] or None — exact field names to drop
    in_mask_pattern=None,    # str regex — fields matching are kept (OR with in_mask)
    out_mask_pattern=None,   # str regex — fields matching are dropped (OR with out_mask)
)
```

Compares `y_true.get_json() == y_pred.get_json()` after masking. Returns 1.0 on full match, 0.0 otherwise. Both exact (`in_mask`/`out_mask`) and regex (`in_mask_pattern`/`out_mask_pattern`) masking are available as constructor kwargs — they are forwarded to the data model's own `in_mask(mask=..., pattern=...)` / `out_mask(mask=..., pattern=...)` methods.

### CosineSimilarity

```python
synalinks.rewards.CosineSimilarity(
    embedding_model=None,   # Optional: resolved at call time via synalinks.default_embedding_model()
    axis=-1,
    name="cosine_similarity",
    in_mask=None,
    out_mask=None,
    in_mask_pattern=None,
    out_mask_pattern=None,
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
    instructions=None,       # Default instructions for the judge generator
    name="lm_as_judge",
    in_mask=None,
    out_mask=None,
    in_mask_pattern=None,
    out_mask_pattern=None,
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
    in_mask_pattern=None,
    out_mask_pattern=None,
)
```

Calls `judge_program([y_true, y_pred])` and returns `float(result.get("reward", 0.0))`.
Use for multi-stage / rubric judging. If the underlying call fails and the program
returns `None`, `ProgramAsJudge` warns and returns `0.0`.

### RubricsAsJudge

Grades a prediction against **named, weighted criteria** and combines them in
Python (weighted mean of the normalized scores), instead of one holistic score.

```python
synalinks.rewards.RubricsAsJudge(
    language_model=None,     # LM judge (or pass decision_model instead)
    rubrics=[                # list of dict / str, or a preset name (str)
        {"name": "correct", "description": "The answer is correct.", "weight": 2},
        {"name": "concise", "description": "No preamble, no filler."},
    ],
    prompt_template=None,
    examples=None,
    instructions=None,
    score_type=None,         # per-criterion scale, default synalinks.Rating20 (LM only)
    reduction="mean",
    name="rubrics_as_judge",
    in_mask=None, out_mask=None, in_mask_pattern=None, out_mask_pattern=None,
    decision_model=None,     # grade with a DecisionModel (see decision-models.md)
)
```

- A rubric item is a dict with `name`, `description` and optional `weight`
  (default `1.0`), or a bare string (its description, name derived from it).
  Names are normalized to snake_case JSON properties; weights must be
  positive; names must be unique after normalization.
- `rubrics="faithfulness"` (a string) selects a **built-in preset**.
- The output has a `critique`, one score per criterion (on `score_type` with a
  language model, normalized to `[0, 1]` with a decision model), and `reward`
  (the weighted mean of the normalized scores).
- With `decision_model=`, every criterion is a score question over five levels,
  all in **one call**; the `critique` lists each criterion's level and
  confidence. `score_type` then raises.

### Rubric presets

`synalinks.rewards.list_rubrics()` lists them, `get_rubric(name)` returns the
rubric dicts. Each preset also has a ready-made reward class taking the same
arguments as `RubricsAsJudge` (minus `rubrics`), e.g.
`synalinks.rewards.Faithfulness(language_model=lm)` or
`synalinks.rewards.Toxicity(decision_model=dm)`:

| Area | Rewards |
|------|---------|
| Answers | `AnswerRelevancy`, `Faithfulness`, `Hallucination`, `PromptAlignment`, `TopicAdherence`, `ArgumentCorrectness`, `Summarization` |
| RAG | `ContextualPrecision`, `ContextualRecall`, `ContextualRelevancy`, `CitationFaithfulness` |
| Safety | `Bias`, `Toxicity`, `PIILeakage`, `Misuse`, `NonAdvice` |
| Agents | `TaskCompletion`, `GoalAccuracy`, `PlanQuality`, `PlanAdherence`, `StepEfficiency`, `AgentLoopDetection`, `ToolCorrectness`, `ToolUse`, `ToolPermission` |
| Conversations | `ConversationCompleteness`, `KnowledgeRetention`, `RoleAdherence`, `RoleViolation`, `TurnFaithfulness`, `TurnRelevancy` |

The judge only sees `y_true` and `y_pred`: for presets about context, tools or
plans, the program's output must include them (e.g. `return_inputs=True`, or an
agent's trajectory).

### AgentAsJudge

A `FunctionCallingAgent` judge: it **calls tools** to gather evidence (run the
predicted code, query a database, recompute a result) before critiquing and
grading. Use when correctness can't be judged from the text alone.

```python
synalinks.rewards.AgentAsJudge(
    language_model=lm,
    tools=[synalinks.Tool(calculate)],   # at least one tool is required
    instructions=None, final_instructions=None,
    score_type=synalinks.Rating20,       # normalized to [0, 1]
    max_iterations=5,
    use_chain_of_thought=False,
    workdir=None, skills=None,
    reduction="mean", name="agent_as_judge",
    in_mask=None, out_mask=None, in_mask_pattern=None, out_mask_pattern=None,
)
```

### DeepAgentAsJudge

A `DeepAgent` judge working in a sandboxed copy-on-write copy of `workdir`
(`read_file`, `list_files`, `search_files`, `write_file`, `edit_file`,
`run_bash`): it can apply a predicted patch, **run the project's tests**, or
execute a generated script. Gold and prediction are written to a JSON file in
the sandbox. The real `workdir` is never touched; by default the sandbox is
reset before each evaluation.

```python
synalinks.rewards.DeepAgentAsJudge(
    language_model=lm,
    workdir="path/to/project",   # seeds the sandbox (tests, sources)
    score_type=synalinks.Rating, # default Rating20; normalized to [0, 1]
    max_iterations=10, timeout=30.0,   # per run_bash command
    sub_language_model=None, max_subagent_depth=0,
    sandbox=None, reset_sandbox=None,  # True when it builds its own sandbox
    tools=None, skills=None,
)
```

### RLMAsJudge

A `RecursiveLanguageModelAgent` judge writing **Python in a persistent
sandbox**: gold and prediction are bound as the `inputs` dict, so it can
execute predicted code, diff long outputs field by field, or delegate semantic
comparisons to a sub-LM via `llm_query` (`recursive=True`). The prompt only sees
a summary of the inputs, so it scales to predictions too large for one context.
It ends with `submit(result={"critique": ..., "reward": ...})`; the reward may
be any value within the `score_type` range (e.g. a computed pass rate).

```python
synalinks.rewards.RLMAsJudge(
    language_model=lm,
    recursive=True, max_llm_calls=50, sub_language_model=None,
    max_iterations=20, timeout=60, max_output_chars=10_000,
    score_type=synalinks.Rating20,
    tools=None, native_tools=None, workdir=None, sandbox=None, sandbox_type=None,
)
```

### Choosing a judge

| Need | Reward |
|------|--------|
| Exact / semantic match with a reference | `ExactMatch`, `CosineSimilarity` |
| One holistic score from reading the output | `LMAsJudge` |
| Several named, weighted criteria | `RubricsAsJudge` / a rubric preset |
| Fast, cheap grading of many criteria | `RubricsAsJudge(decision_model=...)` |
| Verify with tools | `AgentAsJudge` |
| Run tests / patches in a project | `DeepAgentAsJudge` |
| Compute the verdict in code, or huge outputs | `RLMAsJudge` |
| Any custom judge program | `ProgramAsJudge` |

## Built-in Metrics

```python
synalinks.metrics.F1Score(in_mask=["answer"])                # token-set F1 over flattened fields
synalinks.metrics.FBetaScore(beta=0.5, in_mask=["answer"])
synalinks.metrics.Precision(in_mask=["answer"])              # subclass of FBetaScore
synalinks.metrics.Recall(in_mask=["answer"])                 # subclass of FBetaScore
synalinks.metrics.BinaryF1Score(in_mask=["label"])           # bool / Score fields
synalinks.metrics.BinaryFBetaScore(beta=2.0, in_mask=["label"])
synalinks.metrics.BinaryPrecision(in_mask=["label"])
synalinks.metrics.BinaryRecall(in_mask=["label"])
synalinks.metrics.CategoricalF1Score(in_mask=["sources"])    # list / multi-label fields
synalinks.metrics.CategoricalFBetaScore(beta=1.0, in_mask=["sources"])
synalinks.metrics.ListF1Score(in_mask=["sources"])           # alias of CategoricalF1Score
synalinks.metrics.ListFBetaScore(beta=1.0, in_mask=["sources"])  # alias of CategoricalFBetaScore
synalinks.metrics.Mean()                                     # generic averaging
synalinks.metrics.MeanMetricWrapper(fn=my_metric)
synalinks.metrics.Sum()
```

`Precision` / `Recall` (and `BinaryPrecision`/`BinaryRecall`,
`CategoricalPrecision`/`CategoricalRecall`) DO exist — they subclass `FBetaScore`
and reuse its TP/FP/FN state, just swapping the result formula. `F1Score` is
token-set based: it tokenizes the masked fields and compares token overlap. Useful
for short-form answers. For multi-class / multi-label classification with boolean or
`synalinks.Score` fields, use `BinaryF1Score` or the `Categorical*` (aka `List*`)
variants. All F-score / precision / recall metrics accept `in_mask` / `out_mask` /
`in_mask_pattern` / `out_mask_pattern`.

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

`ComposableReward` combines several rewards (instances or async reward
functions) into one weighted reward. Each child keeps its own masks; the outer
`reduction` combines the children (`"mean"` = weighted mean).

```python
synalinks.rewards.ComposableReward(
    rewards=[
        synalinks.rewards.ExactMatch(in_mask=["answer"]),
        synalinks.rewards.Faithfulness(language_model=lm),
        my_async_reward,
    ],
    weights=[2.0, 1.0, 1.0],   # optional, positive; default equal weights
    reduction="mean",           # "mean" | "sum" | "min" | "max" | "none"
    name="composable_reward",
    in_mask=None, out_mask=None, in_mask_pattern=None, out_mask_pattern=None,
)
```

## Batch Rewards

When a reward needs the **whole batch** (group-relative scores, batch
normalization, pairwise comparisons), subclass `BatchReward` (implement
`call(y_true, y_pred)` over lists, returning one float per sample) or wrap an
async batched function:

```python
async def my_batch_reward(y_true, y_pred):   # lists of length batch_size
    return [1.0 if t.get_json() == p.get_json() else 0.0
            for t, p in zip(y_true, y_pred)]

program.compile(
    reward=synalinks.rewards.BatchRewardFunctionWrapper(fn=my_batch_reward),
)
```

A batched function is **never auto-wrapped** by `compile` (its signature can't
be told apart from a per-sample one): always wrap it in
`BatchRewardFunctionWrapper`.

## Pitfalls

1. **Missing decorator** — `register_synalinks_serializable` is required for save/load.
2. **Sync function** — must be `async def`.
3. **Wrong wrapper** — wrap functions in `RewardFunctionWrapper` / `MeanMetricWrapper`. Built-in classes (`ExactMatch`, `CosineSimilarity`, ...) are passed directly. (`MeanRewardWrapper` does not exist — that name was an alias mistake.)
4. **Returning `None`** — breaks the optimizer. Always return a float, even on failure (use 0.0).
5. **Reward leakage** — using `y_true` as part of a custom reward that's later applied at inference (without ground truth) silently breaks evaluation.
6. **`ProgramAsJudge` field name** — the judge program's output schema must use `reward` (not `score`); the wrapper reads `result.get("reward", 0.0)`.
7. **Rubric names** — normalized to snake_case and must stay unique; weights must be positive.
8. **`score_type` with a decision model** — `RubricsAsJudge(decision_model=..., score_type=...)` raises: decision models grade on fixed levels.
9. **Abstentions** — with `min_confidence` (decision models), `y_pred` is `None` when the program abstains; score it between a wrong and a right answer if you tune the threshold.

## See Also

- the **Training** section — `compile()` API
- the **Optimizers** section — How OMEGA uses reward variance for variable selection
