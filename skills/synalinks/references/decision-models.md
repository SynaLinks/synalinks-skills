# Synalinks Decision Models Reference

A **decision model** (`synalinks.DecisionModel`, e.g. TypeSafe's `jev`) answers
typed questions about its input instead of generating text: yes/no, one option
out of a list, or a score along ordered levels, with calibrated probabilities.
Its answer is always one of the options given. It is much faster and cheaper
than a language model, and fits every step that *decides* rather than *writes*:
routing, classification, guards, grading. It cannot reason step by step,
generate text or values, call tools, or read images.

## Setup

```python
import synalinks

# Reads TYPESAFE_API_KEY (and TYPESAFE_BASE_URL) from the environment on every
# call; the key is never stored in a program's config. Use a .env file.
decision_model = synalinks.DecisionModel(model="typesafe/jev-latest")
# Pin a versioned ID (e.g. "typesafe/jev-1.13.0") once thresholds are tuned.
```

Constructor (keyword-only): `model` (required, `<provider>/<model>`, only
`typesafe/` is supported), `api_base`, `timeout` (30), `retry` (5),
`retry_max_wait` (60), `fallback` (another `DecisionModel`), `cache_dir`
(on-disk response cache), `cost_per_token` (overrides the built-in price
table; only input tokens are billed), `name`, `description`, `hooks`.

Default decision model: `synalinks.set_default_decision_model("typesafe/jev-latest")`
/ `synalinks.default_decision_model()`.

## Fields are questions

A decision model is driven by a data model: each field is one question, asked
with the field's `description` (**required**).

| Field type | Question | Output value |
|------------|----------|--------------|
| `bool` | yes/no | `True` when p(yes) >= 0.5 |
| `Literal[...]` / string `Enum` | choice (up to 255 options) | the most probable option |
| Score type (`synalinks.Rating`, `Rating10`, `Rating20`, `Score`, `FineScore`, any numeric `Enum`) | score over the scale | the scale value nearest to the probability-weighted score |
| `synalinks.decision_models.score_schema(instructions, levels)` | score over 2–10 described levels | `{"score", "legend", "probabilities", "confidence"}` |

```python
from typing import Literal

class Triage(synalinks.DataModel):
    is_billing: bool = synalinks.Field(description="Is the ticket about billing?")
    urgency: Literal["low", "medium", "high"] = synalinks.Field(
        description="How urgent is the ticket?",
    )
    frustration: synalinks.Rating = synalinks.Field(
        description="How frustrated is the customer, from 1 (calm) to 5 (angry)?",
    )
```

Describe the meaning of a scale in the `description` (which end is good).
Scales finer than the API's 10 levels (`Score`, `FineScore`, `Rating20`) are
asked over 10 levels spread over their range and mapped back.

Any other field (free-form `str`, number, list, nested free text) raises
`UnsupportedSchemaError` (from `synalinks.decision_models`) — up front, when the
module is built. Check a schema with `decision_model.check_schema(schema)`.

All questions of a data model are answered in **one call**, in parallel: put
every question about an input in the same data model.

## The `decision_model` argument

A decision model is **never** a `language_model`: passing one as
`language_model` raises `ValueError` in every module (it is rejected by
`synalinks.language_models.get`). The modules that can use one take a
`decision_model` argument (last in their signature):

| Module | With a decision model |
|--------|------------------------|
| `Generator` | Answers the questions of its data model. `instructions` and `examples` become the system message the questions are answered in. |
| `Decision` | Asks `question` as is over `labels`. Output `{"choice"}` (no `thinking`). |
| `MultiDecision` | One yes/no question per label ("{question} Does the label 'x' apply?"). Output `{"choices"}` (no `thinking`); can be empty. |
| `Branch` | Its `Decision`/`MultiDecision` decides; the branches keep their own models. |
| `SelfCritique` | Grades inputs on five levels ("Very bad." … "Very good."); `reward` normalized to `[0, 1]`; **no `critique`**, so `return_reward` must stay `True`. |
| `RubricsAsJudge` (+ rubric presets) | Every criterion is a score question over five levels, all in one call; `critique` lists each criterion's level and confidence. `score_type` does not apply. |

```python
team = await synalinks.Decision(
    question="Which team should handle the ticket?",
    labels=["billing", "technical", "sales"],
    decision_model=decision_model,
)(inputs)

(billing, technical) = await synalinks.Branch(
    question="Which team should handle the ticket?",
    labels=["billing", "technical"],
    branches=[billing_agent, technical_agent],   # keep their own LMs
    decision_model=decision_model,
)(inputs)
```

### Resolution order (modules accepting one)

1. `decision_model`
2. a `language_model` other than the default one
3. the default decision model (`set_default_decision_model`)
4. the default language model

The default decision model only applies where it can answer: a `Generator`
uses it only when every field is a question (a text-writing `Generator`, incl.
the one inside `ChainOfThought`, keeps the default LM); `SelfCritique(return_reward=False)`
and `RubricsAsJudge(score_type=...)` skip it.

## Confidence thresholds

- `Decision(min_confidence=...)` — below it, the decision is not taken: the
  module returns **`None`** (abstains) instead of a guess.
- `MultiDecision(threshold=...)` — the probability from which a label is kept
  (default `0.5`).
- `Branch(min_confidence=..., threshold=...)` — forwarded to its decision
  module; when the decision is not taken, **no branch runs**.

Both raise `ValueError` without a decision model. Outputs never contain
probabilities; the thresholds are applied inside the modules.

## Tuning thresholds with KerasTuner

Thresholds are **hyperparameters** (not trained). Search them with
`synalinks.tuners` (see the Hyperparameter Search guide). The reward must
price an abstention (`y_pred is None`) **between** a wrong and a right answer,
otherwise the search learns that abstaining never pays and picks 0.

```python
import synalinks

synalinks.disable_keras_backend()  # MUST run before using the tuners


async def reward(y_true, y_pred):
    if y_pred is None:  # abstained: better than wrong, worse than right
        return 0.5
    return float(y_pred.get("choice") == y_true.get("choice"))


async def build_program(hp):
    synalinks.clear_session()
    inputs = synalinks.Input(data_model=Ticket)
    outputs = await synalinks.Decision(
        question="Which team should handle the ticket?",
        labels=["billing", "technical", "sales"],
        decision_model=decision_model,
        min_confidence=hp.Float("min_confidence", 0.0, 0.9, step=0.1),
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs)
    program.compile(
        reward=reward,
        # Optional: each trial also trains the prompt for its threshold.
        optimizer=synalinks.optimizers.OMEGA(
            language_model=lm, embedding_model=em
        ),
    )
    return program


tuner = synalinks.tuners.GridSearch(
    build_program,
    objective=synalinks.tuners.Objective("val_reward", direction="max"),
)
tuner.search(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=2)
program = await build_program(tuner.get_best_hyperparameters()[0])
await program.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=10)
```

`keras-tuner` is an optional dependency; without `disable_keras_backend()` the
import fails with a missing TensorFlow/JAX/PyTorch error. `hp.Float(..., step=0.1)`
yields float noise (`0.30000000000000004`): round when printing.

## In-context learning & training

Decision models answer in the context of the whole state (system message with
instructions + few-shot examples, then the inputs), so they learn in context:
every optimizer works unchanged on the modules' trainable variables —
`RandomFewShot` selects examples, `OMEGA` evolves the instructions (its own LM
writes candidates; its embedding model keeps them diverse via DNS). Evaluating a
candidate only costs decision model calls.

## Reliability, metrics, tracing

- Retries on 408/409/429/5xx/529 with backoff (`Retry-After` honored); 422 and
  auth errors are not retried. Every answer is validated against its question.
- A call failing every attempt **warns and returns `None`** (or calls
  `fallback`). A missing `TYPESAFE_API_KEY` fails the same way.
- Decision models are counted by the **language model operational metrics**
  (`TotalTokens`, `Cost`, `AvgLatency`, `FailedCalls`, `ErrorRate`, their
  `Reward*`/`Optimizer*` variants), `ProgramCost` and `BudgetStopping` — there
  are no decision-model-specific metrics.
- Observability: `CHAT_MODEL` spans with messages, token usage, cost,
  `synalinks.response_model` (the versioned model that answered) and the raw
  `answers` (with probabilities) in the span outputs; failed calls are marked
  `ERROR`.

## Pitfalls

1. Passing a decision model as `language_model` → `ValueError`. Use `decision_model=`.
2. A field without `description` → `UnsupportedSchemaError`.
3. A free-text field (`str`, number, list) with a decision model → `UnsupportedSchemaError`; use a `LanguageModel` for it.
4. `min_confidence` / `threshold` without a decision model → `ValueError`.
5. Tuning a threshold with a reward that scores `None` as 0 → the search always picks 0.
6. `ChainOfThought` / `thinking` / `critique` fields need generation: not with a decision model.
7. Always handle `None`: failed calls and abstentions both yield `None`.
