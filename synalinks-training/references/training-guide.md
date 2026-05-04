# Synalinks Training Guide

## Overview

Synalinks training optimizes prompts and examples via in-context reinforcement learning. Unlike traditional ML, weights are not updated. Instead:

1. **Rewards** evaluate predictions against ground truth
2. **Optimizers** select/generate better examples and instructions
3. **Trainable variables** (JSON objects) are updated to improve performance

## Training Workflow

```python
import numpy as np
import synalinks

# 1. Create program
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
program = synalinks.Program(inputs=inputs, outputs=outputs)

# 2. Compile with reward and optimizer
program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.RandomFewShot(),
    metrics=[synalinks.metrics.F1Score(in_mask=["answer"])],
)

# 3. Train
history = await program.fit(
    x=x_train,
    y=y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
)

# 4. Evaluate
metrics = await program.evaluate(x=x_test, y=y_test)
```

## Rewards

Rewards guide optimization. Return float typically between 0.0 and 1.0.

### Built-in Rewards

#### ExactMatch

String equality comparison.

```python
synalinks.rewards.ExactMatch(
    in_mask=["answer"],     # Only compare these fields
    out_mask=None,          # Exclude these fields
)
```

#### CosineSimilarity

Semantic similarity via embeddings.

```python
synalinks.rewards.CosineSimilarity(
    embedding_model=em,
    in_mask=["answer"],
)
```

#### LMAsJudge

LLM-based evaluation. Uses a `SelfCritique` module under the hood to score the prediction against (optional) ground truth.

```python
synalinks.rewards.LMAsJudge(
    language_model=lm,
    instructions=["Score accuracy and completeness."],  # list[str], optional
    prompt_template=None,                                # jinja2 template, optional
    examples=None,                                       # few-shot examples, optional
    in_mask=["answer"],
)
```

#### ProgramAsJudge

Use another program as judge.

```python
judge = synalinks.Program(...)
synalinks.rewards.ProgramAsJudge(program=judge)
```

### Custom Rewards

The reward function must be `async` (it is awaited inside `RewardFunctionWrapper.call`).

```python
@synalinks.saving.register_synalinks_serializable()
async def my_reward(y_true, y_pred):
    """Custom reward function.

    Args:
        y_true: Ground truth `JsonDataModel`
        y_pred: Prediction `JsonDataModel`

    Returns:
        float: Reward value (typically 0.0 to 1.0)
    """
    true_val = y_true.get("answer") if y_true else None
    pred_val = y_pred.get("answer") if y_pred else None

    if true_val is None or pred_val is None:
        return 0.0

    # Partial match example
    true_words = set(true_val.lower().split())
    pred_words = set(pred_val.lower().split())
    overlap = len(true_words & pred_words)
    return overlap / max(len(true_words), 1)

program.compile(
    reward=synalinks.rewards.RewardFunctionWrapper(fn=my_reward, in_mask=["answer"]),
    optimizer=...,
)
```

### Field Masking

Use `in_mask` or `out_mask` to filter which fields are compared.

```python
# Only compare "answer" field
reward = synalinks.rewards.ExactMatch(in_mask=["answer"])

# Compare all fields except "thinking"
reward = synalinks.rewards.ExactMatch(out_mask=["thinking"])
```

---

## Metrics

Metrics are monitored but not used for optimization (no backpropagation).

### Built-in Metrics

Word-level F-beta score family (operates on tokenized field values, supports
`average` in `{None, "micro", "macro", "weighted"}`):

```python
synalinks.metrics.F1Score(average="weighted", in_mask=["answer"])
synalinks.metrics.FBetaScore(beta=0.5, average="macro", in_mask=["answer"])
synalinks.metrics.BinaryF1Score(average="weighted")  # for boolean/multi-label fields
synalinks.metrics.ListF1Score(average="weighted")    # for list-valued fields
```

Plus reductions and a regression metric:

```python
synalinks.metrics.Mean()
synalinks.metrics.Sum()
synalinks.metrics.MeanMetricWrapper(fn=my_metric)
synalinks.metrics.CosineSimilarity(embedding_model=em)
```

There is no separate `Precision` or `Recall` class — use `F1Score` (or wrap
your own scorer with `MeanMetricWrapper`).

### Custom Metrics

```python
@synalinks.saving.register_synalinks_serializable()
async def my_metric(y_true, y_pred):
    return float(y_true.get("x") == y_pred.get("x"))

program.compile(
    metrics=[synalinks.metrics.MeanMetricWrapper(fn=my_metric, in_mask=["x"])],
    ...
)
```

---

## Optimizers

Optimizers update trainable variables to maximize reward.

### RandomFewShot

Sample among the best predictions to populate the LM's prompt.

```python
synalinks.optimizers.RandomFewShot(
    nb_min_examples=1,           # Min examples per prompt (default 1)
    nb_max_examples=3,           # Max examples per prompt (default 3)
    sampling="softmax",          # "random", "best", or "softmax" (default "softmax")
    sampling_temperature=0.3,    # Softmax sampling temperature (default 0.3)
    population_size=10,          # Best-candidate buffer (default 10)
)
```

**When to use**: Start here as a baseline. Fast, no extra LLM calls, good for quick experiments.

### OMEGA

**O**pti**M**iz**E**r as **G**enetic **A**lgorithm - State-of-the-art evolutionary optimizer using LLM reasoning and Dominated Novelty Search for quality-diversity optimization.

#### All Parameters

```python
synalinks.optimizers.OMEGA(
    # Required: LLM Components
    language_model=lm,              # For mutation/crossover reasoning
    embedding_model=em,             # For diversity metrics in DNS

    # Dominated Novelty Search (DNS) Parameters
    k_nearest_fitter=5,             # K nearest fitter neighbors for DNS
    distance_function=None,         # Async (c1, c2, embedding_model=None, **kw)
                                    # callable; defaults to cosine distance

    # Temperature Controls
    mutation_temperature=0.3,       # Creativity during mutation
    crossover_temperature=0.3,      # Creativity during crossover
    selection_temperature=0.3,      # Softmax candidate selection sharpness
    reasoning_effort=None,          # "minimal"|"low"|"medium"|"high"|"disable"|"none"|None

    # Genetic Algorithm Configuration
    merging_rate=0.02,              # Base crossover probability (scales with epoch)
    population_size=10,             # Max candidates to maintain

    # Algorithm Selection (for ablation studies)
    algorithm="dns",                # "dns" (default) or "ga" (pure GA)
    selection="softmax",            # "random", "best", or "softmax"

    # Customization
    instructions=None,              # Task-specific mutation guidance (str)
    name=None,
    description=None,
)
```

OMEGA does NOT accept `sampling_temperature`, `few_shot_learning`,
`nb_min_examples`, or `nb_max_examples` — those belong to `RandomFewShot`.

#### How OMEGA Works

OMEGA combines genetic algorithms with quality-diversity optimization:

**1. Mutation (Primary Operation)**
- Selects a trainable variable (prompt, instructions, examples, code)
- Uses ChainOfThought to reason about improvements
- Generates enhanced version constrained to original schema
- Temperature controls creativity (lower = conservative, higher = exploratory)

**2. Crossover (Secondary Operation)**
- Combines two high-performing candidates
- Probability increases with epochs: `merging_rate * epoch_number`
- At epoch 0 with default `merging_rate=0.02`: ~0% crossover
- At epoch 10: ~20% crossover
- Falls back to mutation if no alternative candidates exist

**3. Selection Strategies**
- `softmax` (default): Temperature-scaled probability based on reward
- `best`: Greedy selection of highest-reward candidate
- `random`: Uniform random selection

**4. Dominated Novelty Search (DNS) Competition**
At each epoch end, candidates compete:
```
For each candidate:
  - Find all candidates with higher fitness (fitter neighbors)
  - If none exist: score = 1.0 (keep it - not dominated)
  - Otherwise: score = mean distance to k nearest fitter neighbors
    - High score = novel (different from better solutions)
    - Low score = redundant (similar to better solutions)

Keep candidates above median competition score
Retain top population_size candidates
```

**5. Variable Selection**
OMEGA prioritizes worst-performing variables:
```
probability of selection inversely proportional to exp(avg_reward)
```
This ensures all trainable variables get optimization attention.

#### When to Use OMEGA

**Use OMEGA when:**
- RandomFewShot isn't achieving target performance
- You need to optimize prompts/instructions, not just examples
- You want diverse solution exploration (quality-diversity)
- You have compute budget for extra LLM calls per batch
- Task benefits from creative prompt variations

**Stick with RandomFewShot when:**
- Quick prototyping / baseline establishment
- Limited compute budget
- Only need example selection (not prompt optimization)
- Simple tasks where few-shot examples suffice

#### OMEGA vs RandomFewShot

| Aspect | RandomFewShot | OMEGA |
|--------|---------------|-------|
| **What it optimizes** | Examples only | Full trainable variables (prompts, code, plans) |
| **Generation method** | Random sampling | LLM mutation + crossover with ChainOfThought |
| **Quality-Diversity** | No | Yes (Dominated Novelty Search) |
| **Extra LLM calls** | None | 1+ per batch (mutation/crossover) |
| **Population tracking** | Implicit | Explicit with competition |

#### Usage Examples

**Basic OMEGA:**
```python
lm = synalinks.LanguageModel(model="openai/gpt-4o-mini")
em = synalinks.EmbeddingModel(model="text-embedding-3-small")

program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.OMEGA(
        language_model=lm,
        embedding_model=em,
    ),
)

history = await program.fit(
    x=x_train, y=y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
)
```

**Advanced Configuration:**
```python
optimizer = synalinks.optimizers.OMEGA(
    language_model=lm,
    embedding_model=em,

    # Larger population for more diversity
    population_size=20,

    # Stricter DNS competition (fewer neighbors = stricter novelty)
    k_nearest_fitter=3,

    # More creative mutations
    mutation_temperature=0.5,

    # Greedier candidate selection
    selection_temperature=0.1,

    # Reasoning effort for the mutation/crossover LM
    reasoning_effort="medium",

    # Task-specific guidance
    instructions="Focus on edge cases and robustness",
)
```

**Pure Genetic Algorithm (disable DNS for ablation):**
```python
optimizer = synalinks.optimizers.OMEGA(
    language_model=lm,
    embedding_model=em,
    algorithm="ga",         # Disable DNS competition
    selection="best",       # Greedy selection
)
```

#### OMEGA Gotchas and Limitations

**Common Pitfalls:**

1. **Must provide both models:**
   ```python
   # WRONG - crashes at build() time
   optimizer = OMEGA(language_model=lm)  # Missing embedding_model!

   # CORRECT
   optimizer = OMEGA(language_model=lm, embedding_model=em)
   ```

2. **Temperature must be non-zero:**
   ```python
   # WRONG - breaks softmax
   mutation_temperature=0.0

   # CORRECT
   mutation_temperature=0.1  # Small but non-zero
   ```

3. **Custom distance function must be async:**
   ```python
   # CORRECT signature
   async def my_distance(c1, c2, embedding_model=None, **kwargs):
       return distance_value

   optimizer = OMEGA(..., distance_function=my_distance)
   ```

4. **Merging rate is multiplicative with epochs:**
   ```python
   # merging_rate=0.02 means:
   # Epoch 0:  0% crossover (all mutations)
   # Epoch 5:  10% crossover
   # Epoch 10: 20% crossover
   ```

**Limitations:**

- **Computational cost**: DNS competition is O(N^2) in population size
- **Embedding token limits**: Large JSON variables may exceed embedding model limits
- **LLM quality dependent**: Poor mutation LLM = poor evolution
- **No variable interdependence**: Optimizes variables independently

**Performance Tips:**

- Keep `population_size` <= 20 for reasonable DNS overhead
- Use cheaper/faster models for `language_model` (optimization doesn't need GPT-4)
- Start with `algorithm="dns"`, switch to `"ga"` if diversity isn't helping
- Lower `selection_temperature` (0.1-0.2) for exploitation, higher (0.5+) for exploration

#### Research Background

OMEGA is based on:
- **Dominated Novelty Search**: [arxiv.org/html/2502.00593v1](https://arxiv.org/html/2502.00593v1)
- **GEPA** (prompt evolution): [arxiv.org/pdf/2507.19457](https://arxiv.org/pdf/2507.19457)
- **Chain-of-Thought**: [arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

---

## Training API

### program.compile()

Configure training before fit(). Accepts instances, string identifiers
(case-insensitive, lowercase recommended), or `{"class_name", "config"}` dicts:

```python
program.compile(
    reward=synalinks.rewards.ExactMatch(),
    optimizer=synalinks.optimizers.RandomFewShot(),
    metrics=[synalinks.metrics.F1Score()],
)

# Equivalent — string-form (resolved via rewards.get / optimizers.get / metrics.get)
program.compile(
    reward="exactmatch",
    optimizer="randomfewshot",
    metrics=["mean"],
)
```

`reward_weights`, `run_eagerly`, `steps_per_execution` are also accepted.

When passing string identifiers for optimizers/rewards that need an LM or
embedding model (e.g. `"omega"`, `"lmasjudge"`, `"cosinesimilarity"`), set
the defaults first:

```python
synalinks.set_default_language_model("openai/gpt-4o-mini")
synalinks.set_default_embedding_model("openai/text-embedding-3-small")
```

These accept a model string, a config dict, or an instance, and persist
string identifiers to `~/.synalinks/synalinks.json`.

### program.fit()

Train the program.

```python
history = await program.fit(
    x=x_train,                  # NumPy array of input DataModels
    y=y_train,                  # NumPy array of target DataModels
    batch_size=1,               # Default 1
    minibatch_size=4,           # Random val sub-sample per train batch (default 4)
    epochs=1,                   # Default 1
    verbose="auto",             # "auto", 0, 1, or 2
    callbacks=[...],
    validation_split=0.1,       # Default 0.1 — last fraction held out
    validation_data=(x_val, y_val),  # Overrides validation_split
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=32,   # Default 32
    validation_freq=1,          # Run end-of-epoch validation every N epochs
)
```

There is no public `shuffle` argument — `fit()` iterates batches with
`shuffle=False` internally.

**Return:** History object with training metrics per epoch.

### program.evaluate()

Evaluate on test data.

```python
metrics = await program.evaluate(
    x=x_test,
    y=y_test,
    batch_size=32,
)
```

### program.predict()

Batch prediction.

```python
predictions = await program.predict(
    x=x_test,
    batch_size=32,
)
```

---

## Callbacks

### ProgramCheckpoint

Save best model during training.

```python
checkpoint = synalinks.callbacks.ProgramCheckpoint(
    filepath="best_model.json",
    monitor="val_reward",   # Metric to monitor
    mode="max",             # "max" or "min"
    save_best_only=True,
)

history = await program.fit(..., callbacks=[checkpoint])
```

### Custom Callbacks

Callback hooks are synchronous — `CallbackList` invokes them synchronously
inside the async training loop. Available hooks: `on_{train,test,predict}_{begin,end}`,
`on_epoch_{begin,end}`, `on_{train,test,predict}_batch_{begin,end}`. The
bound program is available as `self.program`.

```python
class MyCallback(synalinks.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Training started")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: reward={logs.get('reward')}")

    def on_train_end(self, logs=None):
        print("Training finished")
```

---

## Data Preparation

### Creating Training Data

```python
import numpy as np

# Input DataModels
x_train = np.array([
    Query(query="What is 2+2?"),
    Query(query="Capital of France?"),
    Query(query="Who wrote Hamlet?"),
], dtype="object")

# Target DataModels
y_train = np.array([
    Answer(answer="4"),
    Answer(answer="Paris"),
    Answer(answer="William Shakespeare"),
], dtype="object")
```

### Using Built-in Datasets

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
```

---

## Saving and Loading

### Full Program

```python
# Save everything (architecture + variables + optimizer state)
program.save("model.json")

# Load
program = synalinks.Program.load("model.json")
```

### Variables Only

```python
# Save variables
program.save_variables("variables.json")

# Load into same architecture
program.load_variables("variables.json")
```

---

## Visualization

### Training History

```python
synalinks.utils.plot_history(
    history,
    to_folder="output",
    to_file="training_history.png",
)
```

### Metrics Comparison

```python
# Run multiple evaluations
results = []
for i in range(5):
    metrics = await program.evaluate(x_test, y_test)
    results.append(metrics)

synalinks.utils.plot_metrics_with_mean_and_std(
    results,
    to_folder="output",
    title="Evaluation Results",
)
```

### Before/After Comparison

```python
comparison = {
    "before_training": baseline_results,
    "after_training": trained_results,
}

synalinks.utils.plot_metrics_comparison_with_mean_and_std(
    comparison,
    to_folder="output",
    title="Training Improvement",
)
```

---

## Training Tips

1. **Start with RandomFewShot** - Simple baseline before advanced optimizers
2. **Use small batches for expensive models** - Balance speed vs accuracy
3. **Monitor validation reward** - Detect overfitting
4. **Use checkpoints** - Save best model automatically
5. **Mask non-essential fields** - Focus reward on important outputs
6. **Run multiple evaluations** - Get statistical significance
7. **Small models can compete** - Trained small models often beat larger ones
