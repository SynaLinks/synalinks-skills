---
name: synalinks-training
description: Use when training Synalinks programs — program.compile() / fit() / evaluate() / predict(), validation_split, validation_data, batch_size, epochs, callbacks (ProgramCheckpoint, custom Callback subclasses), History, in-context reinforcement learning workflow. For reward functions see synalinks-rewards; for optimizer internals see synalinks-optimizers.
---

# Synalinks Training

Train programs with in-context reinforcement learning. Unlike traditional ML, no weights are updated — instead, **trainable variables** (prompts, examples, plans) are evolved by the optimizer to maximize a reward.

## Workflow

```
1. Build a Program (functional / sequential / subclassing)
2. compile()  — attach reward, optimizer, metrics
3. fit()      — train on (x_train, y_train), monitoring validation
4. evaluate() — measure on test set
5. predict()  — batch inference
6. save()     — persist architecture + variables + optimizer state
```

## Quick Start

```python
import numpy as np
import synalinks

class Question(synalinks.DataModel):
    question: str = synalinks.Field(description="A question")

class Answer(synalinks.DataModel):
    thinking: str = synalinks.Field(description="Step-by-step reasoning")
    answer: str = synalinks.Field(description="Final answer")

async def train():
    lm = synalinks.LanguageModel(model="openai/gpt-4o-mini")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="qa")

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
        metrics=[synalinks.metrics.F1Score(in_mask=["answer"])],
    )

    history = await program.fit(
        x=x_train, y=y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
    )

    metrics = await program.evaluate(x=x_test, y=y_test, batch_size=32)
    program.save("trained.json")
```

### String identifiers (Keras-style)

`compile()` accepts string identifiers (case-insensitive), config dicts, or instances:

```python
# Set defaults so string-form optimizers can resolve LM/EM
synalinks.set_default_language_model("openai/gpt-4o-mini")
synalinks.set_default_embedding_model("openai/text-embedding-3-small")

program.compile(
    optimizer="omega",        # or "randomfewshot", "greedyoptimizer"
    reward="exactmatch",      # or "cosinesimilarity", "lmasjudge"
    metrics=["mean"],         # or [{"class_name": "Mean", "config": {}}]
)
```

## Training Data

Training data is **NumPy arrays of DataModel instances**:

```python
import numpy as np

x_train = np.array([
    Question(question="What is 5 + 3?"),
    Question(question="Capital of France?"),
], dtype="object")

y_train = np.array([
    Answer(thinking="5 + 3 = 8", answer="8"),
    Answer(thinking="Paris is the capital of France.", answer="Paris"),
], dtype="object")
```

Built-in datasets return data in this format — see **synalinks-datasets**.

## program.compile()

```python
program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.RandomFewShot(),
    metrics=[synalinks.metrics.F1Score(in_mask=["answer"])],
)
```

- **reward** — drives optimization (see synalinks-rewards)
- **optimizer** — evolves trainable variables (see synalinks-optimizers)
- **metrics** — monitored only; not used for optimization

## program.fit()

```python
history = await program.fit(
    x=x_train,                # NumPy array of input DataModels
    y=y_train,                # NumPy array of target DataModels
    validation_split=0.1,     # Defaults to 0.1 — last fraction held out
    # OR:
    # validation_data=(x_val, y_val),
    epochs=1,                 # Defaults to 1
    batch_size=1,             # Defaults to 1
    minibatch_size=4,         # Random val sub-sample per train batch (default 4)
    validation_batch_size=32, # Defaults to 32
    validation_freq=1,        # Run end-of-epoch validation every N epochs
    callbacks=[...],
)
```

Returns a `History` object with metrics per epoch (visualizable via `synalinks.utils.plot_history` — see synalinks-datasets).

Note: `fit()` always iterates batches with `shuffle=False` internally — there is no public `shuffle` argument.

## program.evaluate()

```python
metrics = await program.evaluate(
    x=x_test,
    y=y_test,
    batch_size=32,
)
```

## program.predict()

Batch inference, no labels required:

```python
predictions = await program.predict(x_test, batch_size=32)
```

## Callbacks

### ProgramCheckpoint

Save the best program by a monitored metric:

```python
checkpoint = synalinks.callbacks.ProgramCheckpoint(
    filepath="best.json",
    monitor="val_reward",   # which metric to track
    mode="max",             # "max" or "min"
    save_best_only=True,
)

history = await program.fit(..., callbacks=[checkpoint])
```

### Custom Callback

```python
class MyCallback(synalinks.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("Training started")

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: reward={logs.get('reward')}")

    def on_train_end(self, logs=None):
        print("Training finished")
```

Callback hooks are synchronous (they're invoked synchronously by `CallbackList`). Available hooks: `on_{train,test,predict}_{begin,end}`, `on_epoch_{begin,end}`, `on_{train,test,predict}_batch_{begin,end}`. The callback's bound program is available as `self.program`.

## Saving and Loading

### Full Program

Saves architecture + trainable variables + optimizer state:

```python
program.save("model.json")
program = synalinks.Program.load("model.json")
```

### Variables Only

Useful for sharing trained prompts without architecture:

```python
program.save_variables("variables.json")
program.load_variables("variables.json")
```

## Visualizing Training

```python
synalinks.utils.plot_history(history, to_folder="output")

# Multiple evaluations for stability analysis
results = []
for _ in range(5):
    results.append(await program.evaluate(x_test, y_test))
synalinks.utils.plot_metrics_with_mean_and_std(results, to_folder="output")

# Before/after comparison
synalinks.utils.plot_metrics_comparison_with_mean_and_std({
    "before_training": baseline,
    "after_training":  trained,
}, to_folder="output")
```

## Training Tips

1. **Start with RandomFewShot** as a baseline before reaching for OMEGA (see synalinks-optimizers)
2. **Mask non-essential fields** in your reward — focus optimization on what matters
3. **Use checkpoints** — save best validation reward
4. **Keep `batch_size` small for expensive models** — each example is an LM call
5. **Run multiple evaluations** — `predict()` is stochastic, average for stability
6. **Small models can compete** — a trained 8B can beat an untrained 70B on narrow tasks

## References

- **references/training-guide.md** — Full training reference (rewards, optimizers, callbacks all in one place)

## See Also

- **synalinks-rewards** — Built-in and custom reward functions, metrics
- **synalinks-optimizers** — RandomFewShot, OMEGA, DNS, when to use each
- **synalinks-datasets** — Built-in datasets and data loading patterns
- **synalinks-core** — Program saving, inspection, configuration
