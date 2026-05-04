---
name: synalinks-datasets
description: Use when loading or building Synalinks training data — built-in datasets (gsm8k, hotpotqa, arcagi load_data / get_input_data_model / get_output_data_model), custom iterables / generator data adapters (from v0.8.004+), NumPy DataModel arrays, or visualization utilities (plot_program, plot_history, plot_metrics_with_mean_and_std, plot_metrics_comparison_with_mean_and_std).
---

# Synalinks Datasets & Visualization

Synalinks training data is **NumPy arrays of DataModel instances** by default, but iterables (with optional `__len__`) are also accepted as of v0.8.004. This skill covers the built-in datasets, how to build your own data sources, and the visualization utilities for inspecting programs and training runs.

## Built-in Datasets

| Dataset | Module | Task |
|---------|--------|------|
| GSM8K | `synalinks.datasets.gsm8k` | Grade-school math word problems |
| HotpotQA | `synalinks.datasets.hotpotqa` | Multi-hop QA with supporting documents |
| ARC-AGI | `synalinks.datasets.arcagi` | Visual reasoning (color grids) |

Each module exposes a uniform API:

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

InputModel  = synalinks.datasets.gsm8k.get_input_data_model()
OutputModel = synalinks.datasets.gsm8k.get_output_data_model()
```

### GSM8K

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

# DataModels:
# - MathQuestion(question: str)
# - NumericalAnswerWithThinking(thinking: str, answer: float)
```

### HotpotQA

```python
(x_train, y_train), (x_test, y_test) = synalinks.datasets.hotpotqa.load_data()
documents = synalinks.datasets.hotpotqa.load_knowledge()

# DataModels:
# - Question(question: str)
# - Answer(answer: str)
# - Document(title: str, text: str)  # for the knowledge base
KnowledgeModel = synalinks.datasets.hotpotqa.get_knowledge_data_model()
```

### ARC-AGI

ARC-AGI loads a single task at a time by name. Use the helpers to discover task names.

```python
# Helpers
training_names   = synalinks.datasets.arcagi.get_arcagi1_training_task_names()
evaluation_names = synalinks.datasets.arcagi.get_arcagi1_evaluation_task_names()

(x_train, y_train), (x_test, y_test) = synalinks.datasets.arcagi.load_data(
    task_name=training_names[0],
    arc_version=1,
)

# Visualization (pass x and either y_true, y_pred, or both)
synalinks.datasets.arcagi.plot_task(x=x_test[0], y_true=y_test[0])
```

## Building Your Own Data

### NumPy Arrays of DataModels (Standard)

```python
import numpy as np

x_train = np.array([
    Question(question="What is 5 + 3?"),
    Question(question="Capital of France?"),
], dtype="object")

y_train = np.array([
    Answer(answer="8"),
    Answer(answer="Paris"),
], dtype="object")
```

`dtype="object"` is required so NumPy treats DataModel instances as opaque objects rather than trying to coerce them.

### Custom Iterables (v0.8.004+)

You can pass any iterable (with optional `__len__`) directly to `fit`/`evaluate`/`predict`. The iterable must yield `(inputs,)` or `(inputs, targets)` tuples — when using a custom iterable you do **not** pass `y` separately.

```python
class MyDataset:
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        for row in self.source:
            yield Question(question=row["q"]), Answer(answer=row["a"])

    def __len__(self):  # optional but enables progress bars
        return len(self.source)

dataset = MyDataset(open_jsonl("train.jsonl"))
history = await program.fit(x=dataset, ...)  # do NOT pass y=
```

Generator-style iterables work too, but without `__len__` you lose progress reporting.

> **Note:** `validation_split` is only supported for NumPy arrays. With iterables/generators, pass `validation_data=(x_val, y_val)` instead.

### Streaming from Disk

```python
def stream_examples(path):
    """Yield (x, y) tuples, lazily."""
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            yield Question(question=row["q"]), Answer(answer=row["a"])

# fit() will pull from this until exhausted
history = await program.fit(x=stream_examples("train.jsonl"), ...)
```

## Visualization

### plot_program

Render the program DAG to a PNG. Useful for understanding control flow, branches, guards.

```python
synalinks.utils.plot_program(
    program,
    to_folder="output",
    show_module_names=True,
    show_schemas=True,
    show_trainable=True,
)
```

### plot_history

Plot training metrics across epochs from `program.fit()`'s return value.

```python
synalinks.utils.plot_history(
    history,
    to_folder="output",
    to_file="training.png",
)
```

### plot_metrics_with_mean_and_std

Aggregate multiple stochastic evaluations:

```python
results = []
for _ in range(5):
    results.append(await program.evaluate(x_test, y_test))

synalinks.utils.plot_metrics_with_mean_and_std(
    results,
    to_folder="output",
    title="Evaluation stability",
)
```

### plot_metrics_comparison_with_mean_and_std

Compare two or more configurations side-by-side:

```python
synalinks.utils.plot_metrics_comparison_with_mean_and_std(
    {
        "baseline":    [eval_baseline_1, eval_baseline_2, ...],
        "trained":     [eval_trained_1, eval_trained_2, ...],
        "self_consistency": [eval_sc_1, eval_sc_2, ...],
    },
    to_folder="output",
    title="Training improvement",
)
```

## Tips

1. **Run multiple evaluations** for any metric you care about — LM outputs are stochastic, single-shot numbers lie.
2. **Save plots and history JSONs** alongside trained programs — provenance for reproducibility.
3. **Use `clear_session()` in notebooks** before plotting, otherwise re-runs accumulate stale module names.
4. **Iterables with `__len__`** are strongly preferred over bare generators — without `__len__`, training has no progress bar. (`validation_split` is unsupported for iterables either way; use `validation_data` instead.)

## References

(This skill keeps no separate reference file — the SKILL.md is the reference.)

## See Also

- **synalinks-training** — `fit()` / `evaluate()` / `predict()`
- **synalinks-rewards** — How metrics flow through evaluations
- **synalinks-knowledge** — `hotpotqa.load_knowledge()` for RAG
- **synalinks-core** — `plot_program` is part of the core utility surface
