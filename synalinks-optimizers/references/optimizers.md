# Synalinks Optimizers Reference

## Overview

Synalinks optimizers update **trainable variables** (JSON objects) to maximize reward. Variables include:

- `instructions` — system / task prompt
- `examples` — list of (input, output) demonstrations
- Custom variables added via `Module.add_variable(...)`

There are no gradients. Variables are evolved by sampling, mutation, and crossover.

## RandomFewShot

```python
synalinks.optimizers.RandomFewShot(
    nb_min_examples=1,
    nb_max_examples=3,
    sampling="softmax",
    sampling_temperature=0.3,
    population_size=10,
)
```

| Parameter | Effect |
|-----------|--------|
| `nb_min_examples` | Minimum examples sampled per prompt (set to 0 to allow zero-shot) |
| `nb_max_examples` | Maximum examples sampled per prompt |
| `sampling` | `"random"` (uniform), `"best"` (top-k by reward), or `"softmax"` (default) |
| `sampling_temperature` | Softmax temperature; lower = exploit, higher = explore |
| `population_size` | Max best-prediction candidates retained internally |

Predictions are scored by reward and reused as few-shot examples in the next batch.

## OMEGA

### Algorithm Outline

```
for each batch:
    pick variable_name to update (inverse-fitness softmax sampling)
    strategy = "crossover" if random() <= merging_rate * optimizer.epochs else "mutation"
    select parent candidate from best_candidates via `selection` strategy
    if strategy == "mutation":
        new_value = LLM.mutate(parent, instructions, mutation_temperature)
    else:  # crossover (falls back to mutation if <2 best_candidates)
        other = select_candidate_to_merge(...)  # uniform random from best_candidates
        new_value = LLM.crossover(parent, other, crossover_temperature)
    add candidate(new_value, mean_reward_on_val_batch) to population

at end of epoch:
    all_candidates = candidates + best_candidates
    if algorithm == "dns":
        for each c in all_candidates:
            for each o in all_candidates (o != c):
                d = distance_function(c, o, embedding_model=...)
                if d < 1.0 / k_nearest_fitter and reward(o) > reward(c):
                    mark c dominated; break
        keep only non-dominated (or [first] if all dominated)
    sort surviving by reward desc; truncate to population_size
```

### Parameters In Detail

```python
synalinks.optimizers.OMEGA(
    instructions=None,         # mutation guidance
    language_model=None,       # mutation/crossover LLM (resolves to default if None)
    embedding_model=None,      # DNS distance metric (resolves to default if None)

    k_nearest_fitter=5,        # k for DNS; radius is 1/k
    distance_function=None,    # async (c1, c2, embedding_model=None, **kwargs) -> float

    mutation_temperature=0.3,
    crossover_temperature=0.3,
    reasoning_effort=None,     # 'minimal'|'low'|'medium'|'high'|'disable'|'none'|None

    merging_rate=0.02,         # crossover prob = merging_rate * optimizer.epochs
    algorithm="dns",           # "dns" or "ga"
    selection="softmax",       # "softmax" | "best" | "random"
    selection_temperature=0.3,
    population_size=10,        # max candidates retained per variable

    name=None,
    description=None,
)
```

`OMEGA` does NOT accept `few_shot_learning`, `nb_min_examples`, `nb_max_examples`, or `sampling_temperature`. Those are `RandomFewShot` parameters.

### Selection Strategies

| Strategy | Behavior | When to use |
|----------|----------|-------------|
| `softmax` | P(c) ∝ exp(reward(c) / temperature) | Default, balances exploit/explore |
| `best` | Argmax — always pick top-reward | Pure exploitation, ablation |
| `random` | Uniform | Pure exploration baseline |

### Variable Selection

Within a batch, OMEGA chooses **which** variable to optimize using inverse-fitness softmax sampling (`Optimizer.select_variable_name_to_update`):

```
avg_reward_i = cumulative_reward_i / nb_visit_i   (or 100000 if never visited)
logits = -avg_reward_i / sampling_temperature
P(var_i) = softmax(logits)
```

Unvisited variables are picked first; otherwise worst-performing variables get attention more often. `sampling_temperature` (default `0.3`, set on the base `Optimizer`) controls how sharply the choice concentrates on the worst variable.

### Custom Distance Function

Default uses cosine distance on embeddings of the JSON variable. Override for domain-specific notions of similarity:

```python
async def edit_distance(c1, c2, embedding_model=None, **kwargs):
    s1 = json.dumps(c1.value)
    s2 = json.dumps(c2.value)
    # ... compute Levenshtein, return float in [0, 1]

optimizer = synalinks.optimizers.OMEGA(
    language_model=lm,
    embedding_model=em,
    distance_function=edit_distance,
)
```

The signature MUST be `async` and accept `embedding_model=None, **kwargs`.

## Tuning Recipes

### "Plateau too early"

```python
synalinks.optimizers.OMEGA(
    ...,
    mutation_temperature=0.6,    # more creative
    population_size=20,           # more diversity
    selection_temperature=0.5,    # more exploration
    algorithm="dns",              # ensure DNS is on
)
```

### "Wandering, not converging"

```python
synalinks.optimizers.OMEGA(
    ...,
    mutation_temperature=0.2,    # conservative
    selection_temperature=0.1,   # exploit
    algorithm="dns",
)
```

### "Population collapses"

```python
synalinks.optimizers.OMEGA(
    ...,
    k_nearest_fitter=10,         # smaller DNS radius (1/k) → more diversity preserved
    population_size=15,
    algorithm="dns",
)
```

### "Need cheap baseline"

```python
synalinks.optimizers.RandomFewShot(
    nb_min_examples=2,
    nb_max_examples=5,
    sampling="softmax",
    sampling_temperature=0.3,
)
```

## Pitfalls

1. **Zero softmax temperature** — `mutation_temperature` / `crossover_temperature` / `selection_temperature` / `sampling_temperature` should always be ≥ 0.1; the LM also rejects pure zero on most providers.
2. **`embedding_model=None` with `algorithm="dns"`** — the default cosine `similarity_distance` will crash at the first epoch end. Either pass an embedding model, register one with `synalinks.set_default_embedding_model(...)`, supply a custom `distance_function` that doesn't need one, or use `algorithm="ga"`.
3. **Huge variables** — embedding model token limits. Mask fields out of trainable variables, or shorten via custom serialization.
4. **Fast LM for both program and optimizer** — cheap programs may not benefit from expensive optimization. Conversely, an underpowered optimizer LM will produce poor mutations.
5. **`merging_rate` math** — uses `merging_rate * optimizer.epochs`; at default `0.02` it reaches 1.0 (always crossover) by epoch 50. Lower `merging_rate` for long training runs.

## See Also

- **synalinks-rewards** — Reward shape determines what OMEGA can optimize for
- **synalinks-training** — `compile()` / `fit()` integration
- **synalinks-providers** — OpenRouter wrappers for OMEGA's LM and embedding model
