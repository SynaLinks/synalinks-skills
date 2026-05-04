---
name: synalinks-optimizers
description: Use when picking, configuring, or tuning a Synalinks optimizer — RandomFewShot (nb_min_examples, nb_max_examples, sampling, sampling_temperature), OMEGA (Dominated Novelty Search, mutation/crossover, k_nearest_fitter, population_size, mutation_temperature, crossover_temperature, selection_temperature, merging_rate, algorithm "dns" vs "ga", selection "softmax"/"best"/"random", reasoning_effort), or evolutionary / quality-diversity prompt optimization in general.
---

# Synalinks Optimizers

Optimizers update **trainable variables** (instructions, examples, plans) to maximize the reward configured in `compile()`. There are no gradients — variables are JSON objects, evolved via random sampling or LLM-driven mutation.

## When to Use Which

| Optimizer | Cost | Optimizes | Use when |
|-----------|------|-----------|----------|
| **RandomFewShot** | Free | Examples only | Baseline, fast experiments, small data, simple tasks |
| **OMEGA** | 1+ extra LM call per batch | Full trainable variables (prompts, code, plans, examples) | RandomFewShot plateaus, prompt-shape matters, you have compute budget |

**Always start with RandomFewShot** to establish a baseline before reaching for OMEGA.

## RandomFewShot

Sampling baseline. Selects high-reward predictions seen during training and uses them as few-shot examples.

```python
synalinks.optimizers.RandomFewShot(
    nb_min_examples=1,            # Min examples per prompt
    nb_max_examples=3,            # Max examples per prompt
    sampling="softmax",           # "random" | "best" | "softmax" (default)
    sampling_temperature=0.3,     # Softmax temperature (only when sampling="softmax")
    population_size=10,           # Max candidates retained
)
```

- Fast — no extra LM calls
- Only manipulates `examples`, never `instructions`
- Good for sanity-checking your reward / data pipeline
- To disable few-shot entirely, set `nb_min_examples=0` and `nb_max_examples=0`

## OMEGA

**O**pti**M**iz**E**r as **G**enetic **A**lgorithm — LLM-driven evolutionary optimizer with Dominated Novelty Search (DNS) for quality-diversity.

### Quick Start

```python
lm = synalinks.LanguageModel(model="openai/gpt-4o-mini")
em = synalinks.EmbeddingModel(model="openai/text-embedding-3-small")

program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.OMEGA(
        language_model=lm,
        embedding_model=em,
    ),
)
```

`language_model` and `embedding_model` default to `None`. When unset, OMEGA falls back at call time to whatever you registered via `synalinks.set_default_language_model(...)` / `synalinks.set_default_embedding_model(...)`. The DNS branch needs an embedding model — pass one explicitly or set a default before training.

You can also use the Keras-style string identifier (resolves via `synalinks.optimizers.get`):

```python
program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer="omega",   # or "OMEGA", "randomfewshot", etc. — case-insensitive
)
```

### All Parameters

```python
synalinks.optimizers.OMEGA(
    # Models (resolved from defaults if None at call time)
    language_model=None,            # For mutation/crossover reasoning
    embedding_model=None,           # For diversity metrics in DNS

    # DNS parameters
    k_nearest_fitter=5,             # K nearest fitter neighbors
    distance_function=None,         # Custom async distance (default: cosine)
    population_size=10,             # Max candidates to maintain

    # Temperature controls (all default 0.3, MUST be non-zero for softmax)
    mutation_temperature=0.3,       # Creativity during mutation
    crossover_temperature=0.3,      # Creativity during crossover
    selection_temperature=0.3,      # Candidate selection sharpness

    # LM reasoning
    reasoning_effort=None,          # 'minimal'|'low'|'medium'|'high'|'disable'|'none'|None

    # Genetic algorithm
    merging_rate=0.02,              # Crossover probability per epoch (multiplicative)
    algorithm="dns",                # "dns" or "ga" (pure GA, ablation)
    selection="softmax",            # "random", "best", or "softmax"

    # Customization
    instructions=None,              # Task-specific mutation guidance
    name=None,
    description=None,
)
```

OMEGA does NOT take `few_shot_learning`, `nb_min_examples`, `nb_max_examples`, or `sampling_temperature` — those belong to `RandomFewShot`. OMEGA optimizes the full trainable variable, not just `examples`.

### How OMEGA Works

**1. Mutation** — Selects a trainable variable, uses ChainOfThought to reason about improvements, generates a new version constrained to the original schema.

**2. Crossover** — Combines two high-performing candidates. Probability scales with epoch: `merging_rate * epoch_number`. Default `0.02` → ~20% crossover at epoch 10.

**3. Selection** — Picks parents from the population:
- `softmax` (default) — temperature-scaled probability ∝ reward
- `best` — greedy
- `random` — uniform

**4. DNS competition** — At each epoch end (when `algorithm="dns"`), candidates compete:
```
For each candidate c:
  for each other candidate o:
    d = distance_function(c, o, embedding_model=...)   # cosine in [0, 1]
    if d < 1.0 / k_nearest_fitter and reward(o) > reward(c):
      mark c as dominated; break
keep all non-dominated candidates (or [candidates[0]] if all dominated)
then sort surviving + best_candidates by reward, truncate to population_size
```
Larger `k_nearest_fitter` → smaller neighborhood radius → easier to be non-dominated → more diversity preserved.

**5. Variable selection** — Inherited from `Optimizer.select_variable_name_to_update`. Uses inverse-fitness softmax: `P(var_i) ∝ exp(-avg_reward_i / sampling_temperature)`. Unvisited variables score `+infty` and are chosen first; afterwards worst-performing variables get attention more often.

### Tuning OMEGA

| Symptom | Adjustment |
|---------|------------|
| Stuck on local optimum | Raise `mutation_temperature` (0.5–0.8); increase `population_size` |
| Population collapses to one solution | Use `algorithm="dns"`, raise `selection_temperature` (more exploration), increase `k_nearest_fitter` (shrinks DNS radius → preserves diversity) |
| Too slow | Smaller `population_size` (≤10); cheaper `language_model` |
| Embedding token limits | Smaller `population_size`, mask large fields out of trainable variables |
| Need just example selection | Use RandomFewShot — OMEGA is overkill |

### Advanced Configuration

```python
optimizer = synalinks.optimizers.OMEGA(
    language_model=lm,
    embedding_model=em,

    # Larger population for diversity
    population_size=20,

    # Larger k = smaller DNS radius (1/k) = more candidates survive
    k_nearest_fitter=10,

    # More creative mutations
    mutation_temperature=0.5,

    # Greedier selection
    selection_temperature=0.1,

    # Lower-effort reasoning during mutation/crossover
    reasoning_effort="low",

    # Task-specific guidance during mutation
    instructions="Focus on edge cases and robustness.",
)
```

For few-shot example selection on top of OMEGA, run RandomFewShot first to establish a baseline, or compose your own training loop.

### Pure Genetic Algorithm (Ablation)

Disable DNS to study its contribution:

```python
optimizer = synalinks.optimizers.OMEGA(
    language_model=lm,
    embedding_model=em,
    algorithm="ga",         # Disables DNS competition
    selection="best",       # Greedy
)
```

## Common OMEGA Pitfalls

1. **Missing `embedding_model`** — does not crash at `build()`, but the default cosine `distance_function` will fail when `algorithm="dns"` runs `competition()` at epoch end. Pass one explicitly, register a default via `synalinks.set_default_embedding_model(...)`, or set `algorithm="ga"`.
2. **Zero temperature** — breaks softmax. Use `0.1` minimum, never `0.0`.
3. **Custom `distance_function` not async** — must be `async def my_distance(c1, c2, embedding_model=None, **kwargs)`.
4. **`merging_rate` is multiplicative** — implementation uses `merging_rate * optimizer.epochs`. At default `0.02`, crossover is ~20% at epoch 10, ~40% at epoch 20.
5. **`population_size > 20`** — DNS is O(N²) and embeds every candidate; becomes very slow.
6. **Using a too-strong LM for `language_model`** — optimization doesn't need GPT-4. Cheaper models work fine and save cost.

## OpenRouter for OMEGA

`OpenRouterEmbeddingModel` (see synalinks-providers) is OMEGA-compatible — it includes string conversion for `tree.flatten()` output, which can contain non-string leaves.

```python
from your_openrouter_module import create_openrouter_language_model, OpenRouterEmbeddingModel

lm_optimizer = create_openrouter_language_model("anthropic/claude-3.5-sonnet")
em = OpenRouterEmbeddingModel(
    "qwen/qwen3-embedding-8b",
    provider={"only": ["nebius"], "allow_fallbacks": False},
)

program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.OMEGA(
        language_model=lm_optimizer,
        embedding_model=em,
    ),
)
```

## Research Background

- **Dominated Novelty Search**: https://arxiv.org/html/2502.00593v1
- **GEPA (prompt evolution)**: https://arxiv.org/pdf/2507.19457
- **Chain-of-Thought**: https://arxiv.org/abs/2201.11903

## References

- **references/optimizers.md** — Full algorithm details, every parameter, ablation patterns

## See Also

- **synalinks-training** — `compile()` / `fit()` API
- **synalinks-rewards** — The reward signal OMEGA maximizes
- **synalinks-providers** — OpenRouter LM and embedding wrappers for OMEGA
