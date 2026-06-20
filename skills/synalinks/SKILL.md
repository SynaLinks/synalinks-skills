---
name: synalinks
description: Use for anything involving the Synalinks neuro-symbolic LM framework (Keras-inspired) — DataModel/Field/Input, JSON operators (+ & | ^ ~), synalinks.ops, LanguageModel/EmbeddingModel and provider prefixes (openai/anthropic/ollama/groq/openrouter/bedrock/...); the Program class and its four building APIs (Functional/Sequential/Subclassing/Mixed), save/load, summary; generation modules (Generator, ChainOfThought, SelfCritique, Identity, PythonSynthesis) and custom Module subclassing; control flow (Decision, Branch, And/Or/Xor, parallel branches, self-consistency, XOR guards); agents (FunctionCallingAgent, RecursiveLanguageModelAgent/RLM, DeepAgent, Tool, MCP, subagents); KnowledgeBase/RAG (DuckDB, EmbedKnowledge/UpdateKnowledge/RetrieveKnowledge, hybrid search); training (compile/fit/evaluate/predict, callbacks, ProgramCheckpoint); rewards & metrics (ExactMatch, CosineSimilarity, LMAsJudge, ProgramAsJudge, F1Score, custom rewards, masking); optimizers (RandomFewShot, OMEGA/DNS); datasets (gsm8k, hotpotqa, arcagi) and visualization. Synalinks is Keras-shaped, so without guidance LMs mix Keras/LangChain/DSPy syntax — this skill constrains usage to idiomatic Synalinks.
---

# Synalinks

Synalinks is an open-source, Keras-inspired framework for building neuro-symbolic
LM applications (RAGs, agents, self-evolving reasoning) with in-context
reinforcement learning. This skill is the single entry point: each section below
is a scannable overview that points to a **runnable script + its captured run log**
and a **deep-dive reference doc** under `references/`.

## Core abstractions

- **DataModel** — Pydantic-style schema for structured I/O (replaces tensors).
- **Module** — async computational unit processing JSON data (replaces layers).
- **Program** — DAG of modules with conditional logic (replaces models).
- **Reward / Optimizer** — guide training by *maximizing reward* (no gradients;
  prompts/examples/plans are the trainable variables).

## Universal gotchas (apply everywhere)

1. **All module constructors are keyword-only** (`*` after `self`). Call
   `Generator(data_model=Answer, language_model=lm)`, never positionally. The
   one exception is `Tool(func, *, ...)` — `func` stays positional.
2. **`instructions` MUST be a string**, never a list — passing a list raises
   `ValidationError`.
3. **All `call()` methods are async** — `await` every module/program call.
4. **`language_model`/`embedding_model` may be omitted** if a process-wide
   default is set (`set_default_language_model(...)`); resolved at call time.
5. **Always check for `None` results** — a failed LM call or a non-activated
   branch yields `None`.

## Quick Start

```python
import synalinks, asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")

async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="simple_qa")
    result = await program(Query(query="What is the capital of France?"))
    print(result.prettify_json())

asyncio.run(main())
```

Runnable: [`scripts/simple_qa.py`](scripts/simple_qa.py) → log [`references/simple_qa.log`](references/simple_qa.log).

---

## Core: DataModel, operators, models

`DataModel` defines structured I/O; `Field(description=...)` shapes what the LM
emits. Complex types, nested models, `Optional`/`Enum`/`List` all work.
`ChatMessages` is the special conversational data model. `LanguageModel` and
`EmbeddingModel` are `Module` subclasses accepting `**default_kwargs`
(`temperature`, `top_p`, `max_tokens`, `reasoning_effort`, `dimensions`, ...)
forwarded to every call, plus `fallback=` (string/dict/instance, auto-coerced).

**JSON operators** (circuit-like logic on DataModels):

| Op | Symbol | Behavior |
|----|--------|----------|
| Concat | `+` | Merge fields; **raises** if either is None |
| And | `&` | Merge; None if either is None |
| Or | `\|` | Non-None value; concatenates if both present |
| Xor | `^` | One if exactly one present, else None |
| Not | `~` | Always None (drops a branch) |
| Contains | `in` | Field-subset check |

Module forms: `synalinks.And()/Or()/Xor()/Not()` (take a list). `synalinks.ops`
has `concat`, `in_mask`/`out_mask`, `factorize`, `logical_*`. Config helpers:
`enable_logging()`, `enable_observability()`, `set_seed()`, `clear_session()`.

- Deep dives: [`references/api-reference.md`](references/api-reference.md),
  [`references/data-models.md`](references/data-models.md)
- [`scripts/subclassing_modules.py`](scripts/subclassing_modules.py) →
  [`references/subclassing_modules.log`](references/subclassing_modules.log)

## Programs: the four building APIs

`Program` inherits from both `Trainer` and `Module` (a Program is itself a
module — nest freely).

| API | When |
|-----|------|
| **Functional** | Default. Multi-IO graphs, branching, parallelism. |
| **Sequential** | Strictly linear; **requires `description`** (else `ValueError`). |
| **Subclassing** | Custom control flow in `call()`; **must** implement `get_config`/`from_config` to be serializable. |
| **Mixed (Subclass + Functional)** | Class API over an introspectable graph; build it in `build()` and call `super().__init__()` twice. Materialize on a symbolic `Input`, not concrete data. |

Save/load: `program.save("p.json")` / `Program.load(...)` (config + variables +
optimizer/reward state). `to_json()` is inspect-only (no variables).
Inspection: `summary(expand_nested=True)`, `get_module(...)`,
`trainable_variables`, `get_state_tree`/`set_state_tree`. Register custom
subclasses with `@synalinks.saving.register_synalinks_serializable()`.

- Deep dive: [`references/programs.md`](references/programs.md)
- [`scripts/four_apis.py`](scripts/four_apis.py) →
  [`references/four_apis.log`](references/four_apis.log)

## Modules: generation & custom subclassing

- **Generator** — core structured-output LM module. Trainable vars:
  `instructions`, `examples`. Knobs: `return_inputs`, `temperature`,
  `reasoning_effort`, `use_inputs_schema`, `streaming`.
- **ChainOfThought** — Generator that prepends a `thinking` field
  (`reasoning_effort="low"` by default).
- **SelfCritique** — produces a `critique` and optional `reward` in `[0,1]`.
- **Identity** — pass-through placeholder.
- **PythonSynthesis** — generates/executes Python in a sandbox (no LM call; the
  script is the trainable variable; needs advanced optimizers).
- **SequentialPlanSynthesis** — *internal only* (not in public API; import from
  `synalinks.src.modules.synthesis.sequential_plan_synthesis`).

Custom modules: subclass `Module`, implement async `call()`,
`compute_output_spec()`, and `get_config`/`from_config`; use `add_variable(...)`
for trainable state; return `None` to short-circuit logical flows.

- Deep dive: [`references/modules-catalog.md`](references/modules-catalog.md)
- [`scripts/chain_of_thought.py`](scripts/chain_of_thought.py) →
  [`references/chain_of_thought.log`](references/chain_of_thought.log)
- [`scripts/custom_module.py`](scripts/custom_module.py) →
  [`references/custom_module.log`](references/custom_module.log)

## Control flow: routing, parallelism, guards

- **Decision** — single-label routing; output `{thinking, choice}` constrained
  to `labels`. **MultiDecision** — multi-label (`choices: list[str]`).
- **Branch** — route to one of N modules; `branches`/`labels` same length;
  non-activated branches return `None`; each branch trains separately;
  `decision_type=MultiDecision` for multi-fire. Merge with `|` (not `+`).
  `return_decision`/`inject_decision` default True.
- **Parallel branches** — modules sharing an input run concurrently via asyncio.
- **Self-consistency** — N Generators at `temperature>0`, merge with `&`, then a
  final Generator over `inputs & merged`.
- **XOR guards** — `warning ^ inputs` to bypass; `warning | answer` to replace.
  A custom module returning `None` allows-through.

- Deep dive: [`references/control-flow.md`](references/control-flow.md)
- [`scripts/conditional_branches.py`](scripts/conditional_branches.py) →
  [`references/conditional_branches.log`](references/conditional_branches.log)
- [`scripts/guard_patterns.py`](scripts/guard_patterns.py) →
  [`references/guard_patterns.log`](references/guard_patterns.log)

## Agents: tools, RLM, DeepAgent, MCP

- **FunctionCallingAgent** — autonomous/interactive tool use; `max_iterations`,
  `return_inputs_with_trajectory` (emits the `ChatMessages` trajectory with
  `tool_calls` inside assistant messages).
- **Tools** — `async def`, fully type-annotated, docstringed, decorated with
  `@synalinks.utils.register_synalinks_serializable()`, returning a dict with a
  `log`/status field. All params **required** (no defaults). `Tool(func)`.
- **RLM** (`RecursiveLanguageModelAgent`) — only tool is `run_python_code` in a
  persistent REPL sandbox; `recursive=True` exposes `llm_query`/
  `llm_query_batched` over a `sub_language_model` for long-context work;
  terminates via in-sandbox `submit(...)`. `max_iterations`, `max_llm_calls`.
- **DeepAgent** — sandboxed coding agent over a `workdir` with
  read/write/edit/search/`run_bash`; host-safe (operates on a copy); inspect via
  `agent.sandbox.diff()` (sandbox lives on the *module*, not the Program).
- **Subagents** — `max_subagent_depth>0` enables `spawn/merge/discard_subagent`
  on isolated `Sandbox.fork`s. **MCP** — `MultiServerMCPClient({...})`,
  `await client.get_tools()`.

- Deep dive: [`references/agents-tools.md`](references/agents-tools.md)
- [`scripts/tool_agent.py`](scripts/tool_agent.py) →
  [`references/tool_agent.log`](references/tool_agent.log) (use `ollama/qwen3:8b`)
- [`scripts/rlm_agent.py`](scripts/rlm_agent.py) →
  [`references/rlm_agent.log`](references/rlm_agent.log)
- [`scripts/deep_agent.py`](scripts/deep_agent.py) →
  [`references/deep_agent.log`](references/deep_agent.log)

## Knowledge & RAG

`KnowledgeBase` is DuckDB-backed (`uri="duckdb://./db.db"` or `:memory:`), one
table per DataModel — **first field is the primary key**. Add
`graph_uri="ladybug://..."` for a property-graph store. Modules
(keyword-only): `EmbedKnowledge` (mask to one field), `UpdateKnowledge` (upsert),
`RetrieveKnowledge` (LM generates the query; `search_type` in
`similarity`/`fulltext`/`hybrid`), `StampKnowledge`. Direct methods:
`fulltext_search`, `similarity_search`, `hybrid_search`, `get`, `getall`,
`query`. Pipe `Generator → UpdateKnowledge` to extract-and-store.

- Deep dive: [`references/knowledge-base.md`](references/knowledge-base.md)
- [`scripts/rag_example.py`](scripts/rag_example.py) →
  [`references/rag_example.log`](references/rag_example.log)

## Training (in-context RL)

Workflow: build → `compile(reward=, optimizer=, metrics=)` → `fit(x, y, ...)` →
`evaluate(x, y)` → `predict(x)` → `save(...)`. Data is NumPy arrays of DataModel
instances (`dtype="object"`) or custom iterables. `fit` knobs:
`validation_split`/`validation_data`, `epochs`, `batch_size`, `minibatch_size`,
`callbacks`. Callbacks: `ProgramCheckpoint(monitor="val_reward", mode="max")`,
custom `Callback` subclasses (sync hooks). `compile` accepts string identifiers
(`"omega"`, `"exactmatch"`, `"f1score"`).

- Deep dive: [`references/training-guide.md`](references/training-guide.md)
- [`scripts/training_example.py`](scripts/training_example.py) →
  [`references/training_example.log`](references/training_example.log) (logging off — `fit()` progress bar only)

## Rewards & metrics

Rewards drive optimization; metrics are passive. Both return a float in `[0,1]`
and support `in_mask`/`out_mask`(`_pattern`) field masking. Built-in rewards:
`ExactMatch`, `CosineSimilarity`, `LMAsJudge`, `ProgramAsJudge` (judge program
must output a float field named `reward`). Custom: async `(y_true, y_pred) ->
float` decorated and wrapped in `RewardFunctionWrapper` — **always handle
`None`**. Metrics: `F1Score`, `FBetaScore`, `BinaryF1Score`, `ListF1Score`,
`Precision`/`Recall`, `MeanMetricWrapper(fn=...)`.

- Deep dive: [`references/rewards-metrics.md`](references/rewards-metrics.md)
- [`scripts/custom_reward.py`](scripts/custom_reward.py) →
  [`references/custom_reward.log`](references/custom_reward.log)

## Optimizers

Optimizers evolve trainable variables (no gradients). **Always start with
RandomFewShot** (free, examples-only) before reaching for **OMEGA** (LLM-driven
evolutionary optimizer with Dominated Novelty Search for quality-diversity;
optimizes the full trainable variable). OMEGA needs an `embedding_model` for the
DNS branch (pass one or set a default); temperatures must be non-zero;
`population_size` ≤ ~20 (DNS is O(N²)). `algorithm="ga"` ablates DNS.

- Deep dive: [`references/optimizers.md`](references/optimizers.md)
- [`scripts/omega_example.py`](scripts/omega_example.py) →
  [`references/omega_example.log`](references/omega_example.log) (`.fit` on ollama — progress bar only)

## Providers

- **Constrained `json_schema`**: `openai/`, `azure/`, `ollama/`, `mistral/`,
  `gemini/`, `xai/`, `deepseek/`, `together_ai/`, `hosted_vllm/`.
- **Tool-call** (no native schema, or heterogeneous backends): `groq/`,
  `cohere/`, `openrouter/`, `bedrock/`.
- **`response_format`** (litellm auto-routes): `anthropic/`.

Some prefixes are rewritten in `__init__`: `ollama/`→`ollama_chat/` (default
`api_base` `http://localhost:11434`), `vllm/`→`hosted_vllm/` (`HOSTED_VLLM_API_BASE`
or `http://localhost:8000`), `doubleword/`→`openai/` + the Doubleword `api_base`.

An OpenAI-compatible server **without** a dedicated prefix (LMStudio, a custom
gateway, OpenRouter *embeddings*) uses the `openai/` prefix + `api_base` + a dummy
`OPENAI_API_KEY`; local servers also need `litellm.register_model({...})` so cost
tracking returns 0. **Azure** keeps its own `azure/<deployment>` prefix (same
constrained `json_schema` path as `openai/`) and needs `AZURE_API_KEY` +
`AZURE_API_BASE` + `AZURE_API_VERSION`.

- Helper: [`scripts/lmstudio_setup.py`](scripts/lmstudio_setup.py) →
  [`references/lmstudio_setup.log`](references/lmstudio_setup.log) (demo runs against local ollama)

## Datasets & visualization

Built-in datasets expose `load_data()` / `get_input_data_model()` /
`get_output_data_model()`: `synalinks.datasets.gsm8k`, `.hotpotqa`
(`load_knowledge()` for RAG), `.arcagi` (per-task, `plot_task(...)`). Build your
own with NumPy arrays of DataModels (`dtype="object"`) or custom iterables
(optional `__len__` enables progress bars; use `validation_data`, not
`validation_split`). Visualization: `synalinks.utils.plot_program`,
`plot_history`, `plot_metrics_with_mean_and_std`,
`plot_metrics_comparison_with_mean_and_std`.

- [`scripts/custom_dataset.py`](scripts/custom_dataset.py) →
  [`references/custom_dataset.log`](references/custom_dataset.log)
- [`scripts/visualization.py`](scripts/visualization.py) →
  [`references/visualization.log`](references/visualization.log) (both `.fit` — progress bar only, no logging)

---

## Scripts → logs index

Every runnable script states its `Usage:` and (where a captured run exists) its
`Run log:` in its module docstring.

| Script | Run log | Topic |
|--------|---------|-------|
| `scripts/simple_qa.py` | `references/simple_qa.log` | Core |
| `scripts/subclassing_modules.py` | `references/subclassing_modules.log` | Core / Modules |
| `scripts/four_apis.py` | `references/four_apis.log` | Programs |
| `scripts/chain_of_thought.py` | `references/chain_of_thought.log` | Modules |
| `scripts/custom_module.py` | `references/custom_module.log` | Modules |
| `scripts/conditional_branches.py` | `references/conditional_branches.log` | Control flow |
| `scripts/guard_patterns.py` | `references/guard_patterns.log` | Control flow |
| `scripts/tool_agent.py` | `references/tool_agent.log` | Agents |
| `scripts/rlm_agent.py` | `references/rlm_agent.log` | Agents (RLM) |
| `scripts/deep_agent.py` | `references/deep_agent.log` | Agents (DeepAgent) |
| `scripts/rag_example.py` | `references/rag_example.log` | Knowledge / RAG |
| `scripts/custom_reward.py` | `references/custom_reward.log` | Rewards |
| `scripts/training_example.py` | `references/training_example.log` | Training (`.fit`, progress bar only) |
| `scripts/omega_example.py` | `references/omega_example.log` | Optimizers (`.fit`, progress bar only) |
| `scripts/custom_dataset.py` | `references/custom_dataset.log` | Datasets (`.fit`, progress bar only) |
| `scripts/visualization.py` | `references/visualization.log` | Datasets (`.fit`, progress bar only) |
| `scripts/lmstudio_setup.py` | `references/lmstudio_setup.log` | Providers (helper; demo on ollama) |

## Reference docs

`api-reference.md`, `data-models.md`, `programs.md`, `modules-catalog.md`,
`control-flow.md`, `agents-tools.md`, `knowledge-base.md`, `training-guide.md`,
`rewards-metrics.md`, `optimizers.md` — all under `references/`.
