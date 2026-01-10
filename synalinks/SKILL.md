---
name: synalinks
description: Build neuro-symbolic LLM applications with Synalinks framework. Use when working with DataModel, Program, Generator, Module, training LLM pipelines, in-context learning, structured output, JSON operators, Branch/Decision control flow, FunctionCallingAgent, RAG/KAG, or Keras-like LLM workflows.
---

# Synalinks Framework

Synalinks is an open-source Keras-inspired framework for building neuro-symbolic LLM applications with in-context reinforcement learning.

## Core Concepts

- **DataModel**: Pydantic-style schema defining structured I/O (replaces tensors)
- **Module**: Computational unit processing JSON data (replaces layers)
- **Program**: DAG of modules with conditional logic (replaces models)
- **Rewards**: Guide training (maximize reward, not minimize loss)
- **Optimizers**: Update prompts/examples via LLM reasoning (no gradients)

## Quick Start

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer")

async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_qa",
        description="A simple Q&A program",
    )

    result = await program(Query(query="What is the capital of France?"))
    print(result.prettify_json())

asyncio.run(main())
```

## Four Ways to Build Programs

### 1. Functional API (Recommended for most cases)

```python
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
program = synalinks.Program(inputs=inputs, outputs=outputs)
```

### 2. Sequential API (Simple linear chains)

```python
program = synalinks.Sequential([
    synalinks.Input(data_model=Query),
    synalinks.Generator(data_model=Answer, language_model=lm),
])
```

### 3. Subclassing (Advanced custom logic)

```python
class MyProgram(synalinks.Program):
    def __init__(self, language_model):
        super().__init__()
        self.gen = synalinks.Generator(data_model=Answer, language_model=language_model)

    async def call(self, inputs, training=False):
        return await self.gen(inputs)

    def get_config(self):
        return {"name": self.name, "language_model": synalinks.saving.serialize_synalinks_object(self.language_model)}

    @classmethod
    def from_config(cls, config):
        lm = synalinks.saving.deserialize_synalinks_object(config.pop("language_model"))
        return cls(language_model=lm)
```

### 4. Mixed (Functional + Subclassing)

```python
class MyProgram(synalinks.Program):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    async def build(self, inputs):
        outputs = await synalinks.Generator(data_model=Answer, language_model=self.language_model)(inputs)
        super().__init__(inputs=inputs, outputs=outputs)
```

## JSON Operators (Circuit-like Logic)

| Operator | Symbol | Behavior |
|----------|--------|----------|
| Concatenate | `+` | Merge fields; raises Exception if either is None |
| Logical And | `&` | Merge fields; returns None if either is None |
| Logical Or | `\|` | Merge fields; returns non-None value if one is None |
| Logical Xor | `^` | Returns None if both present; otherwise returns the non-None value |
| Contains | `in` | Check if one DataModel's fields are a subset of another |

```python
# Concatenation (strict)
combined = x1 + x2  # Exception if either is None

# Logical And (safe concatenation)
combined = inputs & branch_output  # None if branch not taken

# Logical Or (merge branches)
result = branch1_output | branch2_output  # Returns whichever is not None

# Logical Xor (computation bypass for guards)
guarded = warning ^ inputs  # Returns inputs only if warning is None

# Contains operator
print(Query in (Query + Answer))  # True
print(Query in Answer)            # False
```

### Module Alternatives to Operators

You can also use explicit module classes instead of operators:

```python
# Using And module (equivalent to &)
merged = await synalinks.And()([b0, b1, b2])

# Using Or module (equivalent to |)
result = await synalinks.Or()([b0, b1, b2])
```

## Control Flow

### Parallel Branches (auto-detected)

```python
x1 = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
x2 = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
# Both run in parallel via asyncio
```

### Decision Making

```python
decision = await synalinks.Decision(
    question="Evaluate query difficulty",
    labels=["easy", "difficult"],
    language_model=lm,
)(inputs)
```

### Conditional Branching

```python
(easy_answer, hard_answer) = await synalinks.Branch(
    question="Evaluate query difficulty",
    labels=["easy", "difficult"],
    branches=[
        synalinks.Generator(data_model=Answer, language_model=lm),
        synalinks.Generator(data_model=AnswerWithThinking, language_model=lm),
    ],
    language_model=lm,
    return_decision=False,   # If True, returns decision alongside branch outputs
    inject_decision=False,   # If True, injects decision into branch inputs
)(inputs)

# Merge with logical or - non-activated branches return None
final = easy_answer | hard_answer
```

**Key Branch behaviors:**
- Non-activated branches return `None` (not executed, not just empty)
- Each branch module gets optimized separately during training (specialized modules)
- Labels constrain LLM output - prevents hallucination by enforcing valid choices

### Self-Consistency Pattern (Parallel Reasoning)

Use parallel branches with temperature > 0 to generate multiple answers, then merge:

```python
async def build_self_consistency_program(lm):
    inputs = synalinks.Input(data_model=Query)

    # Generate multiple answers with randomness
    b0 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)
    b1 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)
    b2 = await synalinks.Generator(data_model=AnswerWithRationale, language_model=lm, temperature=1.0)(inputs)

    # Merge all answers
    merged = b0 & b1 & b2

    # Final answer that critically analyzes all attempts
    outputs = await synalinks.Generator(
        data_model=AnswerWithRationale,
        language_model=lm,
        instructions="Critically analyze the given answers to produce the final answer.",
    )(inputs & merged)

    return synalinks.Program(inputs=inputs, outputs=outputs)
```

### Input/Output Guards (XOR Pattern)

Use XOR (^) to bypass computation based on conditions:

**Input Guard** - Block processing if input is invalid:

```python
class InputGuard(synalinks.Module):
    """Block invalid inputs."""
    async def call(self, inputs, training=False):
        if self._is_blocked(inputs):
            return synalinks.ChatMessage(role="assistant", content="Cannot process this request")
        return None  # Allow through

async def build_guarded_program(lm):
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    # Check input
    warning = await InputGuard()(inputs)

    # XOR: if warning exists, inputs becomes None (bypassing generator)
    guarded_inputs = warning ^ inputs

    # Generator only runs if guarded_inputs is not None
    answer = await synalinks.Generator(language_model=lm)(guarded_inputs)

    # OR: return warning if it exists, otherwise return answer
    outputs = warning | answer

    return synalinks.Program(inputs=inputs, outputs=outputs)
```

**Output Guard** - Replace invalid outputs:

```python
async def build_output_guarded_program(lm):
    inputs = synalinks.Input(data_model=synalinks.ChatMessages)

    answer = await synalinks.Generator(language_model=lm)(inputs)

    # Check output
    warning = await OutputGuard()(answer)

    # XOR + OR: if warning exists, replace answer with warning
    outputs = (answer ^ warning) | warning

    return synalinks.Program(inputs=inputs, outputs=outputs)
```

## Training Programs

```python
# Compile with reward and optimizer
program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.RandomFewShot(),
    metrics=[synalinks.metrics.F1Score(in_mask=["answer"])],
)

# Train
history = await program.fit(
    x=x_train,
    y=y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[synalinks.callbacks.ProgramCheckpoint(filepath="best.json", monitor="val_reward", mode="max")],
)

# Evaluate
metrics = await program.evaluate(x=x_test, y=y_test, batch_size=32)

# Batch prediction
predictions = await program.predict(x_test, batch_size=32)
```

## Built-in Rewards

- `synalinks.rewards.ExactMatch(in_mask=["field"])` - Exact string match
- `synalinks.rewards.CosineSimilarity(embedding_model=em, in_mask=["field"])` - Semantic similarity
- `synalinks.rewards.LMAsJudge(language_model=lm)` - LLM-based evaluation

### Custom Rewards

```python
@synalinks.saving.register_synalinks_serializable()
async def my_reward(y_true, y_pred):
    # Return float between 0.0 and 1.0
    return 1.0 if y_true.get("answer") == y_pred.get("answer") else 0.0

program.compile(reward=synalinks.rewards.MeanRewardWrapper(fn=my_reward))
```

## Built-in Optimizers

### RandomFewShot (Baseline)
```python
synalinks.optimizers.RandomFewShot(
    few_shot_learning=False,    # Enable few-shot examples
    nb_min_examples=1,          # Min examples per prompt
    nb_max_examples=3,          # Max examples per prompt
)
```
Use as baseline. Fast, no extra LLM calls. Only manipulates examples, not prompts.

### OMEGA (Advanced Evolutionary Optimizer)

**O**pti**M**iz**E**r as **G**enetic **A**lgorithm - Uses LLM-based mutation/crossover with Dominated Novelty Search for quality-diversity optimization.

```python
synalinks.optimizers.OMEGA(
    # Required
    language_model=lm,              # For mutation/crossover reasoning
    embedding_model=em,             # For diversity metrics

    # DNS Parameters
    k_nearest_fitter=5,             # K nearest fitter neighbors
    population_size=10,             # Max candidates to maintain

    # Temperature Controls (all default 0.3)
    mutation_temperature=0.3,       # Creativity during mutation
    crossover_temperature=0.3,      # Creativity during crossover
    selection_temperature=0.3,      # Candidate selection sharpness

    # Genetic Algorithm
    merging_rate=0.02,              # Base crossover probability (increases with epochs)
    algorithm="dns",                # "dns" or "ga" (pure genetic algorithm)
    selection="softmax",            # "random", "best", or "softmax"

    # Few-Shot (inherited from RandomFewShot)
    few_shot_learning=False,
    nb_min_examples=1,
    nb_max_examples=3,

    # Custom guidance
    instructions=None,              # Task-specific mutation guidance
)
```

**When to use OMEGA:**
- RandomFewShot isn't achieving target performance
- Need to optimize prompts/instructions, not just examples
- Want diverse solution exploration (quality-diversity)
- Have compute budget for extra LLM calls

**Key differences from RandomFewShot:**
| Aspect | RandomFewShot | OMEGA |
|--------|---------------|-------|
| Optimizes | Examples only | Full trainable variables (prompts, code, plans) |
| Generation | Random sampling | LLM mutation + crossover with ChainOfThought |
| Quality-Diversity | No | Yes (Dominated Novelty Search) |
| Extra LLM calls | None | 1+ per batch |

**OMEGA Gotchas:**
1. Must provide BOTH `language_model` AND `embedding_model`
2. Temperature must be non-zero (breaks softmax otherwise)
3. `merging_rate` is multiplicative: at epoch 10 with default 0.02, crossover is 20%
4. Keep `population_size` <= 20 for reasonable DNS overhead

**Using OMEGA with OpenRouter:**
The `OpenRouterEmbeddingModel` wrapper (see "Using OpenRouter" section) is compatible with OMEGA. It includes automatic string conversion for `tree.flatten()` output, which may contain non-string leaf values from trainable variables.

```python
from your_openrouter_module import create_openrouter_language_model, OpenRouterEmbeddingModel

# Expensive model for optimization reasoning
lm_optimizer = create_openrouter_language_model("anthropic/claude-3.5-sonnet")

# Embedding model for DNS diversity metrics
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

See **references/training-guide.md** for complete OMEGA documentation including DNS algorithm details, all parameters, and advanced usage patterns.

## Agents with Tools

```python
@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate mathematical expression."""
    return {"result": eval(expression), "log": "Success"}

tools = [synalinks.Tool(calculate)]

outputs = await synalinks.FunctionCallingAgent(
    data_model=FinalAnswer,
    tools=tools,
    language_model=lm,
    max_iterations=5,
    autonomous=True,
)(inputs)
```

### MCP Tools

```python
mcp_client = synalinks.MultiServerMCPClient({
    "math": {"url": "http://localhost:8183/mcp/", "transport": "streamable_http"},
})
tools = await mcp_client.get_tools()
```

## RAG with Knowledge Base

```python
class Document(synalinks.DataModel):
    id: str = synalinks.Field(description="Document ID")
    title: str = synalinks.Field(description="Title")
    content: str = synalinks.Field(description="Content")

knowledge_base = synalinks.KnowledgeBase(
    uri="duckdb://./documents.db",
    data_models=[Document],
    embedding_model=embedding_model,
    metric="cosine",
)

context = await synalinks.RetrieveKnowledge(
    knowledge_base=knowledge_base,
    language_model=lm,
    search_type="hybrid",
    k=5,
    return_inputs=True,
)(inputs)

answer = await synalinks.Generator(
    data_model=Answer,
    language_model=lm,
    instructions="Answer based on retrieved context.",
)(context)
```

## Saving and Loading

```python
# Save entire program (architecture + variables + optimizer state)
program.save("my_program.json")
program = synalinks.Program.load("my_program.json")

# Save variables only
program.save_variables("my_program.variables.json")
program.load_variables("my_program.variables.json")
```

## Visualization

```python
synalinks.utils.plot_program(
    program,
    to_folder="output",
    show_module_names=True,
    show_schemas=True,
    show_trainable=True,
)

synalinks.utils.plot_history(history, to_folder="output")
```

## Configuration

```python
synalinks.enable_logging()  # Enable debug logging
synalinks.enable_observability()  # Enable tracing
synalinks.clear_session()  # Clear session for reproducible naming (important in notebooks)
```

## Program Inspection

```python
# Access modules
print(f"Number of modules: {len(program.modules)}")
module = program.get_module(index=0)        # By index (topological order)
module = program.get_module(name="generator")  # By name

# Access variables
print(f"Total variables: {len(program.variables)}")
print(f"Trainable variables: {len(program.trainable_variables)}")

# View default prompt (trainable)
print(program.trainable_variables[0]["instructions"])

# Program summary
program.summary()
```

## Built-in Datasets

```python
# GSM8K (math word problems)
(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

# Get DataModels for dataset
InputModel = synalinks.datasets.gsm8k.get_input_data_model()
OutputModel = synalinks.datasets.gsm8k.get_output_data_model()
```

## Important Notes and Gotchas

### instructions Parameter
The `instructions` parameter in Generator MUST be a **string**, not a list:

```python
# CORRECT
synalinks.Generator(
    data_model=Answer,
    language_model=lm,
    instructions="Be concise. Focus on key points. Use examples.",
)

# WRONG - will raise ValidationError
synalinks.Generator(
    data_model=Answer,
    language_model=lm,
    instructions=["Be concise", "Focus on key points"],  # ERROR!
)
```

### Model Compatibility Summary

| Provider | Structured Output | Notes |
|----------|------------------|-------|
| openai/* | Works | Recommended |
| anthropic/* | Works | Uses tool-calling internally |
| ollama/* | Works | Local models |
| mistral/* | Works | |
| gemini/* | Works | |
| groq/* | Works | Requires patch (see below) |
| groq/compound-* | Works | Requires patch, uses json_object mode |
| openrouter/* | Works | Requires patch (see below) |
| Local (LMStudio, vLLM) | Works | Requires model registration (see below) |

### Handling None Results
Always check for None results from program execution:

```python
result = await program(input_data)
if result is None:
    print("LLM call failed - check API key and model compatibility")
    return
```

### Mixed Subclassing Pattern
When using the mixed pattern (subclassing + functional), call `super().__init__()` TWICE:

```python
class MyProgram(synalinks.Program):
    def __init__(self, language_model):
        super().__init__()  # First call - basic init
        self.language_model = language_model

    async def build(self, inputs):
        outputs = await synalinks.Generator(...)(inputs)
        super().__init__(inputs=inputs, outputs=outputs)  # Second call - with graph
```

### Using Groq Models

Groq requires a patch to work with Synalinks structured output. The issue is that:
1. Synalinks uses tool-calling for Groq, but Groq's tool-calling is unreliable for structured output
2. `ChatMessage` serializes with `tool_calls` field, which Groq rejects

**Solution**: Patch Synalinks to use proper structured output:
- Regular Groq models (llama-3.1-8b-instant, etc.): Uses `json_schema` response format
- Compound models (compound-beta, compound-beta-mini): Uses `json_object` mode with schema in prompt
  (compound models don't support json_schema)

```python
import os
import copy
import json
import warnings
from functools import wraps

import litellm
import synalinks
from synalinks.src.backend import ChatRole
from synalinks.src.language_models.language_model import LanguageModel
from synalinks.src.utils.nlp_utils import shorten_text


def _clean_messages_for_groq(messages: list) -> list:
    """Remove tool_calls and tool_call_id from messages."""
    cleaned = []
    for msg in messages:
        clean_msg = {"role": msg.get("role"), "content": msg.get("content", "")}
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            clean_msg["tool_call_id"] = msg["tool_call_id"]
        cleaned.append(clean_msg)
    return cleaned


_original_call = None


async def _patched_call(self, messages, schema=None, streaming=False, **kwargs):
    """Patched __call__ that uses json_schema for Groq instead of tool-calling."""
    formatted_messages = messages.get_json().get("messages", [])
    input_kwargs = copy.deepcopy(kwargs)
    schema = copy.deepcopy(schema)

    # Clean messages for Groq
    if self.model.startswith("groq"):
        formatted_messages = _clean_messages_for_groq(formatted_messages)

    if schema:
        if self.model.startswith("groq"):
            # Use json_schema instead of tool-calling
            kwargs.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "structured_output", "schema": schema},
                }
            })
        elif self.model.startswith("anthropic"):
            kwargs.update({
                "tools": [{"name": "structured_output", "description": "Generate a valid JSON output",
                           "input_schema": {"type": "object", "properties": schema.get("properties"),
                                           "required": schema.get("required")}}],
                "tool_choice": {"type": "tool", "name": "structured_output"},
            })
        elif self.model.startswith("ollama") or self.model.startswith("mistral"):
            kwargs.update({"response_format": {"type": "json_schema", "json_schema": {"schema": schema}, "strict": True}})
        elif self.model.startswith("openai") or self.model.startswith("azure"):
            if "properties" in schema:
                for prop_key, prop_value in schema["properties"].items():
                    if "$ref" in prop_value and "description" in prop_value:
                        del prop_value["description"]
            kwargs.update({"response_format": {"type": "json_schema", "json_schema": {"name": "structured_output", "strict": True, "schema": schema}}})
        elif self.model.startswith("gemini") or self.model.startswith("xai") or self.model.startswith("hosted_vllm"):
            kwargs.update({"response_format": {"type": "json_schema", "json_schema": {"schema": schema}, "strict": True}})
        else:
            raise ValueError(f"LM provider '{self.model.split('/')[0]}' not supported")

    if self.api_base:
        kwargs.update({"api_base": self.api_base})
    if streaming and schema:
        streaming = False
    if streaming:
        kwargs.update({"stream": True})

    for i in range(self.retry):
        try:
            response_str = ""
            response = await litellm.acompletion(model=self.model, messages=formatted_messages,
                                                  timeout=self.timeout, caching=self.caching, **kwargs)
            if hasattr(response, "_hidden_params") and "response_cost" in response._hidden_params:
                self.last_call_cost = response._hidden_params["response_cost"]
                if self.last_call_cost is not None:
                    self.cumulated_cost += self.last_call_cost

            if self.model.startswith("anthropic") and schema:
                response_str = response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            else:
                response_str = response["choices"][0]["message"]["content"].strip()

            if schema:
                return json.loads(response_str)
            else:
                return {"role": ChatRole.ASSISTANT, "content": response_str, "tool_call_id": None, "tool_calls": []}
        except Exception as e:
            warnings.warn(f"Error calling {self}: {shorten_text(str(e))}")
        import asyncio
        await asyncio.sleep(1)

    return self.fallback(messages, schema=schema, streaming=streaming, **input_kwargs) if self.fallback else None


def patch_synalinks_for_groq():
    """Patch Synalinks to support Groq structured output. Call once at startup."""
    global _original_call
    if _original_call is None:
        _original_call = LanguageModel.__call__
        LanguageModel.__call__ = _patched_call


def create_groq_language_model(model_name: str, **kwargs) -> synalinks.LanguageModel:
    """Create a Groq LanguageModel with automatic patching."""
    patch_synalinks_for_groq()
    return synalinks.LanguageModel(model=f"groq/{model_name}", **kwargs)


# Usage - regular Groq model:
lm = create_groq_language_model("llama-3.1-8b-instant")

# Usage - compound model (for agentic workflows):
lm = create_groq_language_model("compound-beta")
```

### Using OpenRouter

OpenRouter provides access to many models through a unified API. Synalinks requires a patch to recognize OpenRouter as a provider.

**Key considerations:**
- OpenRouter is OpenAI-compatible, so the patch treats it like OpenAI for structured output
- Provider routing lets you specify which backend provider to use (e.g., DeepInfra, nebius)
- LiteLLM does NOT support OpenRouter embeddings - use direct API calls instead

**Solution for LLM completions:**

```python
import os
import copy
import json
import warnings
from typing import Any, Dict, Optional

import litellm
import synalinks
from synalinks.src.backend import ChatRole
from synalinks.src.language_models.language_model import LanguageModel
from synalinks.src.utils.nlp_utils import shorten_text


# Store provider routing config per model instance
_provider_configs: Dict[int, Dict[str, Any]] = {}
_original_call = None


async def _patched_call_openrouter(self, messages, schema=None, streaming=False, **kwargs):
    """Patched __call__ that handles OpenRouter as OpenAI-compatible."""
    formatted_messages = messages.get_json().get("messages", [])
    input_kwargs = copy.deepcopy(kwargs)
    schema = copy.deepcopy(schema)

    if schema:
        if self.model.startswith("openrouter"):
            # OpenRouter is OpenAI-compatible
            if "properties" in schema:
                for prop_key, prop_value in schema["properties"].items():
                    if "$ref" in prop_value and "description" in prop_value:
                        del prop_value["description"]
            kwargs.update({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "structured_output", "strict": True, "schema": schema},
                }
            })
        elif self.model.startswith("anthropic"):
            kwargs.update({
                "tools": [{"name": "structured_output", "description": "Generate a valid JSON output",
                           "input_schema": {"type": "object", "properties": schema.get("properties"),
                                           "required": schema.get("required")}}],
                "tool_choice": {"type": "tool", "name": "structured_output"},
            })
        elif self.model.startswith("ollama") or self.model.startswith("mistral"):
            kwargs.update({"response_format": {"type": "json_schema", "json_schema": {"schema": schema}, "strict": True}})
        elif self.model.startswith("openai") or self.model.startswith("azure"):
            if "properties" in schema:
                for prop_key, prop_value in schema["properties"].items():
                    if "$ref" in prop_value and "description" in prop_value:
                        del prop_value["description"]
            kwargs.update({"response_format": {"type": "json_schema", "json_schema": {"name": "structured_output", "strict": True, "schema": schema}}})
        elif self.model.startswith("gemini") or self.model.startswith("xai") or self.model.startswith("hosted_vllm"):
            kwargs.update({"response_format": {"type": "json_schema", "json_schema": {"schema": schema}, "strict": True}})
        else:
            raise ValueError(f"LM provider '{self.model.split('/')[0]}' not supported")

    if self.api_base:
        kwargs.update({"api_base": self.api_base})
    if streaming and schema:
        streaming = False
    if streaming:
        kwargs.update({"stream": True})

    # Add provider routing if configured
    instance_id = id(self)
    if instance_id in _provider_configs:
        if "extra_body" not in kwargs:
            kwargs["extra_body"] = {}
        kwargs["extra_body"]["provider"] = _provider_configs[instance_id]

    for i in range(self.retry):
        try:
            response_str = ""
            response = await litellm.acompletion(model=self.model, messages=formatted_messages,
                                                  timeout=self.timeout, caching=self.caching, **kwargs)
            if hasattr(response, "_hidden_params") and "response_cost" in response._hidden_params:
                self.last_call_cost = response._hidden_params["response_cost"]
                if self.last_call_cost is not None:
                    self.cumulated_cost += self.last_call_cost

            if self.model.startswith("anthropic") and schema:
                response_str = response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            else:
                response_str = response["choices"][0]["message"]["content"].strip()

            if schema:
                return json.loads(response_str)
            else:
                return {"role": ChatRole.ASSISTANT, "content": response_str, "tool_call_id": None, "tool_calls": []}
        except Exception as e:
            warnings.warn(f"Error calling {self}: {shorten_text(str(e))}")
        import asyncio
        await asyncio.sleep(1)

    return self.fallback(messages, schema=schema, streaming=streaming, **input_kwargs) if self.fallback else None


def patch_synalinks_for_openrouter():
    """Patch Synalinks to support OpenRouter. Call once at startup."""
    global _original_call
    if _original_call is None:
        _original_call = LanguageModel.__call__
        LanguageModel.__call__ = _patched_call_openrouter


def create_openrouter_language_model(
    model_name: str,
    provider: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> synalinks.LanguageModel:
    """
    Create a Synalinks LanguageModel configured for OpenRouter.

    Args:
        model_name: OpenRouter model name (e.g., "meta-llama/llama-3.1-8b-instruct")
        provider: Optional provider routing (e.g., {"only": ["DeepInfra"], "allow_fallbacks": False})
        **kwargs: Additional LanguageModel arguments
    """
    patch_synalinks_for_openrouter()
    full_model_name = f"openrouter/{model_name}"
    lm = synalinks.LanguageModel(model=full_model_name, **kwargs)
    if provider:
        _provider_configs[id(lm)] = provider
    return lm


# Usage - basic:
lm = create_openrouter_language_model("meta-llama/llama-3.1-8b-instruct")

# Usage - with provider routing (restrict to specific backend):
lm = create_openrouter_language_model(
    "meta-llama/llama-3.1-8b-instruct",
    provider={"only": ["DeepInfra"], "allow_fallbacks": False},
)
```

**Solution for OpenRouter Embeddings:**

LiteLLM does not support OpenRouter embeddings. Use direct API calls instead:

```python
import os
import warnings
from typing import Any, Dict, List, Optional

import httpx


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterEmbeddingModel:
    """
    OpenRouter embedding model using direct API calls.

    LiteLLM doesn't support OpenRouter embeddings, so this bypasses LiteLLM.
    Embedding models on OpenRouter are often only available from one provider,
    so specifying the provider is recommended.

    Args:
        model: OpenRouter model name (e.g., "qwen/qwen3-embedding-8b")
        api_key: API key. If not provided, uses OPENROUTER_API_KEY env var.
        provider: Provider routing (e.g., {"only": ["nebius"], "allow_fallbacks": False})
        retry: Number of retries on failure. Default 5.
        timeout: Request timeout in seconds. Default 30.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[Dict[str, Any]] = None,
        retry: int = 5,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.provider = provider
        self.retry = retry
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")

    async def __call__(self, texts: List[str]) -> Optional[Dict[str, List[List[float]]]]:
        """Generate embeddings for a list of texts.

        Note: Converts non-string values to strings for OMEGA compatibility,
        since tree.flatten() may return non-string leaf values from trainable variables.
        """
        # Ensure all inputs are strings (OMEGA's tree.flatten may return non-strings)
        texts = [str(t) for t in texts]

        for i in range(self.retry):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    payload: Dict[str, Any] = {"model": self.model, "input": texts}
                    if self.provider:
                        payload["provider"] = self.provider

                    response = await client.post(
                        f"{OPENROUTER_API_BASE}/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json=payload,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        vectors = [item.get("embedding", []) for item in data.get("data", [])]
                        return {"embeddings": vectors}
                    else:
                        warnings.warn(f"OpenRouter embedding error ({response.status_code}): {response.text}")
            except Exception as e:
                warnings.warn(f"Error calling OpenRouter embeddings: {e}")
            import asyncio
            await asyncio.sleep(1)
        return None


# Usage - embedding model (nebius hosts qwen3-embedding-8b):
em = OpenRouterEmbeddingModel(
    "qwen/qwen3-embedding-8b",
    provider={"only": ["nebius"], "allow_fallbacks": False},
)
result = await em(["Hello, world!"])
# result = {"embeddings": [[0.02, 0.006, ...]]}

# Batch embeddings:
result = await em(["Hello", "World", "Test"])
# result = {"embeddings": [[...], [...], [...]]}
```

**Finding available providers:**
Use OpenRouter's API to check which providers host a model:
```python
async with httpx.AsyncClient() as client:
    response = await client.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    models = response.json().get("data", [])
    for model in models:
        if "your-model-name" in model.get("id", ""):
            print(f"Providers: {model.get('providers', 'N/A')}")
```

### Using Local Models (LMStudio, vLLM, etc.)

When using local OpenAI-compatible servers like LMStudio, you MUST:
1. Set a dummy API key (any non-empty string)
2. Register the model with LiteLLM before creating the LanguageModel

**Required setup for LMStudio:**

```python
import os
import litellm
import synalinks

# Step 1: Set dummy API key (required by LiteLLM)
os.environ["OPENAI_API_KEY"] = "lm-studio"

# Step 2: Register model with LiteLLM (REQUIRED for local models)
# This tells LiteLLM the model exists and provides cost info (use 0 for local)
litellm.register_model(
    {
        "openai/ibm/granite-4-h-tiny": {  # Use openai/ prefix + your model name
            "max_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }
)

# Step 3: Create LanguageModel with api_base
lm = synalinks.LanguageModel(
    model="openai/ibm/granite-4-h-tiny",
    api_base="http://localhost:1234/v1",
)
```

**Why registration is required**: LiteLLM tracks API costs internally. For models not in its pricing database, it returns `None` for costs, which Synalinks cannot handle. Registering the model with zero costs resolves this.

**Helper pattern for cleaner code:**

```python
def create_lmstudio_language_model(
    model_name: str,
    api_base: str = "http://localhost:1234/v1",
    max_tokens: int = 4096,
    **kwargs,
) -> synalinks.LanguageModel:
    """Create a Synalinks LanguageModel configured for LMStudio."""
    import os
    import litellm

    os.environ["OPENAI_API_KEY"] = "lm-studio"
    full_model_name = f"openai/{model_name}"

    litellm.register_model({
        full_model_name: {
            "max_tokens": max_tokens,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    })

    return synalinks.LanguageModel(
        model=full_model_name,
        api_base=api_base,
        **kwargs,
    )

# Usage:
lm = create_lmstudio_language_model("ibm/granite-4-h-tiny")
```

## References

Read these for detailed information:

- **references/api-reference.md** - Full API for all classes and methods
- **references/data-models.md** - DataModel, Field, and schema patterns
- **references/modules-catalog.md** - All built-in modules (Generator, ChainOfThought, Decision, etc.)
- **references/training-guide.md** - Rewards, optimizers, callbacks, and training workflows
- **references/agents-tools.md** - FunctionCallingAgent, Tool, MCP integration
- **references/knowledge-base.md** - KnowledgeBase (DuckDB), EmbedKnowledge, UpdateKnowledge, RetrieveKnowledge, RAG
