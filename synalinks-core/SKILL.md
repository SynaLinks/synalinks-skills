---
name: synalinks-core
description: Use when working with Synalinks DataModel, Field, Input, JSON operators (+, &, |, ^, ~), synalinks.ops, configuration (enable_logging, enable_observability, set_seed, clear_session), LanguageModel, EmbeddingModel, or Keras-like LLM structured output basics. For the Program class itself (Functional / Sequential / Subclassing / Mixed APIs, save/load, summary, get_module) see synalinks-programs.
---

# Synalinks Core Framework

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

## DataModel

DataModel is the core abstraction for structured I/O. All module inputs and outputs use DataModels.

```python
class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer to the query")
```

### Complex Types

```python
from typing import List, Optional
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class AnalysisResult(synalinks.DataModel):
    summary: str = synalinks.Field(description="Brief summary")
    key_points: List[str] = synalinks.Field(description="List of key points")
    confidence: float = synalinks.Field(description="Confidence score 0-1")
    sentiment: Sentiment = synalinks.Field(description="Detected sentiment")
    source: Optional[str] = synalinks.Field(default=None, description="Source if available")
```

### Nested DataModels

```python
class Address(synalinks.DataModel):
    street: str = synalinks.Field(description="Street address")
    city: str = synalinks.Field(description="City name")

class Person(synalinks.DataModel):
    name: str = synalinks.Field(description="Full name")
    address: Address = synalinks.Field(description="Home address")
```

### Special DataModels

```python
# ChatMessages for conversational applications
inputs = synalinks.Input(data_model=synalinks.ChatMessages)
outputs = await synalinks.Generator(
    language_model=lm,
)(inputs)  # No data_model/schema → emits ChatMessage-like output

synalinks.is_chat_messages(inputs)  # True if ChatMessages
synalinks.is_chat_message(data)     # True if single ChatMessage
```

### DataModel Operations

```python
query = Query(query="What is the capital of France?")
print(query.prettify_json())
value = query.get("query")
schema = Query.get_schema()
print(Query.prettify_schema())
```

See **references/data-models.md** for plain Pydantic interop, truth tables, and best practices.

## Building Programs

A minimal Functional-API program looks like this:

```python
inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.Generator(data_model=Answer, language_model=lm)(inputs)
program = synalinks.Program(inputs=inputs, outputs=outputs)
```

Synalinks supports four building APIs (Functional, Sequential, Subclassing,
Mixed), plus save/load, `summary`, `get_module`, multi-input/multi-output
graphs, and the build/call lifecycle. **For everything Program-related, see
the dedicated synalinks-programs skill.**

## JSON Operators (Circuit-like Logic)

| Operator | Symbol | Behavior |
|----------|--------|----------|
| Concatenate | `+` | Merge fields; raises Exception if either is None |
| Logical And | `&` | Merge fields; returns None if either is None |
| Logical Or | `\|` | Returns non-None value; concatenates if both present |
| Logical Xor | `^` | Returns one if exactly one is present, else None |
| Logical Not | `~` | Inverts None / non-None |
| Contains | `in` | Check if one DataModel's fields are a subset of another |

```python
combined = x1 + x2          # Concatenation (strict)
combined = inputs & branch  # And — None if branch not taken
result = b1 | b2            # Or — merge branches
guarded = warning ^ inputs  # Xor — bypass if warning set
print(Query in (Query + Answer))  # True
```

### Module Alternatives to Operators

```python
merged = await synalinks.And()([b0, b1, b2])
result = await synalinks.Or()([b0, b1, b2])
```

### synalinks.ops Functions

```python
result = await synalinks.ops.concat(x1, x2, name="combined")

result = await synalinks.ops.in_mask(x, mask=["answer"])
result = await synalinks.ops.out_mask(x, mask=["thinking"])
result = await synalinks.ops.in_mask(x, pattern="^input_")

result = await synalinks.ops.factorize(x)   # Group similar fields into lists

result = await synalinks.ops.logical_and(x1, x2)
result = await synalinks.ops.logical_or(x1, x2)
result = await synalinks.ops.logical_xor(x1, x2)
result = await synalinks.ops.logical_not(x)
```

## LanguageModel and EmbeddingModel

`LanguageModel` and `EmbeddingModel` are now `Module` subclasses (so they support hooks). Both accept arbitrary `**default_kwargs` (e.g. `temperature`, `top_p`, `top_k`, `max_tokens`, `reasoning_effort`, `dimensions`) merged into every call; per-call kwargs override.

```python
lm = synalinks.LanguageModel(
    model="ollama/mistral",
    # model="openai/gpt-4o-mini",
    # model="anthropic/claude-3-sonnet-20240229",
    # model="gemini/gemini-2.5-pro",
    # model="mistral/codestral-latest",
    api_base=None,    # Optional endpoint override
    timeout=600,
    retry=5,
    fallback=None,    # str / dict / LanguageModel — coerced via get()
    caching=False,
    temperature=0.0, # Default kwarg, forwarded on every call
)

em = synalinks.EmbeddingModel(
    model="ollama/mxbai-embed-large",
    dimensions=1024,  # Default kwarg, forwarded on every call
)
```

`fallback=` accepts a model string, a config dict, or an existing instance — strings/dicts are resolved via `get()` automatically:

```python
lm = synalinks.LanguageModel(
    model="anthropic/claude-3-sonnet-20240229",
    fallback="gemini/gemini-2.5-flash",  # coerced into a LanguageModel
)
```

### Default models (process-wide)

```python
synalinks.set_default_language_model("openai/gpt-4o-mini")
synalinks.set_default_embedding_model("openai/text-embedding-3-small")

lm = synalinks.default_language_model()   # cached instance
em = synalinks.default_embedding_model()
```

String identifiers persist to `~/.synalinks/synalinks.json`; passing a config dict or instance sets the cached default for the process only. `Generator(language_model=None)` no longer raises — the default LM is resolved at call time inside `ops.predict`.

For Groq, OpenRouter, LMStudio, vLLM, see **synalinks-providers**.

For saving / loading / inspection of programs, see **synalinks-programs**.

## Configuration

```python
synalinks.enable_logging()       # Debug logging
synalinks.enable_observability() # Tracing (Arize Phoenix compatible)
synalinks.set_seed(42)           # Reproducibility
synalinks.clear_session()        # Clear session for reproducible naming (important in notebooks)
```

## Important Gotchas

### `instructions` parameter MUST be a string

```python
# CORRECT
synalinks.Generator(data_model=Answer, language_model=lm,
                    instructions="Be concise. Focus on key points.")

# WRONG - raises ValidationError
synalinks.Generator(data_model=Answer, language_model=lm,
                    instructions=["Be concise", "Focus on key points"])
```

### Always check for None results

```python
result = await program(input_data)
if result is None:
    print("LLM call failed - check API key and model compatibility")
```

## References

- **references/api-reference.md** - Full API for Module, Program, ops, callbacks, utils
- **references/data-models.md** - DataModel, Field, schema patterns, Pydantic interop, truth tables

## See Also

- **synalinks-programs** — Program class, four building APIs, save/load, summary, multi-IO graphs
- **synalinks-modules** — Generator, ChainOfThought, custom modules
- **synalinks-control-flow** — Decision, Branch, guards, parallel/self-consistency patterns
- **synalinks-agents** — FunctionCallingAgent, Tool, MCP
- **synalinks-knowledge** — KnowledgeBase, RAG
- **synalinks-training** — compile/fit/evaluate workflow
- **synalinks-rewards**, **synalinks-optimizers** — training internals
- **synalinks-providers** — Groq / OpenRouter / LMStudio patches
- **synalinks-datasets** — built-in datasets and visualization
