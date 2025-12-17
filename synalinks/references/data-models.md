# Synalinks DataModel Reference

## Overview

DataModel is the core abstraction for structured I/O in Synalinks. It replaces tensors from deep learning with JSON schemas, enabling type-safe, validated data flow through programs.

## Defining DataModels

### Basic Definition

```python
import synalinks

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The answer to the query")
```

### With Default Values

```python
class QueryWithContext(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")
    context: str = synalinks.Field(default="", description="Optional context")
    max_tokens: int = synalinks.Field(default=100, description="Max response length")
```

### With Complex Types

```python
from typing import List, Optional

class AnalysisResult(synalinks.DataModel):
    summary: str = synalinks.Field(description="Brief summary")
    key_points: List[str] = synalinks.Field(description="List of key points")
    confidence: float = synalinks.Field(description="Confidence score 0-1")
    source: Optional[str] = synalinks.Field(default=None, description="Source if available")
```

### With Enums

```python
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentAnalysis(synalinks.DataModel):
    text: str = synalinks.Field(description="The analyzed text")
    sentiment: Sentiment = synalinks.Field(description="Detected sentiment")
    confidence: float = synalinks.Field(description="Confidence score")
```

### Nested DataModels

```python
class Address(synalinks.DataModel):
    street: str = synalinks.Field(description="Street address")
    city: str = synalinks.Field(description="City name")
    country: str = synalinks.Field(description="Country")

class Person(synalinks.DataModel):
    name: str = synalinks.Field(description="Full name")
    address: Address = synalinks.Field(description="Home address")
```

## Field Descriptions

Field descriptions are critical for LLM understanding. They appear in the prompt and guide the model's output.

```python
class DetailedAnswer(synalinks.DataModel):
    thinking: str = synalinks.Field(
        description="Your step-by-step reasoning process. Show your work."
    )
    answer: str = synalinks.Field(
        description="The final concise answer based on your thinking."
    )
    confidence: float = synalinks.Field(
        description="Confidence in the answer from 0.0 (uncertain) to 1.0 (certain)"
    )
```

## DataModel Operations

### Instantiation

```python
query = Query(query="What is the capital of France?")
print(query.prettify_json())
# {"query": "What is the capital of France?"}
```

### Accessing Fields

```python
value = query.get("query")  # "What is the capital of France?"
```

### Schema Inspection

```python
schema = Query.get_schema()
print(Query.prettify_schema())
```

## Metaclass Operations (Schema Composition)

### Concatenation (+)

Creates a new schema combining fields from both DataModels.

```python
# At class level (SymbolicDataModel)
Combined = Query + Answer
# Combined has both "query" and "answer" fields

# At instance level (JsonDataModel)
qa = Query(query="...") + Answer(answer="...")
```

**Field Collision Handling:**
```python
TwoQueries = Query + Query
# Fields: query, query_1 (auto-renamed)
```

### Logical And (&)

Safe concatenation that returns None if either operand is None.

```python
result = inputs & branch_output  # None if branch_output is None
```

### Logical Or (|)

Returns non-None value, or concatenates if both present.

```python
result = branch1 | branch2  # Returns whichever is not None
```

### Logical Xor (^)

Returns one if exactly one is present, None otherwise. Useful for input/output guards.

```python
result = x1 ^ x2  # Returns x1 or x2 if exactly one is non-None

# Guard pattern: bypass computation if warning exists
guarded = warning ^ inputs  # inputs only if warning is None
```

### Logical Not (~)

Inverts None/non-None status.

```python
result = ~x  # None if x exists, raises if x is None
```

## Truth Tables

### Concatenation (+)

| x1 | x2 | Result |
|----|----|----|
| x1 | x2 | x1 + x2 |
| x1 | None | Exception |
| None | x2 | Exception |
| None | None | Exception |

### Logical And (&)

| x1 | x2 | Result |
|----|----|----|
| x1 | x2 | x1 + x2 |
| x1 | None | None |
| None | x2 | None |
| None | None | None |

### Logical Or (|)

| x1 | x2 | Result |
|----|----|----|
| x1 | x2 | x1 + x2 |
| x1 | None | x1 |
| None | x2 | x2 |
| None | None | None |

### Logical Xor (^)

| x1 | x2 | Result |
|----|----|----|
| x1 | x2 | None |
| x1 | None | x1 |
| None | x2 | x2 |
| None | None | None |

## Special DataModels

### ChatMessages (Conversational)

```python
from synalinks.backend import ChatMessages

inputs = synalinks.Input(data_model=ChatMessages)
outputs = await synalinks.Generator(
    language_model=lm,
    prompt_template=synalinks.chat_prompt_template(),
)(inputs)
```

### ChatMessage Helper Functions

```python
# Check if data is ChatMessages (plural - conversation history)
if synalinks.is_chat_messages(inputs):
    messages = inputs["messages"]

# Check if data is ChatMessage (singular - single message)
if synalinks.is_chat_message(data):
    content = data["content"]

# Create symbolic DataModel for output spec
symbolic = synalinks.ChatMessage.to_symbolic_data_model(name="my_module")
```

### Entity Models (Knowledge Graph)

```python
class City(synalinks.Entity):
    name: str = synalinks.Field(description="City name")
    population: int = synalinks.Field(description="Population count")

class IsCapitalOf(synalinks.Relation):
    pass  # Defines relationship between entities
```

## Best Practices

1. **Always include descriptions** - They guide LLM output
2. **Use specific types** - `int`, `float`, `bool` vs generic `str`
3. **Keep field names semantic** - `answer` not `a` or `output1`
4. **Use enums for constrained choices** - Guarantees valid values
5. **Nest for complex structures** - Better organization and reuse
6. **Prefer Optional over defaults** - Clearer intent for nullable fields

## Example: Complete Q&A Schema

```python
class UserQuery(synalinks.DataModel):
    question: str = synalinks.Field(description="The user's question")
    context: Optional[str] = synalinks.Field(default=None, description="Additional context")

class ThinkingStep(synalinks.DataModel):
    step_number: int = synalinks.Field(description="Step number in reasoning chain")
    thought: str = synalinks.Field(description="The reasoning for this step")

class DetailedAnswer(synalinks.DataModel):
    thinking: List[str] = synalinks.Field(description="Step-by-step reasoning")
    answer: str = synalinks.Field(description="Final answer")
    confidence: float = synalinks.Field(description="Confidence score 0-1")
    sources: List[str] = synalinks.Field(default_factory=list, description="Supporting sources")
```
