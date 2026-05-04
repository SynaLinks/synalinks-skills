---
name: synalinks-agents
description: Use when working with Synalinks agents — FunctionCallingAgent (autonomous / interactive, max_iterations, return_inputs_with_trajectory), Tool definitions (async, type-annotated, register_synalinks_serializable), MCP integration via MultiServerMCPClient, parallel tool calling, or agent execution trajectories.
---

# Synalinks Agents and Tools

Build tool-using AI agents with structured outputs, autonomous or interactive execution, and MCP integration.

## Quick Start

```python
import synalinks
import asyncio

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")

class FinalAnswer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The final answer")

@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate mathematical expression.

    Args:
        expression: Mathematical expression like '2 + 2'
    """
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return {"result": round(float(result), 4), "log": "Success"}
    except Exception as e:
        return {"result": None, "log": f"Error: {e}"}

async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")
    tools = [synalinks.Tool(calculate)]

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=FinalAnswer,
        tools=tools,
        language_model=lm,
        max_iterations=5,
        autonomous=True,
    )(inputs)

    agent = synalinks.Program(inputs=inputs, outputs=outputs, name="calc_agent")
    result = await agent(Query(query="What is 15 * 7?"))
    print(result.prettify_json())

asyncio.run(main())
```

## Defining Tools

### Tool Requirements

1. **Async function** — must be `async def`
2. **Type annotations** — every parameter must have a type hint
3. **Docstring** — used by the LLM to decide when to call the tool
4. **Serializable** — decorate with `@synalinks.utils.register_synalinks_serializable()`
5. **Return dict** — return a dict with results and a `log` / status field

### Basic Tool

```python
@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression: Mathematical expression like '2 + 2' or '(10 * 5) / 2'

    Returns:
        dict with 'result' and 'log' keys
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"result": result, "log": "Success"}
    except Exception as e:
        return {"result": None, "log": f"Error: {e}"}

tool = synalinks.Tool(calculate)
```

### Multiple Parameters

All parameters must be **required** (no defaults). LLM providers require every
parameter in the structured-output JSON schema to be required, so Tool defaults
are unsafe.

```python
@synalinks.utils.register_synalinks_serializable()
async def search_database(
    query: str,
    limit: int,
    category: str,
):
    """Search the database for matching records.

    Args:
        query: Search query string
        limit: Maximum number of results (e.g. 10)
        category: Category filter ('all', 'products', 'users')
    """
    results = await db.search(query, limit=limit, category=category)
    return {"results": results, "count": len(results), "log": f"Found {len(results)} results"}
```

### Tool with Validation

```python
@synalinks.utils.register_synalinks_serializable()
async def safe_calculate(expression: str):
    """Safely calculate mathematical expressions.

    Args:
        expression: Expression with numbers and +, -, *, /, (, ), . only
    """
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return {"result": None, "log": "Error: Invalid characters."}
    try:
        return {"result": round(float(eval(expression, {"__builtins__": None}, {})), 4), "log": "Success"}
    except Exception as e:
        return {"result": None, "log": f"Error: {e}"}
```

## FunctionCallingAgent

```python
synalinks.FunctionCallingAgent(
    data_model=FinalAnswer,              # Required: final output schema
    tools=tools,                         # Required: list of Tool instances
    language_model=lm,                   # Optional if default LM is set
    max_iterations=5,                    # Max tool calls before final answer
    autonomous=True,                     # Run autonomously vs interactive
    return_inputs_with_trajectory=True,  # Include full execution trajectory
    prompt_template=None,
    instructions="",                     # MUST be a string
)
```

All arguments are **keyword-only** — `FunctionCallingAgent(data_model, tools, lm)`
(positional) raises `TypeError`. Same for `Generator`, `ChainOfThought`, and other
Module subclasses. `Tool(my_function)` is the exception: its first positional arg
is the wrapped function.

`language_model` may be omitted if `synalinks.set_default_language_model(...)`
was called — defaults flow via `ops.predict`.

**`instructions` MUST be a string**, not a list — see synalinks-modules for the full discussion.

### Execution Trajectory

When `return_inputs_with_trajectory=True`:

```json
{
  "query": "What is 15 * 7?",
  "trajectory": [
    {"tool": "calculate", "input": {"expression": "15 * 7"}, "output": {"result": 105, "log": "Success"}}
  ],
  "answer": "105"
}
```

## Autonomous vs Interactive

### Autonomous

Agent runs until `max_iterations` or final answer:

```python
synalinks.FunctionCallingAgent(autonomous=True, max_iterations=5, ...)
```

### Interactive

Agent pauses for human confirmation at each step:

```python
synalinks.FunctionCallingAgent(autonomous=False, ...)
```

## MCP Integration

### MultiServerMCPClient

```python
mcp_client = synalinks.MultiServerMCPClient({
    "math_server":   {"url": "http://localhost:8183/mcp/", "transport": "streamable_http"},
    "search_server": {"url": "http://localhost:8184/mcp/", "transport": "streamable_http"},
})

tools = await mcp_client.get_tools()

outputs = await synalinks.FunctionCallingAgent(
    data_model=FinalAnswer,
    tools=tools,
    language_model=lm,
)(inputs)
```

### MCP Server Example

```python
# mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math Server")

@mcp.tool()
def calculate(expression: str) -> dict:
    """Calculate mathematical expression."""
    return {"result": eval(expression)}

# Run with: uvicorn mcp_server:app --port 8183
app = mcp.streamable_http_app()
```

## Best Practices

**Tool design**

1. Clear docstrings — the LLM uses these to decide when to call the tool
2. Specific parameter types (`int`, `float`, `bool`) over generic `str`
3. Informative error messages so the LLM can recover
4. Return a `log` / `status` field for debugging

**Agent design**

1. Cap `max_iterations` to prevent runaway loops
2. Use a specific `data_model` to structure the final answer
3. Capture trajectories during development, drop them in production for cost savings
4. Provide `instructions` as a single string

## References

- **references/agents-tools.md** — Complete agent and tool API with examples

## See Also

- **synalinks-core** — DataModel, Program basics
- **synalinks-modules** — Generator, ChainOfThought, custom modules
- **synalinks-training** — Training agents with rewards (e.g. trajectory-aware rewards)
- **synalinks-knowledge** — Combining KnowledgeBase retrieval with agents
- **synalinks-providers** — Groq/OpenRouter/LMStudio for the underlying LM
