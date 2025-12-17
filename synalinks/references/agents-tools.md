# Synalinks Agents and Tools

## Overview

Synalinks provides a powerful agent framework for building tool-using AI systems. The `FunctionCallingAgent` module enables autonomous or interactive tool use with structured outputs.

## Defining Tools

### Basic Tool Definition

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

# Create Tool instance
tool = synalinks.Tool(calculate)
```

### Tool Requirements

1. **Async function** - Must be `async def`
2. **Type annotations** - All parameters must have type hints
3. **Docstring** - Required for LLM understanding
4. **Serializable** - Decorate with `@synalinks.utils.register_synalinks_serializable()`
5. **Return dict** - Should return dict with results and status

### Multiple Parameters

```python
@synalinks.utils.register_synalinks_serializable()
async def search_database(
    query: str,
    limit: int = 10,
    category: str = "all",
):
    """Search the database for matching records.

    Args:
        query: Search query string
        limit: Maximum number of results (default: 10)
        category: Category filter ('all', 'products', 'users')

    Returns:
        dict with search results and metadata
    """
    results = await db.search(query, limit=limit, category=category)
    return {
        "results": results,
        "count": len(results),
        "log": f"Found {len(results)} results",
    }
```

### Tool with Validation

```python
@synalinks.utils.register_synalinks_serializable()
async def safe_calculate(expression: str):
    """Safely calculate mathematical expressions.

    Args:
        expression: Expression with numbers and +, -, *, /, (, ), . only
    """
    # Validate input
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return {
            "result": None,
            "log": "Error: Invalid characters. Only numbers and operators allowed.",
        }

    try:
        result = eval(expression, {"__builtins__": None}, {})
        return {"result": round(float(result), 4), "log": "Success"}
    except Exception as e:
        return {"result": None, "log": f"Error: {e}"}
```

---

## FunctionCallingAgent

### Basic Usage

```python
class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="The user query")

class FinalAnswer(synalinks.DataModel):
    answer: str = synalinks.Field(description="The final answer")

tools = [
    synalinks.Tool(calculate),
    synalinks.Tool(search_database),
]

inputs = synalinks.Input(data_model=Query)
outputs = await synalinks.FunctionCallingAgent(
    data_model=FinalAnswer,
    tools=tools,
    language_model=lm,
    max_iterations=5,
    autonomous=True,
)(inputs)

agent = synalinks.Program(
    inputs=inputs,
    outputs=outputs,
    name="my_agent",
)

result = await agent(Query(query="What is 15 * 7?"))
```

### Parameters

```python
synalinks.FunctionCallingAgent(
    data_model=FinalAnswer,           # Required: final output schema
    tools=tools,                       # Required: list of Tool instances
    language_model=lm,                 # Required: LanguageModel
    max_iterations=5,                  # Max tool calls before final answer
    autonomous=True,                   # Run autonomously vs interactive
    return_inputs_with_trajectory=True, # Include full execution trajectory
    prompt_template=None,              # Custom prompt template
    instructions=[],                   # Additional instructions
)
```

### Execution Trajectory

When `return_inputs_with_trajectory=True`, output includes:

```json
{
  "query": "What is 15 * 7?",
  "trajectory": [
    {
      "tool": "calculate",
      "input": {"expression": "15 * 7"},
      "output": {"result": 105, "log": "Success"}
    }
  ],
  "answer": "105"
}
```

---

## MCP Integration

### MultiServerMCPClient

Connect to Model Context Protocol servers.

```python
mcp_client = synalinks.MultiServerMCPClient({
    "math_server": {
        "url": "http://localhost:8183/mcp/",
        "transport": "streamable_http",
    },
    "search_server": {
        "url": "http://localhost:8184/mcp/",
        "transport": "streamable_http",
    },
})

# Get tools from all servers
tools = await mcp_client.get_tools()

# Use with agent
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

# Run with: uvicorn mcp_server:app
app = mcp.get_asgi_app()
```

---

## Interactive vs Autonomous

### Autonomous Mode

Agent runs independently until max_iterations or final answer.

```python
outputs = await synalinks.FunctionCallingAgent(
    autonomous=True,
    max_iterations=5,
    ...
)(inputs)
```

### Interactive Mode

Agent pauses for human confirmation at each step.

```python
outputs = await synalinks.FunctionCallingAgent(
    autonomous=False,
    ...
)(inputs)
```

---

## Tool Selection

### ToolCalling Module

For single tool selection without agent loop.

```python
outputs = await synalinks.ToolCalling(
    tools=tools,
    language_model=lm,
)(inputs)
# Returns selected tool call and result
```

---

## Best Practices

### Tool Design

1. **Clear docstrings** - LLM uses these to understand when to use tool
2. **Specific parameter types** - Use `int`, `float`, `bool` vs `str` when possible
3. **Informative error messages** - Help LLM recover from errors
4. **Return status** - Include "log" or "status" field for debugging

### Agent Design

1. **Limit max_iterations** - Prevent infinite loops
2. **Use specific output schema** - Guide final answer format
3. **Include trajectory** - Useful for debugging and transparency
4. **Add instructions** - Guide agent behavior

### Example: Complete Agent

```python
import synalinks
import asyncio

synalinks.enable_logging()

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User question")

class Answer(synalinks.DataModel):
    reasoning: str = synalinks.Field(description="Step-by-step reasoning")
    answer: str = synalinks.Field(description="Final answer")
    confidence: float = synalinks.Field(description="Confidence 0-1")

@synalinks.utils.register_synalinks_serializable()
async def search_web(query: str):
    """Search the web for information.

    Args:
        query: Search query
    """
    # Simulate search
    return {"results": [...], "log": "Found 10 results"}

@synalinks.utils.register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate math expression.

    Args:
        expression: Math expression (numbers and +,-,*,/ only)
    """
    try:
        return {"result": eval(expression), "log": "Success"}
    except:
        return {"result": None, "log": "Invalid expression"}

async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")

    tools = [
        synalinks.Tool(search_web),
        synalinks.Tool(calculate),
    ]

    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.FunctionCallingAgent(
        data_model=Answer,
        tools=tools,
        language_model=lm,
        max_iterations=5,
        autonomous=True,
        return_inputs_with_trajectory=True,
        instructions=[
            "Always explain your reasoning",
            "Use tools when needed, not for simple questions",
            "Provide confidence estimate based on source quality",
        ],
    )(inputs)

    agent = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="research_agent",
        description="Agent that searches and calculates",
    )

    synalinks.utils.plot_program(agent, to_folder=".")

    result = await agent(Query(query="What is 25% of 480?"))
    print(result.prettify_json())

asyncio.run(main())
```
