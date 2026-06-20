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

All parameters must be **required** (no defaults) — LLM providers require every
parameter to be required in their structured-output JSON schema.

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
    data_model=FinalAnswer,            # Required: final output schema
    tools=tools,                       # Required: list of Tool instances
    language_model=lm,                 # Optional if a default LM is set
    max_iterations=5,                  # Max tool calls before final answer
    autonomous=True,                   # Run autonomously vs interactive
    return_inputs_with_trajectory=True, # Include full execution trajectory
    prompt_template=None,              # Custom prompt template
    instructions="",                   # Instructions string (NOT a list)
    final_instructions=None,           # Optional: overrides instructions for the final generator
    temperature=0.0,
    use_chain_of_thought=False,
    reasoning_effort=None,
    streaming=False,
)
```

All arguments are **keyword-only** (note the `*,` after `self` in the source).
`language_model` may be omitted if `synalinks.set_default_language_model(...)`
was called — `ops.predict` resolves the default at call time.

### Execution Trajectory

When `return_inputs_with_trajectory=True`, the output is the **`ChatMessages`
trajectory** (a `messages` list) concatenated with the final structured answer.
Tool calls are not a flat `{tool, input, output}` list — they live inside the
messages: the `assistant` turn carries a `tool_calls` array, and each result is
a `tool` message referencing the call's `tool_call_id`.

```json
{
  "messages": [
    {"role": "user", "content": "What is 15 * 7?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {"name": "calculate", "arguments": {"expression": "15 * 7"}}
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": {"result": 105, "log": "Success"}}
  ],
  "answer": "105"
}
```

With no `schema`/`data_model`, the agent returns the `ChatMessages` trajectory
directly; the final answer is the last `assistant` message
(`result.get("messages")[-1].get("content")`). The `ChatMessage` keys are a
subset of the chat-completion message keys (`role`, `content`, `tool_calls`,
`tool_call_id`, ...).

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

# Run with: uvicorn mcp_server:app --port 8183
app = mcp.streamable_http_app()
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

## RecursiveLanguageModelAgent (RLM)

`synalinks.RLM` / `synalinks.RecursiveLanguageModelAgent` — a
`FunctionCallingAgent` subclass for code-driven reasoning and long-context work.
The LM's only tool is `run_python_code(code)`, executed in a persistent REPL
sandbox; state carries across turns. The agent terminates by calling the
in-sandbox `submit(result=...)`.

```python
synalinks.RLM(
    schema=None,                  # final-answer JSON schema (or use data_model)
    data_model=Answer,            # final-answer DataModel; omit both → schemaless ChatMessages
    language_model=primary,       # per-turn code generation + final formatting
    sub_language_model=cheap,     # used by llm_query / llm_query_batched (defaults to language_model)
    instructions=None,
    final_instructions=None,
    temperature=0.0,
    use_chain_of_thought=False,   # emit a per-turn `thinking` field
    reasoning_effort=None,
    tools=None,                   # extra Tools, exposed inside the sandbox as global functions
    autonomous=True,              # False → one code turn per call, requires ChatMessages input
    return_inputs_with_trajectory=True,
    max_iterations=20,            # code-execution turns before forcing final answer
    timeout=60,                   # per-turn execution budget (seconds)
    recursive=True,               # expose llm_query / llm_query_batched helpers
    max_llm_calls=50,             # hard cap on sub-LM calls per invocation
    max_output_chars=10_000,      # REPL output truncation per observation
    workdir=None,                 # seeds sandbox FS; AGENTS.md auto-injected if present
    skills=None,                  # Agent Skill roots → <available_skills> context
    sandbox=None,                 # reuse a Sandbox across calls (persist REPL state)
    sandbox_type=None,            # Sandbox subclass to build (default MirageSandbox)
    max_subagent_depth=0,         # >0 enables spawn/merge/discard subagent tools
)
```

Key behaviors:

- **In-sandbox helpers, not tool calls.** `submit`, `llm_query`,
  `llm_query_batched`, and any user `tools` are plain functions inside the
  sandbox, callable only from the code passed to `run_python_code`. These three
  names are always reserved (even when `recursive=False`).
- **Recursive long-context.** With `recursive=True`, write Python that slices /
  filters a long input and delegates semantic sub-tasks to `sub_language_model`
  via `llm_query` / `llm_query_batched`, instead of dumping everything into one
  context window.
- **Budget.** `max_llm_calls` is per agent invocation (fresh budget each call,
  independent across concurrent calls). Exhausted budget returns an error string
  so the LM falls back to code-side aggregation.
- **Tool naming gotcha.** A tool registers under `tool._func.__name__`, so
  `Tool(_helper)` is `_helper` in scripts — rename the function, don't alias.

> Paper: [Recursive Language Models](https://arxiv.org/abs/2512.24601).

## DeepAgent

`synalinks.DeepAgent` — a `FunctionCallingAgent` subclass for coding /
file-manipulation tasks. It mounts a `workdir` in a `MirageSandbox` and exposes
filesystem + shell tools, all operating on the sandbox copy (host-safe).

Built-in tools: `read_file` (paginated, 1-based line range), `list_files`
(glob), `search_files` (glob + regex grep), `write_file`, `edit_file` (exact
string replacement), `run_bash` (pipes/redirects/globs/loops/`python3`).

```python
synalinks.DeepAgent(
    schema=None,
    data_model=None,              # final-answer schema; omit → ChatMessages output
    language_model=lm,
    sub_language_model=None,      # drives subagents; defaults to language_model
    instructions=None,           # default built from the workdir
    final_instructions=None,
    temperature=0.0,
    use_chain_of_thought=False,
    reasoning_effort=None,
    tools=None,                   # appended to the built-in file/shell tools
    autonomous=True,
    return_inputs_with_trajectory=True,
    max_iterations=10,            # coding needs more rounds than RAG/SQL
    streaming=False,
    timeout=30.0,                 # per-run_bash budget (seconds)
    workdir="/tmp/my_project",    # seeds the sandbox FS; omit → empty in-memory FS
    sandbox=None,                 # e.g. a Sandbox.fork of another agent's FS
    skills=None,
    max_subagent_depth=0,
)
```

Inspect the result via the **DeepAgent module** instance — `deep_agent.sandbox.changes()`
/ `deep_agent.sandbox.diff()`. The sandbox lives on the module, not on the wrapping
`Program`, so keep a reference to the module (`deep_agent = synalinks.DeepAgent(...)`;
`outputs = await deep_agent(inputs)`) rather than only the `Program`. The real
`workdir` is never modified. Typical input is `synalinks.ChatMessages`.

## Subagents (RLM & DeepAgent)

`max_subagent_depth > 0` adds `spawn_subagents` / `merge_subagent` /
`discard_subagent`, running subagents in parallel on isolated `Sandbox.fork`s
whose changes land only on an explicit merge. `1` is the recommended value
(spawned subagents can't themselves spawn). Requires a fork-capable sandbox.

- **DeepAgent** forks are filesystem branches → fold back **all** subagents' file
  changes.
- **RLM** forks inherit REPL state + files, but only **one** REPL namespace can
  be adopted per batch (`merge_subagent(..., adopt_repl=True)`) — the REPL
  serializes as a whole and parallel namespaces can't be unioned.

## Best Practices

### Choosing an agent

- **FunctionCallingAgent** — discrete external tools (search, calculators, APIs,
  MCP servers), parallel tool calls, structured final answer.
- **RLM** — code-driven reasoning, computation over data, or very long inputs
  that should be processed recursively rather than stuffed into one context.
- **DeepAgent** — reading/writing/editing files and running shell commands over a
  project directory (coding tasks), host-safe via the sandbox.

### Tool Design

1. **Clear docstrings** - LLM uses these to understand when to use tool
2. **Specific parameter types** - Use `int`, `float`, `bool` vs `str` when possible
3. **Informative error messages** - Help LLM recover from errors
4. **Return status** - Include "log" or "status" field for debugging

### Agent Design

1. **Limit max_iterations** - Prevent infinite loops
2. **Use specific output schema** - Guide final answer format
3. **Include trajectory** - Useful for debugging and transparency
4. **Add instructions** - Guide agent behavior (single string, not a list)

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
        instructions=(
            "Always explain your reasoning. "
            "Use tools when needed, not for simple questions. "
            "Provide confidence estimate based on source quality."
        ),
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
