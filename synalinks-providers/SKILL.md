---
name: synalinks-providers
description: Use when integrating Synalinks with LM providers — picking the right model prefix (openai/, anthropic/, ollama/, groq/, cohere/, openrouter/, bedrock/, deepseek/, together_ai/, doubleword/, hosted_vllm/ (alias vllm/), gemini/, xai/, mistral/, azure/), env vars per provider, structured-output dispatch (constrained json_schema vs tool-call), local OpenAI-compatible servers (LMStudio, vLLM) requiring litellm.register_model and a dummy OPENAI_API_KEY, and OpenRouter embeddings (LiteLLM doesn't support them — use OpenRouterEmbeddingModel).
---

# Synalinks Provider Integrations

Configuring `LanguageModel` and `EmbeddingModel` for various providers, plus the few cases that still need extra setup (local servers, OpenRouter embeddings).

As of Synalinks **v0.8.006+**, structured output works out of the box for all listed cloud providers — no patches required.

## Provider Cheat Sheet

| Provider | Prefix | Env var | Dispatch path |
|----------|--------|---------|---------------|
| OpenAI | `openai/` | `OPENAI_API_KEY` | constrained `json_schema` |
| Azure OpenAI | `azure/` | `AZURE_API_KEY` + base | constrained `json_schema` |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` | `response_format` (LiteLLM auto-routes) |
| Ollama | `ollama/` | (local, no key) | constrained `json_schema` |
| Mistral | `mistral/` | `MISTRAL_API_KEY` | constrained `json_schema` |
| Gemini | `gemini/` | `GEMINI_API_KEY` | constrained `json_schema` |
| xAI | `xai/` | `XAI_API_KEY` | constrained `json_schema` |
| Groq | `groq/` | `GROQ_API_KEY` | tool-call structured output |
| Cohere | `cohere/` | `COHERE_API_KEY` | tool-call structured output |
| OpenRouter | `openrouter/` | `OPENROUTER_API_KEY` | tool-call structured output |
| Bedrock | `bedrock/` | `AWS_ACCESS_KEY_ID` etc. | tool-call structured output |
| DeepSeek | `deepseek/` | `DEEPSEEK_API_KEY` | constrained `json_schema` |
| Together AI | `together_ai/` | `TOGETHER_AI_API_KEY` | constrained `json_schema` |
| Doubleword | `doubleword/` | `OPENAI_API_KEY` (Doubleword key) | rewritten to `openai/` w/ `api_base` |
| LMStudio / vLLM | `openai/<name>` + `api_base` | dummy `OPENAI_API_KEY` | requires `litellm.register_model` |
| vLLM (native) | `hosted_vllm/` (alias `vllm/`) | (local; `HOSTED_VLLM_API_BASE` opt) | constrained `json_schema` |

The `vllm/` prefix is rewritten to `hosted_vllm/` internally. When `hosted_vllm/` is used without an explicit `api_base`, Synalinks reads `HOSTED_VLLM_API_BASE` (default `http://localhost:8000`).

### Why the two dispatch paths exist

- **Constrained path** (`response_format` with `json_schema`): preferred when the provider natively supports OpenAI-style structured output. The model is forced to emit valid JSON matching the schema.
- **Tool-call path**: a phantom tool named `structured_output` is registered with the schema as its parameters; the provider's tool-calling forces the model to "call" it. Used for providers that lack native JSON schema (Cohere, most Bedrock models) or proxy heterogeneous backends with mixed support (OpenRouter).

You don't pick the path — Synalinks dispatches based on the model prefix.

## Constructor Defaults & Fallback Coercion

`LanguageModel` and `EmbeddingModel` both subclass `Module` (so they support hooks) and accept arbitrary `**default_kwargs` that are forwarded to every call; per-call kwargs override the instance defaults.

```python
lm = synalinks.LanguageModel(
    model="openai/gpt-4o-mini",
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    max_tokens=2048,
    reasoning_effort="medium",  # forwarded only if litellm.supports_reasoning(model)
)

em = synalinks.EmbeddingModel(
    model="openai/text-embedding-3-small",
    dimensions=512,
)
```

`fallback=` accepts a string, a config dict, or an existing `LanguageModel` / `EmbeddingModel` instance — it is auto-coerced via the package-level `get()`:

```python
lm = synalinks.LanguageModel(
    model="anthropic/claude-sonnet-4-5",
    fallback="openai/gpt-4o-mini",  # string is fine
)
```

When all retries fail and a fallback is set, the original call is replayed against the fallback with the user-supplied kwargs (instance defaults are not re-applied).

## Framework-Wide Defaults

`synalinks.set_default_language_model(identifier)` and `synalinks.set_default_embedding_model(identifier)` register a default usable across the framework (e.g. by `Generator`, optimizers, rewards). String identifiers are persisted to `~/.synalinks/synalinks.json`; dicts and instances are kept in-process only.

```python
synalinks.set_default_language_model("openai/gpt-4o-mini")
synalinks.set_default_embedding_model("openai/text-embedding-3-small")

# Read back (lazy-constructs from a persisted string on first access)
lm = synalinks.default_language_model()
em = synalinks.default_embedding_model()
```

Pass `None` to clear.

## Quick Examples

### OpenAI
```python
import os
os.environ["OPENAI_API_KEY"] = "..."
lm = synalinks.LanguageModel(model="openai/gpt-4o-mini")
```

### Anthropic
```python
os.environ["ANTHROPIC_API_KEY"] = "..."
lm = synalinks.LanguageModel(model="anthropic/claude-sonnet-4-5")
```

### Groq
```python
os.environ["GROQ_API_KEY"] = "..."
lm = synalinks.LanguageModel(model="groq/llama-3.1-8b-instant")
```

### OpenRouter
```python
os.environ["OPENROUTER_API_KEY"] = "..."
lm = synalinks.LanguageModel(model="openrouter/anthropic/claude-3-haiku")
```

### Cohere
```python
os.environ["COHERE_API_KEY"] = "..."
lm = synalinks.LanguageModel(model="cohere/command-r-plus")
```

### DeepSeek
```python
os.environ["DEEPSEEK_API_KEY"] = "..."
lm = synalinks.LanguageModel(model="deepseek/deepseek-chat")
```

### Together AI
```python
os.environ["TOGETHER_AI_API_KEY"] = "..."
lm = synalinks.LanguageModel(model="together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")
```

### AWS Bedrock
```python
os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_REGION_NAME"] = "us-east-1"
lm = synalinks.LanguageModel(model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
```

### Doubleword
The `doubleword/` prefix is rewritten to `openai/` internally with `api_base="https://api.doubleword.ai/v1"` — so it goes through the strict-schema OpenAI path.
```python
os.environ["OPENAI_API_KEY"] = "<doubleword key>"
lm = synalinks.LanguageModel(model="doubleword/qwen-qwen3-5-397b-a17b-fp8-dottxt")
```

## Local OpenAI-Compatible Servers (LMStudio, vLLM)

Local servers that don't ship to LiteLLM's pricing database still need three things:

1. Dummy `OPENAI_API_KEY` (any non-empty string)
2. `litellm.register_model({...})` to give it `max_tokens` and zero costs
3. `api_base` pointing at the local server

```python
import os, litellm, synalinks

os.environ["OPENAI_API_KEY"] = "lm-studio"

litellm.register_model({
    "openai/ibm/granite-4-h-tiny": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",
        "mode": "chat",
    }
})

lm = synalinks.LanguageModel(
    model="openai/ibm/granite-4-h-tiny",
    api_base="http://localhost:1234/v1",
)
```

**Why:** LiteLLM tracks API costs internally. Without registration it returns `None` for unknown models; Synalinks's cost arithmetic then fails.

Helper: **scripts/lmstudio_setup.py** wraps this into a one-liner.

For vLLM, the `hosted_vllm/` prefix may also work without registration (it routes through the constrained-schema path). If it doesn't, fall back to the LMStudio recipe.

## OpenRouter Embeddings

LiteLLM does **not** support OpenRouter embeddings. Use `OpenRouterEmbeddingModel` (direct API calls):

```python
from openrouter_embeddings import OpenRouterEmbeddingModel

em = OpenRouterEmbeddingModel(
    "qwen/qwen3-embedding-8b",
    provider={"only": ["nebius"], "allow_fallbacks": False},
)

result = await em(["Hello, world!"])
# {"embeddings": [[0.02, 0.006, ...]]}
```

`OpenRouterEmbeddingModel` is **OMEGA-compatible** — it coerces non-string leaves to strings (needed because `tree.flatten()` over trainable variables may yield non-strings). See **synalinks-optimizers**.

Helper: **scripts/openrouter_embeddings.py**

## Provider Routing (OpenRouter)

OpenRouter lets you pin a request to a specific backend. Pass `provider` via LiteLLM's `extra_body` — `LanguageModel`'s `**default_kwargs` forwards it to every call:

```python
lm = synalinks.LanguageModel(
    model="openrouter/meta-llama/llama-3.1-8b-instruct",
    extra_body={"provider": {"only": ["DeepInfra"], "allow_fallbacks": False}},
)
```

(For pre-`**default_kwargs` versions, see the legacy patch in `scripts/openrouter_patch_legacy.py` for an instance-aware routing setup.)

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `LM provider 'X' not supported` | Old Synalinks version | Upgrade to ≥ 0.8.006 |
| `litellm.NotFoundError` on local model | Model not registered | `litellm.register_model({...})` |
| Cost arithmetic fails on local model | LiteLLM returns `None` for cost | Register with zero costs |
| OpenRouter embeddings 404 | Using LiteLLM | Use `OpenRouterEmbeddingModel` |
| Groq returns invalid JSON | Schema mismatch | Inspect with `synalinks.enable_logging()` |
| Anthropic structured output fails | Old LiteLLM version | Upgrade `litellm` |

## Legacy Patches (Pre-0.8.006)

Earlier Synalinks versions required monkey-patches for Groq and OpenRouter. Those scripts remain in this skill for historical reference but **shouldn't be needed today**:

- `scripts/groq_patch_legacy.py` — strips `tool_calls` from messages, switches Groq to `json_schema` mode
- `scripts/openrouter_patch_legacy.py` — registers `openrouter/` prefix, supports per-instance provider routing

**Current users:** just upgrade Synalinks. The dispatcher handles all of this natively.

## References

- **references/providers.md** — Per-provider dispatch internals, debugging tips
- **scripts/lmstudio_setup.py** — LMStudio / vLLM helper
- **scripts/openrouter_embeddings.py** — `OpenRouterEmbeddingModel`
- **scripts/groq_patch_legacy.py**, **scripts/openrouter_patch_legacy.py** — Pre-0.8.006 patches

## See Also

- **synalinks-core** — `LanguageModel`, `EmbeddingModel` basics
- **synalinks-optimizers** — `OpenRouterEmbeddingModel` for OMEGA
