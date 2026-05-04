# Synalinks Providers Reference

## Module Locations (Internal)

`LanguageModel` and `EmbeddingModel` live under `synalinks.src.modules.{language_models,embedding_models}` (they subclass `Module` and support hooks). The public API (`synalinks.LanguageModel`, `synalinks.EmbeddingModel`) is unchanged. Legacy `synalinks.src.language_models.*` / `synalinks.src.embedding_models.*` import paths are gone — patches and skills must use the new locations.

## Dispatch Logic (v0.8.006+)

`LanguageModel.call` dispatches structured-output requests based on the model prefix (`call`, not `__call__` — the `Module` base class wraps `call` in the public `__call__`):

### Constrained `json_schema` path

These providers natively support OpenAI-style structured output. Synalinks sets `response_format` with the schema and the LM is forced to emit valid JSON:

| Prefix | Notes |
|--------|-------|
| `openai/`, `azure/` | Strict mode; strips `description` from `$ref` properties |
| `ollama/`, `mistral/` | Strict mode |
| `gemini/`, `xai/`, `hosted_vllm/` | Strict mode |
| `deepseek/`, `together_ai/` | OpenAI-compatible APIs (treated as `openai/` payload) |
| `doubleword/` | Rewritten to `openai/` + default `api_base="https://api.doubleword.ai/v1"` |
| `vllm/` | Rewritten to `hosted_vllm/` (`HOSTED_VLLM_API_BASE` env or `http://localhost:8000`) |
| `ollama/` | Rewritten to `ollama_chat/` (better chat-template performance); default `api_base="http://localhost:11434"` |

### Tool-call path

Used for providers that lack native JSON schema or proxy heterogeneous backends:

| Prefix | Why |
|--------|-----|
| `groq/` | Better than json_schema in practice for many Groq models |
| `cohere/` | No native JSON schema |
| `openrouter/` | Heterogeneous backends — tool-calls are most reliable |
| `bedrock/` | Most Bedrock models lack native JSON schema |

A phantom tool named `structured_output` is created on the fly with `schema["properties"]` as its `parameters`; the model is forced to call it via `tool_choice`. On the response side, only `groq/` reads the JSON from `tool_calls[0].function.arguments` — the other tool-call providers return their JSON in `message.content` (LiteLLM normalizes it).

### Special path

| Prefix | Notes |
|--------|-------|
| `anthropic/` | Uses `response_format` — LiteLLM auto-routes to native `output_format` (Sonnet 4.5 / Opus 4.1) or tool-call (older models) |

### Prompt caching

For `gemini/` and `anthropic/`, the first message (system prompt) is annotated with `cache_control={"type": "ephemeral"}` automatically. This is what allows training to reuse the system instructions across many calls cheaply.

### Reasoning effort

If `reasoning_effort` is passed (and not `"none"` / `"disable"`) and `litellm.supports_reasoning(model)` is true, the kwarg is forwarded to LiteLLM. When the output schema also has a `thinking` field, that field is stripped from the schema before the call (the LM produces a native `reasoning_content` trace) and re-injected into the parsed JSON afterwards.

## Legacy Patches (Pre-0.8.006)

Older Synalinks versions raised `LM provider 'groq' not supported` and `LM provider 'openrouter' not supported`. The legacy patches in `scripts/groq_patch_legacy.py` and `scripts/openrouter_patch_legacy.py` monkey-patched `LanguageModel.__call__` with a provider-aware dispatcher.

**On 0.8.006+, none of this is needed.** The legacy scripts are kept for users on older Synalinks who can't upgrade.

If you're patching, only one patch can be active per process — copy the relevant branches into a single dispatcher.

## OpenRouter Embeddings

LiteLLM has no embeddings support for OpenRouter. `OpenRouterEmbeddingModel` makes direct HTTPS calls to `https://openrouter.ai/api/v1/embeddings`.

Important: the wrapper coerces inputs to strings. This is a workaround for OMEGA, where `tree.flatten()` over trainable variables can yield non-string leaves (numbers, dicts) that the embedding endpoint would reject.

```python
async def __call__(self, texts: List[str]):
    texts = [str(t) for t in texts]   # OMEGA-friendly coercion
    ...
```

Returns `{"embeddings": [[...], [...], ...]}`.

## LMStudio / vLLM Setup Details

```python
litellm.register_model({
    "openai/<model_name>": {
        "max_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",
        "mode": "chat",
    }
})
```

Without registration, LiteLLM's cost tracking returns `None`, which Synalinks then attempts to use in arithmetic and crashes.

### Why `openai/` prefix for local models

Local servers like LMStudio implement the OpenAI chat completions API. Using the `openai/` prefix routes them through Synalinks's OpenAI structured-output path — which is what they actually support. The `api_base` override redirects calls to `localhost`.

## Debugging

```python
synalinks.enable_logging()  # see request/response payloads
```

Common breakage points:

- **`response.choices[0].message.content` is None** — provider returned a tool call, not content. For `groq/` the dispatcher extracts from `tool_calls[0].function.arguments`; if you see this for another provider, LiteLLM did not normalize the tool-call into `content` (upgrade `litellm`).
- **`json.JSONDecodeError`** — provider returned text that wasn't JSON. Check that the schema was correctly set in `response_format`.
- **`extra_body` parameter ignored** — for OpenRouter routing, ensure LiteLLM forwards `extra_body` (it does in current versions).

## See Also

- **synalinks-core** — `LanguageModel`, `EmbeddingModel`
- **synalinks-optimizers** — OpenRouter embedding model for OMEGA
