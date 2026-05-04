---
name: synalinks-knowledge
description: Use when working with Synalinks KnowledgeBase (DuckDB-backed), EmbedKnowledge, UpdateKnowledge, RetrieveKnowledge, StampKnowledge, RAG pipelines, hybrid / fulltext / similarity search, default-EmbeddingModel configuration, or document extraction-and-storage flows.
---

# Synalinks Knowledge & RAG

Build retrieval-augmented programs over a unified DuckDB knowledge base. Supports BM25 full-text, vector similarity, and hybrid (RRF) search.

## Quick Start

```python
import synalinks
import asyncio

class Document(synalinks.DataModel):
    id: str = synalinks.Field(description="Document ID")
    title: str = synalinks.Field(description="Document title")
    content: str = synalinks.Field(description="Document content")

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="Answer based on retrieved context")

async def main():
    lm = synalinks.LanguageModel(model="openai/gpt-4o-mini")
    em = synalinks.EmbeddingModel(model="openai/text-embedding-3-small")

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://./docs.db",
        data_models=[Document],
        embedding_model=em,
        metric="cosine",
    )

    inputs = synalinks.Input(data_model=Query)
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
        instructions="Answer based on the retrieved context only. If irrelevant, say you don't know.",
    )(context)

    rag = synalinks.Program(inputs=inputs, outputs=answer, name="rag_qa")
    result = await rag(Query(query="What is Python?"))
    print(result.prettify_json())

asyncio.run(main())
```

## KnowledgeBase

```python
knowledge_base = synalinks.KnowledgeBase(
    uri="duckdb://./my_database.db",     # or duckdb://:memory:
    data_models=[Document, Invoice],     # one table per DataModel
    embedding_model=em,                  # optional, required for similarity/hybrid
    metric="cosine",                     # "cosine" | "l2seq" | "ip"
    wipe_on_start=False,                 # clear DB on init
)
```

**The first field of each DataModel is the primary key.**

## Knowledge Modules

### EmbedKnowledge

Generate embeddings for a DataModel so it can be searched by similarity.

```python
embedded = await synalinks.EmbedKnowledge(
    embedding_model=em,
    in_mask=["content"],   # keep only fields to embed
    # out_mask=["id"],     # OR exclude fields
)(inputs)
```

After masking, exactly one field should remain — the field that gets embedded.

### UpdateKnowledge

Upsert a record into the knowledge base. Uses the DataModel's first field as the primary key.

```python
stored = await synalinks.UpdateKnowledge(
    knowledge_base=knowledge_base,
)(extracted_data)
```

### RetrieveKnowledge

LM-driven retrieval — the LM generates a search query, then the knowledge base is queried.

```python
results = await synalinks.RetrieveKnowledge(
    knowledge_base=knowledge_base,
    language_model=lm,
    search_type="hybrid",   # "similarity" | "fulltext" | "hybrid"
    k=10,
    return_inputs=True,     # forward inputs alongside retrieved context
    return_query=True,      # include the LM-generated query in output
)(inputs)
```

## Search Types

| Type | Backend | When to use |
|------|---------|-------------|
| `fulltext` | DuckDB BM25 | Exact terms, codes, IDs, named entities |
| `similarity` | Vector (cosine/l2seq/ip) | Semantic matches, paraphrases |
| `hybrid` | Reciprocal Rank Fusion of both | Default — best general-purpose |

## Direct Search Methods

```python
# Full-text (BM25)
results = await knowledge_base.fulltext_search("query", k=10)

# Vector similarity
results = await knowledge_base.similarity_search("query", k=10)

# Hybrid (RRF)
results = await knowledge_base.hybrid_search("query", k=10, k_rank=60)

# Lookup by primary key
record = await knowledge_base.get("id_value")

# Paginated scan
records = await knowledge_base.getall(
    Document.to_symbolic_data_model(),
    limit=50,
    offset=0,
)

# Raw SQL (params is a list bound to ? placeholders, in order)
results = await knowledge_base.query(
    "SELECT * FROM Invoice WHERE total > ?",
    params=[100.0],
)
```

## Extraction → Storage Pipeline

Pipe a Generator into UpdateKnowledge to extract structured data and persist it:

```python
class DocumentText(synalinks.DataModel):
    text: str = synalinks.Field(description="Raw document text")

inputs = synalinks.Input(data_model=DocumentText)

extracted = await synalinks.Generator(
    data_model=Invoice,
    language_model=lm,
    instructions="Extract invoice information from the document.",
)(inputs)

stored = await synalinks.UpdateKnowledge(knowledge_base=knowledge_base)(extracted)

ingest = synalinks.Program(inputs=inputs, outputs=stored, name="invoice_ingest")
```

## Default EmbeddingModel

When `embedding_model=None` is passed to `EmbedKnowledge` / `RetrieveKnowledge` / `KnowledgeBase` (or `ops.embedding`), the framework resolves the default at call time:

```python
synalinks.set_default_embedding_model("openai/text-embedding-3-small")
# String identifiers persist into the on-disk config; instances do not.

# Later, anywhere in the program:
em = synalinks.default_embedding_model()  # returns the configured instance, or None
```

`EmbeddingModel` accepts `**default_kwargs` forwarded to every call, and `fallback=` accepts a string, dict, or `EmbeddingModel` instance:

```python
em = synalinks.EmbeddingModel(
    model="openai/text-embedding-3-small",
    dimensions=512,                          # forwarded to litellm.aembedding
    fallback="ollama/mxbai-embed-large",     # str / dict / EmbeddingModel
)
```

## Keyword-only Arguments

The knowledge modules (`EmbedKnowledge`, `UpdateKnowledge`, `RetrieveKnowledge`, `StampKnowledge`) all use keyword-only constructors (`def __init__(self, *, ...)`). Always pass arguments by name:

```python
# Correct
synalinks.UpdateKnowledge(knowledge_base=kb)

# Wrong — TypeError
synalinks.UpdateKnowledge(kb)
```

## Best Practices

1. **Specific field descriptions** — they shape what the LLM extracts and how Retrieval generates queries
2. **First field = primary key** — design DataModels accordingly
3. **Use hybrid search by default** — falls back to whichever signal is stronger
4. **Batch ingestion** with `program.predict(...)` — see synalinks-training
5. **Combine with agents** — pass `tools = [synalinks.Tool(knowledge_base.fulltext_search), ...]` to a FunctionCallingAgent for tool-driven retrieval (see synalinks-agents)

## References

- **references/knowledge-base.md** — Complete API, all search methods, full RAG example

## See Also

- **synalinks-core** — DataModel, EmbeddingModel basics
- **synalinks-modules** — Generator (used as the LM half of every RAG pipeline)
- **synalinks-agents** — Use a KnowledgeBase as a tool inside an agent
- **synalinks-providers** — OpenRouter embedding model wrapper for non-LiteLLM-supported embedders
