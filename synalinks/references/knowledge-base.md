# Synalinks Knowledge Base and RAG

## Overview

Synalinks provides a unified knowledge base system using DuckDB for storing and retrieving structured data. It supports full-text search (BM25) and vector similarity search for building RAG (Retrieval-Augmented Generation) applications.

## Knowledge Base Setup

### KnowledgeBase Class

```python
import synalinks

class Document(synalinks.DataModel):
    id: str = synalinks.Field(description="Document ID")
    title: str = synalinks.Field(description="Document title")
    content: str = synalinks.Field(description="Document content")

embedding_model = synalinks.EmbeddingModel(model="openai/text-embedding-3-small")

knowledge_base = synalinks.KnowledgeBase(
    uri="duckdb://my_database.db",
    data_models=[Document],
    embedding_model=embedding_model,
    metric="cosine",
    wipe_on_start=False,
)
```

### Configuration Options

- **uri**: DuckDB connection string (`duckdb://./path/to/db.db` or `duckdb://:memory:`)
- **data_models**: List of DataModel classes to create tables for
- **embedding_model**: Optional EmbeddingModel for vector similarity search
- **metric**: Distance metric ("cosine", "l2seq", "ip")
- **wipe_on_start**: Whether to clear database on initialization

---

## Data Models

Define your data models using `synalinks.DataModel`. The first field is used as the primary key.

```python
class Invoice(synalinks.DataModel):
    invoice_number: str = synalinks.Field(description="Invoice number")
    vendor: str = synalinks.Field(description="Vendor name")
    total: float = synalinks.Field(description="Total amount")
    description: str = synalinks.Field(description="Description of items")

class Customer(synalinks.DataModel):
    customer_id: str = synalinks.Field(description="Customer ID")
    name: str = synalinks.Field(description="Customer name")
    email: str = synalinks.Field(description="Email address")

knowledge_base = synalinks.KnowledgeBase(
    uri="duckdb://./business.db",
    data_models=[Invoice, Customer],
    embedding_model=embedding_model,
)
```

---

## Knowledge Modules

### EmbedKnowledge

Generate embeddings for data models to enable similarity search.

```python
inputs = synalinks.Input(data_model=Document)

embedded = await synalinks.EmbedKnowledge(
    embedding_model=embedding_model,
    in_mask=["content"],
)(inputs)
```

**Parameters:**
- `embedding_model`: The embedding model to use
- `in_mask`: Fields to include for embedding (keep only these)
- `out_mask`: Fields to exclude from embedding (remove these)

**Note:** Each data model should have exactly one field for embedding after masking.

### UpdateKnowledge

Store data models in the knowledge base.

```python
stored = await synalinks.UpdateKnowledge(
    knowledge_base=knowledge_base,
)(extracted_data)
```

Uses the first field as the primary key for upsert operations.

### RetrieveKnowledge

Retrieve relevant records using LM-generated search queries.

```python
results = await synalinks.RetrieveKnowledge(
    knowledge_base=knowledge_base,
    language_model=language_model,
    search_type="hybrid",
    k=10,
    return_inputs=True,
    return_query=True,
)(query_input)
```

**Search Types:**
- `"similarity"`: Vector-based semantic search
- `"fulltext"`: BM25-based full-text search
- `"hybrid"`: Combines both using Reciprocal Rank Fusion (default)

---

## Direct Search Methods

### Full-Text Search

```python
results = await knowledge_base.fulltext_search(
    "search query",
    k=10,
)
```

### Similarity Search

```python
results = await knowledge_base.similarity_search(
    "semantic query",
    k=10,
)
```

### Hybrid Search

```python
results = await knowledge_base.hybrid_search(
    "search query",
    k=10,
    k_rank=60,
)
```

### Get by ID

```python
record = await knowledge_base.get("id_value")
```

### Get All Records

```python
records = await knowledge_base.getall(
    Document.to_symbolic_data_model(),
    limit=50,
    offset=0,
)
```

### Raw SQL Query

```python
results = await knowledge_base.query(
    "SELECT * FROM Invoice WHERE total > ?",
    params={"1": 100.0},
)
```

---

## RAG Pipeline

### Simple RAG

```python
class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="Answer based on retrieved context")

async def create_rag_program():
    language_model = synalinks.LanguageModel(model="openai/gpt-4.1-mini")
    embedding_model = synalinks.EmbeddingModel(model="openai/text-embedding-3-small")

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://./documents.db",
        data_models=[Document],
        embedding_model=embedding_model,
    )

    inputs = synalinks.Input(data_model=Query)

    context = await synalinks.RetrieveKnowledge(
        knowledge_base=knowledge_base,
        language_model=language_model,
        search_type="hybrid",
        k=5,
        return_inputs=True,
    )(inputs)

    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
        instructions="Answer based on the retrieved context. If context is not relevant, say you don't know.",
    )(context)

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_rag",
    )

program = await create_rag_program()
result = await program(Query(query="What is the capital of France?"))
```

---

## Knowledge Extraction

### Extracting Structured Data

```python
class DocumentText(synalinks.DataModel):
    text: str = synalinks.Field(description="Raw document text")

class ExtractedInfo(synalinks.DataModel):
    title: str = synalinks.Field(description="Document title")
    summary: str = synalinks.Field(description="Brief summary")
    key_points: list = synalinks.Field(description="Key points from the document")

inputs = synalinks.Input(data_model=DocumentText)

extracted = await synalinks.Generator(
    data_model=ExtractedInfo,
    language_model=language_model,
    instructions="Extract the title, summary, and key points from the document.",
)(inputs)
```

### Extraction and Storage Pipeline

```python
inputs = synalinks.Input(data_model=DocumentText)

extracted = await synalinks.Generator(
    data_model=Invoice,
    language_model=language_model,
    instructions="Extract invoice information from the document.",
)(inputs)

stored = await synalinks.UpdateKnowledge(
    knowledge_base=knowledge_base,
)(extracted)

program = synalinks.Program(
    inputs=inputs,
    outputs=stored,
    name="invoice_extraction",
)
```

---

## Best Practices

1. **Define clear data models** - Specific field descriptions improve extraction quality
2. **Use meaningful field names** - LLMs understand natural language field names
3. **Include description fields** - Add a `description` or `content` field for text search
4. **Use hybrid search** - Combines keyword matching and semantic similarity
5. **Batch processing** - Use `program.predict()` for processing multiple records
6. **First field as ID** - The first field in your DataModel is the primary key

---

## Complete Example

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
    answer: str = synalinks.Field(description="Answer")

async def main():
    language_model = synalinks.LanguageModel(model="openai/gpt-4.1-mini")
    embedding_model = synalinks.EmbeddingModel(model="openai/text-embedding-3-small")

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://./docs.db",
        data_models=[Document],
        embedding_model=embedding_model,
        wipe_on_start=True,
    )

    # Store some documents
    docs = [
        Document(id="1", title="Python Basics", content="Python is a programming language..."),
        Document(id="2", title="Machine Learning", content="ML is a subset of AI..."),
    ]
    for doc in docs:
        await knowledge_base.update(doc.to_json_data_model())

    # Build RAG pipeline
    inputs = synalinks.Input(data_model=Query)

    context = await synalinks.RetrieveKnowledge(
        knowledge_base=knowledge_base,
        language_model=language_model,
        search_type="hybrid",
        k=3,
        return_inputs=True,
    )(inputs)

    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
        instructions="Answer using retrieved context only.",
    )(context)

    rag = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="rag_qa",
    )

    synalinks.utils.plot_program(rag, to_folder=".")

    result = await rag(Query(query="What is Python?"))
    print(result.prettify_json())

asyncio.run(main())
```
