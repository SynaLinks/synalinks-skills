# Synalinks Knowledge Base and RAG

## Overview

Synalinks provides native support for hybrid graph + vector databases, enabling Retrieval-Augmented Generation (RAG) and Knowledge-Augmented Generation (KAG) workflows.

## Knowledge Base Setup

### KnowledgeBase Class

```python
knowledge_base = synalinks.KnowledgeBase(
    uri="memgraph://localhost:7687",   # Database URI
    entity_models=[City, Country],      # Entity schemas
    relation_models=[IsCapitalOf],      # Relation schemas
    embedding_model=embedding_model,    # For vector search
    metric="cosine",                    # Similarity metric
    wipe_on_start=False,               # Clear DB on init
)
```

### Supported Databases

- **Memgraph** - `memgraph://localhost:7687`
- **Neo4j** - `neo4j://localhost:7687`

---

## Entity and Relation Models

### Entity Definition

```python
class City(synalinks.Entity):
    name: str = synalinks.Field(description="City name")
    population: int = synalinks.Field(description="Population count")
    country: str = synalinks.Field(description="Country name")

class Country(synalinks.Entity):
    name: str = synalinks.Field(description="Country name")
    capital: str = synalinks.Field(description="Capital city name")

class Place(synalinks.Entity):
    name: str = synalinks.Field(description="Place name")
    description: str = synalinks.Field(description="Place description")

class Event(synalinks.Entity):
    name: str = synalinks.Field(description="Event name")
    date: str = synalinks.Field(description="Event date")
```

### Relation Definition

```python
class IsCapitalOf(synalinks.Relation):
    """Indicates a city is the capital of a country."""
    pass

class IsLocatedIn(synalinks.Relation):
    """Indicates something is located in a place."""
    pass

class IsCityOf(synalinks.Relation):
    """Indicates a city belongs to a country."""
    pass

class TookPlaceIn(synalinks.Relation):
    """Indicates an event occurred at a location."""
    pass
```

---

## Retrieval Modules

### EntityRetriever

Retrieve entities by semantic similarity search.

```python
class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User query")

inputs = synalinks.Input(data_model=Query)

retrieval_result = await synalinks.EntityRetriever(
    entity_models=[City, Country, Place],
    knowledge_base=knowledge_base,
    language_model=lm,
    return_inputs=True,
    return_query=True,              # Include generated search query
    k=5,                            # Number of results
)(inputs)
```

**Output includes:**
- Original inputs
- Generated search query
- Retrieved entities with similarity scores

### TripletRetriever

Retrieve relationship triplets (entity-relation-entity).

```python
triplets = await synalinks.TripletRetriever(
    knowledge_base=knowledge_base,
    language_model=lm,
    return_inputs=True,
)(inputs)
```

**Output includes:**
- Retrieved triplets: (subject, relation, object)
- Relevance scores

### SimilaritySearch

Direct vector similarity search.

```python
results = await synalinks.SimilaritySearch(
    knowledge_base=knowledge_base,
    k=10,
)(query_embedding)
```

### TripletSearch

Direct triplet pattern matching.

```python
results = await synalinks.TripletSearch(
    knowledge_base=knowledge_base,
    pattern={"subject_type": "City", "relation": "IsCapitalOf"},
)(inputs)
```

---

## Knowledge Extraction

### Entity Extraction

```python
class ExtractedEntities(synalinks.DataModel):
    entities: List[synalinks.Entity] = synalinks.Field(
        description="Extracted entities from text"
    )

extractor_output = await synalinks.Generator(
    data_model=ExtractedEntities,
    language_model=lm,
    instructions=[
        "Extract all named entities from the text",
        "Include people, places, organizations, events",
    ],
)(text_input)
```

### Relation Extraction

```python
class ExtractedRelations(synalinks.DataModel):
    triplets: List[dict] = synalinks.Field(
        description="Extracted (subject, relation, object) triplets"
    )

relations = await synalinks.Generator(
    data_model=ExtractedRelations,
    language_model=lm,
)(entity_output)
```

### UpdateKnowledge

Add extracted knowledge to the database.

```python
await synalinks.UpdateKnowledge(
    entity_models=[City, Country, Place, Event],
    relation_models=[IsCapitalOf, IsLocatedIn],
    knowledge_base=knowledge_base,
)(extraction_output)
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
    lm = synalinks.LanguageModel(model="ollama/mistral")
    em = synalinks.EmbeddingModel(model="ollama/mxbai-embed-large")

    kb = synalinks.KnowledgeBase(
        uri="memgraph://localhost:7687",
        entity_models=[City, Country],
        relation_models=[IsCapitalOf],
        embedding_model=em,
    )

    inputs = synalinks.Input(data_model=Query)

    # Retrieve relevant context
    context = await synalinks.EntityRetriever(
        entity_models=[City, Country],
        knowledge_base=kb,
        language_model=lm,
        return_inputs=True,
        return_query=True,
    )(inputs)

    # Generate answer with context
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
        instructions=[
            "Answer based on the retrieved context",
            "If context is not relevant, say you don't know",
        ],
        return_inputs=True,
    )(context)

    return synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="simple_rag",
    )

program = await create_rag_program()
result = await program(Query(query="What is the capital of France?"))
```

### KAG (Knowledge-Augmented Generation)

```python
async def create_kag_program():
    inputs = synalinks.Input(data_model=Query)

    # Entity retrieval
    entities = await synalinks.EntityRetriever(
        entity_models=[City, Country, Place],
        knowledge_base=kb,
        language_model=lm,
    )(inputs)

    # Triplet retrieval for relationships
    triplets = await synalinks.TripletRetriever(
        knowledge_base=kb,
        language_model=lm,
    )(entities)

    # Combine and generate
    combined = entities & triplets
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
        instructions=[
            "Use entity facts and relationships to answer",
            "Cite specific facts from the knowledge graph",
        ],
    )(combined)

    return synalinks.Program(inputs=inputs, outputs=outputs)
```

---

## Multi-Stage Extraction

### One-Stage Extraction

Extract entities and relations in single pass.

```python
class KnowledgeExtraction(synalinks.DataModel):
    entities: List[dict] = synalinks.Field(description="Extracted entities")
    relations: List[dict] = synalinks.Field(description="Extracted relations")

extraction = await synalinks.Generator(
    data_model=KnowledgeExtraction,
    language_model=lm,
)(text_input)
```

### Two-Stage Extraction

Extract entities first, then relations.

```python
# Stage 1: Entities
entities = await synalinks.Generator(
    data_model=Entities,
    language_model=lm,
)(text_input)

# Stage 2: Relations (with entity context)
combined = text_input & entities
relations = await synalinks.Generator(
    data_model=Relations,
    language_model=lm,
)(combined)
```

### Multi-Stage Extraction

Progressive refinement.

```python
# Stage 1: Coarse extraction
coarse = await synalinks.Generator(
    data_model=CoarseEntities,
    language_model=lm,
)(text)

# Stage 2: Fine-grained entities
fine = await synalinks.Generator(
    data_model=FineEntities,
    language_model=lm,
)(coarse)

# Stage 3: Relation extraction
relations = await synalinks.Generator(
    data_model=Relations,
    language_model=lm,
)(fine)

# Stage 4: Validation
validated = await synalinks.SelfCritique(
    language_model=lm,
)(relations)
```

---

## Best Practices

1. **Define clear entity schemas** - Specific fields improve extraction quality
2. **Use meaningful relation names** - LLM understands natural language
3. **Include entity descriptions** - Guide the LLM during extraction
4. **Combine entity + triplet retrieval** - Get both facts and relationships
5. **Add instructions for RAG** - Tell LLM how to use retrieved context
6. **Validate extractions** - Use SelfCritique for quality control
7. **Batch population** - Use UpdateKnowledge for bulk inserts

---

## Complete Example

```python
import synalinks
import asyncio

class Document(synalinks.DataModel):
    text: str = synalinks.Field(description="Document text")

class City(synalinks.Entity):
    name: str = synalinks.Field(description="City name")

class Country(synalinks.Entity):
    name: str = synalinks.Field(description="Country name")

class IsCapitalOf(synalinks.Relation):
    pass

class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User query")

class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="Answer")

async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")
    em = synalinks.EmbeddingModel(model="ollama/mxbai-embed-large")

    kb = synalinks.KnowledgeBase(
        uri="memgraph://localhost:7687",
        entity_models=[City, Country],
        relation_models=[IsCapitalOf],
        embedding_model=em,
        wipe_on_start=True,
    )

    # Build RAG pipeline
    inputs = synalinks.Input(data_model=Query)
    context = await synalinks.EntityRetriever(
        entity_models=[City, Country],
        knowledge_base=kb,
        language_model=lm,
        return_inputs=True,
    )(inputs)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
        instructions=["Answer using retrieved context only"],
    )(context)

    rag = synalinks.Program(inputs=inputs, outputs=outputs, name="rag_qa")

    synalinks.utils.plot_program(rag, to_folder=".")

    result = await rag(Query(query="What is the capital of France?"))
    print(result.prettify_json())

asyncio.run(main())
```
