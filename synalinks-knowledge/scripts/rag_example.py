#!/usr/bin/env python3
"""Minimal RAG pipeline using Synalinks KnowledgeBase.

Usage:
    uv run -- python scripts/rag_example.py

Demonstrates:
1. Defining a Document DataModel
2. Building a DuckDB KnowledgeBase with embeddings
3. Ingesting documents
4. RetrieveKnowledge + Generator for question answering
"""

import asyncio
import synalinks


class Document(synalinks.DataModel):
    """A document chunk."""
    id: str = synalinks.Field(description="Document ID")
    title: str = synalinks.Field(description="Document title")
    content: str = synalinks.Field(description="Document content")


class Query(synalinks.DataModel):
    query: str = synalinks.Field(description="User query")


class Answer(synalinks.DataModel):
    answer: str = synalinks.Field(description="Answer based on retrieved context")


async def main():
    lm = synalinks.LanguageModel(model="ollama/mistral")
    em = synalinks.EmbeddingModel(model="ollama/mxbai-embed-large")

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://./demo_docs.db",
        data_models=[Document],
        embedding_model=em,
        metric="cosine",
        wipe_on_start=True,
    )

    # Ingest a couple of documents directly
    docs = [
        Document(id="1", title="Python", content="Python is a high-level programming language."),
        Document(id="2", title="ML", content="Machine learning is a subset of AI."),
        Document(id="3", title="Synalinks", content="Synalinks is a Keras-inspired framework for neuro-symbolic LLM apps."),
    ]
    for doc in docs:
        await knowledge_base.update(doc.to_json_data_model())

    # Build the RAG pipeline
    inputs = synalinks.Input(data_model=Query)

    context = await synalinks.RetrieveKnowledge(
        knowledge_base=knowledge_base,
        language_model=lm,
        search_type="hybrid",
        k=3,
        return_inputs=True,
    )(inputs)

    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
        instructions="Answer using the retrieved context. If irrelevant, say you don't know.",
    )(context)

    rag = synalinks.Program(inputs=inputs, outputs=outputs, name="rag_qa")

    result = await rag(Query(query="What is Synalinks?"))
    print(result.prettify_json())


if __name__ == "__main__":
    asyncio.run(main())
