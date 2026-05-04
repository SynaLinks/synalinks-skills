#!/usr/bin/env python3
"""OpenRouter embeddings for Synalinks.

LiteLLM does NOT support OpenRouter embeddings, so this module bypasses
LiteLLM entirely and calls OpenRouter's /embeddings endpoint directly.

The wrapper coerces all inputs to strings, which makes it OMEGA-compatible:
`tree.flatten()` over trainable variables can yield non-string leaves
(numbers, dicts, etc.) that the embeddings API would otherwise reject.

Usage:
    from openrouter_embeddings import OpenRouterEmbeddingModel

    em = OpenRouterEmbeddingModel(
        "qwen/qwen3-embedding-8b",
        provider={"only": ["nebius"], "allow_fallbacks": False},
    )
    result = await em(["Hello, world!"])
    # {"embeddings": [[0.02, 0.006, ...]]}
"""

import asyncio
import os
import warnings
from typing import Any, Dict, List, Optional

import httpx


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterEmbeddingModel:
    """OpenRouter embeddings via direct HTTP calls (LiteLLM does not support them).

    Args:
        model: OpenRouter model name (e.g. "qwen/qwen3-embedding-8b")
        api_key: API key. Defaults to OPENROUTER_API_KEY env var.
        provider: Optional OpenRouter provider routing.
        retry: Retries on failure (default 5).
        timeout: HTTP timeout in seconds (default 30).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[Dict[str, Any]] = None,
        retry: int = 5,
        timeout: float = 30.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.provider = provider
        self.retry = retry
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")

    async def __call__(self, texts: List[Any]) -> Optional[Dict[str, List[List[float]]]]:
        # OMEGA-friendly coercion — tree.flatten() may return non-strings.
        texts = [str(t) for t in texts]

        for _ in range(self.retry):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    payload: Dict[str, Any] = {"model": self.model, "input": texts}
                    if self.provider:
                        payload["provider"] = self.provider

                    response = await client.post(
                        f"{OPENROUTER_API_BASE}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        vectors = [item.get("embedding", []) for item in data.get("data", [])]
                        return {"embeddings": vectors}

                    warnings.warn(
                        f"OpenRouter embedding error ({response.status_code}): {response.text}"
                    )
            except Exception as e:
                warnings.warn(f"Error calling OpenRouter embeddings: {e}")
            await asyncio.sleep(1)
        return None


if __name__ == "__main__":
    async def main():
        em = OpenRouterEmbeddingModel(
            "qwen/qwen3-embedding-8b",
            provider={"only": ["nebius"], "allow_fallbacks": False},
        )
        result = await em(["Hello, world!", "Bonjour le monde"])
        if result:
            for i, vec in enumerate(result["embeddings"]):
                print(f"Text {i}: {len(vec)}-dim embedding")
        else:
            print("Failed to get embeddings")

    asyncio.run(main())
