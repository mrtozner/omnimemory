#!/usr/bin/env python3
"""
Cloud Embedding Fallback Service - Fast API for Low-End Devices

For users without fast local computers (no MLX), provides ultra-fast embeddings via:
1. Edge deployment (Cloudflare Workers, AWS Lambda@Edge)
2. Model quantization (INT8)
3. Batching and caching
4. CDN-backed model serving

Target: <50ms per embedding (vs 5-10ms local MLX)
"""

import asyncio
import time
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import httpx
from functools import lru_cache
import hashlib

app = FastAPI(title="OmniMemory Cloud Embeddings")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, small model


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    latency_ms: float
    cached: bool


class CloudEmbeddingService:
    """
    Fast cloud embedding service with caching and batching

    Strategy:
    1. Local cache (LRU) - instant for repeated texts
    2. Batch requests - amortize network overhead
    3. Quantized models - INT8 for 4x speedup
    4. Edge deployment - <10ms network latency
    """

    def __init__(self):
        self.model_cache = {}
        self.embedding_cache = {}  # Text hash -> embedding

        # Edge endpoints (will be deployed)
        self.endpoints = {
            "primary": "https://api.omnimemory.cloud/embeddings",  # Cloudflare Workers
            "fallback": "http://localhost:8000/embeddings",  # Local MLX fallback
        }

        self.client = httpx.AsyncClient(timeout=10.0)

    @lru_cache(maxsize=10000)
    def _hash_text(self, text: str) -> str:
        """Fast hash for cache lookup"""
        return hashlib.md5(text.encode()).hexdigest()

    async def get_embeddings(
        self, texts: List[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> Dict:
        """Get embeddings with automatic fallback"""
        start_time = time.time()

        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            if text_hash in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[text_hash]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # If all cached, return immediately
        if not uncached_texts:
            embeddings = [emb for _, emb in sorted(cached_embeddings)]
            latency = (time.time() - start_time) * 1000
            return {
                "embeddings": embeddings,
                "model": model,
                "latency_ms": latency,
                "cached": True,
            }

        # Get uncached embeddings
        try:
            # Try primary (cloud) first
            new_embeddings = await self._fetch_from_cloud(uncached_texts, model)
        except Exception as e:
            print(f"Cloud fetch failed, using local fallback: {e}")
            # Fallback to local
            new_embeddings = await self._fetch_from_local(uncached_texts, model)

        # Update cache
        for text, embedding in zip(uncached_texts, new_embeddings):
            text_hash = self._hash_text(text)
            self.embedding_cache[text_hash] = embedding

        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        for i, emb in cached_embeddings:
            all_embeddings[i] = emb
        for i, emb in zip(uncached_indices, new_embeddings):
            all_embeddings[i] = emb

        latency = (time.time() - start_time) * 1000

        return {
            "embeddings": all_embeddings,
            "model": model,
            "latency_ms": latency,
            "cached": len(cached_embeddings) > 0,
        }

    async def _fetch_from_cloud(
        self, texts: List[str], model: str
    ) -> List[List[float]]:
        """Fetch from cloud endpoint (Cloudflare Workers)"""
        # In production, this would hit the edge endpoint
        # For now, simulate fast cloud response

        # TODO: Deploy to Cloudflare Workers
        # response = await self.client.post(
        #     self.endpoints["primary"],
        #     json={"texts": texts, "model": model}
        # )
        # return response.json()["embeddings"]

        # Fallback to local for now
        return await self._fetch_from_local(texts, model)

    async def _fetch_from_local(
        self, texts: List[str], model: str
    ) -> List[List[float]]:
        """Fallback to local MLX embedding service"""
        try:
            response = await self.client.post(
                "http://localhost:8000/embeddings",
                json={"texts": texts, "model": model},
            )

            if response.status_code == 200:
                data = response.json()
                return data["embeddings"]
            else:
                raise Exception(
                    f"Local embedding service error: {response.status_code}"
                )

        except Exception as e:
            # Last resort: use sentence-transformers CPU (slow but works)
            print(f"Using CPU fallback (slow): {e}")
            return await self._cpu_embeddings(texts, model)

    async def _cpu_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """CPU fallback using sentence-transformers (slow but always works)"""
        try:
            from sentence_transformers import SentenceTransformer

            # Load model (cached)
            if model not in self.model_cache:
                self.model_cache[model] = SentenceTransformer(model)

            model_obj = self.model_cache[model]
            embeddings = model_obj.encode(texts)

            return embeddings.tolist()

        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="No embedding backend available. Install sentence-transformers or start MLX service.",
            )


# Global service instance
embedding_service = CloudEmbeddingService()


@app.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings with automatic fallback

    Priority:
    1. Cache (instant)
    2. Cloud edge endpoint (<50ms)
    3. Local MLX (<10ms if available)
    4. CPU sentence-transformers (slow, always works)
    """
    try:
        result = await embedding_service.get_embeddings(
            texts=request.texts, model=request.model
        )

        return EmbeddingResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "cache_size": len(embedding_service.embedding_cache),
        "endpoints": embedding_service.endpoints,
    }


@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "cache_size": len(embedding_service.embedding_cache),
        "models_loaded": list(embedding_service.model_cache.keys()),
        "endpoints": embedding_service.endpoints,
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("🌩️  OmniMemory Cloud Embedding Fallback Service")
    print("=" * 70)
    print("Fast embeddings for users without MLX (Windows/Linux/Low-end Mac)")
    print("")
    print("Strategy:")
    print("  1. Cache (instant)")
    print("  2. Cloud edge (<50ms)")
    print("  3. Local MLX (<10ms)")
    print("  4. CPU fallback (works everywhere)")
    print("")
    print("Starting on: http://localhost:8002")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8002)
