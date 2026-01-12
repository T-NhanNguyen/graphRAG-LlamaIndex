"""
Embedding Provider - GPU-accelerated embeddings via Docker Model Runner.

Adapted from rag-tool2 pattern with async batching for efficient corpus indexing.

Following coding framework guidelines:
- Batch APIs over single calls
- Contextual logging with progress
- Typed returns
"""
import asyncio
import logging
from typing import List, Optional

import httpx
import numpy as np

from graphrag_config import settings

logger = logging.getLogger(__name__)


class DockerModelRunnerEmbeddings:
    """
    GPU-accelerated embeddings via Docker Model Runner.
    
    Uses OpenAI-compatible /embeddings endpoint with parallel batching
    for efficient corpus processing.
    """
    
    def __init__(self, baseUrl: Optional[str] = None, model: Optional[str] = None, 
                 concurrency: int = None, batchSize: int = None):
        """
        Initialize embedding provider.
        
        Args:
            baseUrl: Docker Model Runner endpoint
            model: Embedding model name
            concurrency: Number of parallel requests
            batchSize: Texts per batch
        """
        self.baseUrl = baseUrl or settings.EMBEDDING_URL
        self.model = model or settings.EMBEDDING_MODEL
        self.concurrency = concurrency or settings.EMBEDDING_CONCURRENCY
        self.batchSize = batchSize or settings.EMBEDDING_BATCH_SIZE
        
        # Remove trailing slash if present
        if self.baseUrl.endswith("/"):
            self.baseUrl = self.baseUrl[:-1]
        
        logger.info(f"Embedding provider: {self.baseUrl} using {self.model}")
    
    async def _embedBatch(self, client: httpx.AsyncClient, batch: List[str]) -> List[List[float]]:
        """Execute single batch embedding request with retry logic."""
        endpoint = f"{self.baseUrl}/v1/embeddings"
        
        payload = {
            "model": self.model,
            "input": batch
        }
        
        maxRetries = 3
        backoffSeconds = 2.0
        
        for attempt in range(maxRetries):
            try:
                response = await client.post(endpoint, json=payload, timeout=120.0)
                response.raise_for_status()
                
                data = response.json()
                embeddings = [item["embedding"] for item in data.get("data", [])]
                return embeddings
                
            except Exception as exc:
                logger.warning(f"Batch embedding attempt {attempt + 1}/{maxRetries} failed: {exc}")
                
                if attempt < maxRetries - 1:
                    await asyncio.sleep(backoffSeconds * (2 ** attempt))
                else:
                    # All retries exhausted - fail loudly instead of returning zeros
                    logger.error(f"Batch embedding failed after {maxRetries} attempts - raising error")
                    raise RuntimeError(f"Embedding failed after {maxRetries} retries: {exc}")
    
    async def _embedAllAsync(self, texts: List[str]) -> List[List[float]]:
        """Orchestrate parallel embedding of all texts."""
        if not texts:
            return []
        
        # Split into batches
        batches = [texts[i:i + self.batchSize] for i in range(0, len(texts), self.batchSize)]
        allEmbeddings: List[List[float]] = [None] * len(batches)
        
        semaphore = asyncio.Semaphore(self.concurrency)
        
        async def processBatch(idx: int, batch: List[str]):
            async with semaphore:
                async with httpx.AsyncClient() as client:
                    result = await self._embedBatch(client, batch)
                    allEmbeddings[idx] = result
                    logger.info(f"Embedded batch {idx + 1}/{len(batches)} ({len(batch)} texts)")
        
        # Execute all batches
        tasks = [processBatch(i, batch) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)
        
        # Flatten results
        flattened = []
        for batchResult in allEmbeddings:
            if batchResult:
                flattened.extend(batchResult)
        
        return flattened
    
    def embedDocuments(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts with parallel batching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} texts in batches of {self.batchSize}")
        
        # Run async embedding
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self._embedAllAsync(texts))
    
    def embedQuery(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        results = self.embedDocuments([text])
        return results[0] if results else [0.0] * settings.EMBEDDING_DIMENSION
    
    def isAvailable(self) -> bool:
        """Check if embedding endpoint is reachable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.baseUrl}/v1/models")
                return response.status_code == 200
        except Exception:
            return False


def getEmbeddings() -> DockerModelRunnerEmbeddings:
    """Factory function for embedding provider."""
    return DockerModelRunnerEmbeddings()
