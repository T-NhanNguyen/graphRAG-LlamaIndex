"""
Fusion Retrieval Engine - Hybrid BM25 + Vector search with RRF scoring.
"""
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from graphrag_config import settings
from duckdb_store import DuckDBStore
from bm25_index import BM25Scorer, BM25Indexer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieval result with scores and metadata."""
    chunkId: str
    text: str
    vectorScore: float = 0.0
    bm25Score: float = 0.0
    fusedScore: float = 0.0
    vectorRank: int = 0
    bm25Rank: int = 0
    embedding: Optional[List[float]] = None  # Added for deduplication


class FusionRetriever:
    """
    Hybrid retrieval combining BM25 and vector similarity.
    
    Supports three fusion strategies:
    1. Alpha blending: α*vector + (1-α)*bm25
    2. Reciprocal Rank Fusion (RRF): Σ 1/(k + rank_i)
    3. Combined: Normalize scores then apply alpha + RRF
    """
    
    def __init__(self, store: DuckDBStore, embeddingFunction=None):
        """
        Initialize fusion retriever.
        
        Args:
            store: DuckDB storage instance
            embeddingFunction: Callable that takes text -> embedding vector
        """
        self.store = store
        self.embeddingFunction = embeddingFunction
        self.bm25Scorer = BM25Scorer(store)
        
        # Configuration
        self.alpha = settings.FUSION_ALPHA
        self.rrfK = settings.RRF_K
        self.topK = settings.TOP_K
    
    def setEmbeddingFunction(self, func) -> None:
        """Set or update the embedding function."""
        self.embeddingFunction = func
    
    def _normalizeScores(self, scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0, 1] range."""
        # Filter out None values to prevent min/max errors
        valid_scores = [float(s) for s in scores if s is not None]
        
        if not valid_scores:
            return []
        
        minScore = min(valid_scores)
        maxScore = max(valid_scores)
        
        if maxScore == minScore:
            return [0.5] * len(valid_scores)
        
        return [(s - minScore) / (maxScore - minScore) for s in valid_scores]
    
    def _calculateRrfScore(self, ranks: List[int]) -> float:
        """
        Calculate Reciprocal Rank Fusion score.
        
        RRF = Σ 1/(k + rank_i) where k is a constant (default 60)
        Higher RRF = better ranking across methods.
        """
        return sum(1.0 / (self.rrfK + rank) for rank in ranks if rank > 0)
    
    def search(self, query: str, topK: Optional[int] = None, 
               useRrf: bool = True, useAlpha: bool = True) -> List[RetrievalResult]:
        """
        Perform hybrid fusion retrieval.
        
        Args:
            query: Natural language query
            topK: Number of results (default from settings)
            useRrf: Whether to apply RRF scoring
            useAlpha: Whether to apply alpha blending
            
        Returns:
            List of RetrievalResult sorted by fused score
        """
        topK = topK or self.topK
        
        if not self.embeddingFunction:
            logger.warning("No embedding function set, falling back to BM25 only")
            return self._bm25OnlySearch(query, topK)
        
        # Execute both searches (could parallelize for larger corpora)
        vectorResults = self._vectorSearch(query, topK * 3)  # Over-fetch for fusion
        bm25Results = self._bm25Search(query, topK * 3)
        
        # Merge results
        fusedResults = self._fuseResults(vectorResults, bm25Results, useRrf, useAlpha)
        
        # Sort by fused score and return top K
        fusedResults.sort(key=lambda r: r.fusedScore, reverse=True)
        topResults = fusedResults[:topK]
        
        # Enrich with embeddings for downstream deduplication
        self._enrichWithEmbeddings(topResults)
        
        return topResults
    
    def _enrichWithEmbeddings(self, results: List[RetrievalResult]) -> None:
        """
        Bulk fetch embeddings for results (in-place update).
        Required for downstream deduplication by the query engine.
        """
        if not results:
            return
        
        try:
            chunkIds = [r.chunkId for r in results]
            placeholders = ', '.join(['?'] * len(chunkIds))
            rows = self.store.connection.execute(f"""
                SELECT chunk_id, embedding
                FROM documents
                WHERE chunk_id IN ({placeholders})
            """, chunkIds).fetchall()
            
            embeddingMap = {row[0]: row[1] for row in rows if row[1] is not None}
            
            for result in results:
                result.embedding = embeddingMap.get(result.chunkId)
        except Exception as exc:
            logger.warning(f"Failed to enrich results with embeddings: {exc}")
    
    def _vectorSearch(self, query: str, k: int) -> List[Tuple[str, float, str]]:
        """Perform vector similarity search."""
        if not self.embeddingFunction:
            return []
        
        try:
            queryEmbedding = self.embeddingFunction(query)
            results = self.store.vectorSimilaritySearch(queryEmbedding, k)
            return results
        except Exception as exc:
            logger.error(f"Vector search failed: {exc}")
            return []
    
    def _bm25Search(self, query: str, k: int) -> List[Tuple[str, float, str]]:
        """Perform BM25 keyword search."""
        try:
            return self.bm25Scorer.search(query, k)
        except Exception as exc:
            logger.error(f"BM25 search failed: {exc}")
            return []
    
    def _bm25OnlySearch(self, query: str, topK: int) -> List[RetrievalResult]:
        """Fallback to BM25-only search."""
        bm25Results = self._bm25Search(query, topK)
        return [
            RetrievalResult(
                chunkId=chunkId,
                text=text,
                bm25Score=score,
                fusedScore=score,
                bm25Rank=i + 1
            )
            for i, (chunkId, score, text) in enumerate(bm25Results)
        ]
    
    def _fuseResults(self, vectorResults: List[Tuple[str, float, str]], 
                     bm25Results: List[Tuple[str, float, str]],
                     useRrf: bool, useAlpha: bool) -> List[RetrievalResult]:
        """
        Fuse vector and BM25 results into unified ranking.
        
        Handles documents that appear in only one of the result sets
        by assigning worst-case rank for the missing method.
        """
        # Build lookup maps
        vectorMap: Dict[str, Tuple[float, int, str]] = {}
        for rank, (chunkId, score, text) in enumerate(vectorResults, 1):
            vectorMap[chunkId] = (score, rank, text)
        
        bm25Map: Dict[str, Tuple[float, int, str]] = {}
        for rank, (chunkId, score, text) in enumerate(bm25Results, 1):
            bm25Map[chunkId] = (score, rank, text)
        
        # Combine all unique chunk IDs
        allChunkIds = set(vectorMap.keys()) | set(bm25Map.keys())
        
        if not allChunkIds:
            return []
        
        # Normalize scores
        vectorScores = [vectorMap[cid][0] for cid in allChunkIds if cid in vectorMap]
        bm25Scores = [bm25Map[cid][0] for cid in allChunkIds if cid in bm25Map]
        
        # Ensure we handle None scores by treating them as 0.0 during normalization
        vectorScores = [s if s is not None else 0.0 for s in vectorScores]
        bm25Scores = [s if s is not None else 0.0 for s in bm25Scores]
        
        normVectorScores = self._normalizeScores(vectorScores)
        normBm25Scores = self._normalizeScores(bm25Scores)
        
        # Create normalized score lookups
        vectorIdx = 0
        normVectorMap = {}
        for cid in allChunkIds:
            if cid in vectorMap:
                normVectorMap[cid] = normVectorScores[vectorIdx]
                vectorIdx += 1
        
        bm25Idx = 0
        normBm25Map = {}
        for cid in allChunkIds:
            if cid in bm25Map:
                normBm25Map[cid] = normBm25Scores[bm25Idx]
                bm25Idx += 1
        
        # Worst-case rank for missing results
        worstVectorRank = len(vectorResults) + 1
        worstBm25Rank = len(bm25Results) + 1
        
        # Build fused results
        results = []
        for chunkId in allChunkIds:
            # Get scores and ranks (with defaults for missing)
            if chunkId in vectorMap:
                vectorScore, vectorRank, text = vectorMap[chunkId]
                normVectorScore = normVectorMap.get(chunkId, 0.0)
            else:
                vectorScore, vectorRank = 0.0, worstVectorRank
                normVectorScore = 0.0
                text = bm25Map[chunkId][2]  # Get text from BM25 result
            
            if chunkId in bm25Map:
                bm25Score, bm25Rank, text = bm25Map[chunkId]
                normBm25Score = normBm25Map.get(chunkId, 0.0)
            else:
                bm25Score, bm25Rank = 0.0, worstBm25Rank
                normBm25Score = 0.0
            
            # Calculate fused score
            fusedScore = 0.0
            
            if useAlpha:
                # Alpha blending of normalized scores
                fusedScore += self.alpha * normVectorScore + (1 - self.alpha) * normBm25Score
            
            if useRrf:
                # Add RRF component
                rrfScore = self._calculateRrfScore([vectorRank, bm25Rank])
                if useAlpha:
                    fusedScore = (fusedScore + rrfScore) / 2  # Average with RRF
                else:
                    fusedScore = rrfScore
            
            results.append(RetrievalResult(
                chunkId=chunkId,
                text=text,
                vectorScore=vectorScore,
                bm25Score=bm25Score,
                fusedScore=fusedScore,
                vectorRank=vectorRank,
                bm25Rank=bm25Rank
            ))
        
        return results
    
    def searchWithEntityContext(self, query: str, topK: Optional[int] = None) -> Dict:
        """
        Fusion search with entity extraction from results.
        
        Returns chunks plus any entities mentioned in top results.
        This enables graph-enhanced retrieval for local search.
        """
        results = self.search(query, topK)
        
        if not results:
            return {"chunks": [], "entities": [], "relationships": []}
        
        # Get chunk IDs from results
        chunkIds = [r.chunkId for r in results]
        
        # Find entities linked to these chunks
        entities = []
        relationships = []
        
        allEntities = self.store.getAllEntities()
        for entity in allEntities:
            # Check if entity is sourced from any of the retrieved chunks
            if any(cid in (entity.sourceChunkIds or []) for cid in chunkIds):
                # Get relationships for this entity
                rels = self.store.getRelationshipsForEntity(entity.entityId)
                
                # This ensures cleaner retrieval results by focusing on the connected knowledge graph.
                if rels:
                    entities.append(entity)
                    relationships.extend(rels)
        
        # Deduplicate relationships
        seenRelIds = set()
        uniqueRels = []
        for rel in relationships:
            if rel.relationshipId not in seenRelIds:
                seenRelIds.add(rel.relationshipId)
                uniqueRels.append(rel)
        
        return {
            "chunks": results,
            "entities": entities,
            "relationships": uniqueRels
        }


def getFusionRetriever(store: DuckDBStore, embeddingFunction=None) -> FusionRetriever:
    """
    Factory function for fusion retriever.
    
    Args:
        store: DuckDB store instance
        embeddingFunction: Optional callable for embeddings
        
    Returns:
        Configured FusionRetriever instance
    """
    return FusionRetriever(store, embeddingFunction)
