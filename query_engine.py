# GraphRAG Query Engine - Search interface supporting connection-based and thematic reasoning.
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from graphrag_config import settings, SearchType
from duckdb_store import DuckDBStore, Entity, Relationship, getStore
from fusion_retrieval import FusionRetriever, RetrievalResult, getFusionRetriever
from embedding_provider import getEmbeddings

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    # Structured query response for agent consumption.
    query: str
    searchType: str
    chunks: List[Dict]
    entities: List[Dict]
    relationships: List[Dict]
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Agent-optimized stats
    evidence: Optional[Dict[str, List[Dict]]] = None  # Direct vs extended relationships


class GraphRAGQueryEngine:
    # Query engine supporting connection-based (graph context), thematic (community), and keyword search.
    
    def __init__(self, store: Optional[DuckDBStore] = None):
        # Initialize query engine with storage and retrieval components.
        self.store = store or getStore()
        
        # Initialize embedding function for retrieval
        embeddings = getEmbeddings()
        self.retriever = getFusionRetriever(self.store, embeddings.embedQuery)
        
        logger.info("Query engine initialized")
    
    def _entityToDict(self, entity: Entity) -> Dict:
        # Convert Entity to serializable dict.
        return {
            "id": entity.entityId,
            "name": entity.name,
            "type": entity.entityType.value if hasattr(entity.entityType, 'value') else str(entity.entityType),
            "description": entity.description,
            "sourceChunks": entity.sourceChunkIds
        }
    
    def _relationshipToDict(self, rel: Relationship, entityNameMap: Dict[str, tuple[str, str]] = None) -> Dict:
        # Convert Relationship to serializable dict with optional entity name enrichment.
        result = {
            "id": rel.relationshipId,
            "source": rel.sourceEntityId,
            "target": rel.targetEntityId,
            "type": rel.relationshipType,
            "description": rel.description,
            "weight": rel.weight
        }
        
        # Enrich with entity names for zero-lookup reasoning chains
        if entityNameMap:
            sourceInfo = entityNameMap.get(rel.sourceEntityId)
            targetInfo = entityNameMap.get(rel.targetEntityId)
            if sourceInfo:
                result["sourceName"] = sourceInfo[0]
                result["sourceType"] = sourceInfo[1]
            if targetInfo:
                result["targetName"] = targetInfo[0]
                result["targetType"] = targetInfo[1]
        
        return result
    
    def _resultToDict(self, result: RetrievalResult) -> Dict:
        # Convert RetrievalResult to serializable dict.
        return {
            "chunkId": result.chunkId,
            "text": result.text,
            "vectorScore": round(result.vectorScore or 0.0, 4),
            "bm25Score": round(result.bm25Score or 0.0, 4),
            "fusedScore": round(result.fusedScore or 0.0, 4),
            "vectorRank": result.vectorRank,
            "bm25Rank": result.bm25Rank,
            "embedding": result.embedding  # Preserve for deduplication
        }
    
    def _deduplicateChunks(self, chunks: List[Dict]) -> tuple[List[Dict], int]:
        # Remove semantically similar chunks based on cosine similarity to maximize info density.
        if not settings.CHUNK_DEDUP_ENABLED or len(chunks) <= 1:
            return chunks, 0
        
        threshold = settings.CHUNK_DEDUP_SIMILARITY_THRESHOLD
        uniqueChunks = []
        duplicatesRemoved = 0
        
        for chunk in chunks:
            embeddingCurrent = chunk.get("embedding")
            if not embeddingCurrent:
                uniqueChunks.append(chunk)
                continue
            
            isDuplicate = False
            for uniqueChunk in uniqueChunks:
                embeddingExisting = uniqueChunk.get("embedding")
                if not embeddingExisting:
                    continue
                
                similarity = self._cosineSimilarity(embeddingCurrent, embeddingExisting)
                if similarity >= threshold:
                    isDuplicate = True
                    duplicatesRemoved += 1
                    break
            
            if not isDuplicate:
                uniqueChunks.append(chunk)
        
        return uniqueChunks, duplicatesRemoved
    
    def _cosineSimilarity(self, vec1: List[float], vec2: List[float]) -> float:
        # Calculate cosine similarity between two vectors.
        import numpy as np
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    
    def localSearch(self, query: str, topK: Optional[int] = None) -> QueryResult:
        # Local search with entity graph context. Returns QueryResult with evidence for agent reasoning.
        topK = topK or settings.TOP_K
        
        # Get fusion results with entity context
        results = self.retriever.searchWithEntityContext(query, topK)
        
        # Convert to serializable format
        chunks = [self._resultToDict(r) for r in results.get("chunks", [])]
        rawEntities = results.get("entities", [])
        rawRelationships = results.get("relationships", [])
        
        # Deduplicate chunks to maximize information density
        deduplicatedChunks, duplicatesRemoved = self._deduplicateChunks(chunks)
        
        # Build entity name lookup for relationship enrichment
        allEntityIds = set()
        for rel in rawRelationships:
            allEntityIds.add(rel.sourceEntityId)
            allEntityIds.add(rel.targetEntityId)
        
        entityNameMap = {}
        if allEntityIds:
            entityNameMap = self.store.getEntityNamesByIds(list(allEntityIds))
        
        # Enrich relationships with entity names
        enrichedRelationships = [
            self._relationshipToDict(r, entityNameMap) 
            for r in rawRelationships
        ]
        
        # Group relationships by relevance
        directRelationships, extendedRelationships = self._groupRelationshipsByChunk(
            enrichedRelationships,
            deduplicatedChunks,
            rawEntities
        )
        
        # Convert entities
        entities = [self._entityToDict(e) for e in rawEntities]
        
        # Build agent-optimized output
        return QueryResult(
            query=query,
            searchType=SearchType.FIND_CONNECTIONS.value,
            chunks=deduplicatedChunks,
            entities=entities,
            relationships=directRelationships + extendedRelationships,  # For backward compat
            metadata={
                "chunkCount": len(deduplicatedChunks),
                "duplicatesRemoved": duplicatesRemoved,
                "entityCount": len(entities),
                "directRelationshipCount": len(directRelationships),
                "extendedRelationshipCount": len(extendedRelationships)
            },
            evidence={
                "directRelationships": directRelationships,
                "extendedRelationships": extendedRelationships
            }
        )
    
    def _groupRelationshipsByChunk(self, relationships: List[Dict], 
                                    chunks: List[Dict], 
                                    entities: List) -> tuple[List[Dict], List[Dict]]:
        # Separate relationships into direct (from top chunks) vs extended (graph context).
        windowSize = settings.RELATIONSHIP_DIRECT_CHUNK_WINDOW
        topChunkIds = {chunk["chunkId"] for chunk in chunks[:windowSize]}
        
        # Build entity ID -> sourceChunkIds mapping
        entityToChunks = {e.entityId: set(e.sourceChunkIds or []) for e in entities}
        
        directRels = []
        extendedRels = []
        
        for rel in relationships:
            sourceChunks = entityToChunks.get(rel["source"], set())
            targetChunks = entityToChunks.get(rel["target"], set())
            
            # Direct if either entity appears in top chunks
            if sourceChunks & topChunkIds or targetChunks & topChunkIds:
                directRels.append(rel)
            else:
                extendedRels.append(rel)
        
        return directRels, extendedRels
    
    def globalSearch(self, query: str, topK: Optional[int] = None) -> QueryResult:
        # Global search for thematic queries using community summaries.
        logger.info("Performing global search via community summaries")
        # Stage 1.5 fallback: If no communities, use fusion
        # Future: self.store.searchCommunities(query)
        result = self.localSearch(query, topK)
        result.searchType = SearchType.EXPLORE_THEMATIC.value
        return result
    
    def fusionSearch(self, query: str, topK: Optional[int] = None,
                     alpha: Optional[float] = None) -> QueryResult:
        # Pure fusion retrieval without entity context.
        topK = topK or settings.TOP_K
        
        if alpha is not None:
            originalAlpha = self.retriever.alpha
            self.retriever.alpha = alpha
        
        results = self.retriever.search(query, topK)
        
        if alpha is not None:
            self.retriever.alpha = originalAlpha
        
        chunks = [self._resultToDict(r) for r in results]
        
        return QueryResult(
            query=query,
            searchType=SearchType.KEYWORD_SEARCH.value,
            chunks=chunks,
            entities=[],
            relationships=[]
        )
    
    def search(self, query: str, searchType: str = "find_connections", 
               topK: Optional[int] = None) -> QueryResult:
        # Unified search interface.
        searchType = searchType.lower()
        
        if searchType == "find_connections":
            return self.localSearch(query, topK)
        elif searchType == "explore_thematic":
            return self.globalSearch(query, topK)
        elif searchType == "keyword_search":
            return self.fusionSearch(query, topK)
        else:
            logger.warning(f"Unknown search type '{searchType}', using find_connections")
            return self.localSearch(query, topK)
    
    def getEntityNeighborhood(self, entityName: str, hops: int = 1) -> Dict:
        # Get entity and its neighborhood from the graph up to specified hops.
        entity = self.store.getEntityByName(entityName)
        
        if not entity:
            return {"error": f"Entity '{entityName}' not found"}
        
        visited = {entity.entityId}
        entities = [self._entityToDict(entity)]
        relationships = []
        
        # BFS traversal
        frontier = [entity.entityId]
        
        for _ in range(hops):
            nextFrontier = []
            
            for entityId in frontier:
                rels = self.store.getRelationshipsForEntity(entityId)
                
                for rel in rels:
                    relationships.append(self._relationshipToDict(rel))
                    
                    # Find connected entity
                    connectedId = rel.targetEntityId if rel.sourceEntityId == entityId else rel.sourceEntityId
                    
                    if connectedId not in visited:
                        visited.add(connectedId)
                        nextFrontier.append(connectedId)
                        
                        # Lookup connected entity
                        for e in self.store.getAllEntities():
                            if e.entityId == connectedId:
                                entities.append(self._entityToDict(e))
                                break
            
            frontier = nextFrontier
        
        return {
            "centerEntity": entityName,
            "entities": entities,
            "relationships": relationships,
            "hops": hops
        }


def main():
    # CLI entry point for queries.
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="GraphRAG Query Engine")
    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument("--type", "-t", default="find_connections",
                       choices=["find_connections", "explore_thematic", "keyword_search"],
                       help="Search type")
    parser.add_argument("--topk", "-k", type=int, default=settings.TOP_K,
                       help="Number of results")
    parser.add_argument("--alpha", "-a", type=float,
                       help="Fusion alpha (0=BM25, 1=vector)")
    parser.add_argument("--entity", "-e", help="Get entity neighborhood")
    parser.add_argument("--db", default=settings.DUCKDB_PATH,
                       help="DuckDB path")
    parser.add_argument("--json", action="store_true", help="Output as agent-optimized JSON")
    parser.add_argument("--agent-tson", action="store_true", help="Output as agent-optimized TSON")
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = GraphRAGQueryEngine(getStore(args.db))
    
    # Entity lookup mode
    if args.entity:
        result = engine.getEntityNeighborhood(args.entity)
        if args.json or args.agent_tson:
            outputData = result
            if args.agent_tson:
                from json_to_tson import convertToTSON
                outputData["entities"] = convertToTSON(outputData["entities"])
                outputData["relationships"] = convertToTSON(outputData["relationships"])
            print(json.dumps(outputData, indent=2))
        else:
            print(f"\n=== Neighborhood: {args.entity} ===")
            print(f"Entities ({len(result['entities'])}):")
            for e in result["entities"]:
                print(f"  • {e['name']} ({e['entityType']})")
            print(f"\nRelationships ({len(result['relationships'])}):")
            for r in result["relationships"]:
                print(f"  • {r['sourceEntityId']} -> {r['relationshipType']} -> {r['targetEntityId']}")
    # Query mode
    elif args.query:
        # Run search
        result = engine.search(args.query, searchType=args.type, topK=args.topk)
    
        if args.json or args.agent_tson:
            outputData = asdict(result)
            
            # Remove internal embeddings from output
            if "chunks" in outputData:
                for c in outputData["chunks"]:
                    c.pop("embedding", None)
            
            # Apply evidence-aware limits to prevent bloating while preserving reasoning chains
            # Direct relationships: High-trust, grounded in retrieved chunks
            # Extended relationships: Speculative graph exploration
            entityLimit = args.topk
            directRelLimit = min(args.topk * 3, 20)      # Generous for evidence-backed rels
            extendedRelLimit = max(args.topk // 2, 2)    # Conservative for discovery hints
            
            if outputData.get("entities"):
                outputData["entities"] = outputData["entities"][:entityLimit]
            
            if outputData.get("evidence"):
                if outputData["evidence"].get("directRelationships"):
                    outputData["evidence"]["directRelationships"] = outputData["evidence"]["directRelationships"][:directRelLimit]
                if outputData["evidence"].get("extendedRelationships"):
                    outputData["evidence"]["extendedRelationships"] = outputData["evidence"]["extendedRelationships"][:extendedRelLimit]
            
            # Backward-compat flat relationships list (sum of both evidence types)
            if outputData.get("relationships"):
                outputData["relationships"] = outputData["relationships"][:directRelLimit + extendedRelLimit]
            
            # Sync metadata with final truncated counts
            if outputData.get("metadata"):
                outputData["metadata"]["entityCount"] = len(outputData.get("entities", []))
                outputData["metadata"]["relationshipCount"] = len(outputData.get("relationships", []))
                if outputData.get("evidence"):
                    outputData["metadata"]["directRelationshipCount"] = len(outputData["evidence"].get("directRelationships", []))
                    outputData["metadata"]["extendedRelationshipCount"] = len(outputData["evidence"].get("extendedRelationships", []))
            
            if args.agent_tson:
                from json_to_tson import convertToTSON
                # Convert the evidence lists
                if outputData.get("evidence") is not None:
                    for key in ["directRelationships", "extendedRelationships"]:
                        if outputData["evidence"].get(key):
                            outputData["evidence"][key] = convertToTSON(outputData["evidence"][key])
                # Convert top level lists
                for key in ["chunks", "entities", "relationships"]:
                    if key in outputData:
                        outputData[key] = convertToTSON(outputData[key])
            
            print(json.dumps(outputData, indent=2))
        else:
            print(f"\n=== Query: {result.query} ===")
            print(f"Search type: {result.searchType}")
            print(f"\n--- Top Chunks ---")
            for i, chunk in enumerate(result.chunks, 1):
                print(f"\n{i}. Score: {chunk['fusedScore']:.3f} (vec:{chunk['vectorScore']:.3f}, bm25:{chunk['bm25Score']:.3f})")
                print(f"   {chunk['text'][:200]}...")
            
            if result.entities:
                print(f"\n--- Entities ({len(result.entities)}) ---")
                # Showing only top-K to keep terminal readable
                for e in result.entities[:args.topk]:
                    print(f"  • {e['name']} ({e['type']})")
            
            if result.relationships:
                print(f"\n--- Relationships ({len(result.relationships)}) ---")
                # Showing only top-K*2 to keep terminal readable
                for r in result.relationships[:args.topk * 2]:
                    # Show entity names if available (agent-optimized output)
                    if 'sourceName' in r and 'targetName' in r:
                        print(f"  • {r['sourceName']} {r['type']} {r['targetName']}")
                        if r.get('description'):
                            print(f"    \"{r['description'][:80]}...\"")
                    else:
                        # Fallback for old format
                        print(f"  • {r['type']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
