import os
import json
import uuid
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

import duckdb
import numpy as np

from graphrag_config import settings, EntityType

# Pipeline status enum for tracking indexing progress
class PipelineStatus:
    PENDING = 'pending'
    CHUNKED = 'chunked'
    EMBEDDED = 'embedded'
    EXTRACTED = 'extracted'
    COMPLETE = 'complete'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SourceDocument:
    # Represents an ingested source file
    id: str
    sourcePath: str
    rawContent: str
    pipelineStatus: str = PipelineStatus.PENDING
    createdAt: Optional[str] = None


@dataclass
class DocumentChunk:
    # Unified text chunk with embedding.
    # Corresponds to the 'Documents' table in the proposed schema
    chunkId: str
    sourceDocumentId: str
    text: str
    index: int
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Entity:
    # Represents a knowledge graph entity node
    entityId: str
    name: str
    canonicalName: str
    entityType: EntityType
    description: str
    sourceDocumentIds: List[str]
    sourceChunkIds: List[str]


@dataclass
class Relationship:
    # Represents a relationship edge between two entities
    relationshipId: str
    sourceEntityId: str
    targetEntityId: str
    relationshipType: str
    description: str
    weight: float = 1.0


@dataclass
class CommunitySummary:
    # Represents a thematic summary of a Leiden community
    communityId: str
    level: int
    entityIds: List[str]
    summary: str
    summaryEmbedding: Optional[List[float]] = None


class DuckDBStore:
    """
    Persistent DuckDB storage manager for GraphRAG.
    
    Stores all data in a single .duckdb file without in-memory HNSW indices.
    Optimized for on-demand tool usage where 2-5s query time is acceptable
    to avoid 30s+ startup overhead from loading HNSW into memory.
    """
    
    # Schema version for migrations
    SCHEMA_VERSION = 1

    # Type Priority for Deduplication (Lower number = Higher Priority)
    TYPE_PRIORITY = {
        EntityType.CONCEPT: 1,
        EntityType.TECHNOLOGY: 2,
        EntityType.ORGANIZATION: 3,
        EntityType.PRODUCT: 4,
        EntityType.LOCATION: 5,
        EntityType.EVENT: 6,
        EntityType.PERSON: 7
    }
    
    def __init__(self, dbPath: Optional[str] = None):
        # Initialize DuckDB connection and ensure schema exists
        self.dbPath = dbPath or settings.DUCKDB_PATH
        
        # Ensure directory exists
        dbDir = os.path.dirname(self.dbPath)
        if dbDir and not os.path.exists(dbDir):
            os.makedirs(dbDir, exist_ok=True)
            logger.info(f"Created database directory: {dbDir}")
        
        self.connection = duckdb.connect(self.dbPath)
        self._initializeSchema()
        logger.info(f"DuckDB initialized at: {self.dbPath}")
    
    def _initializeSchema(self) -> None:
        # Create all tables with the redesigned GraphRAG schema
        
        # 1. Source Documents (Metadata about the original files)
        # pipeline_status tracks indexing progress: pending -> chunked -> embedded -> extracted -> complete
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS source_documents (
                id VARCHAR PRIMARY KEY,
                source_path VARCHAR NOT NULL,
                raw_content TEXT,
                pipeline_status VARCHAR DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. Documents/Chunks (Unified Text + Embedding)
        self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                chunk_id VARCHAR PRIMARY KEY,
                source_document_id VARCHAR NOT NULL,
                text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding FLOAT[{settings.EMBEDDING_DIMENSION}],
                metadata JSON,
                FOREIGN KEY (source_document_id) REFERENCES source_documents(id)
            )
        """)
        
        # 3. Entities
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                canonical_name VARCHAR,
                entity_type VARCHAR NOT NULL,
                description TEXT,
                source_document_ids VARCHAR[],
                source_chunk_ids VARCHAR[],
                UNIQUE(name)
            )
        """)
        
        # 4. Relationships
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id VARCHAR PRIMARY KEY,
                source_entity_id VARCHAR NOT NULL,
                target_entity_id VARCHAR NOT NULL,
                relationship_type VARCHAR NOT NULL,
                description TEXT,
                weight FLOAT DEFAULT 1.0
            )
        """)
        
        # 5. Community Summaries (Hierarchical abstractions)
        self.connection.execute(f"""
            CREATE TABLE IF NOT EXISTS community_summaries (
                community_id VARCHAR PRIMARY KEY,
                level INTEGER NOT NULL,
                entity_ids VARCHAR[],
                summary TEXT NOT NULL,
                summary_embedding FLOAT[{settings.EMBEDDING_DIMENSION}]
            )
        """)
        
        # 6. Sparse vectors (BM25 data per chunk) - Continued for hybrid search
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS sparse_vectors (
                chunk_id VARCHAR PRIMARY KEY,
                tokenized_terms VARCHAR[],
                term_frequencies JSON,
                doc_length INTEGER
            )
        """)
        
        # 7. BM25 Stats & Metadata
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS bm25_stats (
                term VARCHAR PRIMARY KEY,
                doc_frequency INTEGER DEFAULT 0
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS corpus_metadata (
                key VARCHAR PRIMARY KEY,
                value VARCHAR
            )
        """)
        
        # - Indexing -
        
        # Entity Name index for fast lookup
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        
        # Relationship bidirectional indexes
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_entity_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_entity_id)")
        
        # HNSW Indexing for Vector Search
        # Attempt to load VSS extension and create HNSW index
        try:
            self.connection.execute("INSTALL vss; LOAD vss;")
            # Enable experimental persistence for HNSW (required for most VSS versions)
            self.connection.execute("SET hnsw_enable_experimental_persistence = true;")
            
            # Create HNSW index on the unified documents table
            # Note: Index creation might fail if table is empty or extension has version mismatch
            # but we define the intent here.
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_vss 
                ON documents USING HNSW (embedding)
                WITH (metric = 'cosine')
            """)
            logger.info("VSS extension loaded and HNSW index ensured with experimental persistence")
        except Exception as e:
            logger.warning(f"Native VSS/HNSW indexing unavailable, falling back to brute-force SQL: {e}")
        
        logger.info("Database schema initialized")
    
    # - Source Document Operations -
    
    def insertSourceDocument(self, doc: SourceDocument) -> str:
        # Insert a source document and return its ID
        docId = doc.id or str(uuid.uuid4())
        try:
            self.connection.execute("""
                INSERT INTO source_documents (id, source_path, raw_content)
                VALUES (?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    source_path = EXCLUDED.source_path,
                    raw_content = EXCLUDED.raw_content
            """, [docId, doc.sourcePath, doc.rawContent])
            logger.info(f"Inserted source document: {doc.sourcePath}")
            return docId
        except Exception as exc:
            logger.error(f"Failed to insert source document {doc.sourcePath}: {exc}")
            raise
    
    def getSourceDocument(self, docId: str) -> Optional[SourceDocument]:
        # Retrieve a source document by ID
        result = self.connection.execute(
            "SELECT id, source_path, raw_content, pipeline_status, created_at FROM source_documents WHERE id = ?",
            [docId]
        ).fetchone()
        
        if result:
            return SourceDocument(
                id=result[0],
                sourcePath=result[1],
                rawContent=result[2],
                pipelineStatus=result[3],
                createdAt=str(result[4]) if result[4] else None
            )
        return None
    
    def getSourceDocumentByPath(self, sourcePath: str) -> Optional[SourceDocument]:
        # Check if a source document with this source path already exists
        result = self.connection.execute(
            "SELECT id, source_path, raw_content, pipeline_status, created_at FROM source_documents WHERE source_path = ?",
            [sourcePath]
        ).fetchone()
        
        if result:
            return SourceDocument(
                id=result[0],
                sourcePath=result[1],
                rawContent=result[2],
                pipelineStatus=result[3],
                createdAt=str(result[4]) if result[4] else None
            )
        return None
    
    def getSourceDocumentByFilename(self, filePath: str) -> Optional[SourceDocument]:
        """
        Match document by filename and parent folder (portable path matching).
        
        Used when mount paths change but the actual files are the same.
        Matches on: filename + immediate parent folder name.
        
        Example: /app/input/docs/report.md matches /app/data/docs/report.md
        
        Args:
            filePath: New file path to match against stored paths
            
        Returns:
            SourceDocument if a match is found, None otherwise
        """
        from pathlib import Path
        path = Path(filePath)
        filename = path.name
        parentFolder = path.parent.name if path.parent else ""
        
        # This prevents false matches when same filename exists in different folders
        result = self.connection.execute("""
            SELECT id, source_path, raw_content, pipeline_status, created_at 
            FROM source_documents 
            WHERE source_path LIKE ?
              AND source_path LIKE ?
            LIMIT 1
        """, [f"%/{parentFolder}/{filename}", f"%{filename}"]).fetchone()
        
        if result:
            logger.debug(f"Matched '{filePath}' to existing document '{result[1]}'")
            return SourceDocument(
                id=result[0],
                sourcePath=result[1],
                rawContent=result[2],
                pipelineStatus=result[3],
                createdAt=str(result[4]) if result[4] else None
            )
        return None
    
    def updateSourceDocumentPath(self, docId: str, newPath: str) -> None:
        # used after mount path migration
        self.connection.execute(
            "UPDATE source_documents SET source_path = ? WHERE id = ?",
            [newPath, docId]
        )
        logger.info(f"Updated document path to: {newPath}")
    
    def updatePipelineStatus(self, docId: str, status: str) -> None:
        self.connection.execute(
            "UPDATE source_documents SET pipeline_status = ? WHERE id = ?",
            [status, docId]
        )
        logger.debug(f"Updated pipeline status for {docId}: {status}")
    
    def getIncompleteDocuments(self) -> List[tuple]:
        # Find source documents that have not completed all pipeline stages.
        # List of (id, source_path, pipeline_status) tuples
        results = self.connection.execute("""
            SELECT id, source_path, pipeline_status
            FROM source_documents
            WHERE pipeline_status != 'complete'
            ORDER BY created_at
        """).fetchall()
        
        if results:
            logger.info(f"Found {len(results)} incomplete documents")
        return results
    
    def getDocumentsByStatus(self, status: str) -> List[tuple]:
        # Get all documents at a specific pipeline stage
        return self.connection.execute(
            "SELECT id, source_path FROM source_documents WHERE pipeline_status = ?",
            [status]
        ).fetchall()
    
    def clearAllTables(self, dropTables: bool = False) -> Dict[str, int]:
        # Clear all data from all tables. Returns count of deleted rows per table.
        # If dropTables is True, drops the tables instead of deleting rows.
        deletedCounts = {}
        
        # Dependency order for dropping (children before parents)
        tables = [
            'community_summaries',
            'relationships',
            'sparse_vectors',
            'bm25_stats',
            'documents',
            'entities', 
            'source_documents',
            'corpus_metadata'
        ]
        
        for table in tables:
            try:
                # Check if table exists before getting count
                exists = self.connection.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table}'").fetchone()[0] > 0
                
                if not exists:
                    deletedCounts[table] = 0
                    continue

                count = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if dropTables:
                    self.connection.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    logger.debug(f"Dropped table {table}")
                else:
                    self.connection.execute(f"DELETE FROM {table}")
                deletedCounts[table] = count
            except Exception as e:
                logger.warning(f"Could not clear table {table}: {e}")
                deletedCounts[table] = 0
        
        if dropTables:
            logger.info("Dropped all tables for schema recreation")
            self._initializeSchema()
        else:
            logger.info(f"Cleared all tables: {deletedCounts}")
            
        return deletedCounts
    
    def resetDatabase(self) -> Dict[str, int]:
        # Nuclear reset: Close connection, delete file, reconnect.
        # This guarantees a clean slate with current config dimensions
        deletedCounts = {}
        tables = ['community_summaries', 'relationships', 'sparse_vectors', 
                  'bm25_stats', 'documents', 'entities', 'source_documents', 'corpus_metadata']
        for table in tables:
            try:
                deletedCounts[table] = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except:
                deletedCounts[table] = 0
        
        # Close the connection
        self.connection.close()
        logger.info(f"Closed connection to {self.dbPath}")
        
        # Delete the file
        if os.path.exists(self.dbPath):
            os.remove(self.dbPath)
            logger.info(f"Deleted database file: {self.dbPath}")
        
        # Reconnect (this will recreate the file and schema with current config)
        self.connection = duckdb.connect(self.dbPath)
        self._initializeSchema()
        logger.info(f"Database reset complete. Reconnected to fresh database at {self.dbPath}")
        
        return deletedCounts
    
    # - Unified Chunk/Document Operations -
    
    def insertDocumentChunks(self, chunks: List[DocumentChunk]) -> int:
        # Batch insert chunks with embeddings. Returns count inserted
        if not chunks:
            return 0
        
        data = [
            (c.chunkId, c.sourceDocumentId, c.text, c.index, 
             c.embedding, json.dumps(c.metadata) if c.metadata else None)
            for c in chunks
        ]
        
        try:
            self.connection.executemany("""
                INSERT INTO documents (chunk_id, source_document_id, text, chunk_index, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    text = EXCLUDED.text,
                    chunk_index = EXCLUDED.chunk_index,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
            """, data)
            logger.info(f"Inserted {len(data)} unified chunks")
            return len(chunks)
        except Exception as exc:
            logger.error(f"Failed to insert chunks: {exc}")
            raise
            
    def updateEmbeddings(self, chunkIds: List[str], embeddings: List[List[float]]) -> int:
        # Update only the embeddings for a set of chunks
        if not chunkIds or not embeddings:
            return 0
        data = [(emb, cid) for cid, emb in zip(chunkIds, embeddings)]
        try:
            self.connection.executemany(
                "UPDATE documents SET embedding = ? WHERE chunk_id = ?",
                data
            )
            logger.info(f"Updated {len(chunkIds)} embeddings")
            return len(chunkIds)
        except Exception as exc:
            logger.error(f"Failed to update embeddings: {exc}")
            raise
    
    def getChunksForSourceDocument(self, sourceDocumentId: str) -> List[DocumentChunk]:
        # Get all chunks for a source document, ordered by index
        results = self.connection.execute("""
            SELECT chunk_id, source_document_id, text, chunk_index, embedding, metadata
            FROM documents
            WHERE source_document_id = ?
            ORDER BY chunk_index
        """, [sourceDocumentId]).fetchall()
        
        return [
            DocumentChunk(
                chunkId=r[0],
                sourceDocumentId=r[1],
                text=r[2],
                index=r[3],
                embedding=r[4],
                metadata=json.loads(r[5]) if r[5] else None
            )
            for r in results
        ]
    
    def getAllChunks(self) -> List[DocumentChunk]:
        # Get all chunks in the corpus
        results = self.connection.execute("""
            SELECT chunk_id, source_document_id, text, chunk_index, embedding, metadata
            FROM documents
            ORDER BY source_document_id, chunk_index
        """).fetchall()
        
        return [
            DocumentChunk(
                chunkId=r[0],
                sourceDocumentId=r[1],
                text=r[2],
                index=r[3],
                embedding=r[4],
                metadata=json.loads(r[5]) if r[5] else None
            )
            for r in results
        ]
    
    # - Entity Operations -
    
    def insertEntities(self, entities: List[Entity]) -> int:
        """
        Batch insert entities with conflict resolution.
        If an entity with the same name exists, we merge them based on type priority.
        Uses native DuckDB UPSERT (ON CONFLICT) for row-stability with Foreign Keys.
        """
        if not entities:
            return 0
        
        def _getPriority(entityType):
            if isinstance(entityType, str):
                try:
                    # Map common strings to EntityType
                    t_str = entityType.upper()
                    if t_str == 'CONCEPT': entityType = EntityType.CONCEPT
                    elif t_str == 'TECHNOLOGY': entityType = EntityType.TECHNOLOGY
                    elif t_str == 'ORGANIZATION': entityType = EntityType.ORGANIZATION
                    elif t_str == 'PRODUCT': entityType = EntityType.PRODUCT
                    elif t_str == 'LOCATION': entityType = EntityType.LOCATION
                    elif t_str == 'EVENT': entityType = EntityType.EVENT
                    elif t_str == 'PERSON': entityType = EntityType.PERSON
                except:
                    return 99
            return self.TYPE_PRIORITY.get(entityType, 99)

        consolidated = {}
        for e in entities:
            name = e.name
            p = _getPriority(e.entityType)
            type_val = e.entityType.value if isinstance(e.entityType, EntityType) else e.entityType
            
            if name not in consolidated:
                consolidated[name] = {
                    'entityId': e.entityId,
                    'name': e.name,
                    'canonicalName': e.canonicalName,
                    'entityType': type_val,
                    'description': e.description or "",
                    'sourceDocumentIds': list(e.sourceDocumentIds),
                    'sourceChunkIds': list(e.sourceChunkIds),
                    'priority': p
                }
            else:
                existing = consolidated[name]
                # Keep highest priority type
                if p < existing['priority']:
                    existing['entityType'] = type_val
                    existing['priority'] = p
                
                # Merge descriptions if not redundant
                if e.description and e.description not in existing['description']:
                    if existing['description']:
                        existing['description'] += " | " + e.description
                    else:
                        existing['description'] = e.description
                
                # Merge IDs
                existing['sourceDocumentIds'] = list(set(existing['sourceDocumentIds'] + e.sourceDocumentIds))
                existing['sourceChunkIds'] = list(set(existing['sourceChunkIds'] + e.sourceChunkIds))

        formattedBatch = [
            (c['entityId'], c['name'], c['canonicalName'], c['entityType'], 
             c['description'], c['source_document_ids'] if 'source_document_ids' in c else c['sourceDocumentIds'], 
             c['source_chunk_ids'] if 'source_chunk_ids' in c else c['sourceChunkIds'])
            for c in consolidated.values()
        ]

        try:
            # Get existing entity names for efficient lookup
            existingNames = set()
            existingResult = self.connection.execute("SELECT LOWER(name) FROM entities").fetchall()
            existingNames = {r[0] for r in existingResult}
            
            # Separate into updates and inserts
            toUpdate = []
            toInsert = []
            for batch_item in formattedBatch:
                entity_name = batch_item[1]  # name is index 1
                if entity_name.lower() in existingNames:
                    toUpdate.append(batch_item)
                else:
                    toInsert.append(batch_item)
            
            # 1. UPDATE existing entities (by name) - merge metadata
            for item in toUpdate:
                entityId, name, canonicalName, entityType, description, sourceDocIds, sourceChunkIds = item
                self.connection.execute("""
                    UPDATE entities 
                    SET 
                        description = CASE 
                            WHEN description IS NULL THEN ?
                            WHEN ? IS NULL THEN description
                            WHEN instr(description, ?) = 0 THEN description || ' | ' || ?
                            ELSE description 
                        END,
                        source_document_ids = list_distinct(list_concat(source_document_ids, ?)),
                        source_chunk_ids = list_distinct(list_concat(source_chunk_ids, ?))
                    WHERE LOWER(name) = LOWER(?)
                """, [description, description, description, description, sourceDocIds, sourceChunkIds, name])
            
            if toUpdate:
                logger.info(f"Updated {len(toUpdate)} existing entities")
            
            # 2. INSERT new entities
            if toInsert:
                self.connection.executemany("""
                    INSERT INTO entities (entity_id, name, canonical_name, entity_type, description, source_document_ids, source_chunk_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, toInsert)
                logger.info(f"Inserted {len(toInsert)} new entities")
            
            logger.info(f"Processed {len(consolidated)} unique entities ({len(toUpdate)} updated, {len(toInsert)} new)")
            return len(consolidated)
        except Exception as exc:
            logger.error(f"Failed to insert entities: {exc}")
            raise

    
    def getEntityByName(self, name: str) -> Optional[Entity]:
        result = self.connection.execute("""
            SELECT entity_id, name, canonical_name, entity_type, description, source_document_ids, source_chunk_ids
            FROM entities
            WHERE LOWER(name) = LOWER(?) OR LOWER(canonical_name) = LOWER(?)
        """, [name, name]).fetchone()
        
        if result:
            return Entity(
                entityId=result[0],
                name=result[1],
                canonicalName=result[2],
                entityType=EntityType(result[3]) if result[3] in [e.value for e in EntityType] else result[3],
                description=result[4],
                sourceDocumentIds=result[5] or [],
                sourceChunkIds=result[6] or []
            )
        return None
    
    def getAllEntities(self) -> List[Entity]:
        # Get all entities in the knowledge graph
        results = self.connection.execute("""
            SELECT entity_id, name, canonical_name, entity_type, description, source_document_ids, source_chunk_ids
            FROM entities
        """).fetchall()
        
        return [
            Entity(
                entityId=r[0],
                name=r[1],
                canonicalName=r[2],
                entityType=EntityType(r[3]) if r[3] in [e.value for e in EntityType] else r[3],
                description=r[4],
                sourceDocumentIds=r[5] or [],
                sourceChunkIds=r[6] or []
            )
            for r in results
        ]
    
    def getEntityIdsByNames(self, names: List[str]) -> Dict[str, str]:
        """
        Bulk fetch entity IDs by their names (case-insensitive).
        
        Used to resolve relationship entity IDs after entity upsert/merge,
        since duplicate entities may be merged and original IDs discarded.
        
        Args:
            names: List of entity names to look up
            
        Returns:
            Dict mapping lowercase name -> persisted entity_id
        """
        if not names:
            return {}
        
        # Use a temp table for efficient bulk lookup
        try:
            self.connection.execute("DROP TABLE IF EXISTS temp_name_lookup")
            self.connection.execute("CREATE TEMPORARY TABLE temp_name_lookup (name VARCHAR)")
            self.connection.executemany(
                "INSERT INTO temp_name_lookup VALUES (?)", 
                [[n] for n in names]
            )
            
            results = self.connection.execute("""
                SELECT LOWER(ent.name), ent.entity_id
                FROM entities ent
                INNER JOIN temp_name_lookup lookup ON LOWER(ent.name) = LOWER(lookup.name)
            """).fetchall()
            
            self.connection.execute("DROP TABLE IF EXISTS temp_name_lookup")
            
            return {r[0]: r[1] for r in results}
        except Exception as exc:
            logger.error(f"Failed to bulk fetch entity IDs: {exc}")
            return {}
    
    def getEntityNamesByIds(self, entityIds: List[str]) -> Dict[str, tuple[str, str]]:
        """
        Bulk fetch entity names and types by their IDs.
        
        Used for relationship enrichment - allows agents to construct reasoning chains
        without additional lookups by adding entity names directly to relationship objects.
        
        Args:
            entityIds: List of entity IDs to look up
            
        Returns:
            Dict mapping entity_id -> (name, entity_type)
        """
        if not entityIds:
            return {}
        
        try:
            placeholders = ', '.join(['?'] * len(entityIds))
            results = self.connection.execute(f"""
                SELECT entity_id, name, entity_type
                FROM entities
                WHERE entity_id IN ({placeholders})
            """, entityIds).fetchall()
            
            return {r[0]: (r[1], r[2]) for r in results}
        except Exception as exc:
            logger.error(f"Failed to bulk fetch entity names: {exc}")
            return {}
    
    # - Relationship Operations -
    
    def insertRelationships(self, relationships: List[Relationship]) -> int:
        # Batch insert relationships
        if not relationships:
            return 0
        
        data = [
            (r.relationshipId, r.sourceEntityId, r.targetEntityId,
             r.relationshipType, r.description, r.weight)
            for r in relationships
        ]
        
        try:
            self.connection.executemany("""
                INSERT INTO relationships 
                (relationship_id, source_entity_id, target_entity_id, relationship_type, description, weight)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (relationship_id) DO UPDATE SET
                    relationship_type = EXCLUDED.relationship_type,
                    description = EXCLUDED.description,
                    weight = EXCLUDED.weight
            """, data)
            logger.info(f"Inserted {len(relationships)} relationships")
            return len(relationships)
        except Exception as exc:
            logger.error(f"Failed to insert relationships: {exc}")
            raise

    def insertCommunitySummaries(self, summaries: List[CommunitySummary]) -> int:
        # Batch insert community summaries
        if not summaries:
            return 0
            
        data = [
            (s.communityId, s.level, s.entityIds, s.summary, s.summaryEmbedding)
            for s in summaries
        ]
        
        try:
            self.connection.executemany("""
                INSERT INTO community_summaries (community_id, level, entity_ids, summary, summary_embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (community_id) DO UPDATE SET
                    level = EXCLUDED.level,
                    entity_ids = EXCLUDED.entity_ids,
                    summary = EXCLUDED.summary,
                    summary_embedding = EXCLUDED.summary_embedding
            """, data)
            logger.info(f"Inserted {len(summaries)} community summaries")
            return len(summaries)
        except Exception as exc:
            logger.error(f"Failed to insert community summaries: {exc}")
            raise
    
    def getRelationshipsForEntity(self, entityId: str) -> List[Relationship]:
        results = self.connection.execute("""
            SELECT relationship_id, source_entity_id, target_entity_id, 
                   relationship_type, description, weight
            FROM relationships
            WHERE source_entity_id = ? OR target_entity_id = ?
        """, [entityId, entityId]).fetchall()
        
        return [
            Relationship(
                relationshipId=r[0],
                sourceEntityId=r[1],
                targetEntityId=r[2],
                relationshipType=r[3],
                description=r[4],
                weight=r[5]
            )
            for r in results
        ]
    
    # - Vector Operations -
    
    def vectorSimilaritySearch(self, queryEmbedding: List[float], topK: int = 10) -> List[Tuple[str, float, str]]:
        """
        Perform vector similarity search.
        
        Optimization hierarchy:
        1. VSS HNSW (ANN search) - if extension loaded and index exists
        2. Native SQL Brute-Force (linear scan) - using list_cosine_similarity
        3. Python Fallback - if DuckDB array functions fail
        """
        if not queryEmbedding:
            return []
            
        # Try VSS-specific distance for HNSW utilization
        # Note: VSS uses distance (lower is better), so we convert it to similarity (1 - dist)
        try:
            # Check if vss extension is likely loaded by trying a light query
            results = self.connection.execute(f"""
                SELECT 
                    chunk_id, 
                    1.0 - array_cosine_distance(embedding, ?::FLOAT[{settings.EMBEDDING_DIMENSION}]) as similarity,
                    text
                FROM documents
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT ?
            """, [queryEmbedding, topK]).fetchall()
            return results
        except Exception:
            # Fallback to standard DuckDB native SQL (brute force but faster than Python)
            try:
                results = self.connection.execute(f"""
                    SELECT 
                        chunk_id, 
                        list_cosine_similarity(embedding, ?::FLOAT[{settings.EMBEDDING_DIMENSION}]) as similarity,
                        text
                    FROM documents
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT ?
                """, [queryEmbedding, topK]).fetchall()
                return results
            except Exception as exc:
                logger.error(f"SQL Similarity search failed, falling back to Python: {exc}")
                # Fallback to python-based calculation
                results = self.connection.execute("SELECT chunk_id, embedding, text FROM documents").fetchall()
                if not results: return []
                
                qVec = np.array(queryEmbedding)
                qNorm = np.linalg.norm(qVec)
                if qNorm == 0: return []
                
                similarities = []
                for cid, emb, txt in results:
                    if emb:
                        dVec = np.array(emb)
                        dNorm = np.linalg.norm(dVec)
                        if dNorm > 0:
                            sim = float(np.dot(qVec, dVec) / (qNorm * dNorm))
                            similarities.append((cid, sim, txt))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:topK]
    
    # - BM25/Sparse Vector Operations -
    
    def insertSparseVectors(self, chunkIds: List[str], tokenizedTerms: List[List[str]], 
                            termFrequencies: List[Dict[str, int]], docLengths: List[int]) -> int:
        # Batch insert sparse vector data for BM25
        if not chunkIds:
            return 0
        
        data = [
            (cid, terms, json.dumps(tf), dl)
            for cid, terms, tf, dl in zip(chunkIds, tokenizedTerms, termFrequencies, docLengths)
        ]
        
        try:
            self.connection.executemany("""
                INSERT INTO sparse_vectors (chunk_id, tokenized_terms, term_frequencies, doc_length)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    tokenized_terms = EXCLUDED.tokenized_terms,
                    term_frequencies = EXCLUDED.term_frequencies,
                    doc_length = EXCLUDED.doc_length
            """, data)
            logger.info(f"Inserted {len(data)} sparse vectors")
            return len(data)
        except Exception as exc:
            logger.error(f"Failed to insert sparse vectors: {exc}")
            raise
    
    def updateBm25Stats(self, termDocFrequencies: Dict[str, int]) -> None:
        # Update corpus-level term document frequencies for IDF
        for term, df in termDocFrequencies.items():
            self.connection.execute("""
                INSERT INTO bm25_stats (term, doc_frequency)
                VALUES (?, ?)
                ON CONFLICT (term) DO UPDATE SET
                    doc_frequency = bm25_stats.doc_frequency + EXCLUDED.doc_frequency
            """, [term, df])
        logger.info(f"Updated BM25 stats for {len(termDocFrequencies)} terms")
    
    def getBm25Stats(self) -> Tuple[int, float, Dict[str, int]]:
        # Get corpus statistics: (total_docs, avg_doc_length, term_doc_frequencies)
        totalDocs = self.connection.execute(
            "SELECT COUNT(*) FROM sparse_vectors"
        ).fetchone()[0]
        
        avgLength = self.connection.execute(
            "SELECT AVG(doc_length) FROM sparse_vectors"
        ).fetchone()[0] or 0.0
        
        termDfs = {}
        results = self.connection.execute(
            "SELECT term, doc_frequency FROM bm25_stats"
        ).fetchall()
        for term, df in results:
            termDfs[term] = df
        
        return totalDocs, float(avgLength), termDfs
    
    # - Community Summary Operations -
    
    def insertCommunitySummaries(self, summaries: List[CommunitySummary]) -> int:
        # Batch insert community summaries
        if not summaries:
            return 0
            
        data = [
            (s.communityId, s.level, s.entityIds, s.summary, s.summaryEmbedding)
            for s in summaries
        ]
        
        try:
            self.connection.executemany("""
                INSERT INTO community_summaries (community_id, level, entity_ids, summary, summary_embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (community_id) DO UPDATE SET
                    level = EXCLUDED.level,
                    entity_ids = EXCLUDED.entity_ids,
                    summary = EXCLUDED.summary,
                    summary_embedding = EXCLUDED.summary_embedding
            """, data)
            logger.info(f"Inserted {len(summaries)} community summaries")
            return len(summaries)
        except Exception as exc:
            logger.error(f"Failed to insert community summaries: {exc}")
            raise
            
    # - Corpus Metadata -
        
    def getStats(self) -> Dict[str, int]:
        # Get corpus statistics for debugging/observability
        stats = {}
        
        tables = ['source_documents', 'documents', 'entities', 'relationships', 
                  'community_summaries', 'sparse_vectors']
        
        for table in tables:
            try:
                count = self.connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                stats[table] = count
            except:
                stats[table] = -1
        
        return stats
    
    def deleteGarbageChunks(self, chunkIds: List[str]) -> int:
        # Prune garbage chunks from all tables.
        # Returns the number of chunks successfully removed
        if not chunkIds:
            return 0
            
        try:
            self.connection.executemany("DELETE FROM sparse_vectors WHERE chunk_id = ?", [[cid] for cid in chunkIds])
            self.connection.executemany("DELETE FROM documents WHERE chunk_id = ?", [[cid] for cid in chunkIds])
            
            logger.info(f"Pruned {len(chunkIds)} garbage chunks from database")
            return len(chunkIds)
        except Exception as exc:
            logger.error(f"Failed to prune garbage chunks: {exc}")
            logger.error(f"Failed to prune garbage chunks: {exc}")
            raise

    def getEmbeddingCentroid(self) -> Optional[List[float]]:
        # Calculate the average embedding (centroid) for all documents
        try:
            # Fetch all embeddings
            res = self.connection.execute("SELECT embedding FROM documents WHERE embedding IS NOT NULL").fetchall()
            if not res:
                return None
            
            embs = [np.array(r[0]) for r in res]
            centroid = np.mean(embs, axis=0)
            return centroid.tolist()
        except Exception as e:
            logger.error(f"Failed to calculate embedding centroid: {e}")
            return None

    def getEmbeddingCentroidForDocument(self, sourceDocumentId: str) -> Optional[List[float]]:
        """
        Calculate the semantic center of a single document's chunks.
        
        Args:
            sourceDocumentId: The document to calculate centroid for
            
        Returns:
            Centroid vector as list of floats, or None if no embeddings found
        """
        try:
            res = self.connection.execute("""
                SELECT embedding FROM documents 
                WHERE source_document_id = ? AND embedding IS NOT NULL
            """, [sourceDocumentId]).fetchall()
            
            if not res:
                return None
            
            embs = [np.array(r[0]) for r in res]
            centroid = np.mean(embs, axis=0)
            return centroid.tolist()
        except Exception as exc:
            logger.error(f"Failed to calculate per-document centroid for {sourceDocumentId}: {exc}")
            return None

    def getOutlierChunkIds(self, centroid: List[float], threshold: float = 0.85) -> List[str]:
        """Find chunk IDs that are distant from the provided centroid."""
        if not centroid:
            return []
            
        try:
            # Find those with similarity < threshold
            res = self.connection.execute(f"""
                SELECT chunk_id
                FROM documents
                WHERE embedding IS NOT NULL 
                AND list_cosine_similarity(embedding, ?::FLOAT[{settings.EMBEDDING_DIMENSION}]) < ?
            """, [centroid, threshold]).fetchall()
            
            return [r[0] for r in res]
        except Exception as e:
            logger.error(f"Failed to find outlier chunk ids: {e}")
            return []

    def getOutlierChunkIdsForDocument(self, sourceDocumentId: str, centroid: List[float], threshold: float) -> List[str]:
        """
        Find chunks within a document that are distant from that document's centroid.
        
        Args:
            sourceDocumentId: Document to search within
            centroid: The document's centroid embedding
            threshold: Minimum cosine similarity to be considered valid (e.g., 0.85)
            
        Returns:
            List of chunk IDs that are below the similarity threshold
        """
        if not centroid:
            return []
            
        try:
            res = self.connection.execute(f"""
                SELECT chunk_id
                FROM documents
                WHERE source_document_id = ?
                AND embedding IS NOT NULL 
                AND list_cosine_similarity(embedding, ?::FLOAT[{settings.EMBEDDING_DIMENSION}]) < ?
            """, [sourceDocumentId, centroid, threshold]).fetchall()
            
            return [r[0] for r in res]
        except Exception as exc:
            logger.error(f"Failed to find outliers for document {sourceDocumentId}: {exc}")
            return []

    def pruneStrandedEntities(self) -> int:
        """
        Delete all entities that have no incoming or outgoing relationships.
        
        Returns:
            Number of entities deleted
        """
        try:
            # Count stranded entities first
            countResult = self.connection.execute("""
                SELECT COUNT(*) FROM entities 
                WHERE entity_id NOT IN (
                    SELECT source_entity_id FROM relationships
                    UNION
                    SELECT target_entity_id FROM relationships
                )
            """).fetchone()[0]
            
            if countResult > 0:
                # Delete entities where entity_id is not found in source or target of relationships
                self.connection.execute("""
                    DELETE FROM entities 
                    WHERE entity_id NOT IN (
                        SELECT source_entity_id FROM relationships
                        UNION
                        SELECT target_entity_id FROM relationships
                    )
                """)
                logger.info(f"Pruned {countResult} stranded entities")
            
            return countResult
        except Exception as exc:
            logger.error(f"Failed to prune stranded entities: {exc}")
            return 0
    
    def getCorpusStats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the corpus.
        
        Returns:
            Dict with counts for documents, chunks, entities, relationships.
        """
        try:
            stats = {}
            
            result = self.connection.execute("SELECT COUNT(*) FROM source_documents").fetchone()
            stats['documentCount'] = result[0] if result else 0
            
            result = self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()
            stats['chunkCount'] = result[0] if result else 0
            
            result = self.connection.execute("SELECT COUNT(*) FROM entities").fetchone()
            stats['entityCount'] = result[0] if result else 0
            
            result = self.connection.execute("SELECT COUNT(*) FROM relationships").fetchone()
            stats['relationshipCount'] = result[0] if result else 0
            
            result = self.connection.execute("SELECT COUNT(*) FROM community_summaries").fetchone()
            stats['communityCount'] = result[0] if result else 0
            
            return stats
        except Exception as exc:
            logger.error(f"Failed to get corpus stats: {exc}")
            return {}

    def close(self) -> None:
        # Close the database connection
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

# Multi-instance store cache for supporting multiple databases
_storeInstances: Dict[str, DuckDBStore] = {}


def getStore(dbPath: Optional[str] = None, refresh: bool = False) -> DuckDBStore:
    """
    Get or create a DuckDB store instance for the given path.
    
    Implements multi-instance caching to support concurrent access to
    different databases without repeated initialization overhead.
    
    Args:
        dbPath: Path to .duckdb file. Uses settings.DUCKDB_PATH if None.
        refresh: If True, close existing connection and create new one.
    
    Returns:
        DuckDBStore instance for the specified database.
    """
    path = dbPath or settings.DUCKDB_PATH
    
    path = os.path.abspath(path)
    
    if refresh and path in _storeInstances:
        _storeInstances[path].close()
        del _storeInstances[path]
        logger.info(f"Refreshed store instance for: {path}")
    
    if path not in _storeInstances:
        _storeInstances[path] = DuckDBStore(path)
    
    return _storeInstances[path]


def closeAllStores() -> None:
    # Close all cached store instances. Useful for cleanup
    global _storeInstances
    for path, store in _storeInstances.items():
        try:
            store.close()
        except Exception as e:
            logger.warning(f"Error closing store {path}: {e}")
    _storeInstances = {}
    logger.info("Closed all store instances")
