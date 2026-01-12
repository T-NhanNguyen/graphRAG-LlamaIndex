"""
GraphRAG Indexing Pipeline - End-to-end document processing.

Orchestrates:
1. Markdown file ingestion
2. Text chunking
3. Embedding generation (parallel batched)
4. BM25 sparse indexing
5. Entity/relationship extraction (LLM)
6. DuckDB persistence

Following coding framework guidelines:
- Bulk operations for efficiency
- Progress logging for observability
- Configurable parameters
"""
import os
import time
import uuid
import logging
import glob
import json
import re
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from graphrag_config import settings
from duckdb_store import DuckDBStore, SourceDocument, DocumentChunk, Entity, Relationship, getStore, PipelineStatus
from bm25_index import BM25Indexer
from embedding_provider import DockerModelRunnerEmbeddings, getEmbeddings
from llm_client import LocalLLMClient, getLLMClient, getRelationshipClient
from entity_extractor import BaseEntityExtractor, ExtractorFactory
from garbage_filter import garbageFilter, garbageLogger

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class IndexingStats:
    """Statistics from indexing run."""
    documentsProcessed: int = 0
    documentsSkipped: int = 0
    chunksCreated: int = 0
    entitiesExtracted: int = 0
    relationshipsExtracted: int = 0
    embeddingsGenerated: int = 0
    bm25TokensIndexed: int = 0


class SemanticChunker:
    """
    Semantic chunker that RESPECTS Markdown section boundaries.
    
    Two-phase approach:
    1. Split on headers FIRST (sections are never merged across headers)
    2. Split oversized sections using paragraph/sentence boundaries
    """
    
    def __init__(self, chunkSize: int = None, chunkOverlap: int = None):
        self.chunkSize = chunkSize or settings.CHUNK_SIZE
        self.chunkOverlap = chunkOverlap or settings.CHUNK_OVERLAP
        self.minChunkLength = settings.MIN_CHUNK_LENGTH
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text respecting section boundaries and cleaning artifacts.
        
        Args:
            text: Full Markdown text to chunk
            
        Returns:
            List of text chunks where sections are never merged
        """
        # Phase 0: Pre-process: strip artifacts and normalize whitespace
        text = self._sanitizeArtifacts(text)
        cleanedText = self._collapseExcessiveWhitespace(text)
        
        # Phase 1: Split into sections at header boundaries
        sections = self._splitOnHeaders(cleanedText)
        
        # Phase 2: Split oversized sections (within each section only)
        chunks = []
        for section in sections:
            section = section.strip()
            if not section or len(section) < self.minChunkLength:
                continue
            
            if len(section) <= self.chunkSize:
                # Section fits, keep it intact
                chunks.append(section)
            else:
                # Section too large, split it (but never merge with other sections)
                sectionChunks = self._splitOversizedSection(section)
                chunks.extend(sectionChunks)
        
        return chunks
    
    def _splitOnHeaders(self, text: str) -> List[str]:
        """
        Split on Markdown headers using fast line-by-line parsing.
        Each section stays separate - sections are NEVER merged.
        """
        sections = []
        currentLines = []
        
        for line in text.split('\n'):
            # Check if line is a Markdown header
            stripped = line.lstrip()
            if stripped.startswith('#'):
                # Count consecutive hashes
                hashCount = 0
                for char in stripped:
                    if char == '#':
                        hashCount += 1
                    else:
                        break
                
                # Valid header: 1-6 hashes followed by space
                isHeader = (1 <= hashCount <= 6 and 
                           len(stripped) > hashCount and 
                           stripped[hashCount] == ' ')
                
                if isHeader:
                    # Save current section before starting new one
                    if currentLines:
                        sections.append('\n'.join(currentLines))
                    currentLines = [line]
                    continue
            
            currentLines.append(line)
        
        # Don't forget the last section
        if currentLines:
            sections.append('\n'.join(currentLines))
        
        return sections
    
    def _splitOversizedSection(self, section: str) -> List[str]:
        """
        Split an oversized section using paragraph/sentence boundaries.
        Uses single-pass algorithm for speed.
        """
        chunks = []
        start = 0
        textLength = len(section)
        
        while start < textLength:
            end = min(start + self.chunkSize, textLength)
            
            # Find best break point within this window
            if end < textLength:
                breakPoint = self._findBreakPoint(section, start, end)
                if breakPoint > start:
                    end = breakPoint
            
            chunk = section[start:end].strip()
            if len(chunk) >= self.minChunkLength:
                chunks.append(chunk)
            
            # Advance with overlap, ensuring progress
            nextStart = end - self.chunkOverlap
            if nextStart <= start:
                nextStart = end
            start = nextStart
        
        return chunks
    
    def _findBreakPoint(self, text: str, start: int, end: int) -> int:
        """Find best break point: paragraph > sentence > newline > space."""
        searchStart = start + (self.chunkSize // 2)
        searchRange = text[searchStart:end]
        
        # 1. Paragraph break (double newline)
        idx = searchRange.rfind('\n\n')
        if idx != -1:
            return searchStart + idx + 2
        
        # 2. Sentence boundary
        for punct in ['. ', '? ', '! ']:
            idx = searchRange.rfind(punct)
            if idx != -1:
                return searchStart + idx + len(punct)
        
        # 3. Single newline
        idx = searchRange.rfind('\n')
        if idx != -1:
            return searchStart + idx + 1
        
        # 4. Word boundary
        idx = searchRange.rfind(' ')
        if idx != -1:
            return searchStart + idx + 1
        
        return end
    
    def _collapseExcessiveWhitespace(self, rawText: str) -> str:
        """Normalize noisy PDF/Statutory extractions while preserving structure."""
        # Collapse massive newline gaps (limit to double newline for parity)
        text = re.sub(r'\n{3,}', '\n\n', rawText)
        # Collapse massive horizontal gaps but preserve some indentation (up to 4 spaces)
        text = re.sub(r' {5,}', '    ', text)
        return text

    def _sanitizeArtifacts(self, text: str) -> str:
        """Strip persistent metadata noise like Page markers."""
        # Strip [[Page \d+ STAT. \d+]]
        text = re.sub(r'\[\[Page \d+ STAT\. \d+\]\]', '', text)
        # Strip other common noisy markers if needed (placeholder for future regexes)
        return text


# Backwards compatibility alias
TextChunker = SemanticChunker


class GraphRAGIndexer:
    """
    Main indexing pipeline for GraphRAG.
    
    Processes markdown files and builds:
    - Document/chunk storage
    - Dense vector embeddings
    - BM25 sparse vectors
    - Entity knowledge graph
    """
    
    def __init__(self, store: Optional[DuckDBStore] = None,
                 embeddings: Optional[DockerModelRunnerEmbeddings] = None,
                 llmClient: Optional[LocalLLMClient] = None,
                 entityExtractor: Optional[BaseEntityExtractor] = None,
                 enableEntityExtraction: bool = True):
        """
        Initialize indexer with optional component injection.
        
        Args:
            store: DuckDB storage (creates new if None)
            embeddings: Embedding provider (creates new if None)
            llmClient: LLM for entity extraction (creates new if None)
            entityExtractor: Custom entity extractor (standardizes model switching)
            enableEntityExtraction: Whether to run LLM entity extraction
        """
        self.store = store or getStore()
        self.embeddings = embeddings or getEmbeddings()
        self.llmClient = llmClient or getLLMClient()
        self.relationshipClient = getRelationshipClient()  # Dedicated client for grunt work
        self.enableEntityExtraction = enableEntityExtraction
        
        # Initialize entity extractor using the factory (defaults to LLM_ONLY)
        self.entityExtractor = entityExtractor or ExtractorFactory.getExtractor(llmClient=self.llmClient)
        
        self.chunker = TextChunker()
        self.bm25Indexer = BM25Indexer(self.store)
        self.stats = IndexingStats()
    
    def _loadFileContent(self, filePath: str) -> Optional[str]:
        """Load file content."""
        try:
            with open(filePath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as exc:
            logger.error(f"Failed to load {filePath}: {exc}")
            return None
    
    def _discoverFiles(self, inputDir: str) -> List[str]:
        """Find all markdown and text files in input directory."""
        inputPath = Path(inputDir)
        
        if not inputPath.exists():
            logger.warning(f"Input directory does not exist: {inputDir}")
            return []
        
        # Find .md and .txt files recursively
        mdFiles = list(inputPath.glob("**/*.md"))
        txtFiles = list(inputPath.glob("**/*.txt"))
        allFiles = mdFiles + txtFiles
        
        logger.info(f"Discovered {len(allFiles)} files ({len(mdFiles)} md, {len(txtFiles)} txt) in {inputDir}")
        return [str(f) for f in allFiles]
    
    def isAlreadyIndexed(self, filePath: str) -> bool:
        """Check if a source document path has already been indexed."""
        existing = self.store.getSourceDocumentByPath(filePath)
        return existing is not None
    
    def indexDocument(self, filePath: str, skipIfExists: bool = True) -> tuple[Optional[str], str]:
        """
        Index a single document.
        
        Args:
            filePath: Path to file
            skipIfExists: If True, skip documents already marked 'complete'
            
        Returns:
            Tuple of (docId, status). docId is None if document is already complete and skipIfExists is True.
        """
        # Check if already indexed
        existing = self.store.getSourceDocumentByPath(filePath)
        if existing:
            if skipIfExists and existing.pipelineStatus == PipelineStatus.COMPLETE:
                logger.debug(f"Skipping complete document: {filePath}")
                self.stats.documentsSkipped += 1
                return None, PipelineStatus.COMPLETE
            logger.info(f"Resuming incomplete document: {filePath} (status: {existing.pipelineStatus})")
            return existing.id, existing.pipelineStatus
        
        # Load content
        content = self._loadFileContent(filePath)
        if not content:
            return None, PipelineStatus.PENDING
        
        # Create source document record
        docId = str(uuid.uuid4())
        doc = SourceDocument(
            id=docId,
            sourcePath=filePath,
            rawContent=content,
            pipelineStatus=PipelineStatus.PENDING
        )
        self.store.insertSourceDocument(doc)
        self.stats.documentsProcessed += 1
        
        # Chunk the document immediately
        chunkTexts = self.chunker.chunk(content)
        if not chunkTexts:
            logger.warning(f"No chunks created for {filePath}")
            self.store.updatePipelineStatus(docId, PipelineStatus.COMPLETE)
            return docId, PipelineStatus.COMPLETE
        
        # Create chunk records
        chunks = []
        garbage_count = 0
        for idx, text in enumerate(chunkTexts):
            chunkId = str(uuid.uuid4())
            reason = garbageFilter.isGarbagePre(text)
            if reason:
                garbageLogger.log(chunkId, text, reason, {"source": filePath, "idx": idx})
                garbage_count += 1
                continue
                
            chunks.append(DocumentChunk(
                chunkId=chunkId,
                sourceDocumentId=docId,
                text=text,
                index=idx,
                metadata={"source": filePath}
            ))

        if chunks:
            self.store.insertDocumentChunks(chunks)
        self.stats.chunksCreated += len(chunks)
        
        if garbage_count > 0:
            logger.info(f"Skipped {garbage_count} garbage chunks from {filePath}")
            
        # Success at chunking stage
        self.store.updatePipelineStatus(docId, PipelineStatus.CHUNKED)
        return docId, PipelineStatus.CHUNKED
    
    # helper for internal logic, renamed old indexDocument content
    def _chunkDocumentContent(self, docId: str, filePath: str, content: str) -> List[DocumentChunk]:
        return [] # logic moved into indexDocument for atomicity
    
    def generateEmbeddings(self, chunks: List[DocumentChunk]) -> int:
        """
        Generate and store embeddings for chunks.
        
        Args:
            chunks: List of DocumentChunks to embed
            
        Returns:
            Number of embeddings stored
        """
        if not chunks:
            return 0
        
        # Extract texts
        texts = [c.text for c in chunks]
        chunkIds = [c.chunkId for c in chunks]
        
        # Generate embeddings in batches
        embeddings = self.embeddings.embedDocuments(texts)
        
        # Store in DuckDB
        self.store.updateEmbeddings(chunkIds, embeddings)
        self.stats.embeddingsGenerated += len(embeddings)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return len(embeddings)
    
    def indexBM25(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks for BM25 sparse retrieval."""
        if not chunks:
            return
        
        termStats = self.bm25Indexer.indexChunks(chunks)
        self.stats.bm25TokensIndexed += len(termStats)
        
        logger.info(f"Indexed {len(termStats)} unique terms for BM25")
    
    def extractEntities(self, chunks: List[DocumentChunk]) -> None:
        """
        Extract entities and relationships from chunks using LLM.
        
        Uses hybrid batching strategy:
        1. Process documents sequentially (prevents cross-doc contamination)
        2. Within each document, batch chunks together (reduces overhead)
        3. Concurrent batch requests within document scope
        
        Args:
            chunks: Chunks to process
        """
        if not self.enableEntityExtraction:
            logger.info("Entity extraction disabled, skipping")
            return
        
        if not self.llmClient.isAvailable():
            logger.warning("LLM endpoint not available, skipping entity extraction")
            return
        
        allEntities = []
        allRelationships = []
        
        # Group chunks by document for sequential per-document processing
        chunksByDocument = {}
        for chunk in chunks:
            docId = chunk.sourceDocumentId
            if docId not in chunksByDocument:
                chunksByDocument[docId] = []
            chunksByDocument[docId].append(chunk)
        
        batchSize = settings.LLM_CHUNKS_PER_BATCH
        concurrency = settings.LLM_CONCURRENCY
        
        logger.info(f"Extracting entities from {len(chunks)} chunks across {len(chunksByDocument)} documents")
        logger.info(f"Using batch size={batchSize}, concurrency={concurrency}")
        
        # Process each document sequentially
        for docIdx, (docId, docChunks) in enumerate(chunksByDocument.items()):
            logger.info(f"Processing document {docIdx + 1}/{len(chunksByDocument)} ({len(docChunks)} chunks)")
            
            # Split document chunks into batches
            batches = [docChunks[i:i + batchSize] for i in range(0, len(docChunks), batchSize)]
            
            # 1. Extract Entities (Parallel across batches)
            # Uses the configured entity extractor (LLM or GLiNER)
            documentEntityCollection = {} 
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_batch = {
                    executor.submit(self.entityExtractor.extractEntitiesBatch, batch): batch
                    for batch in batches
                }
                
                for future in as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        batchExtractedEntities = future.result()
                        documentEntityCollection.update(batchExtractedEntities)
                        
                        entityCount = sum(len(extractedEntities) for extractedEntities in batchExtractedEntities.values())
                        logger.info(f"  Extracted {entityCount} entities from {len(batch)} chunks")
                    except Exception as exc:
                        logger.error(f"Entity extraction failed for batch: {exc}")

            # 2. Extract Relationships (LLM only - Batch discovery)
            # IMPORTANT: Pass only BATCH-LOCAL entities to match original design (~20 entities/batch)
            # See: batched_relationship_extraction_analysis.md Section 7.1.1
            # Original assumed: "Entity List: 20 entities × 15 tokens = 300 tokens"
            # Passing all document entities (e.g., 1350 from GLiNER) would balloon context to 5500+ tokens
            if documentEntityCollection:
                for batch in batches:
                    # Build batch-local entity map (only entities from chunks in THIS batch)
                    batchChunkIds = {c.chunkId for c in batch}
                    batchEntityMap = {
                        cid: entities 
                        for cid, entities in documentEntityCollection.items() 
                        if cid in batchChunkIds
                    }
                    
                    if batchEntityMap:
                        batchRelationshipCollection = self.relationshipClient.extractRelationshipsBatch(batch, batchEntityMap)
                        allRelationships.extend(batchRelationshipCollection.values())
            
            # Accumulate all entities from this document into the master list
            for entitiesFromChunk in documentEntityCollection.values():
                allEntities.extend(entitiesFromChunk)

        # Store entities and relationships
        if allEntities:
            # Build a mapping: original entity ID -> entity name (for later resolution)
            entityIdToName = {e.entityId: e.name for e in allEntities}
            
            # Insert entities (upsert/merge may discard some IDs)
            self.store.insertEntities(allEntities)
            self.stats.entitiesExtracted += len(allEntities)
        
        if allRelationships:
            # Relationships is a list of lists from dict.values()
            flatRelationships = []
            for rel_list in allRelationships:
                if isinstance(rel_list, list):
                    flatRelationships.extend(rel_list)
                else: # single list case
                    flatRelationships.append(rel_list)
            
            # Resolve entity IDs: original IDs may have been discarded during merge
            # Get the actual persisted IDs by looking up entity names
            if flatRelationships and allEntities:
                # Collect all entity names referenced by relationships
                entityNamesToResolve = set()
                for rel in flatRelationships:
                    if rel.sourceEntityId in entityIdToName:
                        entityNamesToResolve.add(entityIdToName[rel.sourceEntityId])
                    if rel.targetEntityId in entityIdToName:
                        entityNamesToResolve.add(entityIdToName[rel.targetEntityId])
                
                # Bulk fetch persisted entity IDs by name
                persistedIdMap = self.store.getEntityIdsByNames(list(entityNamesToResolve))
                
                # Build original ID -> persisted ID mapping
                originalToPersistedId = {}
                for origId, name in entityIdToName.items():
                    persistedId = persistedIdMap.get(name.lower())
                    if persistedId:
                        originalToPersistedId[origId] = persistedId
                
                # Update relationship entity IDs and filter out invalid ones
                validRelationships = []
                skippedCount = 0
                for rel in flatRelationships:
                    newSourceId = originalToPersistedId.get(rel.sourceEntityId, rel.sourceEntityId)
                    newTargetId = originalToPersistedId.get(rel.targetEntityId, rel.targetEntityId)
                    
                    # Only include if both entities exist in persisted map
                    if newSourceId in persistedIdMap.values() and newTargetId in persistedIdMap.values():
                        rel.sourceEntityId = newSourceId
                        rel.targetEntityId = newTargetId
                        validRelationships.append(rel)
                    else:
                        skippedCount += 1
                
                if skippedCount > 0:
                    logger.warning(f"Skipped {skippedCount} relationships with unresolved entity IDs")
                
                flatRelationships = validRelationships
            
            if flatRelationships:
                self.store.insertRelationships(flatRelationships)
                self.stats.relationshipsExtracted += len(flatRelationships)
        
        # Post-extraction optimization: Remove stranded entities
        # Only connected knowledge remains in the database to prevent noise bloat.
        strandedEntitiesRemoved = self.store.pruneStrandedEntities()
        
        logger.info(f"Extracted {len(allEntities)} entities and {self.stats.relationshipsExtracted} relationships")
        if strandedEntitiesRemoved > 0:
            logger.info(f"Cleaned up {strandedEntitiesRemoved} stranded entities (no relationships found)")
    
    def pruneNoise(self, chunkIds: Optional[List[str]] = None, 
                   runLLMScore: bool = False) -> Dict[str, int]:
        """
        Prune non-meaningful chunks based on post-extraction metrics.
        
        Args:
            chunkIds: Specific chunks to evaluate (defaults to all)
            runLLMScore: Whether to run the expensive LLM quality scoring
            
        Returns:
            Dict with pruning stats
        """
        logger.info("Starting noise pruning phase...")
        chunks = self.store.getAllChunks() if chunkIds is None else \
                 [c for c in self.store.getAllChunks() if c.chunkId in chunkIds]
        
        if not chunks:
            return {"pruned": 0}
            
        garbage_ids = []
        remaining_chunks = chunks  # Start with all chunks
        
        # NOTE: Entity-count filtering REMOVED (2026-01-08)
        # 
        # Rationale:
        # - Entity count is NOT a reliable quality indicator
        # - GLiNER produces bimodal distribution: 0 entities (99%) or 3+ entities (1%)
        # - Many legitimate chunks have 0 entities (introductions, summaries, methodology)
        # - Existing filters already catch garbage (repetition, entropy, malformed text)
        # - Domain-agnostic approach is more transferable across use cases
        #
        # For future consideration: See project_documentation/contextual_entity_filtering.md
        # for a smarter alternative using neighbor-aware filtering.
        
        # 1. LLM Quality Scoring (Expensive, optional)
        if runLLMScore and remaining_chunks:
            concurrency = settings.LLM_CONCURRENCY
            logger.info(f"Running LLM quality scoring on {len(remaining_chunks)} chunks with {concurrency} workers...")
            
            scored_garbage = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_chunk = {
                    executor.submit(self.llmClient.scoreQuality, c.text): c
                    for c in remaining_chunks
                }
                
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        score = future.result()
                        if score < settings.FILTER_QUALITY_THRESHOLD:
                            scored_garbage.append((chunk.chunkId, chunk.text, score, chunk.metadata))
                    except Exception as exc:
                        logger.error(f"Quality scoring failed for chunk {chunk.chunkId}: {exc}")
            
            for cid, text, score, meta in scored_garbage:
                garbage_ids.append(cid)
                garbageLogger.log(cid, text, 
                                  f"LLM Quality Score: {score:.2f} < {settings.FILTER_QUALITY_THRESHOLD}", 
                                  meta)
                # Remove from remaining_chunks so they aren't processed as outliers redundantly
                remaining_chunks = [c for c in remaining_chunks if c.chunkId != cid]
                                      
        # 3. Per-Document Embedding Outlier Detection
        # Compare each chunk to its OWN document's centroid (not global corpus)
        # This preserves topic diversity across documents while catching noise within each
        processedDocIds = set()
        for chunk in remaining_chunks:
            docId = chunk.sourceDocumentId
            if docId in processedDocIds:
                continue
            processedDocIds.add(docId)
            
            # Get chunks for this document
            docChunks = [c for c in remaining_chunks if c.sourceDocumentId == docId]
            
            # Skip small documents where centroid isn't statistically meaningful
            if len(docChunks) < settings.MIN_CHUNKS_FOR_OUTLIER_DETECTION:
                continue
            
            # Calculate this document's centroid
            docCentroid = self.store.getEmbeddingCentroidForDocument(docId)
            if not docCentroid:
                continue
            
            # Find outliers within this document
            docOutliers = self.store.getOutlierChunkIdsForDocument(
                docId, docCentroid, settings.FILTER_EMBEDDING_OUTLIER_THRESHOLD
            )
            
            for oid in docOutliers:
                if oid not in garbage_ids:
                    chunk_obj = next((c for c in docChunks if c.chunkId == oid), None)
                    if chunk_obj:
                        garbage_ids.append(oid)
                        garbageLogger.log(oid, chunk_obj.text,
                                          f"Embedding outlier (per-document)",
                                          chunk_obj.metadata)
        
        if garbage_ids:
            self.store.deleteGarbageChunks(garbage_ids)
            
        # 4. Final Knowledge Graph Cleanup
        # Remove any entities that become stranded after chunk deletion or were already isolated
        strandedEntitiesRemoved = self.store.pruneStrandedEntities()
            
        return {
            "pruned": len(garbage_ids),
            "pruned_entities": strandedEntitiesRemoved
        }

    def resetDatabase(self) -> Dict[str, int]:
        """
        Nuclear reset: Delete the database file and recreate with current config.
        This avoids FK constraint issues and guarantees fresh schema dimensions.
        
        Returns:
            Dict of table names to deleted row counts
        """
        logger.warning("Resetting database - deleting file for clean recreation")
        return self.store.resetDatabase()
    
    def indexDirectory(self, inputDir: str = None, skipIfExists: bool = True) -> IndexingStats:
        """
        Index all markdown and text files in a directory.
        Now with automatic pipeline resume based on document status.
        """
        inputDir = inputDir or settings.INPUT_DIR
        allFiles = self._discoverFiles(inputDir)
        if not allFiles:
            return self.stats
        
        # Stage 1: Identification and Chunking
        # indexDocument handles chunking for new files and status retrieval for existing ones
        activeDocs = [] # List of (docId, current_status)
        for filePath in allFiles:
            docId, status = self.indexDocument(filePath, skipIfExists=skipIfExists)
            if docId:
                activeDocs.append((docId, status))
        
        if not activeDocs:
            logger.info("No documents needing processing.")
            return self.stats
            
        # Stage 2: Embedding Generation (Resume point for documents < EMBEDDED)
        chunksToEmbed = []
        docsToUpdateEmbedStatus = []
        for docId, status in activeDocs:
            if status == PipelineStatus.CHUNKED:
                chunks = self.store.getChunksForSourceDocument(docId)
                chunksToEmbed.extend(chunks)
                docsToUpdateEmbedStatus.append(docId)
        
        if chunksToEmbed:
            self.generateEmbeddings(chunksToEmbed)
            for docId in docsToUpdateEmbedStatus:
                self.store.updatePipelineStatus(docId, PipelineStatus.EMBEDDED)
            # Update local status for subsequent stages
            activeDocs = [(d, PipelineStatus.EMBEDDED if d in docsToUpdateEmbedStatus else s) 
                         for d, s in activeDocs]
        
        # Stage 3: BM25 Indexing
        # We re-index BM25 for any doc that was at least CHUNKED but not yet COMPLETE
        # (BM25 indexing is cheap and idempotent enough to run on every modified document)
        allActiveChunks = []
        for docId, _ in activeDocs:
            allActiveChunks.extend(self.store.getChunksForSourceDocument(docId))
        
        if allActiveChunks:
            self.indexBM25(allActiveChunks)
            
        # Stage 4: Entity Extraction (Resume point for documents < EXTRACTED)
        chunksToExtract = []
        docsToUpdateExtractStatus = []
        for docId, status in activeDocs:
            if status in [PipelineStatus.CHUNKED, PipelineStatus.EMBEDDED]:
                if self.enableEntityExtraction:
                    chunks = self.store.getChunksForSourceDocument(docId)
                    chunksToExtract.extend(chunks)
                    docsToUpdateExtractStatus.append(docId)
                else:
                    # If entity extraction is disabled, jump straight to complete
                    self.store.updatePipelineStatus(docId, PipelineStatus.COMPLETE)
        
        if chunksToExtract:
            self.extractEntities(chunksToExtract)
            for docId in docsToUpdateExtractStatus:
                self.store.updatePipelineStatus(docId, PipelineStatus.EXTRACTED)
                self.store.updatePipelineStatus(docId, PipelineStatus.COMPLETE)

        logger.info(f"Indexing complete: {self.stats}")
        return self.stats
    
    def indexFile(self, filePath: str) -> IndexingStats:
        """
        Index a single file.
        
        Args:
            filePath: Path to markdown file
            
        Returns:
            Indexing statistics
        """
        docId = self.indexDocument(filePath)
        
        if docId:
            chunks = self.store.getChunksForSourceDocument(docId)
            self.generateEmbeddings(chunks)
            self.indexBM25(chunks)
            self.extractEntities(chunks)
        
        return self.stats


def main():
    """CLI entry point for indexing."""
    start_time = time.time()
    import argparse
    
    parser = argparse.ArgumentParser(description="GraphRAG Indexing Pipeline")
    parser.add_argument("--input", "-i", default=settings.INPUT_DIR,
                       help="Input directory with markdown files")
    parser.add_argument("--file", "-f", help="Single file to index")
    parser.add_argument("--no-entities", action="store_true",
                       help="Skip entity extraction")
    parser.add_argument("--extraction-mode", choices=["local_llm", "gliner_llm"], 
                       help="Override entity extraction strategy (local_llm or gliner_llm)")
    parser.add_argument("--reset", action="store_true",
                       help="Clear database and reindex all documents")
    parser.add_argument("--prune", action="store_true",
                       help="Run post-extraction pruning on existing index")
    parser.add_argument("--llm-prune", action="store_true",
                       help="Use LLM scoring during pruning (expensive)")

    parser.add_argument("--db", default=settings.DUCKDB_PATH,
                       help="DuckDB database path")
    
    args = parser.parse_args()
    
    # Initialize store
    store = getStore(args.db)
    
    # Create indexer
    extraction_mode = None
    if args.extraction_mode:
        from graphrag_config import ExtractionMode
        extraction_mode = ExtractionMode(args.extraction_mode)
        
    entity_extractor = None
    if extraction_mode:
        from entity_extractor import ExtractorFactory
        entity_extractor = ExtractorFactory.getExtractor(mode=extraction_mode)
        
    indexer = GraphRAGIndexer(
        store=store,
        entityExtractor=entity_extractor,
        enableEntityExtraction=not args.no_entities
    )
    
    # Handle reset
    if args.reset:
        deleted = indexer.resetDatabase()
        print("\n=== Database Reset ===")
        for table, count in deleted.items():
            if count > 0:
                print(f"  Deleted {count} rows from {table}")
    
    # Run indexing (skip duplicates only if not resetting)
    skipIfExists = not args.reset
    
    if args.file:
        indexer.indexFile(args.file)
    else:
        indexer.indexDirectory(args.input, skipIfExists=skipIfExists)
    
    # Run post-extraction pruning if requested
    if args.prune or args.llm_prune:
        prune_stats = indexer.pruneNoise(runLLMScore=args.llm_prune)
        print(f"\n=== Pruning Summary ===")
        print(f"Chunks Pruned:   {prune_stats['pruned']}")
        print(f"Entities Pruned: {prune_stats.get('pruned_entities', 0)}")
    
    # Print summary
    print("\n=== Indexing Summary ===")
    print(f"Documents:     {indexer.stats.documentsProcessed}")
    print(f"Skipped:       {indexer.stats.documentsSkipped}")
    print(f"Chunks:        {indexer.stats.chunksCreated}")
    print(f"Embeddings:    {indexer.stats.embeddingsGenerated}")
    print(f"BM25 Terms:    {indexer.stats.bm25TokensIndexed}")
    print(f"Entities:      {indexer.stats.entitiesExtracted}")
    print(f"Relationships: {indexer.stats.relationshipsExtracted}")
    
    # Show corpus stats
    dbStats = store.getStats()
    print("\n=== Database Stats ===")
    for table, count in dbStats.items():
        print(f"{table}: {count}")
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\nTotal time elapsed: {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
