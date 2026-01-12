"""
BM25 Indexing Module - Sparse vector indexing for keyword-based retrieval.

Implements Okapi BM25 scoring with:
- Simple whitespace tokenization (can extend with NLTK)
- Term frequency calculation per chunk
- Document frequency tracking for IDF
- Integration with DuckDB storage

Following coding framework guidelines:
- Bulk operations over single-item processing
- Centralized constants from config
- Clear docstrings for LLM tool compatibility
"""
import re
import math
import logging
from typing import List, Dict, Tuple
from collections import Counter

from graphrag_config import settings
from duckdb_store import DuckDBStore, DocumentChunk

logger = logging.getLogger(__name__)


class BM25Indexer:
    """
    BM25 sparse vector indexer for hybrid retrieval.
    
    Tokenizes documents, calculates term frequencies, and stores
    sparse representations for later BM25 scoring during retrieval.
    """
    
    # Stopwords to filter (basic English set)
    STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
        't', 'just', 'now', 'also', 'as', 'if'
    }
    
    def __init__(self, store: DuckDBStore):
        """Initialize with DuckDB store reference."""
        self.store = store
        self.k1 = settings.BM25_K1
        self.b = settings.BM25_B
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase terms with basic cleaning.
        Supports multilingual text including CJK characters.
        
        Args:
            text: Raw text to tokenize
            
        Returns:
            List of lowercase tokens (stopwords removed)
        """
        if not text:
            return []
            
        text = text.lower()
        
        # This ensuring CJK is indexed at the character level for high search recall.
        pattern = r'[a-z0-9]+|[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]'
        tokens = re.findall(pattern, text)
        
        # Filter stopwords and handle min-length rules
        # For CJK, single characters are meaningful tokens; for Latin/others, we keep len > 1
        cjk_regex = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')
        
        filteredTokens = []
        for t in tokens:
            if t in self.STOPWORDS:
                continue
            
            # Keep CJK characters or words with length > 1
            if cjk_regex.match(t) or len(t) > 1:
                filteredTokens.append(t)
        
        return filteredTokens
    
    def calculateTermFrequencies(self, tokens: List[str]) -> Dict[str, int]:
        """Calculate term frequency for each unique token."""
        return dict(Counter(tokens))
    
    def indexChunks(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """
        Index chunks for BM25 retrieval.
        
        Tokenizes each chunk, calculates TF, and stores in DuckDB.
        Returns term -> document frequency mapping for corpus stats.
        
        Args:
            chunks: List of Chunk objects to index
            
        Returns:
            Dict mapping each term to its document frequency
        """
        if not chunks:
            return {}
        
        chunkIds = []
        allTokenizedTerms = []
        allTermFrequencies = []
        allDocLengths = []
        
        # Track document frequency for each term
        termDocFrequency: Dict[str, int] = {}
        
        for chunk in chunks:
            tokens = self.tokenize(chunk.text)
            tf = self.calculateTermFrequencies(tokens)
            
            chunkIds.append(chunk.chunkId)
            allTokenizedTerms.append(tokens)
            allTermFrequencies.append(tf)
            allDocLengths.append(len(tokens))
            
            # Update document frequencies (term appears in this doc)
            for term in set(tokens):
                termDocFrequency[term] = termDocFrequency.get(term, 0) + 1
        
        # Batch insert sparse vectors
        self.store.insertSparseVectors(
            chunkIds, allTokenizedTerms, allTermFrequencies, allDocLengths
        )
        
        # Update corpus-level BM25 stats
        self.store.updateBm25Stats(termDocFrequency)
        
        logger.info(f"Indexed {len(chunks)} chunks with {len(termDocFrequency)} unique terms")
        return termDocFrequency


class BM25Scorer:
    """
    BM25 scoring engine for query-time retrieval.
    
    Calculates BM25 scores using pre-computed term statistics
    stored in DuckDB.
    """
    
    def __init__(self, store: DuckDBStore):
        """Initialize with DuckDB store and load corpus stats."""
        self.store = store
        self.k1 = settings.BM25_K1
        self.b = settings.BM25_B
        
        # Load corpus statistics
        self._loadCorpusStats()
    
    def _loadCorpusStats(self) -> None:
        """Load corpus-level statistics from DuckDB."""
        self.totalDocs, self.avgDocLength, self.termDocFreqs = self.store.getBm25Stats()
        logger.info(f"Loaded BM25 stats: {self.totalDocs} docs, avg length {self.avgDocLength:.1f}")
    
    def refreshStats(self) -> None:
        """Reload corpus stats after new documents are indexed."""
        self._loadCorpusStats()
    
    def _idf(self, term: str) -> float:
        """
        Calculate Inverse Document Frequency for a term.
        
        Uses the standard BM25 IDF formula:
        log((N - df + 0.5) / (df + 0.5) + 1)
        
        where N = total docs, df = doc frequency of term
        """
        df = self.termDocFreqs.get(term, 0)
        
        if df == 0:
            return 0.0
        
        # Standard IDF with smoothing
        idfScore = math.log((self.totalDocs - df + 0.5) / (df + 0.5) + 1.0)
        return max(0.0, idfScore)
    
    def scoreDocument(self, queryTokens: List[str], termFrequencies: Dict[str, int], 
                      docLength: int) -> float:
        """
        Calculate BM25 score for a single document.
        
        Args:
            queryTokens: Tokenized query terms
            termFrequencies: Term -> frequency mapping for the document
            docLength: Number of tokens in the document
            
        Returns:
            BM25 score (higher = more relevant)
        """
        if self.avgDocLength == 0 or self.totalDocs == 0:
            return 0.0
        
        score = 0.0
        
        for term in queryTokens:
            if term not in termFrequencies:
                continue
            
            tf = termFrequencies[term]
            idf = self._idf(term)
            
            # BM25 term score
            # tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * docLength / self.avgDocLength)
            
            termScore = idf * numerator / denominator
            score += termScore
        
        return score
    
    def search(self, query: str, topK: int = None) -> List[Tuple[str, float, str]]:
        """
        Search corpus using BM25 scoring.
        
        Args:
            query: Natural language query
            topK: Number of results to return (defaults to settings.TOP_K)
            
        Returns:
            List of (chunk_id, bm25_score, chunk_text) tuples
        """
        topK = topK or settings.TOP_K
        
        # Tokenize query
        indexer = BM25Indexer(self.store)
        queryTokens = indexer.tokenize(query)
        
        if not queryTokens:
            logger.warning(f"Query '{query}' produced no tokens after filtering")
            return []
        
        # Get all sparse vectors
        results = self.store.connection.execute("""
            SELECT sv.chunk_id, sv.term_frequencies, sv.doc_length, d.text
            FROM sparse_vectors sv
            JOIN documents d ON sv.chunk_id = d.chunk_id
        """).fetchall()
        
        if not results:
            return []
        
        # Score each document
        scored = []
        for chunkId, tfJson, docLength, chunkText in results:
            import json
            tf = json.loads(tfJson) if isinstance(tfJson, str) else tfJson
            
            score = self.scoreDocument(queryTokens, tf, docLength)
            if score > 0:
                scored.append((chunkId, score, chunkText))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:topK]


def createBm25Index(store: DuckDBStore, chunks: List[DocumentChunk]) -> BM25Indexer:
    """
    Convenience function to index chunks for BM25.
    
    Args:
        store: DuckDB store instance
        chunks: Chunks to index
        
    Returns:
        BM25Indexer instance
    """
    indexer = BM25Indexer(store)
    indexer.indexChunks(chunks)
    return indexer


def getBm25Scorer(store: DuckDBStore) -> BM25Scorer:
    """
    Get a BM25 scorer with loaded corpus statistics.
    
    Args:
        store: DuckDB store instance
        
    Returns:
        BM25Scorer ready for querying
    """
    return BM25Scorer(store)
