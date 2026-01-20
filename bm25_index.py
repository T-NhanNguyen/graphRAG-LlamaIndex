# BM25 Indexing Module - Sparse vector indexing for keyword-based retrieval.
import re
import math
import logging
from typing import List, Dict, Tuple
from collections import Counter

from graphrag_config import settings
from duckdb_store import DuckDBStore, DocumentChunk
from stop_words import get_stop_words

logger = logging.getLogger(__name__)


class BM25Indexer:
    # BM25 sparse vector indexer for hybrid retrieval. Tokenizes docs and stores stats for scoring.
    
    def __init__(self, store: DuckDBStore, language: str = None):
        # Initialize with DuckDB store and load language-specific stopwords.
        self.store = store
        self.k1 = settings.BM25_K1
        self.b = settings.BM25_B
        
        # Use configured language or default to English
        lang = language or settings.BM25_LANGUAGE
        
        # Load stopwords for specified language (supports: en, zh, ja, ko, fr, de, es, etc.)
        try:
            self.stopwords = set(get_stop_words(lang))
            logger.info(f"Loaded {len(self.stopwords)} stopwords for language '{lang}'")
        except Exception as e:
            logger.warning(f"Failed to load stopwords for '{lang}': {e}. Using English fallback.")
            self.stopwords = set(get_stop_words(lang))
    
    def tokenize(self, text: str) -> List[str]:
        # Tokenize text into lowercase terms with cleaning. Supports multilingual/CJK characters.
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
            if t in self.stopwords:
                continue
            
            # Keep CJK characters or words with length > 1
            if cjk_regex.match(t) or len(t) > 1:
                filteredTokens.append(t)
        
        return filteredTokens
    
    def calculateTermFrequencies(self, tokens: List[str]) -> Dict[str, int]:
        # Calculate term frequency for each unique token.
        return dict(Counter(tokens))
    
    def indexChunks(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        # Index chunks for BM25 retrieval and return term document frequencies.
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
    # BM25 scoring engine for query-time retrieval using DuckDB stats.
    
    def __init__(self, store: DuckDBStore):
        self.store = store
        self.k1 = settings.BM25_K1
        self.b = settings.BM25_B
        
        # Load corpus statistics
        self._loadCorpusStats()
    
    def _loadCorpusStats(self) -> None:
        # Load corpus-level statistics from DuckDB.
        self.totalDocs, self.avgDocLength, self.termDocFreqs = self.store.getBm25Stats()
        logger.info(f"Loaded BM25 stats: {self.totalDocs} docs, avg length {self.avgDocLength:.1f}")
    
    def refreshStats(self) -> None:
        # Reload corpus stats after new documents are indexed.
        self._loadCorpusStats()
    
    def _idf(self, term: str) -> float:
        # Calculate Inverse Document Frequency (IDF) for a term using BM25 formula.
        df = self.termDocFreqs.get(term, 0)
        
        if df == 0:
            return 0.0
        
        # Standard IDF with smoothing
        idfScore = math.log((self.totalDocs - df + 0.5) / (df + 0.5) + 1.0)
        return max(0.0, idfScore)
    
    def scoreDocument(self, queryTokens: List[str], termFrequencies: Dict[str, int], 
                      docLength: int) -> float:
        # Calculate BM25 score for a single document.
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
        # Search corpus using BM25 scoring. Returns ranked chunk IDs and scores.
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
    # Convenience function to index chunks for BM25.
    indexer = BM25Indexer(store)
    indexer.indexChunks(chunks)
    return indexer


def getBm25Scorer(store: DuckDBStore) -> BM25Scorer:
    # Get a BM25 scorer with loaded corpus statistics.
    return BM25Scorer(store)
