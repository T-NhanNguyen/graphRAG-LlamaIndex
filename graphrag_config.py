"""
GraphRAG Configuration - Centralized constants for agentic tool development.

Following coding framework guidelines:
- Uppercase constants with descriptive names
- Enums for categorical values
- Typed parameters for LLM tool compatibility
"""
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class EntityType(str, Enum):
    """Entity classification types for knowledge graph nodes."""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"


class EmbeddingProvider(str, Enum):
    """Supported embedding providers."""
    DOCKER_MODEL_RUNNER = "docker_model_runner"
    OPENAI = "openai"
    OLLAMA = "ollama"


class SearchType(str, Enum):
    """Query search strategy."""
    FIND_CONNECTIONS = "find_connections" # Hybrid + Graph traversal (local context)
    EXPLORE_THEMATIC = "explore_thematic" # Community-level reasoning (global themes)
    KEYWORD_SEARCH = "keyword_search"     # BM25 + vector hybrid text search only


class ExtractionMode(str, Enum):
    """Entity extraction strategies."""
    LLM_ONLY = "local_llm"       # Default: Traditional LLM
    HYBRID = "gliner_llm"  # Optional: GLiNER entities + LLM relationships


class RelationshipProvider(str, Enum):
    """Relationship extraction providers."""
    LOCAL = "local"           # Use local LLM (Docker Model Runner)
    OPENROUTER = "openrouter"  # Use OpenRouter API


class GraphRAGSettings(BaseSettings):
    """
    Runtime configuration loaded from environment/.env file.
    All magic numbers centralized here per coding framework.
    """
    # Default Embedding Configuration (Loaded from .env)
    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.DOCKER_MODEL_RUNNER
    EMBEDDING_MODEL: str = ""      # Must be set in .env
    EMBEDDING_URL: str = "http://host.docker.internal:12434"
    EMBEDDING_DIMENSION: int = 0           # Required in .env (model-specific)
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_CONCURRENCY: int = 8
    
    # Default LLM Configuration (for entity extraction - uses Docker Model Runner like embeddings) (Loaded from .env)
    LLM_URL: str = "http://host.docker.internal:12434"
    LLM_MODEL: str = ""            # Must be set in .env. Docker Model Runner model
    LLM_TEMPERATURE: float = 0.1  # Low temp for structured extraction
    LLM_MAX_TOKENS: int = 2048     # Generous output room for 8k context
    LLM_CONTEXT_LENGTH: int = 32768 # Configured via Docker CLI
    LLM_CONCURRENCY: int = 8      # Concurrent requests (Note: Increase to 12+ if reducing batch size for speed)
    LLM_CHUNKS_PER_BATCH: int = 4 # Chunks per parallel request (Note: Set to 1-2 for max stability/isolation)
    
    # Entity Extraction Mode
    ENTITY_EXTRACTION_MODE: ExtractionMode = ExtractionMode.LLM_ONLY
    
    # Relationship Extraction Configuration
    RELATIONSHIP_PROVIDER: RelationshipProvider = RelationshipProvider.LOCAL
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "xiaomi/mimo-v2-flash:free"
    
    # GLiNER Configuration
    GLINER_MODEL: str = "urchade/gliner_large-v2.1"
    GLINER_MAX_LENGTH: int = 512    # because GLiNER is an encoder model, doubling its context length (if the model supported it) 
                                    # would quadruple its memory usage (O(n²)). By keeping it at 512, it stays incredibly efficient 
                                    # and fits our 1000-character chunks perfectly.


    
    # DuckDB Storage
    DUCKDB_PATH: str = "./.DuckDB/graphrag.duckdb"
    
    # Indexing Parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MIN_CHUNK_LENGTH: int = 50  # Skip tiny fragments
    
    # BM25 Parameters (Okapi BM25 standard values)
    BM25_K1: float = 1.5   # Term frequency saturation
    BM25_B: float = 0.75   # Document length normalization
    
    # Fusion Retrieval Parameters
    TOP_K: int = 10
    FUSION_ALPHA: float = 0.5  # 0=pure BM25, 1=pure vector
    RRF_K: int = 60            # Reciprocal Rank Fusion constant
    
    # Query Output Optimization (Agent-First Interface)
    CHUNK_DEDUP_SIMILARITY_THRESHOLD: float = 0.85  # Cosine similarity threshold for deduplication
    CHUNK_DEDUP_ENABLED: bool = True                # Toggle chunk deduplication
    RELATIONSHIP_DIRECT_CHUNK_WINDOW: int = 3       # Top-N chunks for "direct" relationship classification
    
    MAX_COMMUNITY_SIZE: int = 50
    COMMUNITY_SUMMARY_PROMPT: str = """
    You are an AI assistant helping to summarize a thematic community of entities in a knowledge graph.
    Given the following entities and their descriptions, provide a concise summary that captures the primary theme and key relationships within this group.
    
    ENTITIES:
    {entity_context}
    
    SUMMARY:
    """

    # --- Extraction Prompts (Balanced for 8192 context) ---
    LLM_SYSTEM_PROMPT: str = "You are a precise entity extraction assistant. Always respond with valid JSON."
    
    ENTITY_EXTRACTION_PROMPT: str = """Extract the most important entities from the following text. 
Return ONLY valid JSON in this format:
{{"entities": [{{"name": "Entity name", "type": "TYPE", "description": "1-sentence description"}}]}}

Types: PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT, PRODUCT, TECHNOLOGY
Rules: Max {max_entities} entities. Concise descriptions. Names capitalized.

Text: {text}

JSON:"""

    RELATIONSHIP_EXTRACTION_PROMPT: str = """Given these entities: {entity_names}
Extract explicit relationships between them from the provided text.

Return ONLY valid JSON in this format:
{{"relationships": [{{"source": "Name", "target": "Name", "type": "VERB", "description": "1-sentence max"}}]}}

Rules: Max {max_relationships} relationships. Types must be concise verbs (e.g., PLAYS_IN, LOCATED_AT).

Text: {text}

JSON:"""
    # Entity Extraction
    MAX_ENTITIES_PER_CHUNK: int = 10
    MAX_RELATIONSHIPS_PER_CHUNK: int = 15
    
    BATCH_ENTITY_EXTRACTION_PROMPT: str = """Extract entities from {num_chunks} independent text chunks.
Follow isolation: Do NOT cross-reference between chunks.

{chunk_blocks}

Return ONLY valid JSON keyed by chunk index:
{{
  "0": {{"entities": [{{"name": "...", "type": "...", "description": "..."}}]}}
}}

Rules: Max {max_entities}/chunk. Types: PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT, PRODUCT, TECHNOLOGY.
JSON:"""

    BATCH_RELATIONSHIP_EXTRACTION_PROMPT: str = """Extract relationships from {num_chunks} text chunks using the provided entities.
The chunks are from the same document. Identify relationships between ANY of the provided entities that are explicitly evidenced in the text.

{chunk_blocks}

Return ONLY valid JSON keyed by chunk index:
{{
  "0": {{"relationships": [{{"source": "Name", "target": "Name", "type": "VERB", "description": "..."}}]}}
}}

Rules: Max {max_relationships}/chunk. Types: concise verbs.
JSON:"""

    # --- Garbage Filtering Parameters ---
    # These filters catch truly low-quality chunks (noise, artifacts, malformed text)
    # without over-pruning legitimate contextual content.
    
    FILTER_REPETITION_THRESHOLD: float = 0.4       # Max ratio of most common character
    FILTER_MIN_ENTROPY: float = 3.0                # Min Shannon entropy (information density)
    FILTER_MAX_ENTROPY: float = 7.0                # Max entropy (catches random noise)
    FILTER_MALFORMED_THRESHOLD: float = 0.05       # Ratio of broken ligatures to word count
    FILTER_MAX_WHITESPACE_DENSITY: float = 0.30    # Max ratio of whitespace to total chars
    
    # REMOVED: FILTER_MIN_ENTITIES (2026-01-08)
    # Reason: Entity count is not a reliable quality signal.
    # - GLiNER produces 0 entities for 99% of legitimate chunks (descriptions, context)
    # - Created arbitrary threshold with no useful granularity (0=keep all, 1=delete 99%)
    # - Domain-dependent (financial docs have more entities than scientific papers)
    # - Existing filters (entropy, repetition, malformed) already catch garbage
    
    FILTER_QUALITY_THRESHOLD: float = 0.75         # LLM quality score threshold (optional)
    FILTER_EMBEDDING_OUTLIER_THRESHOLD: float = 0.50  # Safer threshold for many-topic documents
    MIN_CHUNKS_FOR_OUTLIER_DETECTION: int = 300
    
    # Test isolation - separate database for function tests
    TEST_DB_PATH: str = "./.DuckDB/test_graphrag.duckdb"

    
    QUALITY_SCORING_PROMPT: str = """
    Evaluate the following text chunk for information quality. 
    Garbage text includes navigation menus, repeated footers, binary/random data, or extremely repetitive formatting.
    High-quality text contains meaningful sentences, facts, or descriptions.
    
    Score 1.0 for high-quality content, 0.0 for pure garbage, and values in between for mixed content.
    Return ONLY a single float number between 0.0 and 1.0.
    
    TEXT:
    {text}
    
    SCORE:"""

    # Paths
    INPUT_DIR: str = "./input"
    OUTPUT_DIR: str = "./output"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Singleton instance
settings = GraphRAGSettings()
