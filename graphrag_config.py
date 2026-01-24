# GraphRAG Configuration - Centralized constants and settings for engine and agentic tools.
from enum import Enum
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class EntityType(str, Enum):
    # Entity classification types for knowledge graph nodes.
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"


class EmbeddingProvider(str, Enum):
    # Supported embedding providers.
    DOCKER_MODEL_RUNNER = "docker_model_runner"
    OPENAI = "openai"
    OLLAMA = "ollama"


class SearchType(str, Enum):
    # Query search strategy.
    FIND_CONNECTIONS = "find_connections" # Hybrid + Graph traversal (local context)
    EXPLORE_THEMATIC = "explore_thematic" # Community-level reasoning (global themes)
    KEYWORD_SEARCH = "keyword_search"     # BM25 + vector hybrid text search only


class ExtractionMode(str, Enum):
    # Entity extraction strategies.
    LLM_ONLY = "local_llm"       # Default: Traditional LLM
    HYBRID = "gliner_llm"  # Optional: GLiNER entities + LLM relationships


class RelationshipProvider(str, Enum):
    # Relationship extraction providers.
    LOCAL = "local"           # Use local LLM (Docker Model Runner)
    OPENROUTER = "openrouter"  # Use OpenRouter API


class GraphRAGSettings(BaseSettings):
    # Runtime configuration loaded from .env. Magic numbers centralized here.
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


    
    # DuckDB Storage (Registry managed, defaults provided for safety)
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
    BM25_LANGUAGE: str = "en"  # Stopword language (en, zh, ja, ko, fr, de, es, etc.)
    
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

    # --- Extraction Prompts (Optimized for batch processing with noise resilience) ---
    LLM_SYSTEM_PROMPT: str = """You are a knowledge extraction specialist. Parse text despite formatting issues (extra whitespace, broken lines, OCR artifacts). Ignore noise: navigation menus, footers, social links, author bios, ads, publication metadata. Output ONLY valid JSON."""
    
    ENTITY_EXTRACTION_PROMPT: str = """Extract key entities from the text. Infer meaning from context even with bad formatting.

IGNORE: URLs, social handles, "Share/Follow/Subscribe", author bios, copyright notices, navigation, ads, image captions, "Read more", publication dates/sources.

Types: PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT, PRODUCT, TECHNOLOGY
Format: {{"entities": [{{"name": "Proper Name", "type": "TYPE", "description": "Core role or significance"}}]}}
Rules: Max {max_entities}. Merge duplicates. Capitalize names. Infer full names from partials when obvious.

Text: {text}

JSON:"""

    RELATIONSHIP_EXTRACTION_PROMPT: str = """Entities: {entity_names}
Extract relationships evidenced in text. Infer implicit connections when strongly suggested.

IGNORE: Links, ads, boilerplate, author/publisher info.

Format: {{"relationships": [{{"source": "Name", "target": "Name", "type": "VERB_PHRASE", "description": "Why connected"}}]}}
Rules: Max {max_relationships}. Types: action verbs (LEADS, ACQUIRED_BY, LOCATED_IN, DEVELOPS). Match entity names exactly as given.

Text: {text}

JSON:"""
    # Entity Extraction
    MAX_ENTITIES_PER_CHUNK: int = 10
    MAX_RELATIONSHIPS_PER_CHUNK: int = 15
    
    BATCH_ENTITY_EXTRACTION_PROMPT: str = """Extract entities from {num_chunks} chunks. Process each independently. Tolerate formatting issues.

IGNORE per chunk: URLs, social links, nav menus, footers, "Share/Subscribe", author bios, ads, image refs, publication metadata, copyright.

{chunk_blocks}

Format (keyed by chunk index):
{{
  "0": {{"entities": [{{"name": "Proper Name", "type": "TYPE", "description": "Core significance"}}]}}
}}

Types: PERSON, ORGANIZATION, CONCEPT, LOCATION, EVENT, PRODUCT, TECHNOLOGY
Rules: Max {max_entities}/chunk. Merge duplicates within chunk. Infer full names. Capitalize properly.
JSON:"""

    BATCH_RELATIONSHIP_EXTRACTION_PROMPT: str = """Extract relationships from {num_chunks} chunks using provided entities. Chunks share document context—cross-reference allowed.

IGNORE: Boilerplate, links, author info, ads, formatting artifacts.

{chunk_blocks}

Format (keyed by chunk index):
{{
  "0": {{"relationships": [{{"source": "Name", "target": "Name", "type": "VERB_PHRASE", "description": "Evidence summary"}}]}}
}}

Rules: Max {max_relationships}/chunk. Match entity names exactly. Types: action verbs. Infer implicit connections if strongly evidenced.
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

    
    QUALITY_SCORING_PROMPT: str = """Score text quality (0.0-1.0). Ignore formatting flaws (whitespace, line breaks).

GARBAGE (0.0-0.3): Nav menus, "Share/Subscribe/Follow", social links, author bios, copyright, ads, image captions, cookie notices, random chars, repeated headers/footers, URL lists, "Read more at...", publication boilerplate.

MIXED (0.4-0.6): Some useful content buried in noise.

QUALITY (0.7-1.0): Substantive info—facts, analysis, descriptions, arguments, data. Minor formatting issues acceptable.

TEXT:
{text}

Return ONLY a float:"""

    # Registry-managed paths (Overridden by forDatabase factory)
    INPUT_DIR: str = ""
    OUTPUT_DIR: str = ""
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    @classmethod
    def forDatabase(cls, dbName: str = None) -> "GraphRAGSettings":
        # Create settings instance with database-specific paths from workspace registry.
        from workspace_config import getRegistry
        
        registry = getRegistry()
        dbConfig = registry.getOrDefault(dbName)
        
        # Create new instance with .env settings
        instance = cls()
        
        # Override paths with database-specific values
        instance.DUCKDB_PATH = dbConfig.dbPath
        instance.INPUT_DIR = dbConfig.sourceFolder
        instance.OUTPUT_DIR = dbConfig.outputFolder
        
        return instance


# Singleton instance (default settings from .env)
settings = GraphRAGSettings()


def getSettingsForDatabase(dbName: str = None) -> GraphRAGSettings:
    # Get settings for a specific database or default singleton.
    if dbName is None:
        return settings  # Return global singleton for default
    return GraphRAGSettings.forDatabase(dbName)
