import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass

from graphrag_config import settings, ExtractionMode
from duckdb_store import Entity, DocumentChunk

logger = logging.getLogger(__name__)

class BaseEntityExtractor(ABC):
    # Ensures different extraction models (LLM, GLiNER) can be swapped seamlessly
    @abstractmethod
    def extractEntities(self, text: str, chunkId: str, sourceDocumentId: str = "") -> List[Entity]:
        # Extract entities from a single text chunk
        pass

    @abstractmethod
    def extractEntitiesBatch(self, chunks: List[DocumentChunk]) -> Dict[str, List[Entity]]:
        # Extract entities from a batch of chunks for efficiency
        pass

class LLMEntityExtractor(BaseEntityExtractor):
    # Adapter for existing LLM-based entity extraction.
    # Wraps LocalLLMClient to adhere to the BaseEntityExtractor interface
    def __init__(self, llmClient):
        self.llmClient = llmClient
        logger.info("Initializing LLMEntityExtractor")

    def extractEntities(self, text: str, chunkId: str, sourceDocumentId: str = "") -> List[Entity]:
        return self.llmClient.extractEntities(text, chunkId, sourceDocumentId)

    def extractEntitiesBatch(self, chunks: List[DocumentChunk]) -> Dict[str, List[Entity]]:
        return self.llmClient.extractEntitiesBatch(chunks)

class ExtractorFactory:
    # To instantiate the appropriate entity extractor based on configuration
    @staticmethod
    def getExtractor(mode: Optional[ExtractionMode] = None, llmClient = None) -> BaseEntityExtractor:
        extractionMode = mode or settings.ENTITY_EXTRACTION_MODE
        
        if extractionMode == ExtractionMode.HYBRID:
            from gliner_extractor import GLiNEREntityExtractor
            return GLiNEREntityExtractor()
        else:
            # Default to LLM_ONLY
            if llmClient is None:
                from llm_client import getLLMClient
                llmClient = getLLMClient()
            return LLMEntityExtractor(llmClient)
