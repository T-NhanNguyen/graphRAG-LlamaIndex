import logging
import uuid
import time
import warnings
from typing import List, Dict, Any

from gliner import GLiNER
from entity_extractor import BaseEntityExtractor
from graphrag_config import settings, EntityType
from duckdb_store import Entity, DocumentChunk

logger = logging.getLogger(__name__)

class GLiNEREntityExtractor(BaseEntityExtractor):
    # Local entity extraction using GLiNER transformer encoders for zero-shot NER extraction.
    
    def __init__(self, modelName: str = None, threshold: float = None):
        # Initialize GLiNER extractor with optional model name and confidence threshold.
        self.modelName = modelName or settings.GLINER_MODEL
        # Using FILTER_QUALITY_THRESHOLD as the confidence cutoff as per user feedback
        self.threshold = threshold or settings.FILTER_QUALITY_THRESHOLD
        
        # Mapping EntityType enum to labels used by GLiNER
        # GLiNER works best with Title Case or lowercase labels
        self.labels = ["Person", "Organization", "Concept", "Location", "Event", "Product", "Technology"]
        
        logger.info(f"Loading GLiNER model: {self.modelName} (Threshold: {self.threshold})")
        try:
            # Silence the Transformers/SentencePiece warning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*sentencepiece tokenizer.*")
                self.model = GLiNER.from_pretrained(self.modelName)
            
            # Explicitly set the tokenizer and processor max length to silence truncation warnings.
            # GLiNER versions vary, so we try multiple common attribute locations.
            if hasattr(self.model, 'data_processor'):
                dataProcessor = self.model.data_processor
                if hasattr(dataProcessor, 'transformer_tokenizer'):
                    dataProcessor.transformer_tokenizer.model_max_length = settings.GLINER_MAX_LENGTH
                
                # Some GLiNER versions use max_len or config.max_len in the processor
                for attributeName in ['max_len', 'max_length']:
                    if hasattr(dataProcessor, attributeName):
                        setattr(dataProcessor, attributeName, settings.GLINER_MAX_LENGTH)
                if hasattr(dataProcessor, 'config') and hasattr(dataProcessor.config, 'max_len'):
                    dataProcessor.config.max_len = settings.GLINER_MAX_LENGTH
            
            # Set model level max_len if it exists
            if hasattr(self.model, 'max_len'):
                self.model.max_len = settings.GLINER_MAX_LENGTH

            logger.info("GLiNER model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load GLiNER model {self.modelName}: {exc}")
            raise

    def extractEntities(self, text: str, chunkId: str, sourceDocumentId: str = "") -> List[Entity]:
        # Extract normalized entities from a single text chunk.
        startTime = time.time()
        
        # predict_entities returns a list of dicts: {"start", "end", "text", "label", "score"}
        rawEntities = self.model.predict_entities(
            text, 
            self.labels, 
            threshold=self.threshold,
            flat_ner=True,
            max_length=settings.GLINER_MAX_LENGTH
        )
        
        # Filter entities by length
        filteredRawExtractions = [e for e in rawEntities if 1 < len(e["text"]) < 50]
        
        entities = self._normalizeRawEntities(filteredRawExtractions, chunkId, sourceDocumentId)
        
        latency = time.time() - startTime
        logger.info(f"GLiNER extracted {len(entities)} entities from chunk {chunkId} in {latency:.2f}s")
        
        return entities

    def extractEntitiesBatch(self, chunks: List[DocumentChunk]) -> Dict[str, List[Entity]]:
        # Extract entities from multiple chunks in a single pass.
        if not chunks:
            return {}
            
        startTime = time.time()
        chunkTexts = [chunk.text for chunk in chunks]
        
        # Batch inference
        # GLiNER model.predict_entities can handle a list of texts
        # Note: If memory usage is a concern, we could sub-batch here, 
        # but GLiNER Large is efficient within its context.
        results = {}
        totalEntities = 0
        
        # GLiNER predict_entities generally expects a single string.
        # We loop over chunks for robustness while keeping them isolated.
        for chunk in chunks:
            # Added flat_ner=True to prevent overlapping entities (e.g., "AI" as both Concept/Product)
            # Added max_length from settings to match model architecture
            rawEntities = self.model.predict_entities(
                chunk.text, 
                self.labels, 
                threshold=self.threshold,
                flat_ner=True,
                max_length=settings.GLINER_MAX_LENGTH
            )
            
            # Real entities are rarely longer than 50 characters
            filteredRawExtractions = [
                e for e in rawEntities 
                if len(e["text"]) < 50 and len(e["text"]) > 1
            ]
            
            entities = self._normalizeRawEntities(filteredRawExtractions, chunk.chunkId, chunk.sourceDocumentId)
            results[chunk.chunkId] = entities
            totalEntities += len(entities)
            
        latency = time.time() - startTime
        logger.info(f"GLiNER processed {totalEntities} entities from {len(chunks)} chunks in {latency:.2f}s")
        
        return results

    def _normalizeRawEntities(self, rawEntities: List[Dict[str, Any]], chunkId: str, sourceDocumentId: str) -> List[Entity]:
        # Map GLiNER raw output to the standard Entity schema.
        entities = []
        for rawEntityData in rawEntities:
            try:
                # Map Title Case labels back to EntityType enum values
                label = rawEntityData["label"].upper()
                if label not in [t.value for t in EntityType]:
                    # Fallback to CONCEPT if mismatch
                    label = "CONCEPT"
                
                name = rawEntityData["text"].strip()
                if not name:
                    continue
                    
                entity = Entity(
                    entityId=str(uuid.uuid4()),
                    name=name,
                    canonicalName=name,
                    entityType=EntityType(label),
                    description=f"Extracted by GLiNER ({rawEntityData['label']}) with confidence {rawEntityData['score']:.4f}",
                    sourceDocumentIds=[sourceDocumentId] if sourceDocumentId else [],
                    sourceChunkIds=[chunkId]
                )
                entities.append(entity)
            except Exception as exc:
                logger.warning(f"Failed to normalize GLiNER entity: {exc}")
                continue
        return entities
