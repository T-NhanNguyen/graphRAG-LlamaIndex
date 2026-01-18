"""
Local LLM Client - Entity and relationship extraction via Docker-hosted models.

Connects to OpenAI-compatible endpoints (Ollama, Docker Model Runner) for:
- Entity extraction from text chunks
- Relationship identification between entities
- Structured JSON output parsing

Following coding framework guidelines:
- Typed parameters and returns for LLM tool compatibility
- Retry logic with contextual logging
- No silent failures
"""
import json
import re
import logging
import uuid
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import httpx

from graphrag_config import settings, EntityType, RelationshipProvider
from duckdb_store import Entity, Relationship

from openai import OpenAI

logger = logging.getLogger(__name__)

# Enable httpx logging for LLM requests (matches embedding_provider behavior)
logging.getLogger("httpx").setLevel(logging.INFO)


# Result Dataclass
@dataclass
class ExtractionResult:
    """Result from entity/relationship extraction."""
    entities: List[Entity]
    relationships: List[Relationship]
    rawResponse: str
    success: bool
    errorMessage: Optional[str] = None


class LocalLLMClient:
    """
    Client for local LLM entity extraction.
    
    Uses OpenAI-compatible API format for Ollama/Docker Model Runner.
    """
    
    def __init__(self, baseUrl: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            baseUrl: API endpoint (defaults to settings.LLM_URL)
            model: Model name (defaults to settings.LLM_MODEL)
        """
        self.baseUrl = baseUrl or settings.LLM_URL
        self.model = model or settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.maxTokens = settings.LLM_MAX_TOKENS
        self.maxEntities = settings.MAX_ENTITIES_PER_CHUNK
        self.maxRelationships = settings.MAX_RELATIONSHIPS_PER_CHUNK
        
        logger.info(f"LLM client initialized: {self.baseUrl} using {self.model}")
    
    def _callLLM(self, prompt: str, taskDescription: str = "LLM request") -> Tuple[str, Optional[str]]:
        """
        Make a chat completion request to the LLM.
        
        Returns:
            Tuple of (response_text, error_message)
        """
        # Docker Model Runner uses OpenAI-compatible /v1/chat/completions endpoint
        endpoint = f"{self.baseUrl}/v1/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": settings.LLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.maxTokens,
            "num_ctx": settings.LLM_CONTEXT_LENGTH  # Explicit context window for llama.cpp
        }
        
        # Diagnostic logging: estimate token count (rough: 4 chars per token)
        systemPromptLen = len(settings.LLM_SYSTEM_PROMPT)
        promptLen = len(prompt)
        estimatedInputTokens = (systemPromptLen + promptLen) // 4
        logger.debug(f"LLM request: ~{estimatedInputTokens} input tokens, max_tokens={self.maxTokens}, num_ctx={settings.LLM_CONTEXT_LENGTH}")
        
        try:
            with httpx.Client(timeout=600.0) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
                
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                return content, None
                
        except httpx.TimeoutException:
            error = f"LLM request timed out after 600s"
            logger.error(f"Error: {error}. Prompt snippet: {prompt[:500]}...")
            return "", error
        except httpx.HTTPStatusError as exc:
            error = f"LLM API error: {exc.response.status_code}: {exc.response.text}"
            logger.error(f"Error: {error}. Prompt snippet: {prompt[:500]}...")
            return "", error
        except httpx.ConnectError as exc:
            error = f"Cannot connect to LLM at {self.baseUrl}"
            logger.error(f"Error: {error}. Prompt snippet: {prompt[:500]}...")
            return "", error
        except Exception as exc:
            error = f"LLM request failed: {exc}"
            logger.error(f"[ERROR] {error}. Prompt snippet: {prompt[:500]}...")
            return "", error
    
    def _normalizeJson(self, text: str) -> str:
        """
        Normalize common LLM JSON hallucinations before parsing.
        
        Handles patterns like:
        - "key":": "value" -> "key": "value" (extra colon after key)
        - "key":: "value" -> "key": "value" (double colon)
        """
        normalizedText = re.sub(r'":\s*":\s*"', '": "', text)
        normalizedText = re.sub(r'"::\s*"', '": "', normalizedText)
        return normalizedText
    
    def _parseJson(self, text: str) -> Optional[Dict]:
        """
        Attempt to parse JSON from LLM response, handling common issues.
        Includes a repair mechanism for truncated JSON.
        """
        if not text:
            return None
        
        # Strip thinking tags (DeepSeek R1 style)
        text = re.sub(r'<think>[\s\S]*?</think>', '', text)
        processed_text = text.strip()
        
        processed_text = self._normalizeJson(processed_text)
        
        # 1. Try direct parse
        try:
            return json.loads(processed_text)
        except json.JSONDecodeError:
            pass
        
        # 2. Try to extract JSON from markdown code blocks
        # Improved: handle unclosed markdown blocks and apply repair if needed
        jsonMatch = re.search(r'```(?:json)?\s*([\s\S]*?)(?:```|$)', processed_text)
        if jsonMatch and '{' in jsonMatch.group(1):
            jsonStr = jsonMatch.group(1).strip()
            try:
                return json.loads(jsonStr)
            except json.JSONDecodeError:
                # Try repair on the extracted markdown content
                repaired = self._tryRepairTruncated(jsonStr)
                if repaired:
                    return repaired
        
        # 3. Try to find JSON structure and fix common errors
        jsonMatch = re.search(r'(\{[\s\S]*\})', processed_text)
        if not jsonMatch:
            # Maybe it started but never finished? (Truncated)
            jsonMatch = re.search(r'(\{[\s\S]*)', processed_text)
            
        if jsonMatch:
            jsonStr = jsonMatch.group(1)
            # Try to fix common issues
            # Remove trailing commas before closing braces/brackets
            jsonStr = re.sub(r',\s*([\]}])', r'\1', jsonStr)
            
            try:
                return json.loads(jsonStr)
            except json.JSONDecodeError:
                # 4. Truncated JSON Repair
                # This helps if the model hits max_tokens halfway through an object
                repaired = self._tryRepairTruncated(jsonStr)
                if repaired:
                    return repaired
        
        logger.warning(f"Failed to parse JSON from response: {text[:200]}...")
        return None

    def _tryRepairTruncated(self, jsonStr: str) -> Optional[Dict]:
        """Attempt to close balance unclosed braces and brackets."""
        # Clean up any trailing text that isn't part of the JSON structure
        # (e.g., if it stopped in the middle of a word)
        # Find the last valid structural character
        last_struct = -1
        for i, char in enumerate(reversed(jsonStr)):
            if char in '}],":':
                last_struct = len(jsonStr) - 1 - i
                break
        
        if last_struct == -1: return None
        
        # Cut off any dangling text after the last potentially valid structural point
        potential_json = jsonStr[:last_struct+1]
        
        # Try to balance
        stack = []
        for char in potential_json:
            if char == '{': stack.append('}')
            elif char == '[': stack.append(']')
            elif char == '}': 
                if stack and stack[-1] == '}': stack.pop()
            elif char == ']': 
                if stack and stack[-1] == ']': stack.pop()
        
        # Append missing closing characters in reverse order
        repaired = potential_json + "".join(reversed(stack))
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            # Final attempt: remove the last partial object if it's still failing
            # e.g., if it cut off inside an object in an array
            # Find the last comma and try to close there
            last_comma = potential_json.rfind(',')
            if last_comma != -1:
                potential_json = potential_json[:last_comma]
                stack = []
                for char in potential_json:
                    if char == '{': stack.append('}')
                    elif char == '[': stack.append(']')
                    elif char == '}': 
                        if stack and stack[-1] == '}': stack.pop()
                    elif char == ']': 
                        if stack and stack[-1] == ']': stack.pop()
                repaired = potential_json + "".join(reversed(stack))
                try:
                    return json.loads(repaired)
                except:
                    pass
            return None
    
    def extractEntities(self, text: str, chunkId: str, sourceDocumentId: str = "") -> List[Entity]:
        """
        Extract entities from text chunk.
        
        Args:
            text: Source text to extract from
            chunkId: ID of the chunk (for source tracking)
            sourceDocumentId: ID of the source document
            
        Returns:
            List of Entity objects
        """
        prompt = settings.ENTITY_EXTRACTION_PROMPT.format(
            text=text[:3000],  # Limit context length
            max_entities=self.maxEntities
        )
        
        response, error = self._callLLM(prompt, taskDescription=f"entity extraction for chunk {chunkId}")
        
        if error:
            logger.warning(f"Entity extraction failed for chunk {chunkId}: {error}")
            return []
        
        parsed = self._parseJson(response)
        if not parsed or "entities" not in parsed:
            logger.warning(f"Failed to parse entity response for chunk {chunkId}")
            return []
        
        entities = []
        for e in parsed.get("entities", []):
            try:
                # Standard keys: name, type, description
                entityType = e.get("type", "CONCEPT").upper()
                if entityType not in [t.value for t in EntityType]:
                    entityType = "CONCEPT"
                
                name = e.get("name", "Unknown")
                entity = Entity(
                    entityId=str(uuid.uuid4()),
                    name=name,
                    canonicalName=name,
                    entityType=EntityType(entityType),
                    description=e.get("description", ""),
                    sourceDocumentIds=[sourceDocumentId] if sourceDocumentId else [],
                    sourceChunkIds=[chunkId]
                )
                entities.append(entity)
            except Exception as exc:
                logger.warning(f"Error creating entity: {exc}")
                continue
        
        logger.info(f"Extracted {len(entities)} entities from chunk {chunkId}")
        return entities
    
    def extractRelationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships between entities.
        
        Args:
            text: Source text
            entities: Previously extracted entities
            
        Returns:
            List of Relationship objects
        """
        if not entities:
            return []
        
        # Build entity name mapping for reference
        entityNames = [e.name for e in entities]
        entityMap = {e.name.lower(): e for e in entities}
        
        prompt = settings.RELATIONSHIP_EXTRACTION_PROMPT.format(
            entity_names=", ".join(entityNames),
            text=text[:3000],
            max_relationships=self.maxRelationships
        )
        
        response, error = self._callLLM(prompt, taskDescription="relationship extraction")
        
        if error:
            logger.warning(f"Relationship extraction failed: {error}")
            return []
        
        parsed = self._parseJson(response)
        if not parsed or "relationships" not in parsed:
            logger.warning("Failed to parse relationship response")
            return []
        
        relationships = []
        for r in parsed.get("relationships", []):
            try:
                # Standard keys: source, target, type, description
                sourceName = r.get("source", "").lower()
                targetName = r.get("target", "").lower()
                
                # Find matching entities
                sourceEntity = entityMap.get(sourceName)
                targetEntity = entityMap.get(targetName)
                
                if not sourceEntity or not targetEntity:
                    # Try partial matching
                    for name, entity in entityMap.items():
                        if sourceName in name or name in sourceName:
                            sourceEntity = sourceEntity or entity
                        if targetName in name or name in targetName:
                            targetEntity = targetEntity or entity
                
                if not sourceEntity or not targetEntity:
                    continue
                
                relationship = Relationship(
                    relationshipId=str(uuid.uuid4()),
                    sourceEntityId=sourceEntity.entityId,
                    targetEntityId=targetEntity.entityId,
                    relationshipType=r.get("type", "RELATED_TO").upper(),
                    description=r.get("description", ""),
                    weight=1.0
                )
                relationships.append(relationship)
            except Exception as exc:
                logger.warning(f"Error creating relationship: {exc}")
                continue
        
        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def extract(self, text: str, chunkId: str, sourceDocumentId: str = "") -> ExtractionResult:
        """
        Full extraction pipeline: entities then relationships.
        
        Args:
            text: Source text chunk
            chunkId: Chunk identifier
            sourceDocumentId: Source document identifier
            
        Returns:
            ExtractionResult with entities and relationships
        """
        try:
            entities = self.extractEntities(text, chunkId, sourceDocumentId)
            relationships = self.extractRelationships(text, entities) if entities else []
            
            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                rawResponse="",
                success=True
            )
        except Exception as exc:
            return ExtractionResult(
                entities=[],
                relationships=[],
                rawResponse="",
                success=False,
                errorMessage=str(exc)
            )
    
    def extractEntitiesBatch(self, chunks: List) -> Dict[str, List[Entity]]:
        """
        Extract entities from multiple chunks in a single LLM call.
        
        Uses explicit chunk separators to prevent cross-contamination.
        Each chunk is processed independently by the LLM.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Dict mapping chunk IDs to their extracted entities
        """
        if not chunks:
            return {}
        
        # Build chunk blocks with explicit separators
        chunkBlocks = []
        for idx, chunk in enumerate(chunks):
            chunkBlocks.append(f"===CHUNK_{idx}===\n{chunk.text[:3000]}\n===END_CHUNK_{idx}===")
        
        prompt = settings.BATCH_ENTITY_EXTRACTION_PROMPT.format(
            num_chunks=len(chunks),
            chunk_blocks="\n\n".join(chunkBlocks),
            max_entities=self.maxEntities
        )
        
        chunkIds = [c.chunkId for c in chunks]
        response, error = self._callLLM(prompt, taskDescription=f"batch entity extraction ({len(chunks)} chunks)")
        
        if error:
            logger.warning(f"Batch entity extraction failed: {error}")
            return {cid: [] for cid in chunkIds}
        
        parsed = self._parseJson(response)
        if not parsed:
            logger.warning(f"Failed to parse batch entity response")
            return {cid: [] for cid in chunkIds}
        
        # Parse keyed response format: {"0": {...}, "1": {...}}
        results = {}
        for idx, chunk in enumerate(chunks):
            chunkData = parsed.get(str(idx), {})
            rawEntities = chunkData.get("entities", []) if isinstance(chunkData, dict) else []
            
            entities = []
            for e in rawEntities:
                try:
                    # Standard keys: name, type, description
                    entityType = e.get("type", "CONCEPT").upper()
                    if entityType not in [t.value for t in EntityType]:
                        entityType = "CONCEPT"
                    
                    name = e.get("name", "Unknown")
                    entity = Entity(
                        entityId=str(uuid.uuid4()),
                        name=name,
                        canonicalName=name,
                        entityType=EntityType(entityType),
                        description=e.get("description", ""),
                        sourceDocumentIds=[chunk.sourceDocumentId] if chunk.sourceDocumentId else [],
                        sourceChunkIds=[chunk.chunkId]
                    )
                    entities.append(entity)
                except Exception as exc:
                    logger.warning(f"Error creating entity in batch: {exc}")
                    continue
            
            results[chunk.chunkId] = entities
        
        totalEntities = sum(len(e) for e in results.values())
        logger.info(f"Batch extracted {totalEntities} entities from {len(chunks)} chunks")
        return results
    
    def extractRelationshipsBatch(self, chunks: List, entityMap: Dict[str, List[Entity]]) -> Dict[str, List[Relationship]]:
        """
        Extract relationships from multiple chunks in a single LLM call.
        
        Following bulk fetch guidelines to minimize network loops.
        Provides all local document context (entities from all chunks in batch)
        to allow cross-chunk relationship discovery.
        
        Args:
            chunks: List of DocumentChunk objects
            entityMap: mapping of chunkId to their already extracted entities
            
        Returns:
            Dict mapping chunk IDs to their extracted relationships
        """
        if not chunks:
            return {}
        
        # Build set of all entity names across this batch (document scope)
        allEntityNamesSet = set()
        allEntities = []
        for entities in entityMap.values():
            for e in entities:
                allEntityNamesSet.add(e.name)
                allEntities.append(e)
        
        allEntityNames = ", ".join(sorted(list(allEntityNamesSet)))
        
        # Global entity lookup for the whole batch
        entityLookup = {e.name.lower(): e for e in allEntities}
        
        # Build chunk blocks
        chunkBlocks = []
        for idx, chunk in enumerate(chunks):
            chunkBlocks.append(f"===CHUNK_{idx}===\nEntities present in document section: {allEntityNames}\nText: {chunk.text[:3000]}\n===END_CHUNK_{idx}===")
        
        prompt = settings.BATCH_RELATIONSHIP_EXTRACTION_PROMPT.format(
            num_chunks=len(chunks),
            chunk_blocks="\n\n".join(chunkBlocks),
            max_relationships=self.maxRelationships
        )
        
        chunkIds = [c.chunkId for c in chunks]
        response, error = self._callLLM(prompt, taskDescription=f"batch relationship extraction ({len(chunks)} chunks)")
        
        if error:
            logger.warning(f"Batch relationship extraction failed: {error}")
            return {cid: [] for cid in chunkIds}
        
        parsed = self._parseJson(response)
        if not parsed:
            logger.warning(f"Failed to parse batch relationship response")
            return {cid: [] for cid in chunkIds}
        
        results = {}
        for idx, chunk in enumerate(chunks):
            chunkData = parsed.get(str(idx), {})
            rawRels = chunkData.get("relationships", []) if isinstance(chunkData, dict) else []
            
            relationships = []
            for r in rawRels:
                try:
                    sourceName = r.get("source", "").lower()
                    targetName = r.get("target", "").lower()
                    
                    sourceEntity = entityLookup.get(sourceName)
                    targetEntity = entityLookup.get(targetName)
                    
                    # Fuzzy matching for robustness
                    if not sourceEntity or not targetEntity:
                        for name, entity in entityLookup.items():
                            if sourceName in name or name in sourceName:
                                sourceEntity = sourceEntity or entity
                            if targetName in name or name in targetName:
                                targetEntity = targetEntity or entity
                    
                    if not sourceEntity or not targetEntity:
                        continue
                    
                    relationship = Relationship(
                        relationshipId=str(uuid.uuid4()),
                        sourceEntityId=sourceEntity.entityId,
                        targetEntityId=targetEntity.entityId,
                        relationshipType=r.get("type", "RELATED_TO").upper(),
                        description=r.get("description", ""),
                        weight=1.0
                    )
                    relationships.append(relationship)
                except Exception as exc:
                    logger.warning(f"Error creating relationship in batch: {exc}")
                    continue
            results[chunk.chunkId] = relationships
            
        totalRels = sum(len(r) for r in results.values())
        logger.info(f"Batch extracted {totalRels} relationships from {len(chunks)} chunks")
        return results

    def extractBatch(self, chunks: List) -> List[ExtractionResult]:
        """
        Full extraction pipeline for a batch of chunks.
        
        Uses dual-batch strategy (Entities then Relationships) for maximum efficiency.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List of ExtractionResult objects, one per chunk
        """
        if not chunks:
            return []
        
        # 1. Batch entity extraction (1 LLM call)
        entityMap = self.extractEntitiesBatch(chunks)
        
        # 2. Batch relationship extraction (1 LLM call)
        relationshipMap = self.extractRelationshipsBatch(chunks, entityMap)
        
        results = []
        for chunk in chunks:
            results.append(ExtractionResult(
                entities=entityMap.get(chunk.chunkId, []),
                relationships=relationshipMap.get(chunk.chunkId, []),
                rawResponse="",
                success=True
            ))
        
        return results
    
    def isAvailable(self) -> bool:
        """Check if the LLM endpoint is reachable via Docker Model Runner."""
        try:
            with httpx.Client(timeout=10.0) as client:
                # Docker Model Runner uses /v1/models endpoint
                response = client.get(f"{self.baseUrl}/v1/models")
                if response.status_code == 200:
                    logger.info(f"LLM endpoint available: {self.baseUrl}")
                    return True
                else:
                    logger.warning(f"LLM endpoint returned {response.status_code}")
                    return False
        except Exception as exc:
            logger.warning(f"Warning: LLM endpoint not reachable at {self.baseUrl}: {exc}")
            return False

    def summarizeCommunity(self, communityId: str, entities: List[Entity]) -> str:
        """
        Generate a thematic summary for a group of entities.
        
        Args:
            communityId: Identifier for the community
            entities: List of entities in this cluster
            
        Returns:
            Thematic summary string
        """
        if not entities:
            return "Empty community."
            
        # Build context from entity descriptions
        entityContext = "\n".join([
            f"- {e.name} ({e.entityType.value if isinstance(e.entityType, EntityType) else e.entityType}): {e.description}"
            for e in entities
        ])
        
        prompt = settings.COMMUNITY_SUMMARY_PROMPT.format(entity_context=entityContext)
        
        response, error = self._callLLM(prompt, taskDescription=f"summary for community {communityId}")
        
        if error:
            logger.warning(f"Community summarization failed for {communityId}: {error}")
            return f"Summary unavailable due to error: {error}"
            
        return response.strip()

    def scoreQuality(self, text: str) -> float:
        """
        Score the quality of a text chunk using the LLM.
        
        Args:
            text: Text chunk to score
            
        Returns:
            Float score between 0.0 and 1.0 (default 1.0 on failure)
        """
        prompt = settings.QUALITY_SCORING_PROMPT.format(text=text)
        response, error = self._callLLM(prompt, taskDescription="Quality Scoring")
        
        if error:
            logger.warning(f"Quality scoring failed: {error}. Defaulting to 1.0")
            return 1.0
            
        try:
            # Try to extract the first float found in the response
            # Some models might wrap it in text despite the prompt
            match = re.search(r"[-+]?\d*\.\d+|\d+", response)
            if match:
                score = float(match.group())
                return max(0.0, min(1.0, score))
            return 1.0
        except Exception as e:
            logger.warning(f"Failed to parse quality score from '{response}': {e}. Defaulting to 1.0")
            return 1.0


class OpenRouterClient(LocalLLMClient):
    """
    Client for OpenRouter API extraction.
    
    Compatible with OpenAI SDK, optimized for relationship extraction grunt work.
    """
    
    def __init__(self, apiKey: Optional[str] = None, model: Optional[str] = None):
        """Initialize OpenRouter client."""
        super().__init__()
        self.apiKey = apiKey or settings.OPENROUTER_API_KEY
        self.model = model or settings.OPENROUTER_MODEL
        
        if not self.apiKey:
            logger.error("OpenRouter API key not found in settings or environment!")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.apiKey,
        )
        logger.info(f"OpenRouter client initialized using {self.model}")

    def _callLLM(self, prompt: str, taskDescription: str = "OpenRouter request") -> Tuple[str, Optional[str]]:
        """Make a chat completion request via OpenRouter."""
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/graphrag-local", # Standard identifier
                    "X-Title": "GraphRAG Local Indexer",
                },
                model=self.model,
                messages=[
                    {"role": "system", "content": settings.LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.maxTokens
            )
            
            content = completion.choices[0].message.content
            # Track usage for observability
            usage = getattr(completion, 'usage', None)
            if usage:
                logger.info(f"{taskDescription} usage: {usage.prompt_tokens}p + {usage.completion_tokens}c = {usage.total_tokens} tokens")
            
            return content, None
            
        except Exception as exc:
            error = f"OpenRouter API error: {exc}"
            logger.error(f"Error: {error}")
            return "", error

    def isAvailable(self) -> bool:
        """Check if OpenRouter API is reachable (basic key check)."""
        return bool(self.apiKey) and len(self.apiKey) > 10


def getLLMClient(baseUrl: Optional[str] = None, model: Optional[str] = None) -> LocalLLMClient:
    """
    Factory function for GENERAL LLM client (entities, summarization, pruning).
    Respects the RELATIONSHIP_PROVIDER toggle for architectural consistency.
    """
    provider = settings.RELATIONSHIP_PROVIDER
    
    if provider == RelationshipProvider.OPENROUTER:
        return OpenRouterClient(model=model)
    
    return LocalLLMClient(baseUrl, model)


def getRelationshipClient() -> LocalLLMClient:
    """Factory function for RELATIONSHIP extraction client."""
    provider = settings.RELATIONSHIP_PROVIDER
    
    if provider == RelationshipProvider.OPENROUTER:
        return OpenRouterClient()
    
    return LocalLLMClient()
