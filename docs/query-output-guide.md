# Understanding GraphRAG Query Output

## Search Modes Explained

### 1. `entity_connections` (Default)

**Best for**: Finding specific entities and how they relate to each other

**Output Structure**:

- `chunks`: Top text passages ranked by relevance (BM25 + vector hybrid)
- `entities`: Top entities found in those chunks
- `relationships`: ALL relationships involving those entities (from entire database)
- `evidence`:
  - `directRelationships`: Relationships extracted from the top-ranked chunks (most relevant)
  - `extendedRelationships`: Relationships one hop away (graph traversal)

**Why two relationship lists?**

- `relationships`: Complete context - shows everything we know about these entities
- `evidence.directRelationships`: Focused proof - relationships directly supported by your query results
- `evidence.extendedRelationships`: Extended context - entities connected to your results

**Example**:
Query: "NVIDIA Rubin"

- `relationships`: All 35 NBIS/NVIDIA/Europe relationships in database
- `evidence.directRelationships`: 30 relationships from top 9 chunks (directly about Rubin/Nebius)
- `evidence.extendedRelationships`: 5 relationships one hop away (e.g., Microsoft→hyperscalers→Nebius)

### 2. `thematic_overview`

**Best for**: Understanding broad themes and topics

**Output Structure**:

- `chunks`: Top community summaries (high-level themes)
- `entities`: Entities from those communities
- `relationships`: Relationships within those communities
- No `evidence` section (community-based, not graph traversal)

### 3. `keyword_lookup`

**Best for**: Pure text search without entity graph

**Output Structure**:

- `chunks`: Top text passages (no entity filtering)
- `entities`: Entities found in top chunks
- `relationships`: Relationships between those entities
- No `evidence` section (no graph traversal)

## Reading the Output

### Chunk Fields

- `text`: The actual content
- `vectorScore`: Semantic similarity (0-1, higher = more similar)
- `bm25Score`: Keyword relevance (varies, higher = more relevant)
- `fusedScore`: Combined score using Reciprocal Rank Fusion
- `vectorRank`: Where it ranked in pure semantic search
- `bm25Rank`: Where it ranked in pure keyword search

### Relationship Fields

- `id`: Internal relationship ID
- `source`: Source entity ID
- `target`: Target entity ID
- `type`: Relationship type (verb like USES, DEPLOYS, CONTRACTS)
- `description`: Human-readable explanation
- `weight`: Relationship strength (usually 1.0)
- `sourceName`: Name of source entity (added for readability)
- `sourceType`: Type of source entity
- `targetName`: Name of target entity (added for readability)
- `targetType`: Type of target entity

### Entity Fields

- `id`: Internal entity ID
- `name`: Entity name
- `type`: PERSON, ORGANIZATION, TECHNOLOGY, PRODUCT, LOCATION, EVENT, CONCEPT
- `description`: How/where it was extracted (includes confidence scores)
- `sourceChunks`: Which chunks mention this entity

## Common Patterns

### High BM25, Low Vector

Chunk contains exact keywords but isn't semantically similar (e.g., mentions "NVIDIA" but talks about a different topic)

### High Vector, Low BM25

Chunk is semantically related but doesn't use exact keywords (e.g., talks about "AI accelerators" when you searched "NVIDIA GPU")

### High Fusion Score

Best of both worlds - contains keywords AND is semantically relevant
