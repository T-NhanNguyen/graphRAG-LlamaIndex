# MCP Server Database Switching

> **Status**: Future consideration - not currently in scope

## Overview

Enhance the existing MCP server to support multiple databases, allowing AI agents to switch context between different knowledge bases.

## Current State

- MCP server connects to single hardcoded database
- No runtime database selection

## Proposed Enhancement

### Updated Tool Signatures

```typescript
// All tools gain optional `database` parameter
search(query: string, mode?: string, topK?: number, database?: string)
explore_entity_graph(entityName: string, depth?: number, database?: string)
get_corpus_stats(database?: string)

// New tool for database management
list_databases()  // Returns available databases
```

### Behavior

- `database` parameter defaults to `"default"` if omitted
- Invalid database name returns structured error
- Agent can call `list_databases()` to discover available corpora

## Example Agent Workflow

```
Agent: list_databases()
→ ["financial-analysis", "technical-docs", "default"]

Agent: search("What is the Q4 revenue?", database="financial-analysis")
→ {chunks: [...], entities: [...]}

Agent: search("How does the embedding model work?", database="technical-docs")
→ {chunks: [...], entities: [...]}
```

## Implementation Notes

### mcp_server.ts Changes

```typescript
// Route database param to Python backend
async function handleSearch(params) {
  const db = params.database || "default";
  const result = await callPython(`mcp.py`, [
    `search('${params.query}', database='${db}')`,
  ]);
  return result;
}
```

### mcp.py Changes

```python
def routeCommand(commandString: str):
    # Parse database param from command
    # Load appropriate settings via GraphRAGSettings.forDatabase(dbName)
    # Execute query against that database
```

## Prerequisites from CLI Design

For MCP database switching to work later:

- [ ] `workspace_config.py` must be complete (registry exists)
- [ ] `GraphRAGSettings.forDatabase()` factory must work
- [ ] Query engine must accept injected settings
- [ ] No global singletons that prevent multi-database access
