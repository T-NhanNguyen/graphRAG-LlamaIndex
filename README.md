# GraphRAG LlamaIndex

Full-stack GraphRAG engine optimized for local indexing and lightweight cloud querying. Built on DuckDB, LlamaIndex, and the Model Context Protocol (MCP).

## 1. Context

This project implements a **Decoupled Architecture**:

- **Indexer (Image A)**: Heavy-duty ML environment (PyTorch, GLiNER) for local graph construction.
- **Query (Image B)**: Lightweight API environment (Node.js, DuckDB) for fast cloud deployment (~1GB footprint).

## 2. Building the Images

```bash
# Build specialized images via Docker Compose
docker compose build
```

## 3. Running Indexer & Query

Load the shell aliases for the fastest workflow:

- **macOS (Zsh)**: `source .graphrag-alias.zsh`
- **WSL/Bash**: `source .graphrag-alias.sh`
- **PowerShell**: `. .\.graphrag-alias.ps1`

### Indexing Documents

```bash
# 1. Initialize a database entry
graphrag start my_project --source /app/documents/source_files

# 2. Run the ingestion pipeline (Indexer Image)
graphrag index my_project [--reset] [--prune]
```

### Querying

```bash
# CLI Search (Query Image)
graphrag search my_project "What are the common themes?"

# Start MCP Server for Agents
docker compose up query
```

## 4. Deployment Folder

- `deployment/fly/`: Scripts for zero-latency hosting on Fly.io (optimized for free-tier fly-machines).
- `deployment/aws/`: Infrastructure scripts for ECR, S3 backups, and App Runner deployments.

## 5. Setup & Configuration

### Environment (.env)

Copy `.env.example` and set:

- `OPENAI_API_KEY`: For LLM reasoning and extraction.
- `DOCUMENTS_HOME`: Absolute path to your local data folder (mapped to `/app/documents` in Docker).

### Engine Configuration (core/graphrag_config.py)

Tweak these parameters to refine performance:

- **`SearchType`**: Switch between `entity_connections` (graph-heavy) or `thematic_overview` (summary-heavy).
- **`ExtractionMode`**: Choose `llm` (creative) or `gliner` (fast/cost-effective).

## 6. Local Integration & Testing

Integrate this project as an **MCP Server** in Desktop Agents (Claude Desktop, Cursor, etc.) to give them memory of your documents.

**Add to your MCP Config:**

```json
{
  "mcpServers": {
    "graphrag": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v",
        "C:/Users/<USER>/.graphrag:/root/.graphrag",
        "graphrag-query"
      ]
    }
  }
}
```

## 7. Database Cheatsheet

### Core Commands

- `graphrag list`: Show all registered databases.
- `graphrag status <db>`: Check entity/relationship counts and health.
- `graphrag delete <db> [--files]`: Unregister database entry and optionally remove physical files.
- `graphrag index <db> [--reset]`: Index documents into database (use `--reset` to skip duplicate checks).

### Managed Storage Workflow

To keep your project portable, move database files into a `Managed/` folder inside your data directory.

**Manual Move:**

1. Move `your_db.duckdb` to `[DOCUMENTS_HOME]/Managed/`.
2. Re-register the path:

```bash
graphrag register my_db --db-path /app/documents/Managed/your_db.duckdb
```

---

_For S3 management see [S3_CHEATSHEET.md](deployment/aws/S3_CHEATSHEET.md)._
