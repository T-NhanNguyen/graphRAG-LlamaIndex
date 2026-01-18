# Hosted API Service Deployment

> **Status**: Future consideration - not currently in scope

## Overview

Expose GraphRAG as a REST API for web applications, multi-user access, or cloud deployment.

## Technology

- **FastAPI** for async web framework
- **Uvicorn** as ASGI server
- Optional: Redis for caching, PostgreSQL for user management

## API Endpoints

```
POST   /databases                    → Create database (start)
GET    /databases                    → List all databases
GET    /databases/{name}             → Get database status

POST   /databases/{name}/index       → Start indexing
GET    /databases/{name}/index/status → Poll indexing progress

GET    /databases/{name}/search      → Query (search)
  Query params: ?q=<query>&mode=<type>&topK=<n>

GET    /databases/{name}/entities/{name}/graph → Graph traversal
DELETE /databases/{name}             → Delete database
```

## Example Implementation

```python
from fastapi import FastAPI, HTTPException
from graphrag_cli import GraphRAGService

app = FastAPI(title="GraphRAG API", version="1.0")
service = GraphRAGService()

@app.post("/databases")
async def create_database(name: str, input_dir: str):
    return service.start(name, input_dir)

@app.get("/databases/{name}/search")
async def search(name: str, q: str, mode: str = "find_connections", topK: int = 10):
    return service.search(name, q, mode, topK)
```

## Docker Deployment

```yaml
# docker-compose.yml addition
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ~/.graphrag:/root/.graphrag
    command: ["uvicorn", "api_server:app", "--host", "0.0.0.0"]
```

## Prerequisites from CLI Design

For API server to work later, the CLI must:

- [ ] Have stateless, reusable service layer (not just CLI commands)
- [ ] Return structured data (not just print output)
- [ ] Support concurrent database access
- [ ] Handle errors with proper exceptions (not sys.exit)
