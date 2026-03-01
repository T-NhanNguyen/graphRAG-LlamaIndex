# Fly.io Deployment & Maintenance

This directory contains the configuration and scripts required to deploy the GraphRAG Query Engine to [Fly.io](https://fly.io/).

### 1. `update-knowledge.ps1` (Local Maintenance)

**Used by:** Developers on their local machine.
**What it does:** This is your "One-Click Update" automation.

- It runs the **Indexer** (using Docker) to process your latest documents.
- It refreshes your local **index-vault** (the Knowledge Graph).
- It triggers a `fly deploy` to upload the updated Knowledge Graph to your live Fly.io server.

**How to use (PowerShell):**

```powershell
# Navigate to this folder and run:
./update-knowledge.ps1
```

---

### 2. `start_query.sh` (Container Entrypoint)

**Used by:** The Fly.io server (automatically).
**What it does:** This script runs inside the Docker container when the server starts.

- It identifies the bundled Knowledge Graph (`.duckdb` file).
- It registers the database configuration so the engine knows where to look.
- It launches the **MCP (Model Context Protocol) Server**, making your AI searchable.

_Note: You don't need to run this script manually; Fly.io handles it during boot._

---

## Setup

1. **Install Fly CLI:**  
   `iwr https://fly.io/install.ps1 -useb | iex` (PowerShell)
2. **Login:**  
   `fly auth login`
3. **Set Secrets:**  
   Ensure your cloud server has your API keys:
   ```powershell
   fly secrets set OPENROUTER_API_KEY="sk-or-v1-..." -c ./fly.toml
   ```

- `fly.toml`: Server configuration (Region, Ports, Env Vars).
- `Dockerfile.query`: Instructions to build the lightweight query container.
- `index-vault/`: Staging area where the database is bundled before upload.
