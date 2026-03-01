# GraphRAG Fly Update Script
# Usage: ./update-knowledge.ps1

Write-Host "--- Starting Knowledge Graph Update ---" -ForegroundColor Cyan

# 1. Run Indexer
Write-Host "[1/3] Running Indexer..." -ForegroundColor Gray
docker compose -f ../../docker-compose.yml run --rm graphrag python -m api.graphrag_cli index investment-analysis

if ($LASTEXITCODE -ne 0) {
    Write-Host " Indexing failed. Aborting." -ForegroundColor Red
    exit 1
}

# 2. Sync to staging
Write-Host "[2/3] Refreshing project vault..." -ForegroundColor Gray
# Use the local index-vault directory for bundling with the Docker build
Copy-Item -Recurse -Force -Path C:\Users\nhan\.graphrag\index-vault\investment-analysis -Destination ../../index-vault/

# 3. Deploy to Fly
Write-Host "[3/3] Deploying to Fly.io..." -ForegroundColor Gray
# Run deploy from the root context so Dockerfile.query can find all files correctly
fly deploy ../../ -c ./fly.toml --ha=false

Write-Host "--- Update Complete! ---" -ForegroundColor Green
