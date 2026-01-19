# GraphRAG Alias Configuration for PowerShell

# PowerShell doesn't need GRAPHRAG_REGISTRY_DIR override (~ resolves correctly to C:\Users\<username>)
# But we set it explicitly for consistency and documentation
$env:GRAPHRAG_REGISTRY_DIR = "$env:USERPROFILE\.graphrag"

function graphrag {
    # REPLACE E:\path\to\your\project with the actual path to this repository
    docker compose -f E:\path\to\your\project\docker-compose.yml `
        run --rm graphrag python graphrag_cli.py @args
}

Write-Host "GraphRAG alias loaded (PowerShell mode - registry: $env:GRAPHRAG_REGISTRY_DIR)" -ForegroundColor Green
