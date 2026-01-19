#!/bin/bash
# GraphRAG Alias Configuration for WSL/Bash

# REPLACE <your-windows-username> with your actual Windows username
export GRAPHRAG_REGISTRY_DIR=/mnt/c/Users/<your-windows-username>/.graphrag

graphrag() {
    # REPLACE /path/to/your/project with the actual path to this repository in WSL
    docker compose -f /path/to/your/project/docker-compose.yml \
        run --rm graphrag python graphrag_cli.py "$@"
}

# Export the function so it's available in subshells
export -f graphrag

echo "GraphRAG alias loaded (WSL mode - registry: $GRAPHRAG_REGISTRY_DIR)"
