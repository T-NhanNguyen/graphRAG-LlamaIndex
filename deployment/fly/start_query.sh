#!/bin/bash
set -e

echo "Starting GraphRAG Query Environment..." >&2

# Database Registration 
DB_NAME=${GRAPHRAG_DATABASE:-default}
TARGET_DIR="/root/.graphrag/index-vault/${DB_NAME}"

# Check if S3_BUCKET_NAME is provided for remote DB download
if [ -n "$S3_BUCKET_NAME" ]; then
    echo "S3_BUCKET_NAME is set ($S3_BUCKET_NAME). Syncing database from S3..." >&2
    
    S3_PATH="s3://${S3_BUCKET_NAME}/backups/index-vault/${DB_NAME}"
    echo "Targeting local path: ${TARGET_DIR}" >&2
    
    # Use AWS CLI to sync the database folder from S3
    mkdir -p "$TARGET_DIR"
    aws s3 sync "$S3_PATH" "$TARGET_DIR" --delete
    
    if [ $? -eq 0 ]; then
        echo " Database synced from S3 successfully." >&2
    else
        echo " Failed to sync from S3. Check IAM permissions or bucket name." >&2
    fi
else
    echo " S3_BUCKET_NAME not set. Running with local/mounted database." >&2
fi

# ALWAYS attempt to register the database file if it exists (assumes standard naming convention)
DB_FILE="${TARGET_DIR}/${DB_NAME}.duckdb"
if [ -f "$DB_FILE" ]; then
    echo "Registering database context for $DB_NAME" >&2
    python -m api.graphrag_cli register "$DB_NAME" --db-path "$DB_FILE" >&2
fi

echo "Starting MCP server..." >&2
# Use exec to replace the bash process with the node process (for signal handling)
exec npx tsx api/mcp_server.ts
