"""
Rebuild HNSW index to fix 'Duplicate keys' internal error.
This script drops and recreates the HNSW index on the documents table.
"""
import sys
import os
import argparse
sys.path.insert(0, '.')

from duckdb_store import getStore
from workspace_config import getRegistry
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_hnsw_index(database_name=None):
    """Drop and recreate the HNSW index to resolve duplicate key errors."""
    
    # Determine which database to use (priority: CLI arg > env var > default)
    if database_name:
        logger.info(f"Using database from argument: {database_name}")
    else:
        database_name = os.getenv('GRAPHRAG_DATABASE', 'default')
        logger.info(f"Using database from environment: {database_name}")
    
    # Get database configuration from registry
    registry = getRegistry()
    db_config = registry.getOrDefault(database_name)
    
    logger.info(f"Database path: {db_config.dbPath}")
    
    store = getStore(db_config.dbPath)
    
    try:
        # Load VSS extension
        store.connection.execute("INSTALL vss; LOAD vss;")
        store.connection.execute("SET hnsw_enable_experimental_persistence = true;")
        logger.info("VSS extension loaded")
        
        # Drop existing index
        logger.info("Dropping existing HNSW index...")
        store.connection.execute("DROP INDEX IF EXISTS idx_documents_vss")
        logger.info("✓ Index dropped")
        
        # Check document count
        count = store.connection.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL").fetchone()[0]
        logger.info(f"Found {count} documents with embeddings")
        
        if count == 0:
            logger.warning("No documents with embeddings found. Index will be created but empty.")
        
        # Recreate index
        logger.info("Creating fresh HNSW index...")
        store.connection.execute("""
            CREATE INDEX idx_documents_vss 
            ON documents USING HNSW (embedding)
            WITH (metric = 'cosine')
        """)
        logger.info("✓ Index created successfully")
        
        logger.info(f"\n✅ HNSW index rebuild complete for '{database_name}'!")
        
    except Exception as e:
        logger.error(f"❌ Failed to rebuild index: {e}")
        raise
    finally:
        store.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild HNSW index to fix duplicate key errors")
    parser.add_argument("database", nargs="?", help="Database name (defaults to active database)")
    args = parser.parse_args()
    
    rebuild_hnsw_index(args.database)
