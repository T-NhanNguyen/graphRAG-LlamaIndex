#!/usr/bin/env python3
"""
GraphRAG CLI - Intuitive command-line interface for knowledge graph management.

Commands:
    graphrag start <db> [--input <path>]     Create/initialize a database
    graphrag index <db> [--prune]            Index documents into database
    graphrag search <db> <query> [options]   Query the knowledge graph
    graphrag list                            List all databases
    graphrag status <db>                     Show database statistics
    graphrag delete <db> [--force]           Remove a database
    graphrag register <db> --db-path <path>  Register existing .duckdb file

Design Principles (for future scalability):
- Thin wrapper over service layer (business logic in modules)
- Structured returns (dicts) - CLI formats for human, API uses raw
- Exception-based errors - CLI catches and formats, API propagates
- Dependency injection via settings
"""
import sys
import os
import argparse
import logging
from typing import Optional, Dict, Any

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from workspace_config import getRegistry, DEFAULT_DATABASE_NAME
from graphrag_config import GraphRAGSettings, SearchType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Service Layer - Reusable business logic for all deployment modes
# =============================================================================

class GraphRAGService:
    """
    Core service layer for GraphRAG operations.
    
    Designed for reuse across CLI, API, and MCP interfaces.
    Returns structured data (dicts), never prints directly.
    Raises exceptions on errors (never sys.exit()).
    """
    
    def start(
        self,
        dbName: str,
        inputDir: Optional[str] = None,
        outputDir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initialize a new database.
        
        Args:
            dbName: Database name (required, cannot be empty)
            inputDir: Optional input directory for documents
            outputDir: Optional output directory
        
        Returns:
            Dict with database configuration details.
        
        Raises:
            ValueError: If dbName is empty or invalid
        """
        # Early validation per coding framework
        if not dbName or not dbName.strip():
            raise ValueError("Database name cannot be empty")
        
        registry = getRegistry()
        config = registry.register(
            name=dbName,
            inputDir=inputDir,
            outputDir=outputDir
        )
        
        return {
            "success": True,
            "action": "created",
            "database": config.name,
            "dbPath": config.dbPath,
            "inputDir": config.inputDir,
            "outputDir": config.outputDir
        }
    
    def index(
        self,
        dbName: Optional[str] = None,
        prune: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents into the database.
        
        Returns:
            Dict with indexing statistics.
        """
        from graphrag_config import getSettingsForDatabase
        from duckdb_store import getStore
        from indexer import GraphRAGIndexer
        
        settings = getSettingsForDatabase(dbName)
        store = getStore(settings.DUCKDB_PATH)
        
        indexer = GraphRAGIndexer(store=store)
        stats = indexer.indexDirectory(settings.INPUT_DIR)
        
        # Update last indexed timestamp
        registry = getRegistry()
        resolvedDbName = dbName or DEFAULT_DATABASE_NAME
        registry.updateLastIndexed(resolvedDbName)
        logger.info(f"Indexing complete for database '{resolvedDbName}'")
        
        result = {
            "success": True,
            "database": dbName or DEFAULT_DATABASE_NAME,
            "documentsProcessed": stats.documentsProcessed,
            "documentsSkipped": stats.documentsSkipped,
            "chunksCreated": stats.chunksCreated,
            "entitiesExtracted": stats.entitiesExtracted,
            "relationshipsExtracted": stats.relationshipsExtracted
        }
        
        if prune:
            pruneStats = indexer.pruneNoise(runLLMScore=False)
            result["chunksPruned"] = pruneStats.get("pruned", 0)
        
        return result
    
    def search(
        self,
        dbName: Optional[str],
        query: str,
        searchType: str = "connections",
        topK: int = 10
    ) -> Dict[str, Any]:
        """
        Execute a search query.
        
        Args:
            dbName: Database name
            query: Search query string
            searchType: One of 'connections', 'thematic', 'keyword'
            topK: Number of results
        
        Returns:
            Dict with search results.
        """
        from graphrag_config import getSettingsForDatabase
        from duckdb_store import getStore
        from query_engine import GraphRAGQueryEngine
        
        # Map CLI search types to internal enum
        typeMap = {
            "connections": SearchType.FIND_CONNECTIONS,
            "thematic": SearchType.EXPLORE_THEMATIC,
            "keyword": SearchType.KEYWORD_SEARCH
        }
        internalType = typeMap.get(searchType.lower(), SearchType.FIND_CONNECTIONS)
        
        settings = getSettingsForDatabase(dbName)
        store = getStore(settings.DUCKDB_PATH)
        engine = GraphRAGQueryEngine(store=store)
        
        result = engine.search(query, searchType=internalType.value, topK=topK)
        
        return {
            "success": True,
            "database": dbName or DEFAULT_DATABASE_NAME,
            "query": query,
            "searchType": searchType,
            "chunks": result.chunks,
            "entities": result.entities,
            "relationships": result.relationships
        }
    
    def listDatabases(self) -> Dict[str, Any]:
        """List all registered databases."""
        registry = getRegistry()
        databases = registry.list()
        
        return {
            "success": True,
            "count": len(databases),
            "databases": [
                {
                    "name": db.name,
                    "dbPath": db.dbPath,
                    "inputDir": db.inputDir,
                    "createdAt": db.createdAt,
                    "lastIndexed": db.lastIndexed
                }
                for db in databases
            ]
        }
    
    def status(self, dbName: Optional[str] = None) -> Dict[str, Any]:
        """Get database statistics."""
        from graphrag_config import getSettingsForDatabase
        from duckdb_store import getStore
        
        settings = getSettingsForDatabase(dbName)
        store = getStore(settings.DUCKDB_PATH)
        stats = store.getCorpusStats()
        
        registry = getRegistry()
        config = registry.getOrDefault(dbName)
        
        return {
            "success": True,
            "database": config.name,
            "dbPath": config.dbPath,
            "inputDir": config.inputDir,
            "lastIndexed": config.lastIndexed,
            "stats": stats
        }
    
    def delete(self, dbName: str, deleteFiles: bool = False) -> Dict[str, Any]:
        """Remove a database from registry."""
        # Early validation
        if not dbName or not dbName.strip():
            raise ValueError("Database name cannot be empty")
        
        if dbName.strip().lower() == DEFAULT_DATABASE_NAME.lower():
            raise ValueError("Cannot delete the default database")
        
        registry = getRegistry()
        deleted = registry.delete(dbName, deleteFiles=deleteFiles)
        
        if not deleted:
            raise ValueError(f"Database '{dbName}' not found")
        
        return {
            "success": True,
            "action": "deleted",
            "database": dbName,
            "filesDeleted": deleteFiles
        }
    
    def register(
        self,
        dbName: str,
        dbPath: str,
        inputDir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register an existing .duckdb file."""
        registry = getRegistry()
        config = registry.registerExisting(dbName, dbPath, inputDir)
        
        return {
            "success": True,
            "action": "registered",
            "database": config.name,
            "dbPath": config.dbPath
        }


# =============================================================================
# CLI Interface - Human-friendly command parsing and output formatting
# =============================================================================

def _formatSuccess(message: str) -> None:
    """Print success message with green checkmark."""
    print(message)


def _formatError(message: str) -> None:
    """Print error message with red X."""
    print(f"Error: {message}", file=sys.stderr)


def _formatTable(headers: list, rows: list) -> None:
    """Print a simple ASCII table."""
    if not rows:
        print("(no data)")
        return
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Print header
    headerLine = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(headerLine)
    print("-" * len(headerLine))
    
    # Print rows
    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def _cmdStart(args, service: GraphRAGService) -> int:
    """Handle 'start' command."""
    try:
        result = service.start(
            dbName=args.database,
            inputDir=args.input
        )
        _formatSuccess(f"Database '{result['database']}' initialized")
        print(f"   Database: {result['dbPath']}")
        print(f"   Input:    {result['inputDir']}")
        print(f"   Output:   {result['outputDir']}")
        return 0
    except Exception as e:
        _formatError(str(e))
        return 1


def _cmdIndex(args, service: GraphRAGService) -> int:
    """Handle 'index' command."""
    try:
        dbName = getattr(args, 'database', None)
        print(f"Indexing documents for '{dbName or DEFAULT_DATABASE_NAME}'...")
        
        result = service.index(
            dbName=dbName,
            prune=getattr(args, 'prune', False)
        )
        
        _formatSuccess(f"Indexing complete")
        print(f"   Documents: {result['documentsProcessed']} processed, {result['documentsSkipped']} skipped")
        print(f"   Chunks:    {result['chunksCreated']}")
        print(f"   Entities:  {result['entitiesExtracted']}")
        print(f"   Relations: {result['relationshipsExtracted']}")
        
        if 'chunksPruned' in result:
            print(f"   Pruned:    {result['chunksPruned']} chunks")
        
        return 0
    except Exception as e:
        _formatError(str(e))
        logger.exception("Indexing failed")
        return 1


def _cmdSearch(args, service: GraphRAGService) -> int:
    """Handle 'search' command."""
    try:
        result = service.search(
            dbName=getattr(args, 'database', None),
            query=args.query,
            searchType=getattr(args, 'type', 'connections'),
            topK=getattr(args, 'top_k', 10)
        )
        
        print(f"\nSearch results for \"{result['query']}\":")
        print(f"   Database: {result['database']} | Type: {result['searchType']}")
        print("-" * 60)
        
        # Print chunks
        if result['chunks']:
            print(f"\nTop chunks ({len(result['chunks'])}):")
            for i, chunk in enumerate(result['chunks'][:5]):
                text = chunk.get('text', '')[:200].replace('\n', ' ')
                print(f"   {i+1}. {text}...")
        
        # Print entities
        if result['entities']:
            print(f"\nEntities ({len(result['entities'])}):")
            for ent in result['entities'][:10]:
                print(f"   • {ent.get('name', 'Unknown')} ({ent.get('type', 'N/A')})")
        
        # Print relationships
        if result['relationships']:
            print(f"\nRelationships ({len(result['relationships'])}):")
            for rel in result['relationships'][:5]:
                src = rel.get('sourceName', rel.get('source', '?'))
                tgt = rel.get('targetName', rel.get('target', '?'))
                rtype = rel.get('type', '?')
                print(f"   • {src} {rtype} {tgt}")
                if rel.get('description'):
                    desc = rel['description'][:80]
                    print(f"     \"{desc}...\"")
        
        return 0
    except Exception as e:
        _formatError(str(e))
        return 1


def _cmdList(args, service: GraphRAGService) -> int:
    """Handle 'list' command."""
    try:
        result = service.listDatabases()
        
        print(f"\nRegistered databases ({result['count']}):\n")
        
        if result['databases']:
            _formatTable(
                ["Name", "Last Indexed", "Input Directory"],
                [
                    [db['name'], db['lastIndexed'] or "Never", db['inputDir']]
                    for db in result['databases']
                ]
            )
        else:
            print("   No databases registered. Run 'graphrag start <name>' to create one.")
        
        return 0
    except Exception as e:
        _formatError(str(e))
        return 1


def _cmdStatus(args, service: GraphRAGService) -> int:
    """Handle 'status' command."""
    try:
        result = service.status(dbName=getattr(args, 'database', None))
        
        print(f"\nDatabase status: {result['database']}")
        print("-" * 40)
        print(f"   Path:        {result['dbPath']}")
        print(f"   Input Dir:   {result['inputDir']}")
        print(f"   Last Indexed: {result['lastIndexed'] or 'Never'}")
        
        stats = result.get('stats', {})
        if stats:
            print(f"\n   Documents:     {stats.get('documentCount', 0)}")
            print(f"   Chunks:        {stats.get('chunkCount', 0)}")
            print(f"   Entities:      {stats.get('entityCount', 0)}")
            print(f"   Relationships: {stats.get('relationshipCount', 0)}")
        
        return 0
    except Exception as e:
        _formatError(str(e))
        return 1


def _cmdDelete(args, service: GraphRAGService) -> int:
    """Handle 'delete' command."""
    try:
        if not args.force:
            confirm = input(f"Warning: Delete database '{args.database}'? (y/N): ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                return 0
        
        result = service.delete(
            dbName=args.database,
            deleteFiles=getattr(args, 'files', False)
        )
        
        _formatSuccess(f"Database '{result['database']}' deleted")
        return 0
    except Exception as e:
        _formatError(str(e))
        return 1


def _cmdRegister(args, service: GraphRAGService) -> int:
    """Handle 'register' command."""
    try:
        result = service.register(
            dbName=args.database,
            dbPath=args.db_path,
            inputDir=getattr(args, 'input', None)
        )
        
        _formatSuccess(f"Registered existing database as '{result['database']}'")
        print(f"   Path: {result['dbPath']}")
        return 0
    except FileNotFoundError as e:
        _formatError(f"Database file not found: {args.db_path}")
        return 1
    except Exception as e:
        _formatError(str(e))
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='graphrag',
        description='GraphRAG - Knowledge Graph Management CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # --- start ---
    p_start = subparsers.add_parser('start', help='Initialize a new database')
    p_start.add_argument('database', help='Database name')
    p_start.add_argument('--input', '-i', help='Input directory for documents')
    p_start.set_defaults(func=_cmdStart)
    
    # --- index ---
    p_index = subparsers.add_parser('index', help='Index documents into database')
    p_index.add_argument('database', nargs='?', help='Database name (default: "default")')
    p_index.add_argument('--prune', action='store_true', help='Prune low-quality content after indexing')
    p_index.set_defaults(func=_cmdIndex)
    
    # --- search ---
    p_search = subparsers.add_parser('search', help='Query the knowledge graph, use with -t <connections, thematic, keyword>')
    p_search.add_argument('database', nargs='?', help='Database name')
    p_search.add_argument('query', help='Search query')
    p_search.add_argument('--type', '-t', choices=['connections', 'thematic', 'keyword'],
                          default='connections', help='Search type')
    p_search.add_argument('--top-k', '-k', type=int, default=10, help='Number of results')
    p_search.set_defaults(func=_cmdSearch)
    
    # --- list ---
    p_list = subparsers.add_parser('list', help='List all databases')
    p_list.set_defaults(func=_cmdList)
    
    # --- status ---
    p_status = subparsers.add_parser('status', help='Show database statistics')
    p_status.add_argument('database', nargs='?', help='Database name')
    p_status.set_defaults(func=_cmdStatus)
    
    # --- delete ---
    p_delete = subparsers.add_parser('delete', help='Remove a database')
    p_delete.add_argument('database', help='Database name')
    p_delete.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    p_delete.add_argument('--files', action='store_true', help='Also delete database files')
    p_delete.set_defaults(func=_cmdDelete)
    
    # --- register ---
    p_register = subparsers.add_parser('register', help='Register existing .duckdb file')
    p_register.add_argument('database', help='Name for the database')
    p_register.add_argument('--db-path', required=True, help='Path to .duckdb file')
    p_register.add_argument('--input', '-i', help='Input directory')
    p_register.set_defaults(func=_cmdRegister)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    service = GraphRAGService()
    return args.func(args, service)


if __name__ == "__main__":
    sys.exit(main())
