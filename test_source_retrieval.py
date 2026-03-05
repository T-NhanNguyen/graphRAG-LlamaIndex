import os
import sys
import uuid
import json

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import getStore, DocumentChunk, settings
from tools import getFusionRetriever

def test_source_retrieval():
    test_db = "./.DuckDB/test_source_citation.duckdb"
    
    if os.path.exists(test_db):
        os.remove(test_db)
    
    print(f"Initializing test database at {test_db}...")
    # Use empty embedding provider to skip vector search safely
    settings.EMBEDDING_PROVIDER = ""
    store = getStore(test_db, refresh=True)
    
    print("Manually inserting a test chunk with metadata...")
    doc_id = str(uuid.uuid4())
    chunk_id = str(uuid.uuid4())
    
    # Insert source document first to satisfy foreign key constraint
    from core.duckdb_store import SourceDocument, PipelineStatus
    store.insertSourceDocument(SourceDocument(
        id=doc_id,
        sourcePath="documents/manual_test_source.txt",
        rawContent="Unique test content about GraphRAG and citations.",
        pipelineStatus=PipelineStatus.COMPLETE
    ))
    
    test_chunk = DocumentChunk(
        chunkId=chunk_id,
        sourceDocumentId=doc_id,
        text="Unique test content about GraphRAG and citations.",
        index=0,
        metadata={"source": "documents/manual_test_source.txt"}
    )
    store.insertDocumentChunks([test_chunk])
    
    # Also need to manually index for BM25 since FusionRetriever uses it
    from tools.bm25_index import createBm25Index
    createBm25Index(store, [test_chunk])
    
    print("Running FusionRetriever search (BM25 only path)...")
    retriever = getFusionRetriever(store)
    # Force BM25 only search for this test
    results = retriever.search("GraphRAG", useRrf=False, useAlpha=False)
    
    if not results:
        print("FAIL: No results returned")
        return
    
    result = results[0]
    print(f"Retrieved chunk metadata: {result.metadata}")
    
    source_path = result.metadata.get("source", "")
    source_name = os.path.basename(source_path)
    print(f"Verified source name: {source_name}")
    
    if source_name == "manual_test_source.txt":
        print("SUCCESS: Source citation verified via FusionRetriever!")
    else:
        print(f"FAIL: Expected 'manual_test_source.txt', got '{source_name}'")

if __name__ == "__main__":
    try:
        test_source_retrieval()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
