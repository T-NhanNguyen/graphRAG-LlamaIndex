# GraphRAG FILE TOC

## Core Orchestration

- `indexer.py`: The central pipeline. Coordinates ingestion, semantic chunking, embedding generation, BM25 indexing, and entity/relationship extraction. Supports resume-from-crash and --reset.
- `graphrag_cli.py`: The human/server entry point. Implements the `graphrag` CLI for managing databases, health checks, and running multi-mode searches.
- `query_engine.py`: High-level search interface. Bridges the storage layer and retrieval components to provide `find_connections`, `explore_thematic`, and `keyword_search` results.

## Storage & Configuration

- `duckdb_store.py`: Primary database abstraction. Manages the DuckDB schema (entities, relationships, chunks, embeddings) and handles cross-session persistence.
- `workspace_config.py`: Global registry manager. Handles multi-database tracking and path resolution in `~/.graphrag/registry.json`.
- `graphrag_config.py`: Centralized settings. Defines constants, enums (SearchType, ExtractionMode), and Pydantic-based configuration loaded from `.env`.

## Search & Retrieval

- `fusion_retrieval.py`: Hybrid retrieval engine. Combines BM25 and Vector search results using Reciprocal Rank Fusion (RRF) for improved accuracy.
- `bm25_index.py`: Sparse vector indexing. Implements the BM25 algorithm for keyword-based retrieval, supporting multilingual tokenization.
- `embedding_provider.py`: Vector generation client. Provides parallelized batch embeddings compatible with OpenAI-style endpoints (Ollama/LMStudio).

## Extraction & Processing

- `entity_extractor.py`: Common interface for entity extraction. Defines the `BaseEntityExtractor` and factory for switching between LLM and GLiNER modes.
- `gliner_extractor.py`: Zero-shot NER extraction. Implements entity extraction using the GLiNER transformer model for lower-latency indexing.
- `llm_client.py`: LLM interaction layer. Manages prompts for entity extraction, relationship discovery, and community summarization.
- `garbage_filter.py`: Data quality guard. Uses statistical heuristics (entropy, repetition) and embedding outlier detection to prune "junk" chunks.
- `community_pipeline.py`: Graph analysis tool. Implements hierarchical Leiden clustering and LLM-based summarization to build community-level insights.

## Protocols & Utilities

- `mcp.py`: Python MCP Command Router. Translates internal functions into Model Context Protocol (MCP) compatible responses for agentic tool use.
- `mcp_server.ts`: TypeScript MCP Wrapper. Exposes the Python engine as a standard MCP server for seamless integration with AI agents.
- `rebuild_hnsw_index.py`: Maintenance utility. Rebuilds the vector index in DuckDB to resolve "Duplicate keys" or performance issues.
- `json_to_tson.py`: Token optimization. Converts JSON lists into "Tabular SON" (TSON) format to reduce context window usage.
- `.graphrag-alias.[ps1/sh]`: Shell utility scripts. Provides environment aliases for the `graphrag` CLI (PowerShell and Bash).

## Documentation

- `docs/system-overview.md`: Architectural deep dive into the 📥 input, ⚙️ processing, 💾 storage, and 🔍 query layers.
- `docs/what-is-graphrag.md`: Conceptual introduction to GraphRAG vs. standard RAG.
- `docs/query-output-guide.md`: Guide to interpreting search results, TSON format, and graph traversals.
- `docs/garbage-filtering-explained.md`: Detailed breakdown of the pruning logic and quality thresholds.
