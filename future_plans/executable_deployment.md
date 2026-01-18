# Standalone Executable Deployment

> **Status**: Future consideration - not currently in scope

## Overview

Package GraphRAG as a standalone executable that users can run without Python or Docker installed.

## Technology

- **PyInstaller** or **Nuitka** for Python → native executable
- Creates `graphrag.exe` (Windows) / `graphrag` (Linux/Mac)
- Estimated size: ~150-200MB with ONNX runtime

## Build Process

```bash
# Install build dependencies
pip install pyinstaller

# Build executable
pyinstaller --onefile --name graphrag graphrag_cli.py

# Output: dist/graphrag.exe
```

## User Experience

```bash
# Download and run directly (no installation)
./graphrag start my-corpus --input ./documents
./graphrag index my-corpus
./graphrag search my-corpus "What is the revenue trend?"
```

## Architecture Considerations

### Bundled Components

| Component      | Bundled     | External                    |
| -------------- | ----------- | --------------------------- |
| DuckDB         | ✅ Yes      | -                           |
| BM25 tokenizer | ✅ Yes      | -                           |
| Embeddings     | ⚡ Optional | Docker Model Runner, OpenAI |
| LLM            | -           | ✅ Always external          |

### Embedding Options

1. **Local ONNX** - Bundle small embedding model (~100MB)
2. **External** - Connect to Docker Model Runner or cloud API

## Prerequisites from CLI Design

For executable packaging to work later, the CLI must:

- [ ] Have clean entry point (`graphrag_cli.py`)
- [ ] Use relative imports or proper package structure
- [ ] Not depend on runtime file discovery that would break bundling
- [ ] Support configuration via CLI args (not just env vars)
