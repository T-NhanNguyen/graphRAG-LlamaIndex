# GraphRAG Local - Python 3.11 + Node.js for MCP Server
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Node.js for MCP TypeScript server
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package.json and install MCP dependencies
COPY package.json .
RUN npm install

# Pre-install DuckDB extensions for vector search support
RUN python -c "import duckdb; conn = duckdb.connect(':memory:'); conn.execute('INSTALL vss;')"

# Pre-download GLiNER model to bake into image
RUN python -c "from gliner import GLiNER; GLiNER.from_pretrained('urchade/gliner_large-v2.1')"

# Copy all source files
COPY *.py ./
COPY *.ts ./
COPY .env* ./

# Create required directories
RUN mkdir -p /app/input /app/output /app/.DuckDB

# Set environment
ENV PYTHONUNBUFFERED=1

# Default command: MCP server (for integration)
# Override with docker compose run for other modes
CMD ["npx", "tsx", "mcp_server.ts"]
