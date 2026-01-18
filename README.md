## Setup

### 1. Clone the repo

git clone <your-repo>

### 2. Copy .env.example to .env

`cp .env.example .env`

### 3. Edit .env and set their data directory

> GRAPHRAG_DATA_DIR=/path/to/your/documents

(You can copy and paste the windows address directly)

### 4. Create a database

```
docker compose run --rm graphrag python graphrag_cli.py start my-docs \
 --input /app/data/<subfolder>
```

Your .env settings `GRAPHRAG_DATA_DIR=E:/ai-workspace/analysis-docs` maps to docker as `/app/data`,
so you Just replace SUBFOLDER with whatever folder exists in your analysis-docs directory!

```
E:/ai-workspace/analysis-docs/
├── converted_md/
│   └── Documents/          ← Your investment docs
├── research-papers/        ← Another collection
└── quarterly-reports/      ← Another collection

# Investment analysis (your current one)
docker compose run --rm graphrag python graphrag_cli.py start investment-analysis `
  --input /app/data/converted_md/Documents

# Research papers
docker compose run --rm graphrag python graphrag_cli.py start research `
  --input /app/data/research-papers

# Quarterly reports
docker compose run --rm graphrag python graphrag_cli.py start quarterly `
  --input /app/data/quarterly-reports
```

### 5. Index and query

Search keywords first then use the output to search for thematic or connection with better yields

```
docker compose run --rm graphrag python graphrag_cli.py index <database>
docker compose run --rm graphrag python graphrag_cli.py search <database> "query" --type keyword -k 5
docker compose run --rm graphrag python graphrag_cli.py search <database> "query" --type thematic -k 5
```

Guide for Window Users:

- Opening the folder in File Expolorer:
  `explorer $env:USERPROFILE\.graphrag`
- View the registry file:
  `cat $env:USERPROFILE\.graphrag\registry.json`
- See all registered databases:
  `ls $env:USERPROFILE\.graphrag\databases`

## Moving database

### Adding an entry to ~/.graphrag/registry.json and pointing to your existing file:

```
docker compose run --rm graphrag python graphrag_cli.py register my-database \
  --db-path /app/.DuckDB/graphrag.duckdb \
  --input /app/data/<located-in-another-subfolder>
```

- Immediate Access: You can now run status, search, or index using that name (e.g., graphrag search my-database "...").
- No Data Loss: It doesn't move or modify your actual .duckdb file; it just "bookmarks" it for the CLI.

### If you want to physically move it to the new "Managed" folder:

- Create a folder for your database in your defined `GRAPHRAG_DATA_DIR`
- Move the .duckdb file into that folder and rename it to match
- Register it:

```
docker compose run --rm graphrag python graphrag_cli.py register my-project \
  --db-path /app/data/my-project/my-project.duckdb
```

This design should be portable. it uses Path.home() in `workspace_config.py` to automatically resolves to:

- C:\Users\<username> on Windows
- /home/<username> on Linux
- /Users/<username> on macOS

## Parent Directory & Design Limitations

Because this is designed with docker container for portability, the current setup with a single hardcoded mount (/app/input) means all databases share the same input directory. So my advice is to make a folder somewhere on your PC and organize multiple different topics and interests input folder within.

If you need complete flexibility without predefined slots, look into creating a docker-compose.override to establish a multi drive support.
