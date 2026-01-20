# Workspace Configuration - Global database registry managing ~/.graphrag/registry.json.
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)

# --- Constants (Uppercase per coding framework) ---
import os
DEFAULT_DATABASE_NAME = os.environ.get("GRAPHRAG_DATABASE", "default").strip().lower()
REGISTRY_DIR = Path.home() / ".graphrag"
REGISTRY_FILE = REGISTRY_DIR / "registry.json"
INDEX_VAULT_DIR = REGISTRY_DIR / "index-vault"  # Renamed from "databases"
REGISTRY_VERSION = 1  # For future schema migrations


@dataclass
class DatabaseConfig:
    # Configuration for a single GraphRAG database.
    name: str                           # User-friendly name (e.g., "financial-analysis")
    dbPath: str                         # Absolute path to .duckdb file
    sourceFolder: str                   # Source documents directory
    outputFolder: str                   # Output directory for logs, exports
    createdAt: str = field(default_factory=lambda: datetime.now().isoformat())
    lastIndexed: Optional[str] = None   # ISO timestamp of last indexing run
    
    def toDict(self) -> Dict:
        # Convert to serializable dict.
        return asdict(self)
    
    @classmethod
    def fromDict(cls, data: Dict) -> "DatabaseConfig":
        # Create from dict, handling missing optional fields.
        return cls(
            name=data["name"],
            dbPath=data["dbPath"],
            sourceFolder=data.get("sourceFolder", data.get("inputDir", "")),
            outputFolder=data.get("outputFolder", data.get("outputDir", "")),
            createdAt=data.get("createdAt", datetime.now().isoformat()),
            lastIndexed=data.get("lastIndexed")
        )


class WorkspaceRegistry:
    # Manages global database registry at ~/.graphrag/registry.json as a singleton.
    
    _instance: Optional["WorkspaceRegistry"] = None
    
    def __init__(self):
        # Initialize registry, creating directories if needed.
        self._ensureDirectories()
        self._registry: Dict[str, DatabaseConfig] = {}
        self._load()
    
    @classmethod
    def getInstance(cls) -> "WorkspaceRegistry":
        # Get singleton instance.
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _ensureDirectories(self) -> None:
        # Create registry directories if they don't exist.
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_VAULT_DIR.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Registry directory ensured: {REGISTRY_DIR}")
    
    def _load(self) -> None:
        # Load registry from disk.
        if REGISTRY_FILE.exists():
            try:
                with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._registry = {
                    name: DatabaseConfig.fromDict(config)
                    for name, config in data.get("databases", {}).items()
                }
                logger.info(f"Loaded {len(self._registry)} database(s) from registry")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load registry, starting fresh: {e}")
                self._registry = {}
        else:
            self._registry = {}
            self._save()  # Create empty registry file
    
    def _save(self) -> None:
        # Persist registry to disk.
        data = {
            "version": 1,
            "databases": {name: config.toDict() for name, config in self._registry.items()}
        }
        with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved registry with {len(self._registry)} database(s)")
    
    def register(
        self,
        name: str,
        sourceFolder: Optional[str] = None,
        outputFolder: Optional[str] = None,
        dbPath: Optional[str] = None
    ) -> DatabaseConfig:
        # Register a new database or update existing one with default path resolution.
        if not name or not name.strip():
            raise ValueError("Database name cannot be empty")
        
        name = name.strip().lower().replace(" ", "-")
        
        # Set defaults for paths
        dbDir = INDEX_VAULT_DIR / name
        dbDir.mkdir(parents=True, exist_ok=True)
        
        resolvedDbPath = dbPath or str(dbDir / f"{name}.duckdb")
        resolvedSourceFolder = sourceFolder
        resolvedOutputFolder = outputFolder or str(dbDir / "output")
        
        if resolvedOutputFolder:
            Path(resolvedOutputFolder).mkdir(parents=True, exist_ok=True)
        
        config = DatabaseConfig(
            name=name,
            dbPath=resolvedDbPath,
            sourceFolder=resolvedSourceFolder or "",
            outputFolder=resolvedOutputFolder
        )
        
        isUpdate = name in self._registry
        self._registry[name] = config
        self._save()
        
        action = "Updated" if isUpdate else "Registered"
        logger.info(f"{action} database '{name}' at {resolvedDbPath}")
        
        return config
    
    def get(self, name: Optional[str] = None) -> Optional[DatabaseConfig]:
        # Get database configuration by name, falling back to default.
        targetName = (name or DEFAULT_DATABASE_NAME).strip().lower()
        return self._registry.get(targetName)
    
    def getOrDefault(self, name: Optional[str] = None) -> DatabaseConfig:
        # Get database configuration, creating default if not found.
        config = self.get(name)
        if config is None:
            # Create default database if requested name doesn't exist
            targetName = (name or DEFAULT_DATABASE_NAME).strip().lower()
            if targetName != DEFAULT_DATABASE_NAME and name is not None:
                logger.warning(f"Database '{name}' not found, falling back to '{DEFAULT_DATABASE_NAME}'")
            config = self.register(DEFAULT_DATABASE_NAME)
        return config
    
    def list(self) -> List[DatabaseConfig]:
        # List all registered databases.
        return list(self._registry.values())
    
    def delete(self, name: str, deleteFiles: bool = False) -> bool:
        # Remove database from registry and optionally delete files (NOT REVERTABLE).
        name = name.strip().lower()
        if name not in self._registry:
            return False
        
        config = self._registry[name]
        
        if deleteFiles:
            import shutil
            dbDir = INDEX_VAULT_DIR / name
            if dbDir.exists():
                shutil.rmtree(dbDir)
                logger.info(f"Deleted database files: {dbDir}")
        
        del self._registry[name]
        self._save()
        logger.info(f"Unregistered database '{name}'")
        return True
    
    def updateLastIndexed(self, name: str) -> None:
        # Update the lastIndexed timestamp for a database.
        if name in self._registry:
            self._registry[name].lastIndexed = datetime.now().isoformat()
            self._save()
    
    def registerExisting(self, name: str, dbPath: str, sourceFolder: Optional[str] = None) -> DatabaseConfig:
        # Register an existing .duckdb file under a new name.
        if not Path(dbPath).exists():
            raise FileNotFoundError(f"Database file not found: {dbPath}")
        
        return self.register(
            name=name,
            dbPath=str(Path(dbPath).resolve()),
            sourceFolder=sourceFolder
        )


def getRegistry() -> WorkspaceRegistry:
    # Get the global workspace registry singleton.
    return WorkspaceRegistry.getInstance()
