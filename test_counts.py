import logging
import json
from mcp import routeCommand

logging.disable(logging.CRITICAL)

def test():
    try:
        response = routeCommand("search('NVIDIA GPU')")
        res = json.loads(response)
        
        # Check if the response is TSON or JSON
        chunks = res.get("chunks", [])
        if isinstance(chunks, dict) and chunks.get("format") == "tson":
             chunk_count = len(chunks.get("data", []))
        else:
             chunk_count = len(chunks)
             
        entities = res.get("entities", [])
        if isinstance(entities, dict) and entities.get("format") == "tson":
             entity_count = len(entities.get("data", []))
        else:
             entity_count = len(entities)

        relationships = res.get("relationships", [])
        if isinstance(relationships, dict) and relationships.get("format") == "tson":
             rel_count = len(relationships.get("data", []))
        else:
             rel_count = len(relationships)

        print(f"Chunks count: {chunk_count}")
        print(f"Entities count: {entity_count}")
        print(f"Relationships count: {rel_count}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test()
