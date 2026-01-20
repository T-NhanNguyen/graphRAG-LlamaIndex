import logging
import networkx as nx
from typing import List, Dict, Any, Tuple
from duckdb_store import getStore, DuckDBStore, Entity, CommunitySummary
from llm_client import getLLMClient
from graphrag_config import settings
import graspologic.partition as partition
import pandas as pd

logger = logging.getLogger(__name__)

class LeidenPipeline:
    # Orchestrates hierarchical community detection and summarization
    
    def __init__(self, store: DuckDBStore):
        self.store = store
        self.graph = nx.Graph()
        self.llm = getLLMClient()

    def _loadGraph(self):
        # Build a NetworkX graph from DuckDB entities and relationships
        logger.info("Extracting graph from DuckDB")
        
        # 1. Fetch Entities
        entities = self.store.connection.execute("SELECT entity_id, name, entity_type, description FROM entities").fetchall()
        for eid, name, etype, desc in entities:
            self.graph.add_node(eid, name=name, type=etype, description=desc)
            
        # 2. Fetch Relationships
        relationships = self.store.connection.execute("""
            SELECT source_entity_id, target_entity_id, relationship_type, weight 
            FROM relationships
        """).fetchall()
        for src, tgt, rtype, weight in relationships:
            self.graph.add_edge(src, tgt, type=rtype, weight=weight or 1.0)
            
        logger.info(f"Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def detectCommunities(self) -> Dict[int, Dict[str, List[str]]]:
        # Run hierarchical Leiden clustering.
        # Returns a mapping of {level: {community_id: [entity_ids]}}
        if self.graph.number_of_nodes() == 0:
            self._loadGraph()
            
        if self.graph.number_of_nodes() == 0:
            logger.warning("No entities found in database. Skipping community detection.")
            return {}

        logger.info("Running hierarchical Leiden partitioning")
        
        try:
            # Use graspologic for hierarchical Leiden
            # Defaults are usually sufficient for starting out
            clusters = partition.hierarchical_leiden(self.graph)
            
            hierarchical_map = {}
            for c in clusters:
                level = int(c.level)
                cluster_id = str(c.cluster)
                node_id = str(c.node)
                
                if level not in hierarchical_map:
                    hierarchical_map[level] = {}
                if cluster_id not in hierarchical_map[level]:
                    hierarchical_map[level][cluster_id] = []
                hierarchical_map[level][cluster_id].append(node_id)
                
            logger.info(f"Detected communities across {len(hierarchical_map)} levels")
            return hierarchical_map
            
        except Exception as e:
            logger.error(f"Leiden detection failed: {e}")
            return {}

    def summarizeCommunities(self, hierarchicalMap: Dict[int, Dict[str, List[str]]]):
        # Generate summaries for each detected community and persist to DuckDB
        all_summaries = []
        
        for level, communities in hierarchicalMap.items():
            logger.info(f"Summarizing {len(communities)} communities at level {level}")
            
            for cid, entityIds in communities.items():
                # Fetch full entity objects for context
                community_entities = []
                for eid in entityIds:
                    node_data = self.graph.nodes[eid]
                    entity = Entity(
                        entityId=eid,
                        name=node_data.get('name', 'Unknown'),
                        canonicalName=node_data.get('name', 'Unknown'),
                        entityType=node_data.get('type', 'CONCEPT'),
                        description=node_data.get('description', ''),
                        sourceDocumentIds=[],
                        sourceChunkIds=[]
                    )
                    community_entities.append(entity)
                
                # Generate summary via LLM
                summary_text = self.llm.summarizeCommunity(cid, community_entities)
                
                # Create persistence object
                summary_obj = CommunitySummary(
                    communityId=f"L{level}_C{cid}", # Prefix with level for uniqueness
                    level=level,
                    entityIds=entityIds,
                    summary=summary_text
                )
                all_summaries.append(summary_obj)
                
        # Batch insert into DuckDB
        if all_summaries:
            self.store.insertCommunitySummaries(all_summaries)
            logger.info(f"Successfully persisted {len(all_summaries)} community summaries")

    def run(self):
        # Full pipeline execution.
        hierarchical_map = self.detectCommunities()
        if hierarchical_map:
            self.summarizeCommunities(hierarchical_map)
            logger.info("Leiden community detection and summarization complete.")
        else:
            logger.warning("No communities detected to summarize.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = getStore()
    pipeline = LeidenPipeline(store)
    pipeline.run()
