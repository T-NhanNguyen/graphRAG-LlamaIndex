/**
 * GraphRAG MCP Server - Model Context Protocol interface for knowledge graph queries.
 * 
 * Exposes GraphRAG tools to AI agents via the MCP standard:
 * - search: Unified search with mode-based strategies
 * - explore_entity_graph: Graph traversal from a known entity
 * - get_corpus_stats: Database health and scale statistics
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    ListToolsRequestSchema,
    CallToolRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

const server = new Server({
    name: "graphrag-mcp",
    version: "1.0.0"
}, {
    capabilities: { tools: {} }
});

const tools = [
    {
        name: "search",
        description: "Search the GraphRAG knowledge base. Choose a mode based on your information need:\n- 'entity_connections': Find specific entities and their relationships (WHO/WHAT is connected to X?)\n- 'thematic_overview': Explore high-level themes and patterns (WHAT are the big trends?)\n- 'keyword_lookup': Fast direct retrieval for specific terms (WHERE does X appear?)",
        inputSchema: {
            type: "object",
            properties: {
                query: { 
                    type: "string", 
                    description: "Natural language query" 
                },
                mode: { 
                    type: "string", 
                    enum: ["entity_connections", "thematic_overview", "keyword_lookup"],
                    description: "Search mode: 'entity_connections' (default) for relationships, 'thematic_overview' for big picture, 'keyword_lookup' for direct term search" 
                },
                topK: { 
                    type: "integer", 
                    description: "Number of results (default: 10)" 
                }
            },
            required: ["query"]
        }
    },
    {
        name: "explore_entity_graph",
        description: "Traverse the knowledge graph starting from a specific entity. Returns the entity, its direct connections (other entities), and the relationships between them. Use this to build reasoning chains or verify facts about a known entity.",
        inputSchema: {
            type: "object",
            properties: {
                entityName: { 
                    type: "string", 
                    description: "Exact name of the entity to explore (e.g., 'Microsoft', 'Vistra')" 
                },
                hops: { 
                    type: "integer", 
                    description: "How many relationship hops to traverse (default: 1, max: 3)" 
                }
            },
            required: ["entityName"]
        }
    },
    {
        name: "get_corpus_stats",
        description: "Get statistics about the indexed knowledge base. Returns counts of documents, chunks, entities, and relationships. Use for corpus health checks or to understand the scale of available data.",
        inputSchema: {
            type: "object",
            properties: {}
        }
    }
];

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    // Build Python command string
    // Convert JS object to Python-parseable format
    const pythonArgs = JSON.stringify(args || {}).replace(/"/g, "'");
    const pythonCmd = `${name}(${pythonArgs})`;

    try {
        // Execute mcp.py with the command
        // Using double quotes for outer shell argument to allow inner single quotes
        const { stdout, stderr } = await execAsync(`python mcp.py "${pythonCmd}"`);

        if (stderr && stderr.trim()) {
            // Some libraries write to stderr on success, check if stdout is empty
            if (!stdout.trim()) {
                return { content: [{ type: "text", text: stderr }], isError: true };
            }
        }

        return { content: [{ type: "text", text: stdout }] };
    } catch (error: any) {
        return { content: [{ type: "text", text: error.message }], isError: true };
    }
});

const transport = new StdioServerTransport();
await server.connect(transport);
console.error("GraphRAG MCP Server started");
