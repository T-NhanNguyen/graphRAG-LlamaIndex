# Future Deployment Plans

This folder contains specifications for deployment modes that are **not currently in scope** but inform the design of the CLI implementation.

## Status

| Mode          | Status             | File                                                     |
| ------------- | ------------------ | -------------------------------------------------------- |
| CLI           | ✅ **In Progress** | See `implementation_plan.md`                             |
| Executable    | 🔮 Future          | [executable_deployment.md](./executable_deployment.md)   |
| Hosted API    | 🔮 Future          | [hosted_api_deployment.md](./hosted_api_deployment.md)   |
| MCP Switching | 🔮 Future          | [mcp_database_switching.md](./mcp_database_switching.md) |

## Design Principles

The CLI implementation is designed with these future modes in mind:

1. **Service Layer Pattern** - Business logic in reusable classes, CLI is thin wrapper
2. **Dependency Injection** - Settings and stores passed as parameters, not globals
3. **Structured Returns** - Functions return data structures, not print to stdout
4. **Error Handling** - Exceptions for errors, not sys.exit()
5. **Stateless Operations** - Each command is self-contained

## Scalability Checklist

When implementing CLI, verify these patterns support future modes:

- [ ] `workspace_config.py` can be imported by API server
- [ ] `GraphRAGIndexer` accepts settings injection
- [ ] `GraphRAGQueryEngine` accepts settings injection
- [ ] No hardcoded paths in business logic
- [ ] Commands return structured results (dict/dataclass)
