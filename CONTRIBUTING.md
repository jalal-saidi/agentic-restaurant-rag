# Contributing

This project keeps retrieval, orchestration, transport, and presentation as
separate layers. Contributions should preserve those boundaries so LangGraph
and Agno remain comparable clients of the same MCP service.

## Development setup

The recommended environment is the checked-in development container:

1. Copy `.env.example` to `.env` and add a development model configuration.
2. Stop the root Compose stack if it is running with `docker compose down`.
3. In VS Code, run **Dev Containers: Reopen in Container**.

The development Compose project starts four containers:

- `dev` — the editor, shell, editable package, and test tools;
- `mcp` — the FastMCP retrieval service on port 8001;
- `api` — reload-enabled FastAPI on port 8000;
- `web` — Gradio on port 7860.

The source tree is bind-mounted into every service. FastAPI reloads Python
changes automatically. Restart the MCP or web sidecar after changing code
executed by those processes. Rebuild the container after dependency or
development-image changes.

Python 3.11 or 3.12 local environments remain supported:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
```

## Architecture boundaries

- `connoisseur.core` owns corpus loading, indexing, and retrieval behavior.
- `connoisseur.mcp` publishes those capabilities as typed MCP tools.
- `connoisseur.orchestrators` consumes MCP tools and contains no direct corpus
  reads.
- `connoisseur.api` normalizes HTTP requests and backend selection.
- `connoisseur.client` contains the Gradio client and calls only the API.

Do not add a data-file shortcut to an orchestrator. A retrieval feature should
be added behind MCP so both backends can use it.

## Adding an MCP tool

LangGraph and Agno discover tool definitions at runtime. Adding a tool does not
require adding its name to either orchestrator, but the tool must have a clear
schema and description so a model can select it correctly.

1. Add or extend the underlying operation in `connoisseur.core`. Keep data
   access, validation, and deterministic business rules in this layer.
2. Add a dependency-injectable method to `ConnoisseurTools` in
   `src/connoisseur/mcp/server.py`.
3. Register a typed function with `@server.tool` inside `create_mcp`. Use
   descriptive parameter names, precise type hints, safe defaults, and a
   docstring that explains when the tool should be chosen.
4. Add unit tests for the core behavior and MCP wrapper. Test valid results,
   empty results, filters, and invalid identifiers as applicable.
5. Add or update an MCP construction test that verifies the tool appears in
   the server's `list_tools` response and that its JSON schema is usable.
6. Restart the MCP service, then restart the API. LangGraph discovers and
   caches tools when its backend is first used; restarting the API invalidates
   that cache. Agno discovers tools when it establishes its MCP tool session.
7. Send scenario-level requests through both backends and verify that the new
   tool is selected only when its description fits the request.

For LangGraph, the runtime sequence is:

```text
profile
  -> discover and bind MCP tools
  -> select_tools
  -> ToolNode executes selected calls
  -> select_tools again when follow-up evidence is needed
  -> collect_evidence
  -> trend/style/nutrition specialists in parallel
  -> synthesis
```

The loop is capped by `LANGGRAPH_MAX_TOOL_ROUNDS`. Do not work around this
design by adding a direct `client.call_tool("new_tool", ...)` call to the
LangGraph backend. Improve the tool's name, description, argument schema, or
the generic retrieval-planner instructions if selection quality is poor.

## Tests and quality checks

Run the complete local checks before opening a pull request:

```powershell
ruff check .
mypy
pytest -q
```

The tests should not contact a live model or download an embedding model.
Use injected model, MCP-client, repository, and framework doubles for unit
tests. A framework construction test may build real LangGraph, Agno, FastMCP,
and Gradio objects, but it must remain offline.

For container parity:

```powershell
docker build --target test -t agentic-restaurant-rag:test .
docker run --rm agentic-restaurant-rag:test
```

When a change affects service wiring, also validate the rendered Compose
configuration and service health:

```powershell
docker compose config
docker compose up --build
```

## Change checklist

- Keep public API response contracts backward compatible or document the
  versioned change.
- Include tests for behavior and failure paths.
- Update README configuration and operational guidance for new settings.
- Never commit `.env`, API keys, generated indexes, model caches, or source
  data without confirmed publication rights.
- Keep logs and error responses free of prompts, credentials, and unnecessary
  provider details.
