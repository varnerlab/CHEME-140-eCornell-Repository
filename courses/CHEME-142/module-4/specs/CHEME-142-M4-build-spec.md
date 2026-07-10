# CHEME-142 Module 4 (Model Context Protocol Servers) — Build Spec

*Drafted 2026-07-10. Scope, structure, and example domains validated with the author in a brainstorming session.*

**Status: Spec approved, not yet built.**

## Goal

Build Module 4 of CHEME-142 on Model Context Protocol (MCP) servers: two lecture notebooks (Read-Pages), one demo notebook with code (Watch-Demo, author records a voice-over walkthrough), one ungraded Codio activity, and one graded Codio activity (student + solution). The module completes the course networking arc: M1 files/JSON → M2 REST/HTTP → M3 WebSockets → M4 MCP (JSON-RPC sessions).

## Decisions (from brainstorming)

1. **Student role: both sides.** Students drive an MCP client from Julia against a provided server, and the graded activity has them implement and register a new server tool.
2. **Protocol-only, no LLM.** No API keys or model calls anywhere. Students play the role of the host/agent: initialize, list tools, call tools over JSON-RPC. The lecture explains where the LLM sits in a real deployment.
3. **Server source: minimal Julia implementation in the module's `src/`** (Approach A). MCP's stdio transport is JSON-RPC 2.0 over stdin/stdout; the module ships a small, spec-conformant, tools-only subset (~200–250 lines) that students can read in its entirety. Rejected alternatives: ModelContextProtocol.jl (hides the protocol mechanics the module teaches; third-party API churn risk) and official reference servers via `npx`/`uvx` (adds Node/Python runtime dependency in Codio; server-side work would leave Julia).
4. **Official open-source ecosystem covered in the lecture,** not executed: the reference servers (Everything, Filesystem, Fetch, Time) at github.com/modelcontextprotocol/servers, official SDKs, and the spec's stewardship (released by Anthropic Nov 2024; donated to the Linux Foundation's Agentic AI Foundation Dec 2025). The Watch-Demo closes with a markdown-only pointer to this ecosystem — no `npx` execution, so every notebook runs in Codio with Julia alone.
5. **Two Read-Pages:** an introduction (motivation, architecture, ecosystem) and a protocol-mechanics lecture (JSON-RPC 2.0, lifecycle, tools/list, tools/call, transports).
6. **Tool domain: chemical engineering.** The server exposes ChemE calculations, reinforcing that MCP is how you expose your own computational code to agents.
7. **No non-Julia dependencies.** Deps: `JSON`, `Test`, `PrettyTables`. No network access required by any notebook.

## Deliverables

Naming follows the CHEME-142 pattern `CHEME-142-M4-<Topic>-<DeliveryType>.ipynb`.

1. `CHEME-142-M4-Introduction-MCP-Read-Pages.ipynb` — the problem MCP solves (each host application integrating each tool separately); host/client/server architecture and where the LLM sits; the three server primitives (tools, resources, prompts) with tools as this module's focus; local stdio vs. remote HTTP transports; the open-source ecosystem and spec stewardship; brief security and consent considerations; course-arc tie-back (M2 request/response, M3 persistent sockets, M4 JSON-RPC sessions).
2. `CHEME-142-M4-MCP-ProtocolMessages-Read-Pages.ipynb` — JSON-RPC 2.0 message anatomy (request, response, notification, error object); lifecycle (`initialize` handshake with version and capability negotiation, `initialized` notification, operation, shutdown via stdin close); `tools/list` and `tools/call` request/result schemas including JSON Schema input descriptors; the protocol-error vs. tool-error distinction (JSON-RPC error object vs. `isError: true` result); newline-delimited UTF-8 stdio framing; algorithm blocks for the server dispatch loop and the client call sequence; one complete worked wire exchange shown message-by-message (captured from the module's own implementation).
3. `CHEME-142-M4-MCP-ChemETools-Watch-Demo.ipynb` — spawns `server.jl` as a subprocess and walks a full session with raw JSON shown in both directions at every step: initialize handshake → `tools/list` (PrettyTables summary) → `tools/call` on `molecular_weight` and `antoine_vapor_pressure` → two failure cases (unknown tool → JSON-RPC invalid-params error; out-of-range temperature → `isError: true` tool result) → shutdown. Ends with the markdown-only ecosystem pointer.
4. `CHEME-142-M4-MCP-ToolDiscovery-Ungraded-Codio-Activity.ipynb` — students drive the client against the provided three-tool server: connect, initialize, list tools, read input schemas. Core exercise: given a natural-language request ("What is the saturation pressure of acetone at 320 K?"), play the agent — select the tool, construct the arguments dictionary from its schema, make the call, interpret the result. One error-handling exercise. Self-check `@assert` cells, no grade.
5. `CHEME-142-M4-MCP-ServerTools-Graded-Codio-Activity.ipynb` **+ `-Solution.ipynb`** — students extend the server with a fourth tool, `vanderwaals_pressure`, working entirely in notebook cells (no `src/` edits, so grading reads from the notebook alone). Task 1: implement the handler function. Task 2: write its JSON Schema descriptor, build the `MCPTool`, and register it. Task 3: verify at the wire level — run the real dispatch loop in-process with injected streams, confirm `tools/list` returns four tools, and check `tools/call` results against known values. Wrap-up `@assert`/`@test` cells grade the results; the solution notebook is the completed version.

Support files: `Include.jl`, `Project.toml`, `Manifest.toml`, `server.jl` (entry script), `src/`, `data/`, `test/`.

## Protocol scope (tools-only MCP subset)

- Transport: stdio, newline-delimited UTF-8 JSON-RPC 2.0 messages (one message per line, no embedded newlines).
- Methods implemented: `initialize` (request/response with `protocolVersion`, `capabilities`, `clientInfo`/`serverInfo`), `notifications/initialized`, `tools/list`, `tools/call`.
- Errors: JSON-RPC error objects for parse error (−32700), method not found (−32601), and invalid params (−32602); tool execution failures returned as a `tools/call` result with `isError: true` (the MCP convention the lecture contrasts with protocol errors).
- `protocolVersion` pinned to one published spec revision in all notebooks (exact string finalized during build against the current spec); the lecture notes that version negotiation handles mismatches.
- Resources and prompts: explained in Read-Pages 1, **not implemented** — tools-only keeps `src/` readable end-to-end, which is the point of the self-contained approach.
- Shutdown: client closes the subprocess's stdin; server loop exits when stdin reaches EOF.

## `src/` scaffolding

### `Types.jl`
- `MCPTool` — `name::String`, `description::String`, `inputschema::Dict{String,Any}` (JSON Schema), `handler::Function` (takes the arguments `Dict`, returns a result `String` or throws for the `isError` path).
- `MCPServerConnection` — subprocess handle (from `open(cmd, "r+")`), request-id counter, initialized flag.

### `Server.jl`
- Tool registry (name → `MCPTool`) and `register!(tool)`.
- `serve(registry; input=stdin, output=stdout)` — dispatch loop: read line → parse JSON-RPC → route by method → write one-line response. Runs until EOF. Injectable streams so `test/` can drive the loop in-process.

### `Client.jl`
- `connect(cmd::Cmd)::MCPServerConnection` — spawn the server subprocess.
- `initialize!(conn)`, `listtools(conn)`, `calltool(conn, name, arguments::Dict)`, `Base.close(conn)`.
- Raw `send!(conn, msg)` / `receive!(conn)` exposed, plus a verbose mode that prints each wire message, so notebooks can show the JSON traffic in both directions.

### `Tools.jl`
- The three provided ChemE tools (below) and their registrations. The graded activity's fourth tool is implemented and registered in notebook cells, not here — `src/` ships identical for all students.

### `server.jl` (module root)
- Entry script: include `Include.jl`, register tools, call `serve(...)`. Launched by the client as `julia --project=<module dir> server.jl` — the same shape as a command entry in a host's MCP server configuration, which the lecture points out.

## Tool specifics (numbers finalized during build for clean answers)

1. `molecular_weight` — molar mass (g/mol) from a chemical formula string (element symbols with optional integer counts; parser scope kept small, decided during build). Atomic-weights table for common elements in `data/` (JSON).
2. `antoine_vapor_pressure` — saturation pressure for a named species at temperature T via the Antoine equation; coefficient table in `data/` includes units and valid temperature range, and the tool rejects out-of-range T (source of the demo's invalid-arguments failure case).
3. `ideal_gas_solve` — given any three of P, V, n, T, solve PV = nRT for the fourth (R = 8.314 J mol⁻¹ K⁻¹).
4. `vanderwaals_pressure` *(graded)* — P = nRT/(V − nb) − a n²/V² for a named species; a/b coefficient table in `data/`. Explicit in P, so no root-finding; contrasts directly with `ideal_gas_solve`.

## `test/runtests.jl`

- Protocol tests: drive `serve` in-process with injected streams — initialize response fields, `tools/list` contents, `tools/call` known values, each error path.
- Wire tests: spawn `server.jl` as a real subprocess and repeat the happy path end-to-end.
- `test/runtests.jl` covers the shipped three-tool `src/` only; the graded fourth tool is checked by `@assert`/`@test` cells inside the graded notebook (incomplete in the student version, passing in the solution).

## Build order (vertical slice first)

- **Phase 0 — scaffolding:** `src/`, `server.jl`, `data/`, `Include.jl`, `Project.toml`, `test/`; smoke-test the full client↔server loop in Julia.
- **Phase 1 — Watch-Demo:** validates the authoring + execution pattern and produces the captured wire traffic the lectures quote.
- **Phase 2 — Read-Pages:** introduction, then protocol mechanics (using real captured messages).
- **Phase 3 — ungraded activity.**
- **Phase 4 — graded activity + solution** (solution first, student version derived from it by stubbing).

## Authoring standards (from CLAUDE.md and module precedent)

Each notebook: title + intro, three Learning Objectives (`* __[title]:__ …`), Setup (`include("Include.jl")`), worked sections, Summary + exactly three Key Takeaways, direct concise language, no unnecessary adjectives, all content supported by the material. Notebooks are authored with nbformat/nbconvert and executed top-to-bottom in the local Julia environment with real outputs embedded. Figures: none required; the architecture description is markdown/text, and the author may add a drawn schematic later.

## Verification

- Every notebook executes top-to-bottom in the local Julia environment with outputs embedded.
- `test/runtests.jl` passes against the shipped `src/`; the graded notebook's check cells pass in the solution version.
- Wire examples quoted in Read-Pages 2 match the implementation's actual messages.
- Tool numerical results checked against hand calculations (e.g., water molar mass 18.015 g/mol; Antoine and van der Waals values verified against the coefficient tables' sources).

## Non-goals

- No LLM or API-key integration anywhere in the module.
- No resources or prompts implementation (lecture-level coverage only).
- No HTTP/SSE transport implementation (stdio only; HTTP transport described in the lecture).
- No Node.js/Python runtime dependency; official reference servers are discussed, not executed.
