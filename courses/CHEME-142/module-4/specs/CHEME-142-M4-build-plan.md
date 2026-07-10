# CHEME-142 Module 4 (MCP Servers) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the self-contained Julia MCP server + client in `src/`, and the six notebooks (two Read-Pages, one Watch-Demo, one ungraded activity, one graded pair) per the approved spec, all executed with real outputs.

**Architecture:** A local Julia project (`module-4/`) in the CHEME-142 house style: `Include.jl` activates the local environment and loads `src/` (Types → Server → Client → Tools). The server speaks a tools-only MCP subset (JSON-RPC 2.0, newline-delimited, stdio); `server.jl` is the subprocess entry point the client launches. Notebooks are thin presentation layers over the tested library, executed via `jupyter nbconvert`.

**Tech Stack:** Julia 1.12.6 (kernel `julia-1.12`); packages `IJulia`, `JSON` (pinned 0.21), `PrettyTables` (v3 API), `Test` stdlib; Jupyter nbconvert.

**Model assignment (per author decision):** Phase 0 and Phase 5 are executed inline by the main session (Fable) — protocol correctness and final review. Phases 1–4 (notebook authoring and student-version derivation) are delegated to **one continuing Opus subagent** (single agent for voice consistency across the series; `model: "opus"`), with a Fable review gate after each phase.

## Global Constraints

- **Working directory:** `courses/CHEME-142/module-4/` (all paths relative to this unless noted).
- **Protocol constants (verbatim, everywhere):** `PROTOCOL_VERSION = "2025-06-18"`; server name `"cheme-142-m4-cheme-tools"`; server version `"1.0.0"`.
- **Transport:** stdio only; one JSON-RPC message per line (newline-delimited UTF-8, no embedded newlines). stdout carries protocol messages only; logs go to stderr.
- **Methods implemented (only these):** `initialize`, `notifications/initialized`, `tools/list`, `tools/call`.
- **Error model:** JSON-RPC error objects for parse error (−32700), method not found (−32601), unknown tool / invalid params (−32602), internal error (−32603); tool *execution* failures are successful `tools/call` responses with `"isError": true`.
- **JSON pin:** `JSON = "0.21"` in both `[deps]` and `[compat]` (JSON 1.x breaks the global IJulia kernel precompile — durable course gotcha).
- **PrettyTables v3 API:** `column_labels =` (not `header =`), data as a `Matrix`.
- **Naming:** `CHEME-142-M4-<Topic>-<DeliveryType>.ipynb` (CHEME-142 pattern — no `Lecture-`/`Example-` marker; that is a CHEME-145 convention).
- **Notebook template (every notebook):** title + intro paragraph; blockquoted `> __Learning Objectives__` with exactly three `* __[title]:__ …` bullets ending "Let's get started!"; **Setup** (`include("Include.jl")` — activity/demo notebooks only; Read-Pages have zero code cells); worked sections; `## Summary` with a blockquoted `> __Key Takeaways:__` list of exactly three `* **[title]:** …` bullets, framed by a one-sentence direct summary before and a one-sentence conclusion after. Direct concise language, no unnecessary adjectives; bold step-label punctuation *inside* the markers (`__Initialize:__`, never `__Initialize__:`); `### Algorithm` nests under an `## <Topic>` parent (never H2→H4).
- **Read-Pages notebooks have zero code cells** (theory only). Wire examples appear as ` ```json ` fenced blocks in markdown.
- **Manifest.toml is gitignored repo-wide** — never commit it.
- **Commit message style:** `M4: <what>` (matches repo history), each ending with the Claude co-author trailer.

---

## Notebook authoring & execution procedure (shared by all notebook tasks)

1. **Create the `.ipynb`** with a Python `nbformat` builder script in the scratchpad (raw triple-quoted cell sources; `nbformat.write` handles JSON escaping). Notebook metadata:
   ```json
   {
     "kernelspec": {"display_name": "Julia 1.12.6", "language": "julia", "name": "julia-1.12"},
     "language_info": {"file_extension": ".jl", "mimetype": "application/julia", "name": "julia", "version": "1.12.6"}
   }
   ```
2. **Execute in place:**
   ```bash
   jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.kernel_name=julia-1.12 \
     --ExecutePreprocessor.timeout=600 <path>.ipynb
   ```
   Expected: exit 0, no `CellExecutionError`. Add `--allow-errors` **only** for the graded *student* notebook (its stubs raise intentionally). Read-Pages notebooks are also run through this (no-op on zero code cells) to normalize format.
3. **Verify** embedded outputs against the task's stated expected values.
4. **Review pass (main session):** extract cell sources + output summaries to a markdown file (never diff raw `.ipynb` — base64 blobs) and check against the template and expected values.
5. **Commit.**

---

## File structure

**Create:**
- `Project.toml` — deps `IJulia, JSON, PrettyTables, Test`; `[compat] JSON = "0.21"`.
- `Include.jl` — env activation/instantiation, package loads, path constants, `include` of `src/`.
- `src/Types.jl` — `MCPTool`, `MCPServerConnection`, `build` factories.
- `src/Server.jl` — protocol constants, `register!`, `process_message`, `serve`, `run_session`.
- `src/Client.jl` — `connect`, `send!`, `receive!`, `request!`, `initialize!`, `listtools`, `calltool`, `Base.close`.
- `src/Tools.jl` — the three provided tool handlers + `build_default_registry`.
- `server.jl` — subprocess entry point (lean load: JSON + Types/Server/Tools only).
- `data/atomic-weights.json`, `data/antoine-coefficients.json`, `data/vanderwaals-coefficients.json`.
- `test/runtests.jl` — unit + protocol + wire tests.
- Six notebooks (paths in their tasks).

**Reference (do not modify):** `../module-2/` and `../module-3/` notebooks (voice/format), `../../CHEME-145/module-4/` (Read-Pages/graded-pair structure), `specs/CHEME-142-M4-build-spec.md`.

---

## Phase 0 — Scaffolding & library (main session, Fable)

### Task 0.1: Environment and data tables

**Files:**
- Create: `Project.toml`, `Include.jl`, `data/atomic-weights.json`, `data/antoine-coefficients.json`, `data/vanderwaals-coefficients.json`

**Interfaces:**
- Produces: `include("Include.jl")` loads packages and (after Tasks 0.2–0.4) `src/`; path constants `_ROOT`, `_PATH_TO_SRC`, `_PATH_TO_DATA`, `_PATH_TO_SERVER`.

- [ ] **Step 1: Write `Project.toml`**

```toml
[deps]
IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
JSON = "0.21"
```

- [ ] **Step 2: Write `Include.jl`** (CHEME-142 comment style, CHEME-145 activation robustness)

```julia
# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SERVER = joinpath(_ROOT, "server.jl");

# activate the local environment; instantiate on first run (no Manifest.toml yet) -
using Pkg
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.add(["IJulia", "JSON", "PrettyTables", "Test"]);
    Pkg.instantiate();
end

# load packages -
using JSON
using PrettyTables
using Test

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Server.jl"));
include(joinpath(_PATH_TO_SRC, "Client.jl"));
include(joinpath(_PATH_TO_SRC, "Tools.jl"));
```

(The four `include`d files land with Tasks 0.2–0.4; write `Include.jl` complete now — nothing runs it until the Task 0.4 smoke test.)

- [ ] **Step 3: Write the three data files**

`data/atomic-weights.json` (IUPAC 2021 standard atomic weights, g/mol):

```json
{
  "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81,
  "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
  "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
  "S": 32.06, "Cl": 35.45, "Ar": 39.95, "K": 39.098, "Ca": 40.078,
  "Fe": 55.845, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "I": 126.90
}
```

`data/antoine-coefficients.json` (NIST WebBook form: log₁₀(Psat/bar) = A − B/(T/K + C); verify each entry against NIST during this step and correct if the WebBook differs):

```json
{
  "water":   {"A": 4.6543,  "B": 1435.264, "C": -64.848, "Tmin": 255.9, "Tmax": 373.0},
  "acetone": {"A": 4.42448, "B": 1312.253, "C": -32.445, "Tmin": 259.2, "Tmax": 507.6},
  "ethanol": {"A": 5.24677, "B": 1598.673, "C": -46.424, "Tmin": 292.8, "Tmax": 366.6},
  "benzene": {"A": 4.01814, "B": 1203.835, "C": -53.226, "Tmin": 287.7, "Tmax": 354.1}
}
```

`data/vanderwaals-coefficients.json` (a in Pa·m⁶·mol⁻², b in m³·mol⁻¹, standard tabulated values):

```json
{
  "co2": {"a": 0.3640, "b": 4.267e-5},
  "n2":  {"a": 0.1408, "b": 3.913e-5},
  "o2":  {"a": 0.1378, "b": 3.183e-5},
  "ch4": {"a": 0.2283, "b": 4.278e-5},
  "nh3": {"a": 0.4225, "b": 3.707e-5},
  "h2o": {"a": 0.5536, "b": 3.049e-5}
}
```

- [ ] **Step 4: Spot-check the Antoine table by hand** (values used in later asserts)

water at 350 K: 10^(4.6543 − 1435.264/285.152) ≈ **0.4178 bar** (literature ≈ 41.7 kPa ✓). acetone at 320 K: ≈ **0.7261 bar** (bp 329 K ✓). benzene at 298.15 K: ≈ **0.1268 bar** (literature ≈ 12.7 kPa ✓). If a NIST correction in Step 3 changes these, update the expected values in Tasks 0.5, 1.1, 3.1.

- [ ] **Step 5: Commit**

```bash
git add Project.toml Include.jl data/ && git commit -m "M4: add module environment and ChemE data tables"
```

### Task 0.2: `src/Types.jl` and `src/Server.jl` (protocol core)

**Files:**
- Create: `src/Types.jl`, `src/Server.jl`

**Interfaces:**
- Produces: `MCPTool` (fields `name::String`, `description::String`, `inputschema::Dict{String,Any}`, `handler::Function` — handler takes `Dict{String,Any}`, returns `String`, throws on execution failure); `MCPServerConnection` (fields `process`, `requestid::Int64`, `verbose::Bool`); `build(::Type{MCPTool}, data::NamedTuple)::MCPTool`; `register!(registry, tool)`; `process_message(registry, line)::Union{Nothing,Dict}`; `serve(registry; input, output)`; `run_session(registry, messages::Vector{String})::Vector{String}`.

- [ ] **Step 1: Write `src/Types.jl`**

```julia
abstract type AbstractMCPModel end

"""
    mutable struct MCPTool <: AbstractMCPModel

Descriptor and handler for a single MCP tool. The `handler` takes the
`arguments::Dict{String,Any}` from a `tools/call` request and returns a
result `String`; it throws an exception to signal a tool execution failure.
"""
mutable struct MCPTool <: AbstractMCPModel
    name::String
    description::String
    inputschema::Dict{String,Any}
    handler::Function

    MCPTool() = new(); # empty constructor, fields set by build
end

"""
    mutable struct MCPServerConnection <: AbstractMCPModel

Client-side handle for a running MCP server subprocess.
"""
mutable struct MCPServerConnection <: AbstractMCPModel
    process::Base.Process
    requestid::Int64
    verbose::Bool

    MCPServerConnection() = new(); # empty constructor, fields set by connect
end

"""
    build(::Type{MCPTool}, data::NamedTuple)::MCPTool
"""
function build(::Type{MCPTool}, data::NamedTuple)::MCPTool
    tool = MCPTool();
    tool.name = data.name;
    tool.description = data.description;
    tool.inputschema = data.inputschema;
    tool.handler = data.handler;
    return tool;
end
```

- [ ] **Step 2: Write `src/Server.jl`**

```julia
# protocol constants -
const PROTOCOL_VERSION = "2025-06-18";
const SERVER_NAME = "cheme-142-m4-cheme-tools";
const SERVER_VERSION = "1.0.0";

"""
    register!(registry::Dict{String,MCPTool}, tool::MCPTool)
"""
function register!(registry::Dict{String,MCPTool}, tool::MCPTool)::Dict{String,MCPTool}
    registry[tool.name] = tool;
    return registry;
end

# -- JSON-RPC message builders ------------------------------------------- #
function _response(id, result::Dict)::Dict{String,Any}
    return Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result);
end

function _error(id, code::Int64, message::String)::Dict{String,Any}
    return Dict{String,Any}("jsonrpc" => "2.0", "id" => id,
        "error" => Dict{String,Any}("code" => code, "message" => message));
end

# -- method handlers ------------------------------------------------------ #
function _handle_initialize(id)::Dict{String,Any}
    result = Dict{String,Any}(
        "protocolVersion" => PROTOCOL_VERSION,
        "capabilities" => Dict{String,Any}("tools" => Dict{String,Any}()),
        "serverInfo" => Dict{String,Any}("name" => SERVER_NAME, "version" => SERVER_VERSION));
    return _response(id, result);
end

function _handle_tools_list(registry::Dict{String,MCPTool}, id)::Dict{String,Any}
    tools = Vector{Dict{String,Any}}();
    for name in sort(collect(keys(registry))) # sorted: deterministic output
        tool = registry[name];
        push!(tools, Dict{String,Any}("name" => tool.name,
            "description" => tool.description, "inputSchema" => tool.inputschema));
    end
    return _response(id, Dict{String,Any}("tools" => tools));
end

function _handle_tools_call(registry::Dict{String,MCPTool}, id, params::Dict)::Dict{String,Any}
    name = get(params, "name", nothing);
    if (name === nothing || haskey(registry, name) == false)
        return _error(id, -32602, "Unknown tool: $(name)"); # protocol error
    end
    arguments = get(params, "arguments", Dict{String,Any}());
    text, iserror = try
        (registry[name].handler(arguments), false)
    catch e
        (sprint(showerror, e), true) # tool execution failure: NOT a protocol error
    end
    result = Dict{String,Any}(
        "content" => [Dict{String,Any}("type" => "text", "text" => text)],
        "isError" => iserror);
    return _response(id, result);
end

"""
    process_message(registry::Dict{String,MCPTool}, line::AbstractString)::Union{Nothing,Dict{String,Any}}

Parse one JSON-RPC message and return the response `Dict`, or `nothing`
for notifications (which get no response).
"""
function process_message(registry::Dict{String,MCPTool}, line::AbstractString)::Union{Nothing,Dict{String,Any}}
    message = try
        JSON.parse(line)
    catch
        return _error(nothing, -32700, "Parse error");
    end
    id = get(message, "id", nothing);
    method = get(message, "method", nothing);
    if (method == "initialize")
        return _handle_initialize(id);
    elseif (method == "notifications/initialized")
        return nothing; # notification: no response
    elseif (method == "tools/list")
        return _handle_tools_list(registry, id);
    elseif (method == "tools/call")
        return _handle_tools_call(registry, id, get(message, "params", Dict{String,Any}()));
    elseif (id === nothing)
        return nothing; # unknown notification: ignore per JSON-RPC 2.0
    else
        return _error(id, -32601, "Method not found: $(method)");
    end
end

"""
    serve(registry::Dict{String,MCPTool}; input::IO = stdin, output::IO = stdout)

Dispatch loop: one JSON-RPC message per line until `input` reaches EOF.
Only protocol messages are written to `output`.
"""
function serve(registry::Dict{String,MCPTool}; input::IO = stdin, output::IO = stdout)
    for line in eachline(input)
        (isempty(strip(line))) && continue
        response = try
            process_message(registry, line)
        catch
            _error(nothing, -32603, "Internal error")
        end
        (response === nothing) && continue
        println(output, JSON.json(response)); flush(output);
    end
end

"""
    run_session(registry::Dict{String,MCPTool}, messages::Vector{String})::Vector{String}

Run the dispatch loop in-process over `messages`; return the response lines.
Used by the tests and the graded activity for wire-level verification.
"""
function run_session(registry::Dict{String,MCPTool}, messages::Vector{String})::Vector{String}
    input = IOBuffer(join(messages, "\n") * "\n");
    output = IOBuffer();
    serve(registry; input = input, output = output);
    return filter(!isempty, split(String(take!(output)), "\n")) .|> String;
end
```

- [ ] **Step 3: Smoke-test the protocol core in-process** (from `module-4/`)

```bash
julia --project=. -e '
using Pkg; Pkg.instantiate();
using JSON
include("src/Types.jl"); include("src/Server.jl");
registry = Dict{String,MCPTool}();
r = process_message(registry, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}");
@assert r["result"]["protocolVersion"] == "2025-06-18"
@assert process_message(registry, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}") === nothing
@assert process_message(registry, "not json")["error"]["code"] == -32700
@assert process_message(registry, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"bogus\"}")["error"]["code"] == -32601
println("protocol core OK")'
```

Expected: `protocol core OK` (first run pays the Pkg instantiate cost).

- [ ] **Step 4: Commit**

```bash
git add src/Types.jl src/Server.jl && git commit -m "M4: add MCP protocol core (types, dispatch loop)"
```

### Task 0.3: `src/Tools.jl` and `server.jl`

**Files:**
- Create: `src/Tools.jl`, `server.jl`

**Interfaces:**
- Consumes: `MCPTool`, `build`, `register!`, `serve` from Task 0.2.
- Produces: handlers `_tool_molecular_weight`, `_tool_antoine_vapor_pressure`, `_tool_ideal_gas_solve` (each `Dict{String,Any} -> String`, JSON text results); `build_default_registry()::Dict{String,MCPTool}`; runnable `server.jl`.

- [ ] **Step 1: Write `src/Tools.jl`**

```julia
# load coefficient tables (path relative to this file, so server.jl and Include.jl both work) -
const ATOMIC_WEIGHTS = JSON.parsefile(joinpath(@__DIR__, "..", "data", "atomic-weights.json"));
const ANTOINE_COEFFICIENTS = JSON.parsefile(joinpath(@__DIR__, "..", "data", "antoine-coefficients.json"));

"""
    _parse_formula(formula::AbstractString)::Dict{String,Int64}

Parse a chemical formula of element symbols with optional integer counts,
e.g. "C6H12O6". No parentheses. Throws `ArgumentError` on anything else.
"""
function _parse_formula(formula::AbstractString)::Dict{String,Int64}
    (isempty(formula)) && throw(ArgumentError("Empty formula"));
    counts = Dict{String,Int64}();
    matchedlength = 0;
    for m in eachmatch(r"([A-Z][a-z]?)(\d*)", formula)
        symbol = m.captures[1];
        (haskey(ATOMIC_WEIGHTS, symbol)) || throw(ArgumentError("Unknown element: $(symbol)"));
        count = isempty(m.captures[2]) ? 1 : parse(Int64, m.captures[2]);
        counts[symbol] = get(counts, symbol, 0) + count;
        matchedlength += length(m.match);
    end
    (matchedlength == length(formula)) || throw(ArgumentError("Unable to parse formula: $(formula)"));
    return counts;
end

function _tool_molecular_weight(arguments::Dict{String,Any})::String
    (haskey(arguments, "formula")) || throw(ArgumentError("Missing required argument: formula"));
    formula = arguments["formula"];
    counts = _parse_formula(formula);
    M = sum(count * ATOMIC_WEIGHTS[symbol] for (symbol, count) in counts);
    return JSON.json(Dict("formula" => formula, "molar_mass" => round(M, digits = 3), "units" => "g/mol"));
end

function _tool_antoine_vapor_pressure(arguments::Dict{String,Any})::String
    (haskey(arguments, "species")) || throw(ArgumentError("Missing required argument: species"));
    (haskey(arguments, "T")) || throw(ArgumentError("Missing required argument: T"));
    species = lowercase(arguments["species"]);
    (haskey(ANTOINE_COEFFICIENTS, species)) || throw(ArgumentError(
        "Unknown species: $(species). Known species: $(join(sort(collect(keys(ANTOINE_COEFFICIENTS))), ", "))"));
    p = ANTOINE_COEFFICIENTS[species];
    T = arguments["T"];
    (p["Tmin"] <= T <= p["Tmax"]) || throw(ArgumentError(
        "T = $(T) K is outside the valid range [$(p["Tmin"]), $(p["Tmax"])] K for $(species)"));
    Psat = 10.0^(p["A"] - p["B"] / (T + p["C"]));
    return JSON.json(Dict("species" => species, "T" => T, "Psat" => round(Psat, sigdigits = 5), "units" => "bar"));
end

function _tool_ideal_gas_solve(arguments::Dict{String,Any})::String
    R = 8.314; # J mol⁻¹ K⁻¹
    variables = ("P", "V", "n", "T");
    provided = [v for v in variables if haskey(arguments, v)];
    (length(provided) == 3) || throw(ArgumentError("Provide exactly three of P, V, n, T; got $(length(provided))"));
    unknown = only(v for v in variables if !(v in provided));
    P = get(arguments, "P", nothing); V = get(arguments, "V", nothing);
    n = get(arguments, "n", nothing); T = get(arguments, "T", nothing);
    value = (unknown == "P") ? n * R * T / V :
            (unknown == "V") ? n * R * T / P :
            (unknown == "n") ? P * V / (R * T) : P * V / (n * R);
    units = Dict("P" => "Pa", "V" => "m^3", "n" => "mol", "T" => "K")[unknown];
    return JSON.json(Dict("variable" => unknown, "value" => round(value, sigdigits = 6), "units" => units));
end

"""
    build_default_registry()::Dict{String,MCPTool}

The three ChemE tools this module's server ships with.
"""
function build_default_registry()::Dict{String,MCPTool}
    registry = Dict{String,MCPTool}();
    register!(registry, build(MCPTool, (
        name = "molecular_weight",
        description = "Compute the molar mass (g/mol) of a chemical formula, e.g. H2O or C6H12O6.",
        inputschema = Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}(
                "formula" => Dict{String,Any}("type" => "string", "description" => "Chemical formula, element symbols with optional integer counts")),
            "required" => ["formula"]),
        handler = _tool_molecular_weight)));
    register!(registry, build(MCPTool, (
        name = "antoine_vapor_pressure",
        description = "Saturation pressure (bar) of a named species at temperature T (K) from the Antoine equation.",
        inputschema = Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}(
                "species" => Dict{String,Any}("type" => "string", "description" => "Species name, e.g. water, acetone, ethanol, benzene"),
                "T" => Dict{String,Any}("type" => "number", "description" => "Temperature in K, inside the species' valid range")),
            "required" => ["species", "T"]),
        handler = _tool_antoine_vapor_pressure)));
    register!(registry, build(MCPTool, (
        name = "ideal_gas_solve",
        description = "Solve the ideal gas law PV = nRT for the one variable not provided (SI units: Pa, m^3, mol, K).",
        inputschema = Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}(
                "P" => Dict{String,Any}("type" => "number", "description" => "Pressure in Pa"),
                "V" => Dict{String,Any}("type" => "number", "description" => "Volume in m^3"),
                "n" => Dict{String,Any}("type" => "number", "description" => "Amount in mol"),
                "T" => Dict{String,Any}("type" => "number", "description" => "Temperature in K")),
            "required" => []),
        handler = _tool_ideal_gas_solve)));
    return registry;
end
```

- [ ] **Step 2: Write `server.jl`** (module root; lean load — no notebook packages)

```julia
# CHEME-142 Module 4 MCP server entry point. A host launches this as a
# subprocess: julia --project=<module dir> server.jl
# stdout is reserved for protocol messages; anything else goes to stderr.
import Pkg
Pkg.activate(@__DIR__; io = stderr);

using JSON

include(joinpath(@__DIR__, "src", "Types.jl"));
include(joinpath(@__DIR__, "src", "Server.jl"));
include(joinpath(@__DIR__, "src", "Tools.jl"));

serve(build_default_registry(); input = stdin, output = stdout)
```

- [ ] **Step 3: Smoke-test tools in-process**

```bash
julia --project=. -e '
using JSON
include("src/Types.jl"); include("src/Server.jl"); include("src/Tools.jl");
registry = build_default_registry();
r = process_message(registry, JSON.json(Dict("jsonrpc"=>"2.0","id"=>1,"method"=>"tools/call",
    "params"=>Dict("name"=>"molecular_weight","arguments"=>Dict("formula"=>"H2O")))));
result = JSON.parse(r["result"]["content"][1]["text"]);
@assert isapprox(result["molar_mass"], 18.015; atol = 1e-3)
@assert r["result"]["isError"] == false
bad = process_message(registry, JSON.json(Dict("jsonrpc"=>"2.0","id"=>2,"method"=>"tools/call",
    "params"=>Dict("name"=>"antoine_vapor_pressure","arguments"=>Dict("species"=>"water","T"=>500.0)))));
@assert bad["result"]["isError"] == true
println("tools OK")'
```

Expected: `tools OK`.

- [ ] **Step 4: Smoke-test the real subprocess over the wire**

```bash
printf '%s\n%s\n%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
 | julia --project=. server.jl
```

Expected: exactly two JSON lines on stdout — an `initialize` response with `"protocolVersion":"2025-06-18"`, and a `tools/list` response listing `antoine_vapor_pressure`, `ideal_gas_solve`, `molecular_weight` (alphabetical). Pkg activation chatter, if any, appears on stderr only.

- [ ] **Step 5: Commit**

```bash
git add src/Tools.jl server.jl && git commit -m "M4: add ChemE tool registry and server entry point"
```

### Task 0.4: `src/Client.jl`

**Files:**
- Create: `src/Client.jl`

**Interfaces:**
- Consumes: `MCPServerConnection`, `PROTOCOL_VERSION`.
- Produces: `connect(command::Cmd; verbose=false)::MCPServerConnection`; `send!(conn, message::Dict)`; `receive!(conn)::Dict`; `request!(conn, method, params)::Dict`; `initialize!(conn)::Dict`; `listtools(conn)::Dict`; `calltool(conn, name::String, arguments::Dict)::Dict`; `Base.close(conn)`. Verbose mode prints `→ <json>` / `← <json>` wire lines (what the Watch-Demo shows).

- [ ] **Step 1: Write `src/Client.jl`**

```julia
"""
    connect(command::Cmd; verbose::Bool = false)::MCPServerConnection

Launch an MCP server subprocess and return a connection handle. With
`verbose = true`, every wire message is echoed: `→` outgoing, `←` incoming.
"""
function connect(command::Cmd; verbose::Bool = false)::MCPServerConnection
    connection = MCPServerConnection();
    connection.process = open(command, "r+");
    connection.requestid = 0;
    connection.verbose = verbose;
    return connection;
end

function send!(connection::MCPServerConnection, message::Dict)
    line = JSON.json(message);
    (connection.verbose) && println("→ ", line);
    println(connection.process, line); flush(connection.process);
end

function receive!(connection::MCPServerConnection)::Dict
    line = readline(connection.process);
    (isempty(line)) && error("Server closed the connection");
    (connection.verbose) && println("← ", line);
    return JSON.parse(line);
end

function request!(connection::MCPServerConnection, method::String, params::Dict)::Dict
    connection.requestid += 1;
    send!(connection, Dict("jsonrpc" => "2.0", "id" => connection.requestid,
        "method" => method, "params" => params));
    return receive!(connection);
end

"""
    initialize!(connection::MCPServerConnection)::Dict

Run the MCP initialization handshake: an `initialize` request followed by
the `notifications/initialized` notification (which gets no response).
"""
function initialize!(connection::MCPServerConnection)::Dict
    response = request!(connection, "initialize", Dict(
        "protocolVersion" => PROTOCOL_VERSION,
        "capabilities" => Dict{String,Any}(),
        "clientInfo" => Dict("name" => "cheme-142-notebook-client", "version" => "1.0.0")));
    send!(connection, Dict("jsonrpc" => "2.0", "method" => "notifications/initialized"));
    return response;
end

listtools(connection::MCPServerConnection)::Dict = request!(connection, "tools/list", Dict{String,Any}());

function calltool(connection::MCPServerConnection, name::String, arguments::Dict)::Dict
    return request!(connection, "tools/call", Dict("name" => name, "arguments" => arguments));
end

"""
    Base.close(connection::MCPServerConnection)

Close the server's stdin (EOF ends its dispatch loop) and wait for exit.
"""
function Base.close(connection::MCPServerConnection)
    close(connection.process.in);
    wait(connection.process);
    return nothing;
end
```

- [ ] **Step 2: Smoke-test the full client↔server loop**

```bash
julia --project=. -e '
include("Include.jl");
command = `$(joinpath(Sys.BINDIR, "julia")) --project=$(_ROOT) $(_PATH_TO_SERVER)`;
connection = connect(command);
r0 = initialize!(connection);
@assert r0["result"]["serverInfo"]["name"] == "cheme-142-m4-cheme-tools"
r1 = listtools(connection);
@assert length(r1["result"]["tools"]) == 3
r2 = calltool(connection, "antoine_vapor_pressure", Dict("species" => "acetone", "T" => 320.0));
result = JSON.parse(r2["result"]["content"][1]["text"]);
@assert isapprox(result["Psat"], 0.7261; rtol = 1e-2)
close(connection);
@assert process_exited(connection.process)
println("wire OK")'
```

Expected: `wire OK` (subprocess startup adds a few seconds).

- [ ] **Step 3: Commit**

```bash
git add src/Client.jl && git commit -m "M4: add MCP client (subprocess launch, handshake, tool calls)"
```

### Task 0.5: `test/runtests.jl`

**Files:**
- Create: `test/runtests.jl`

**Interfaces:**
- Consumes: everything from Tasks 0.1–0.4 via `include("../Include.jl")`.

- [ ] **Step 1: Write `test/runtests.jl`**

```julia
include(joinpath(@__DIR__, "..", "Include.jl"));

@testset "formula parsing and molecular weight" begin
    @test _parse_formula("H2O") == Dict("H" => 2, "O" => 1)
    @test isapprox(JSON.parse(_tool_molecular_weight(Dict{String,Any}("formula" => "H2O")))["molar_mass"], 18.015; atol = 1e-3)
    @test isapprox(JSON.parse(_tool_molecular_weight(Dict{String,Any}("formula" => "C6H12O6")))["molar_mass"], 180.156; atol = 1e-3)
    @test_throws ArgumentError _parse_formula("Xx2")
    @test_throws ArgumentError _parse_formula("H2O)")
    @test_throws ArgumentError _tool_molecular_weight(Dict{String,Any}())
end

@testset "antoine vapor pressure" begin
    r = JSON.parse(_tool_antoine_vapor_pressure(Dict{String,Any}("species" => "water", "T" => 350.0)));
    @test isapprox(r["Psat"], 0.4178; rtol = 1e-2) # hand check: 10^(4.6543 - 1435.264/285.152)
    @test_throws ArgumentError _tool_antoine_vapor_pressure(Dict{String,Any}("species" => "water", "T" => 500.0))
    @test_throws ArgumentError _tool_antoine_vapor_pressure(Dict{String,Any}("species" => "octane", "T" => 300.0))
end

@testset "ideal gas solve" begin
    r = JSON.parse(_tool_ideal_gas_solve(Dict{String,Any}("n" => 1.0, "T" => 273.15, "V" => 0.022414)));
    @test r["variable"] == "P"
    @test isapprox(r["value"], 101320.0; rtol = 1e-3) # 1 mol at STP ≈ 1 atm
    @test_throws ArgumentError _tool_ideal_gas_solve(Dict{String,Any}("n" => 1.0, "T" => 273.15))
end

@testset "protocol: dispatch and errors" begin
    registry = build_default_registry();
    r = process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "initialize", "params" => Dict())));
    @test r["result"]["protocolVersion"] == PROTOCOL_VERSION
    @test process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "method" => "notifications/initialized"))) === nothing
    @test process_message(registry, "this is not JSON")["error"]["code"] == -32700
    @test process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "resources/list")))["error"]["code"] == -32601
    r = process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "id" => 3, "method" => "tools/call",
        "params" => Dict("name" => "gibbs_energy", "arguments" => Dict()))));
    @test r["error"]["code"] == -32602
    r = process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "id" => 4, "method" => "tools/call",
        "params" => Dict("name" => "antoine_vapor_pressure", "arguments" => Dict("species" => "water", "T" => 500.0)))));
    @test r["result"]["isError"] == true # execution failure is NOT a protocol error
end

@testset "protocol: in-process session (run_session)" begin
    registry = build_default_registry();
    messages = [
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "initialize", "params" => Dict())),
        JSON.json(Dict("jsonrpc" => "2.0", "method" => "notifications/initialized")),
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "tools/list")),
    ];
    responses = run_session(registry, messages);
    @test length(responses) == 2 # the notification gets no response
    tools = JSON.parse(responses[2])["result"]["tools"];
    @test [t["name"] for t in tools] == ["antoine_vapor_pressure", "ideal_gas_solve", "molecular_weight"]
end

@testset "wire: real subprocess" begin
    command = `$(joinpath(Sys.BINDIR, "julia")) --project=$(_ROOT) $(_PATH_TO_SERVER)`;
    connection = connect(command);
    @test initialize!(connection)["result"]["serverInfo"]["name"] == SERVER_NAME
    @test length(listtools(connection)["result"]["tools"]) == 3
    r = calltool(connection, "molecular_weight", Dict("formula" => "C6H12O6"));
    @test isapprox(JSON.parse(r["result"]["content"][1]["text"])["molar_mass"], 180.156; atol = 1e-3)
    close(connection);
    @test process_exited(connection.process)
end
```

- [ ] **Step 2: Run the suite**

```bash
julia --project=. test/runtests.jl
```

Expected: all testsets pass, zero failures.

- [ ] **Step 3: Commit**

```bash
git add test/runtests.jl && git commit -m "M4: add protocol, tool, and wire test suite"
```

---

## Phase 1 — Watch-Demo (Opus subagent; Fable reviews)

### Task 1.1: `CHEME-142-M4-MCP-ChemETools-Watch-Demo.ipynb`

**Files:**
- Create: `CHEME-142-M4-MCP-ChemETools-Watch-Demo.ipynb`

**Interfaces:**
- Consumes: `connect`, `initialize!`, `listtools`, `calltool`, `Base.close`, `_ROOT`, `_PATH_TO_SERVER` (Task 0.4); JSON/PrettyTables from `Include.jl`.
- Produces: the executed demo whose captured wire lines Task 2.2 quotes.

- [ ] **Step 1: Author the notebook** per the shared procedure. Cell manifest (markdown prose to the CHEME-142 template; every code cell verbatim below):

1. *(md)* Title `# Talking to an MCP Server: ChemE Tools` + intro paragraph (what MCP is in one sentence; what this demo does: launch a local Julia MCP server as a subprocess and drive a full session, watching every message cross the wire) + blockquoted Learning Objectives (three: launch/handshake; discovery via `tools/list`; calls and the two error types via `tools/call`) + "Let's get started!".
2. *(md)* `## Setup, Data, and Prerequisites` — one paragraph: local environment, `src/` MCP implementation, `server.jl` entry point.
3. *(code)*
   ```julia
   include("Include.jl");
   ```
4. *(md)* `### Constants` — the launch command is the same shape as a command entry in a host's MCP configuration file.
5. *(code)*
   ```julia
   server_command = `$(joinpath(Sys.BINDIR, "julia")) --project=$(_ROOT) $(_PATH_TO_SERVER)`;
   ```
6. *(md)* `## Task 1: Connect and Initialize` — subprocess launch over stdio; the handshake: `initialize` request, response with `protocolVersion`/`capabilities`/`serverInfo`, then the `notifications/initialized` notification. `verbose = true` echoes every wire message: `→` outgoing, `←` incoming.
7. *(code)*
   ```julia
   connection = connect(server_command, verbose = true);
   response_initialize = initialize!(connection)
   ```
8. *(md)* transition: pull the server identity out of the response.
9. *(code)*
   ```julia
   response_initialize["result"]["serverInfo"]
   ```
10. *(md)* `## Task 2: Discover the Tools` — `tools/list`; each descriptor: `name`, `description`, `inputSchema` (JSON Schema). This is how an agent learns what a server can do — no docs, no SDK.
11. *(code)*
    ```julia
    response_tools = listtools(connection);
    tools = response_tools["result"]["tools"];
    ```
12. *(code)*
    ```julia
    let
        table = Matrix{String}(undef, length(tools), 2);
        for (i, tool) in enumerate(tools)
            table[i, 1] = tool["name"];
            table[i, 2] = tool["description"];
        end
        pretty_table(table; column_labels = ["name", "description"])
    end
    ```
13. *(md)* transition: look at one input schema in full (the contract an agent uses to build arguments).
14. *(code)*
    ```julia
    JSON.print(tools[1]["inputSchema"], 2) # antoine_vapor_pressure (tools are listed alphabetically)
    ```
15. *(md)* `## Task 3: Call the Tools` — `tools/call` with `name` + `arguments`; result carries a `content` array and an `isError` flag.
16. *(code)*
    ```julia
    response_mw = calltool(connection, "molecular_weight", Dict("formula" => "C6H12O6"));
    result_mw = JSON.parse(response_mw["result"]["content"][1]["text"])
    ```
17. *(md)* transition: glucose is 180.156 g/mol; now a thermodynamic property from a data table the server owns.
18. *(code)*
    ```julia
    response_psat = calltool(connection, "antoine_vapor_pressure", Dict("species" => "acetone", "T" => 320.0));
    result_psat = JSON.parse(response_psat["result"]["content"][1]["text"])
    ```
19. *(md)* `## Task 4: When Calls Fail` — the two failure modes: a *protocol error* (unknown tool → JSON-RPC error object, code −32602) versus a *tool execution error* (tool ran and failed → successful response with `isError: true`). A host retries or reports these differently.
20. *(code)*
    ```julia
    response_unknown = calltool(connection, "gibbs_energy", Dict("species" => "water"))
    ```
21. *(code)*
    ```julia
    response_range = calltool(connection, "antoine_vapor_pressure", Dict("species" => "water", "T" => 500.0))
    ```
22. *(md)* transition: note `response_unknown` has an `error` field and no `result`; `response_range` has a `result` with `isError = true` and the failure text in `content`.
23. *(md)* `## Task 5: Shut Down` — closing the server's stdin is the stdio shutdown signal; the dispatch loop ends at EOF.
24. *(code)*
    ```julia
    close(connection);
    process_exited(connection.process)
    ```
25. *(md)* `## The Open-Source MCP Ecosystem` — markdown-only pointer: the official reference servers (Everything, Filesystem, Fetch, Time) at github.com/modelcontextprotocol/servers, official SDKs, spec at modelcontextprotocol.io; this module's server is the same protocol at teaching scale.
26. *(md)* `## Summary` — direct summary sentence; `> __Key Takeaways:__` with exactly three bullets (launch/handshake over stdio; discovery-then-call with JSON Schema contracts; protocol errors vs. tool execution errors); conclusion sentence pointing to the activities.

- [ ] **Step 2: Execute** per the shared procedure.

- [ ] **Step 3: Verify outputs:** cell 7 shows `→`/`←` lines for three wire messages (initialize request, response, initialized notification); cell 12 table lists the three tools alphabetically; cell 16 shows `molar_mass = 180.156`; cell 18 shows `Psat ≈ 0.7261`; cell 20 shows an `"error"` entry with code −32602; cell 21 shows `"isError" => true`; cell 24 prints `true`.

- [ ] **Step 4: Save the captured wire lines** (copy from cells 7, 11, 16, 20, 21 outputs) to `specs/wire-capture.md` for Task 2.2 to quote verbatim.

- [ ] **Step 5: Commit**

```bash
git add CHEME-142-M4-MCP-ChemETools-Watch-Demo.ipynb specs/wire-capture.md
git commit -m "M4: add ChemE tools MCP watch-demo (executed)"
```

---

## Phase 2 — Read-Pages lectures (same Opus subagent; Fable reviews)

### Task 2.1: `CHEME-142-M4-Introduction-MCP-Read-Pages.ipynb`

**Files:**
- Create: `CHEME-142-M4-Introduction-MCP-Read-Pages.ipynb`

**Interfaces:**
- Consumes: spec section "Deliverables" item 1; module-2 Watch-Demo lecture voice (blockquoted definition boxes: `> __Term__` + explanation paragraphs).

- [ ] **Step 1: Author** (zero code cells). Section manifest:

1. Title `# Introduction to the Model Context Protocol` + intro + Learning Objectives (three: state the integration problem MCP solves and name the host/client/server roles; distinguish the three server primitives and the two transports; describe the open-source ecosystem and its governance) + "Let's get started!".
2. `## The Tool-Integration Problem` — LLMs generate text; acting on the world requires tools; before MCP each host application integrated each tool source separately (M host applications × N tool sources = M·N custom integrations); MCP replaces this with one protocol (M + N implementations). Blockquote boxes for __Host__, __Client__, __Server__ definitions.
3. `## Architecture` — host process (the agent application) owns one client per server; servers expose capabilities; where the LLM sits: the host sends tool descriptors to the model, the model selects a tool and arguments, the host executes the call through the client and returns the result to the model. This module has no LLM: the student plays the host.
4. `## Server Primitives` — blockquote boxes: __Tools__ (model-controlled actions; this module's focus), __Resources__ (application-controlled data), __Prompts__ (user-controlled templates). Only tools are implemented in this module's server.
5. `## Transports` — __stdio__: server as a local subprocess, one JSON-RPC message per line, stdout reserved for protocol messages, logs to stderr; __Streamable HTTP__: remote servers (described only). Tie-back: M2 REST is stateless request/response; M3 WebSockets hold a persistent connection; MCP runs stateful JSON-RPC sessions over transports like these.
6. `## The Open-Source Ecosystem` — spec released by Anthropic (Nov 2024), donated to the Agentic AI Foundation within the Linux Foundation (Dec 2025); official SDKs (TypeScript, Python, and others); reference servers (Everything, Filesystem, Fetch, Time) as readable examples; this module's server implements the same wire protocol in Julia at teaching scale.
7. `## Security Considerations` — brief: servers execute code on the user's machine; hosts require explicit user consent for tool calls; tool descriptions are untrusted input to the model.
8. `## Looking Ahead` — forward pointer to the protocol-mechanics lecture, then demo and activities.
9. `## Summary` + three Key Takeaways (integration problem → one protocol; host/client/server with LLM at the host; open governance + self-contained Julia implementation here) + conclusion sentence.

- [ ] **Step 2–4: Execute (no-op normalize), review, commit** `M4: add MCP introduction lecture (read-pages)`.

### Task 2.2: `CHEME-142-M4-MCP-ProtocolMessages-Read-Pages.ipynb`

**Files:**
- Create: `CHEME-142-M4-MCP-ProtocolMessages-Read-Pages.ipynb`

**Interfaces:**
- Consumes: `specs/wire-capture.md` (Task 1.1 Step 4) — quote the real captured messages, pretty-printed, in ` ```json ` blocks.

- [ ] **Step 1: Author** (zero code cells). Section manifest:

1. Title `# MCP Protocol Messages` + intro + Learning Objectives (three: construct/read the three JSON-RPC 2.0 message shapes; trace the MCP session lifecycle; distinguish protocol errors from tool execution errors) + "Let's get started!".
2. `## JSON-RPC 2.0 Message Anatomy` — request (`jsonrpc`, `id`, `method`, `params`), response (`result` XOR `error`), notification (no `id`, never answered); error object (`code`, `message`) with the code table: −32700 parse, −32601 method not found, −32602 invalid params/unknown tool, −32603 internal.
3. `## Session Lifecycle` — `### Algorithm` block (bold labels, colon inside markers):
   `__Launch:__` host starts the server subprocess. `__Initialize:__` client sends `initialize` with `protocolVersion`, `capabilities`, `clientInfo`; server answers with its version, capabilities, `serverInfo`. `__Confirm:__` client sends `notifications/initialized` (no response). `__Operate:__` `tools/list` and `tools/call` requests in any order. `__Shutdown:__` client closes the server's stdin; the server exits at EOF. Quoted captured initialize exchange.
4. `## Tool Discovery` — `tools/list` result: descriptor = `name` + `description` + `inputSchema` (JSON Schema: `type`, `properties`, `required`); quoted captured exchange; the schema is the contract an agent uses to construct arguments.
5. `## Tool Invocation` — `tools/call` params (`name`, `arguments`), result (`content` array of typed items — text here — plus `isError`); quoted captured success exchange; then the two captured failure exchanges side by side with the rule: *protocol error* = the request itself was bad (error object, no result); *tool execution error* = the request was valid but the tool failed (`isError: true` result). Hosts handle these differently.
6. `## The stdio Transport` — newline-delimited UTF-8, one message per line, no embedded newlines; stdout protocol-only, stderr for logs; `### Algorithm` block for the server dispatch loop (`__Read:__` a line from stdin. `__Parse:__` JSON-RPC; on failure emit −32700. `__Route:__` by `method`; notifications produce no response. `__Respond:__` one line to stdout, flush. `__Repeat:__` until EOF.).
7. `## Summary` + three Key Takeaways (three message shapes; lifecycle initialize→initialized→operate→EOF; two error channels) + conclusion pointing to demo/activities.

- [ ] **Step 2–4: Execute, review** (verify every quoted JSON matches `specs/wire-capture.md`), **commit** `M4: add MCP protocol messages lecture (read-pages)`.

---

## Phase 3 — Ungraded activity (same Opus subagent; Fable reviews)

### Task 3.1: `CHEME-142-M4-MCP-ToolDiscovery-Ungraded-Codio-Activity.ipynb`

**Files:**
- Create: `CHEME-142-M4-MCP-ToolDiscovery-Ungraded-Codio-Activity.ipynb`

**Interfaces:**
- Consumes: client API (Task 0.4). Fully executable end-to-end (ungraded = run-and-modify, house convention); modification prompts direct students to edit and re-run.

- [ ] **Step 1: Author.** Cell manifest:

1. *(md)* Title `# Activity: MCP Tool Discovery and Invocation` + intro + Learning Objectives (three: run the handshake and read `serverInfo`; select a tool and construct schema-conformant arguments from a natural-language request; interpret both failure modes) + "Let's get started!".
2. *(md)* `## Setup, Data, and Prerequisites`; *(code)* `include("Include.jl");`
3. *(md)* `### Constants`; *(code)*
   ```julia
   server_command = `$(joinpath(Sys.BINDIR, "julia")) --project=$(_ROOT) $(_PATH_TO_SERVER)`;
   ```
4. *(md)* `## Task 1: Connect and Initialize`; *(code)*
   ```julia
   connection = connect(server_command, verbose = true);
   response_initialize = initialize!(connection);
   @assert response_initialize["result"]["serverInfo"]["name"] == "cheme-142-m4-cheme-tools"
   response_initialize["result"]["serverInfo"]
   ```
5. *(md)* `## Task 2: Discover the Tools` — prompt: which tool answers a saturation-pressure question, and what does its schema require? *(code)*
   ```julia
   tools = listtools(connection)["result"]["tools"];
   [t["name"] for t in tools]
   ```
   *(code)* `JSON.print(tools[1]["inputSchema"], 2)`
6. *(md)* `## Task 3: Play the Agent` — request: *"What is the saturation pressure of acetone at 320 K, in bar?"* Walk the agent's three decisions: which tool, which arguments (schema-conformant), what does the result mean. *(code)*
   ```julia
   arguments_psat = Dict("species" => "acetone", "T" => 320.0); # from the inputSchema: species (string), T (number, K)
   response_psat = calltool(connection, "antoine_vapor_pressure", arguments_psat);
   result_psat = JSON.parse(response_psat["result"]["content"][1]["text"]);
   @assert isapprox(result_psat["Psat"], 0.7261; rtol = 1e-2)
   result_psat
   ```
   *(md)* modification prompt: change the species to ethanol at the same T, re-run; which entry of the result changed and why is the pressure lower?
7. *(md)* `## Task 4: A Second Request` — *"How many moles of an ideal gas occupy 10 L at 1 bar and 298.15 K?"* — unit discipline: the schema says Pa and m³. *(code)*
   ```julia
   arguments_n = Dict("P" => 1.0e5, "V" => 0.010, "T" => 298.15);
   response_n = calltool(connection, "ideal_gas_solve", arguments_n);
   result_n = JSON.parse(response_n["result"]["content"][1]["text"]);
   @assert result_n["variable"] == "n"
   @assert isapprox(result_n["value"], 0.40342; rtol = 1e-3)
   result_n
   ```
   *(md)* modification prompt: pass all four of P, V, n, T — what comes back, and which failure mode is it?
8. *(md)* `## Task 5: Read the Failure Modes`; *(code)*
   ```julia
   response_range = calltool(connection, "antoine_vapor_pressure", Dict("species" => "water", "T" => 200.0));
   @assert response_range["result"]["isError"] == true
   response_range["result"]["content"][1]["text"]
   ```
   *(code)*
   ```julia
   response_unknown = calltool(connection, "fugacity", Dict("species" => "water"));
   @assert haskey(response_unknown, "error")
   response_unknown["error"]
   ```
   *(md)* one-paragraph explanation prompt: state in your own words why the server answered these two calls through different channels.
9. *(md)* `## Shut Down`; *(code)*
   ```julia
   close(connection);
   ```
10. *(md)* `## Summary` + three Key Takeaways (handshake before use; schema-driven argument construction; two failure channels) + conclusion sentence.

- [ ] **Step 2–4: Execute, verify** (all asserts pass; Task 5 outputs show the range message and the −32602 error), **commit** `M4: add MCP tool-discovery ungraded activity (executed)`.

---

## Phase 4 — Graded activity (same Opus subagent; Fable reviews)

### Task 4.1: `CHEME-142-M4-MCP-ServerTools-Graded-Codio-Activity-Solution.ipynb`

**Files:**
- Create: `CHEME-142-M4-MCP-ServerTools-Graded-Codio-Activity-Solution.ipynb`

**Interfaces:**
- Consumes: `build`, `register!`, `build_default_registry`, `run_session`, `_PATH_TO_DATA`.
- Produces: the completed reference; Task 4.2 derives the student version from it.

- [ ] **Step 1: Author.** Cell manifest:

1. *(md)* Title `# Activity: Extend the MCP Server with a van der Waals Tool` + intro (students add a fourth tool and verify it at the wire level, entirely in this notebook — no `src/` edits) + Learning Objectives (three: implement a tool handler with validation and JSON text results; write a JSON Schema descriptor and register the tool; verify wire-level behavior with an in-process session) + "Let's get started!".
2. *(md)* `## Setup, Data, and Prerequisites` — van der Waals equation displayed:
   $P = \dfrac{nRT}{V - nb} - \dfrac{an^{2}}{V^{2}}$, with $a$ (Pa·m⁶·mol⁻²), $b$ (m³·mol⁻¹) from the provided table; $R = 8.314$ J·mol⁻¹·K⁻¹; domain requirement $V > nb$.
3. *(code)* `include("Include.jl");`
4. *(md)* `### Constants`; *(code)*
   ```julia
   vanderwaals_parameters = JSON.parsefile(joinpath(_PATH_TO_DATA, "vanderwaals-coefficients.json"))
   ```
5. *(md)* `## Task 1: Implement the Handler` — contract restated: takes `arguments::Dict{String,Any}` with keys `species`, `T` (K), `V` (m³), `n` (mol); returns a JSON text `String` with keys `species`, `P`, `units`; throws `ArgumentError` for a missing argument, unknown species, or `V ≤ nb`. *(code — the assessed cell)*
   ```julia
   function vanderwaals_pressure_handler(arguments::Dict{String,Any})::String
       for key in ("species", "T", "V", "n")
           (haskey(arguments, key)) || throw(ArgumentError("Missing required argument: $(key)"));
       end
       species = lowercase(arguments["species"]);
       (haskey(vanderwaals_parameters, species)) || throw(ArgumentError(
           "Unknown species: $(species). Known species: $(join(sort(collect(keys(vanderwaals_parameters))), ", "))"));
       R = 8.314; # J mol⁻¹ K⁻¹
       a = vanderwaals_parameters[species]["a"];
       b = vanderwaals_parameters[species]["b"];
       T = arguments["T"]; V = arguments["V"]; n = arguments["n"];
       (V > n * b) || throw(ArgumentError("V must exceed n*b = $(n * b) m^3"));
       P = n * R * T / (V - n * b) - a * n^2 / V^2;
       return JSON.json(Dict("species" => species, "P" => round(P, sigdigits = 6), "units" => "Pa"));
   end
   ```
6. *(md)* check: CO₂, T = 300 K, V = 1.0 L, n = 1 mol — hand value P ≈ 2.2414 × 10⁶ Pa (ideal: 2.4942 × 10⁶ Pa). *(code)*
   ```julia
   result_check = JSON.parse(vanderwaals_pressure_handler(Dict{String,Any}(
       "species" => "co2", "T" => 300.0, "V" => 1.0e-3, "n" => 1.0)));
   @assert isapprox(result_check["P"], 2.24138e6; rtol = 1e-3)
   result_check
   ```
7. *(md)* `## Task 2: Describe and Register the Tool`. *(code — the assessed cell)*
   ```julia
   vanderwaals_tool = build(MCPTool, (
       name = "vanderwaals_pressure",
       description = "Pressure (Pa) of a real gas from the van der Waals equation of state (SI units).",
       inputschema = Dict{String,Any}("type" => "object",
           "properties" => Dict{String,Any}(
               "species" => Dict{String,Any}("type" => "string", "description" => "Species key, e.g. co2, n2, o2, ch4, nh3, h2o"),
               "T" => Dict{String,Any}("type" => "number", "description" => "Temperature in K"),
               "V" => Dict{String,Any}("type" => "number", "description" => "Volume in m^3"),
               "n" => Dict{String,Any}("type" => "number", "description" => "Amount in mol")),
           "required" => ["species", "T", "V", "n"]),
       handler = vanderwaals_pressure_handler));
   registry = build_default_registry();
   register!(registry, vanderwaals_tool);
   @assert length(registry) == 4
   ```
8. *(md)* `## Task 3: Verify at the Wire Level` — `run_session` pushes real JSON-RPC lines through the same dispatch loop `server.jl` runs; provided plumbing (not assessed). *(code)*
   ```julia
   messages = [
       JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "initialize", "params" => Dict())),
       JSON.json(Dict("jsonrpc" => "2.0", "method" => "notifications/initialized")),
       JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "tools/list")),
       JSON.json(Dict("jsonrpc" => "2.0", "id" => 3, "method" => "tools/call",
           "params" => Dict("name" => "vanderwaals_pressure",
               "arguments" => Dict("species" => "co2", "T" => 300.0, "V" => 1.0e-3, "n" => 1.0)))),
       JSON.json(Dict("jsonrpc" => "2.0", "id" => 4, "method" => "tools/call",
           "params" => Dict("name" => "vanderwaals_pressure",
               "arguments" => Dict("species" => "co2", "T" => 300.0, "V" => 1.0e-5, "n" => 1.0)))),
   ];
   responses = run_session(registry, messages);
   ```
9. *(code — graded checks)*
   ```julia
   response_list = JSON.parse(responses[2]);
   toolnames = [t["name"] for t in response_list["result"]["tools"]];
   @assert "vanderwaals_pressure" in toolnames
   response_call = JSON.parse(responses[3]);
   result_wire = JSON.parse(response_call["result"]["content"][1]["text"]);
   @assert isapprox(result_wire["P"], 2.24138e6; rtol = 1e-3)
   response_domain = JSON.parse(responses[4]);
   @assert response_domain["result"]["isError"] == true # V ≤ nb rejected through the wire
   println("All wire-level checks passed.")
   ```
10. *(md)* `## Task 4: Real-Gas Deviation` — compare with `ideal_gas_solve` through the same registry. *(code)*
    ```julia
    message_ideal = JSON.json(Dict("jsonrpc" => "2.0", "id" => 5, "method" => "tools/call",
        "params" => Dict("name" => "ideal_gas_solve",
            "arguments" => Dict("T" => 300.0, "V" => 1.0e-3, "n" => 1.0))));
    response_ideal = JSON.parse(only(run_session(registry, [message_ideal])));
    P_ideal = JSON.parse(response_ideal["result"]["content"][1]["text"])["value"];
    deviation = (result_wire["P"] - P_ideal) / P_ideal * 100.0;
    @assert isapprox(deviation, -10.14; atol = 0.1)
    println("van der Waals deviates from ideal by $(round(deviation, digits = 2))% at this state.")
    ```
11. *(md)* `## Summary` + three Key Takeaways (handler = validation + computation + structured text result; the schema/registration makes it discoverable; wire-level verification exercises the same dispatch loop the subprocess runs) + conclusion sentence.

- [ ] **Step 2–4: Execute, verify** (all asserts pass; deviation prints −10.14%), **commit** `M4: add MCP server-tools graded activity solution (executed)`.

### Task 4.2: `CHEME-142-M4-MCP-ServerTools-Graded-Codio-Activity.ipynb` (student version)

**Files:**
- Create: `CHEME-142-M4-MCP-ServerTools-Graded-Codio-Activity.ipynb` (derived from 4.1)

- [ ] **Step 1: Derive from the solution** (Python `json` pass over the `.ipynb`): identical except the two assessed cells:
  - Cell 5 (Task 1) body replaced by:
    ```julia
    function vanderwaals_pressure_handler(arguments::Dict{String,Any})::String
        # TODO: validate the arguments (species, T, V, n; known species; V > n*b),
        # TODO: compute P from the van der Waals equation of state,
        # TODO: return a JSON text String with keys species, P, units.
        error("TODO: implement the van der Waals pressure handler");
    end
    ```
  - Cell 7 (Task 2) body replaced by:
    ```julia
    # TODO: build the MCPTool descriptor for vanderwaals_pressure (name, description,
    # TODO: inputSchema with species/T/V/n all required) and register it.
    vanderwaals_tool = nothing;
    registry = build_default_registry();
    error("TODO: build and register the vanderwaals_pressure tool");
    @assert length(registry) == 4
    ```
  All check/verification cells stay verbatim (they define the grade).
- [ ] **Step 2: Execute with `--allow-errors`.** Expected: cell 5's check cell and everything downstream error with the TODO messages; setup cells succeed.
- [ ] **Step 3: Verify** the notebook shows the intended failure trail and no solution text remains (grep the file for `vanderwaals_parameters[species]` — zero hits outside the provided-table cell).
- [ ] **Step 4: Commit** `M4: add MCP server-tools graded activity student version`.

---

## Phase 5 — Final review & wrap-up (main session, Fable)

### Task 5.1: Module-wide review pass

- [ ] **Step 1:** Re-run `julia --project=. test/runtests.jl` — all pass.
- [ ] **Step 2:** Extract all six notebooks' cell sources + output summaries to markdown; run the CLAUDE.md review: LO/KT format (three each, exact `* __[title]:__` / `* **[title]:**` forms), heading nesting (`### Algorithm` under `##`), bold-label punctuation inside markers, direct language, no unsupported claims, notation consistency (P in Pa, T in K, V in m³, n in mol, R = 8.314 J·mol⁻¹·K⁻¹ everywhere; JSON-RPC error codes match `src/Server.jl`).
- [ ] **Step 3:** Verify every ` ```json ` block in Read-Pages 2 against `specs/wire-capture.md`.
- [ ] **Step 4:** Rate each notebook 0–10 per CLAUDE.md Step 6 and report to the author.
- [ ] **Step 5:** Fix findings, re-execute affected notebooks, commit `M4: review pass fixes across module notebooks`.

---

## Self-review (performed at plan-writing time)

- **Spec coverage:** all five deliverables have tasks (1.1, 2.1, 2.2, 3.1, 4.1/4.2); protocol scope → 0.2; tools + data → 0.1/0.3; client → 0.4; tests → 0.5; verification section → 0.5/5.1; authoring standards → shared procedure + 5.1. The spec's "protocolVersion finalized during build" is resolved here to `2025-06-18`.
- **Type consistency:** `inputschema` (struct field, lowercase) vs `"inputSchema"` (wire key, camelCase) is deliberate — the wire follows MCP, the field follows Julia course style; both appear exactly as used.
- **Known judgment calls:** ungraded activity ships fully executable with modification prompts (house convention for ungraded); student graded version uses `error("TODO: …")` stubs + `--allow-errors` (house convention for graded pairs).
