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
