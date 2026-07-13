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
