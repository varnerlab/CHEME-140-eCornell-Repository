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
