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
