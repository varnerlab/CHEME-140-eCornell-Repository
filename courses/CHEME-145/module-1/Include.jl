# setup paths -
const _ROOT = @__DIR__;
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# activate the local environment; instantiate it the first time (no Manifest.toml yet) -
using Pkg
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.add(["JuMP", "Ipopt", "Plots", "Colors", "PrettyTables", "LinearAlgebra", "ForwardDiff", "Random"]);
    Pkg.instantiate();
end

# load external packages -
using JuMP
using Ipopt
using Plots
using Colors
using PrettyTables
using LinearAlgebra
using ForwardDiff
using Random

# color palette (Paul Tol muted) -
colors = Dict{Int,RGB}();
colors[1] = colorant"#0077BB"; # blue
colors[2] = colorant"#33BBEE"; # cyan
colors[3] = colorant"#EE7733"; # orange
colors[4] = colorant"#CC3311"; # red
colors[5] = colorant"#009988"; # teal
colors[6] = colorant"#EE3377"; # magenta

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "Indifference.jl"));
include(joinpath(_PATH_TO_SRC, "Budget.jl"));
include(joinpath(_PATH_TO_SRC, "DiscreteChoice.jl"));
