# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_IMAGES = joinpath(_ROOT, "images");

# load external packages -
using Pkg
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false) # have manifest file, we are good. Otherwise, we need to instantiate the environment
    Pkg.activate("."); Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# using statements -
using Images
using ImageInTerminal
using FileIO
using ImageIO
using OneHotArrays
using Statistics
using LinearAlgebra
using Plots
using Colors
using Distances
using NNlib
using Distributions
using PrettyTables
using DataFrames
using StatsBase
using YAML
using TOML
using JSON
