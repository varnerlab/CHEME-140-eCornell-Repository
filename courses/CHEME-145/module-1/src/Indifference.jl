# ---------------------------------------------------------------------------------------------- #
# Indifference curves for two-good utility functions. An indifference curve is the set of bundles
# (x₁, x₂) that satisfy U(x₁, x₂) = U⋆ for a fixed utility level U⋆. Each method below solves that
# level-set equation for x₂ as a function of a grid of x₁ values, and returns an N×2 matrix whose
# columns are [x₁ x₂].
# ---------------------------------------------------------------------------------------------- #
"""
    indifference(model::MyLinearUtilityFunction, U::Float64, x1::AbstractVector{Float64}) -> Array{Float64,2}

Computes the indifference curve `α₁ x₁ + α₂ x₂ = U` of a two-good linear utility function.
Solves for `x₂ = (U − α₁ x₁) / α₂` over the grid `x1`.

### Returns
- `Array{Float64,2}`: an N×2 matrix with columns `[x₁ x₂]`.
"""
function indifference(model::MyLinearUtilityFunction, U::Float64, x1::AbstractVector{Float64})::Array{Float64,2}
    α = model.α;
    Y = Array{Float64,2}(undef, length(x1), 2);
    for j ∈ eachindex(x1)
        Y[j,1] = x1[j];
        Y[j,2] = (U - α[1]*x1[j])/α[2];
    end
    return Y;
end

"""
    indifference(model::MyCobbDouglasUtilityFunction, U::Float64, x1::AbstractVector{Float64}) -> Array{Float64,2}

Computes the indifference curve `x₁^α₁ x₂^α₂ = U` of a two-good Cobb-Douglas utility function.
Solves for `x₂ = (U / x₁^α₁)^(1/α₂)` over the grid `x1`. Requires `x₁ > 0`.

### Returns
- `Array{Float64,2}`: an N×2 matrix with columns `[x₁ x₂]`.
"""
function indifference(model::MyCobbDouglasUtilityFunction, U::Float64, x1::AbstractVector{Float64})::Array{Float64,2}
    α = model.α;
    Y = Array{Float64,2}(undef, length(x1), 2);
    for j ∈ eachindex(x1)
        Y[j,1] = x1[j];
        Y[j,2] = (U/(x1[j]^α[1]))^(1/α[2]);
    end
    return Y;
end

"""
    indifference(model::MyLeontiefUtilityFunction, U::Float64; xmax::Float64, ymax::Float64) -> Array{Float64,2}

Computes the indifference curve `min(x₁/α₁, x₂/α₂) = U` of a two-good Leontief utility function.
The curve is the L-shaped corner with vertex at `(α₁ U, α₂ U)`: a vertical ray up to `ymax` and a
horizontal ray out to `xmax`. Returns the three-point polyline that traces the L.

### Returns
- `Array{Float64,2}`: a 3×2 matrix with columns `[x₁ x₂]`.
"""
function indifference(model::MyLeontiefUtilityFunction, U::Float64;
    xmax::Float64, ymax::Float64)::Array{Float64,2}
    α = model.α;
    xc = α[1]*U; # corner x-coordinate
    yc = α[2]*U; # corner y-coordinate
    Y = [
        xc   ymax; # top of the vertical ray
        xc   yc;   # corner (vertex)
        xmax yc;   # end of the horizontal ray
    ];
    return Y;
end
