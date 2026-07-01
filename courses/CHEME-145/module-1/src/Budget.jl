# ---------------------------------------------------------------------------------------------- #
# Budget line for a two-good choice problem. The budget line is the set of bundles (x₁, x₂) that
# spend the entire budget: c₁ x₁ + c₂ x₂ = I. We solve for x₂ = (I − c₁ x₁) / c₂ over a grid of
# x₁ values and return an N×2 matrix whose columns are [x₁ x₂].
# ---------------------------------------------------------------------------------------------- #
"""
    budget(problem::MySimpleCobbDouglasChoiceProblem, x1::AbstractVector{Float64}) -> Array{Float64,2}

Computes the budget line `c₁ x₁ + c₂ x₂ = I` of a two-good choice problem. Solves for
`x₂ = (I − c₁ x₁) / c₂` over the grid `x1`.

### Returns
- `Array{Float64,2}`: an N×2 matrix with columns `[x₁ x₂]`.
"""
function budget(problem::MySimpleCobbDouglasChoiceProblem, x1::AbstractVector{Float64})::Array{Float64,2}
    c = problem.c;
    I = problem.I;
    Y = Array{Float64,2}(undef, length(x1), 2);
    for j ∈ eachindex(x1)
        Y[j,1] = x1[j];
        Y[j,2] = (I - c[1]*x1[j])/c[2];
    end
    return Y;
end
