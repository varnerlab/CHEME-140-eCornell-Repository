# ---------------------------------------------------------------------------------------------- #
# Evaluate a utility function U(x) at a bundle x.
# ---------------------------------------------------------------------------------------------- #
"""
    evaluate(model::MyLinearUtilityFunction, x::Array{Float64,1}) -> Float64

Evaluates the linear utility `U(x) = ∑ αᵢ xᵢ` at the bundle `x`.
"""
function evaluate(model::MyLinearUtilityFunction, x::Array{Float64,1})::Float64
    return dot(model.α, x);
end

"""
    evaluate(model::MyCobbDouglasUtilityFunction, x::Array{Float64,1}) -> Float64

Evaluates the Cobb-Douglas utility `U(x) = ∏ xᵢ^αᵢ` at the bundle `x`.
"""
function evaluate(model::MyCobbDouglasUtilityFunction, x::Array{Float64,1})::Float64
    α = model.α;
    value = 1.0;
    for i ∈ eachindex(x)
        value *= x[i]^α[i];
    end
    return value;
end

"""
    evaluate(model::MyLeontiefUtilityFunction, x::Array{Float64,1}) -> Float64

Evaluates the Leontief utility `U(x) = min(x₁/α₁, …, xₙ/αₙ)` at the bundle `x`.
"""
function evaluate(model::MyLeontiefUtilityFunction, x::Array{Float64,1})::Float64
    return minimum(x ./ model.α);
end

# short-form functor syntax: model(x) calls evaluate(model, x) -
(model::MyLinearUtilityFunction)(x::Array{Float64,1}) = evaluate(model, x);
(model::MyCobbDouglasUtilityFunction)(x::Array{Float64,1}) = evaluate(model, x);
(model::MyLeontiefUtilityFunction)(x::Array{Float64,1}) = evaluate(model, x);

# ---------------------------------------------------------------------------------------------- #
# Marginal utility of a Cobb-Douglas utility function.
# ---------------------------------------------------------------------------------------------- #
"""
    marginal_utility(model::MyCobbDouglasUtilityFunction, x::Array{Float64,1}) -> Array{Float64,1}

Computes the marginal utility vector `Ūᵢ = ∂U/∂xᵢ` of a Cobb-Douglas utility function at the
bundle `x`. Uses the closed form `∂U/∂xᵢ = αᵢ U(x) / xᵢ`.

### Returns
- `Array{Float64,1}`: the marginal utility of each good at `x`.
"""
function marginal_utility(model::MyCobbDouglasUtilityFunction, x::Array{Float64,1})::Array{Float64,1}
    α = model.α;
    U = evaluate(model, x);
    MU = Array{Float64,1}(undef, length(x));
    for i ∈ eachindex(x)
        MU[i] = α[i]*U/x[i]; # ∂U/∂xᵢ = αᵢ U / xᵢ
    end
    return MU;
end

# ---------------------------------------------------------------------------------------------- #
# Numerically solve a budget-constrained Cobb-Douglas choice problem with JuMP + Ipopt.
# ---------------------------------------------------------------------------------------------- #
"""
    solve(problem::MySimpleCobbDouglasChoiceProblem) -> Dict{String,Any}

Numerically solves the budget-constrained Cobb-Douglas utility maximization problem with the
JuMP modeling language and the Ipopt interior-point solver. Because the logarithm is strictly
increasing, we maximize the concave transform `∑ αᵢ log(xᵢ)`, which has the same maximizer as
`∏ xᵢ^αᵢ`.

### Returns
A `Dict{String,Any}` with keys:
- `argmax::Array{Float64,1}`: the optimal bundle `x⋆`.
- `budget::Float64`: resources spent at the optimum, `∑ cᵢ x⋆ᵢ`.
- `objective_value::Float64`: the Cobb-Douglas utility `∏ (x⋆ᵢ)^αᵢ` at the optimum.
"""
function solve(problem::MySimpleCobbDouglasChoiceProblem)::Dict{String,Any}

    # initialize -
    α = problem.α;
    c = problem.c;
    I = problem.I;
    bounds = problem.bounds;
    xₒ = problem.initial;
    n = length(α);

    # build the JuMP model, hand it to Ipopt -
    model = Model(Ipopt.Optimizer);
    set_silent(model);

    # decision variables (with bounds and a starting point) -
    @variable(model, bounds[i,1] <= x[i=1:n] <= bounds[i,2], start = xₒ[i]);

    # objective: maximize the log of the Cobb-Douglas utility (concave, same maximizer) -
    @objective(model, Max, sum(α[i]*log(x[i]) for i ∈ 1:n));

    # budget constraint: spend the entire budget -
    @constraint(model, sum(c[i]*x[i] for i ∈ 1:n) == I);

    # solve -
    optimize!(model);

    # package and return the solution -
    x_opt = value.(x) |> collect;
    results = Dict{String,Any}();
    results["argmax"] = x_opt;
    results["budget"] = dot(c, x_opt);
    results["objective_value"] = prod(x_opt[i]^α[i] for i ∈ 1:n);
    return results;
end
