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

# ---------------------------------------------------------------------------------------------- #
# Scalar wealth utilities U(w), their inverses U⁻¹(u), and decision-under-risk quantities.
# The argument w is left untyped (Real) so the functions compose with ForwardDiff for the
# Arrow-Pratt derivatives below.
# ---------------------------------------------------------------------------------------------- #
"""
    evaluate(model::AbstractWealthUtilityFunction, w) -> Real

Evaluates a scalar wealth utility `U(w)`: `ln(w)` (logarithmic), `a w + b` (linear), or `w^τ`
(power), depending on the model type.
"""
evaluate(model::MyLogarithmicUtilityFunction, w) = log(w);
evaluate(model::MyLinearWealthUtilityFunction, w) = model.a*w + model.b;
evaluate(model::MyPowerUtilityFunction, w) = w^(model.τ);

# short-form functor syntax: model(w) calls evaluate(model, w) -
(model::MyLogarithmicUtilityFunction)(w) = evaluate(model, w);
(model::MyLinearWealthUtilityFunction)(w) = evaluate(model, w);
(model::MyPowerUtilityFunction)(w) = evaluate(model, w);

"""
    inverse(model::AbstractWealthUtilityFunction, u) -> Real

Evaluates the inverse utility `U⁻¹(u)`, the wealth whose utility is `u`: `exp(u)` (logarithmic),
`(u − b)/a` (linear), or `u^(1/τ)` (power). Used to compute the certainty equivalent.
"""
inverse(model::MyLogarithmicUtilityFunction, u) = exp(u);
inverse(model::MyLinearWealthUtilityFunction, u) = (u - model.b)/model.a;
inverse(model::MyPowerUtilityFunction, u) = u^(1/model.τ);

"""
    absolute_risk_aversion(model::AbstractWealthUtilityFunction, w::Real) -> Float64

Computes the Arrow-Pratt coefficient of absolute risk aversion `A(w) = −U''(w)/U'(w)`. The first
and second derivatives are obtained by automatic differentiation (ForwardDiff), so the same code
works for any twice-differentiable wealth utility. `A(w) > 0` is risk averse, `A(w) = 0` is risk
neutral, and `A(w) < 0` is risk loving.
"""
function absolute_risk_aversion(model::AbstractWealthUtilityFunction, w::Real)::Float64
    U = x -> evaluate(model, x);
    Up = ForwardDiff.derivative(U, w); # U'(w)
    Upp = ForwardDiff.derivative(x -> ForwardDiff.derivative(U, x), w); # U''(w)
    return -Upp/Up;
end

"""
    relative_risk_aversion(model::AbstractWealthUtilityFunction, w::Real) -> Float64

Computes the Arrow-Pratt coefficient of relative risk aversion `R(w) = −w U''(w)/U'(w) = w A(w)`.
"""
relative_risk_aversion(model::AbstractWealthUtilityFunction, w::Real)::Float64 = w*absolute_risk_aversion(model, w);

# ---------------------------------------------------------------------------------------------- #
# Discrete-lottery quantities: a lottery pays wealth w[i] with probability p[i] (∑ p = 1).
# ---------------------------------------------------------------------------------------------- #
"""
    expected_utility(model::AbstractWealthUtilityFunction, w::Array{Float64,1}, p::Array{Float64,1}) -> Float64

Computes the expected utility `𝔼[U(W)] = ∑ᵢ pᵢ U(wᵢ)` of a discrete lottery with outcomes `w` and
probabilities `p`.
"""
function expected_utility(model::AbstractWealthUtilityFunction,
    w::Array{Float64,1}, p::Array{Float64,1})::Float64
    return sum(p[i]*evaluate(model, w[i]) for i ∈ eachindex(w));
end

"""
    certainty_equivalent(model::AbstractWealthUtilityFunction, w::Array{Float64,1}, p::Array{Float64,1}) -> Float64

Computes the certainty equivalent `CE = U⁻¹(𝔼[U(W)])`, the guaranteed wealth that gives the same
utility as the lottery with outcomes `w` and probabilities `p`.
"""
function certainty_equivalent(model::AbstractWealthUtilityFunction,
    w::Array{Float64,1}, p::Array{Float64,1})::Float64
    return inverse(model, expected_utility(model, w, p));
end
