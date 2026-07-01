# ---------------------------------------------------------------------------------------------- #
# Abstract types. All utility function models are subtypes of AbstractUtilityFunction, and all
# constrained choice problems are subtypes of AbstractChoiceProblem.
# ---------------------------------------------------------------------------------------------- #
abstract type AbstractUtilityFunction end
abstract type AbstractChoiceProblem end

# ---------------------------------------------------------------------------------------------- #
# Scalar utility functions over wealth, U: ℝ₊ → ℝ, used for decision under risk (expected utility,
# certainty equivalent, risk premium, Arrow-Pratt risk aversion).
# ---------------------------------------------------------------------------------------------- #
abstract type AbstractWealthUtilityFunction end

"""
    mutable struct MyLinearUtilityFunction <: AbstractUtilityFunction

Model of a linear utility function `U(x) = α'x = ∑ αᵢ xᵢ`. Models goods that are perfect substitutes.

### Fields
- `α::Array{Float64,1}`: parameter (weight) vector with `αᵢ ≥ 0`.
"""
mutable struct MyLinearUtilityFunction <: AbstractUtilityFunction
    α::Array{Float64,1} # parameter (weight) vector
    MyLinearUtilityFunction() = new(); # empty constructor
end

"""
    mutable struct MyCobbDouglasUtilityFunction <: AbstractUtilityFunction

Model of a Cobb-Douglas utility function `U(x) = ∏ xᵢ^αᵢ`. Models goods that are consumed together.

### Fields
- `α::Array{Float64,1}`: exponent vector with `αᵢ ≥ 0` and `∑ αᵢ = 1`.
"""
mutable struct MyCobbDouglasUtilityFunction <: AbstractUtilityFunction
    α::Array{Float64,1} # exponent vector
    MyCobbDouglasUtilityFunction() = new(); # empty constructor
end

"""
    mutable struct MyLeontiefUtilityFunction <: AbstractUtilityFunction

Model of a Leontief utility function `U(x) = min(x₁/α₁, …, xₙ/αₙ)`. Models perfect complements
(goods consumed in fixed proportions).

### Fields
- `α::Array{Float64,1}`: scaling vector with `αᵢ > 0`.
"""
mutable struct MyLeontiefUtilityFunction <: AbstractUtilityFunction
    α::Array{Float64,1} # scaling vector
    MyLeontiefUtilityFunction() = new(); # empty constructor
end

"""
    mutable struct MySimpleCobbDouglasChoiceProblem <: AbstractChoiceProblem

Model of a budget-constrained Cobb-Douglas utility maximization problem:
`max ∏ xᵢ^αᵢ` subject to `∑ cᵢ xᵢ = I` and `xᵢ ≥ 0`.

### Fields
- `α::Array{Float64,1}`: exponent (preference) vector with `∑ αᵢ = 1`.
- `c::Array{Float64,1}`: unit price of each good.
- `I::Float64`: total budget (income).
- `bounds::Array{Float64,2}`: lower/upper bounds on each good, one row per good `[L U]`.
- `initial::Array{Float64,1}`: initial guess passed to the numerical solver.
"""
mutable struct MySimpleCobbDouglasChoiceProblem <: AbstractChoiceProblem
    α::Array{Float64,1} # exponent (preference) vector
    c::Array{Float64,1} # unit price of each good
    I::Float64 # total budget (income)
    bounds::Array{Float64,2} # bounds on each good
    initial::Array{Float64,1} # initial guess for the solver
    MySimpleCobbDouglasChoiceProblem() = new(); # empty constructor
end

"""
    mutable struct MyLogarithmicUtilityFunction <: AbstractWealthUtilityFunction

Model of a logarithmic (Bernoulli) utility over wealth, `U(w) = ln(w)`. Concave, so it models a
risk-averse decision-maker. Has no parameters.
"""
mutable struct MyLogarithmicUtilityFunction <: AbstractWealthUtilityFunction
    MyLogarithmicUtilityFunction() = new(); # empty constructor
end

"""
    mutable struct MyLinearWealthUtilityFunction <: AbstractWealthUtilityFunction

Model of a linear utility over wealth, `U(w) = a w + b`. Has zero curvature, so it models a
risk-neutral decision-maker.

### Fields
- `a::Float64`: slope, `a > 0`.
- `b::Float64`: intercept.
"""
mutable struct MyLinearWealthUtilityFunction <: AbstractWealthUtilityFunction
    a::Float64 # slope
    b::Float64 # intercept
    MyLinearWealthUtilityFunction() = new(); # empty constructor
end

"""
    mutable struct MyPowerUtilityFunction <: AbstractWealthUtilityFunction

Model of a power utility over wealth, `U(w) = w^τ`. The exponent `τ` sets the risk attitude:
`τ < 1` is concave (risk averse), `τ = 1` is linear (risk neutral), and `τ > 1` is convex
(risk loving).

### Fields
- `τ::Float64`: risk (curvature) parameter, `τ > 0`.
"""
mutable struct MyPowerUtilityFunction <: AbstractWealthUtilityFunction
    τ::Float64 # risk (curvature) parameter
    MyPowerUtilityFunction() = new(); # empty constructor
end
