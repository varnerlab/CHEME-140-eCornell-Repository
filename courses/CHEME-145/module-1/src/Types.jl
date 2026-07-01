# ---------------------------------------------------------------------------------------------- #
# Abstract types. All utility function models are subtypes of AbstractUtilityFunction, and all
# constrained choice problems are subtypes of AbstractChoiceProblem.
# ---------------------------------------------------------------------------------------------- #
abstract type AbstractUtilityFunction end
abstract type AbstractChoiceProblem end

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
