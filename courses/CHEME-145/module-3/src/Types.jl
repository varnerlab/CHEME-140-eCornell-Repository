# ---------------------------------------------------------------------------------------------- #
# Abstract types. Hidden Markov models and solution (algorithm) models.
# ---------------------------------------------------------------------------------------------- #
abstract type AbstractHiddenMarkovModel end
abstract type AbstractSolutionModel end

"""
    mutable struct MyHiddenMarkovModel <: AbstractHiddenMarkovModel

A discrete Hidden Markov Model ℋ = (P, E, π₀).

### Fields
- `P::Array{Float64,2}`: transition matrix, `P[i,j] = P(sₜ₊₁ = j | sₜ = i)`. Rows sum to 1.
- `E::Array{Float64,2}`: emission matrix, `E[i,o] = P(oₜ = o | sₜ = i)`. Rows sum to 1.
- `π₀::Array{Float64,1}`: initial hidden-state distribution. Sums to 1.
"""
mutable struct MyHiddenMarkovModel <: AbstractHiddenMarkovModel
    P::Array{Float64,2}
    E::Array{Float64,2}
    π₀::Array{Float64,1}
    MyHiddenMarkovModel() = new();
end

"""
    mutable struct MyBaumWelchModel <: AbstractSolutionModel

Settings for the Baum-Welch (EM) learning algorithm.

### Fields
- `maxiterations::Int64`: maximum number of EM iterations K_max.
- `ϵ::Float64`: convergence tolerance on |ℒₖ - ℒₖ₋₁|.
"""
mutable struct MyBaumWelchModel <: AbstractSolutionModel
    maxiterations::Int64
    ϵ::Float64
    MyBaumWelchModel() = new();
end

"""
    mutable struct MyBaumWelchSolution

Learned parameters and convergence history returned by `solve(::MyBaumWelchModel, ...)`.

### Fields
- `P::Array{Float64,2}`: learned transition matrix.
- `E::Array{Float64,2}`: learned emission matrix.
- `π₀::Array{Float64,1}`: learned initial distribution.
- `loglikelihood_history::Array{Float64,1}`: total log-likelihood ℒₖ at each iteration.
- `iterations::Int64`: number of EM iterations performed.
"""
mutable struct MyBaumWelchSolution
    P::Array{Float64,2}
    E::Array{Float64,2}
    π₀::Array{Float64,1}
    loglikelihood_history::Array{Float64,1}
    iterations::Int64
    MyBaumWelchSolution() = new();
end
