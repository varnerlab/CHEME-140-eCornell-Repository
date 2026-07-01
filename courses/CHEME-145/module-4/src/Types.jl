# ---------------------------------------------------------------------------------------------- #
# Abstract types. MDP problems, world models, and solution (algorithm) models.
# ---------------------------------------------------------------------------------------------- #
abstract type AbstractMDPModel end
abstract type AbstractWorldModel end
abstract type AbstractSolutionModel end

"""
    mutable struct MyMDPProblemModel <: AbstractMDPModel

A finite Markov decision process (𝒮, 𝒜, T, R, γ).

### Fields
- `𝒮::Array{Int64,1}`: state index set.
- `𝒜::Array{Int64,1}`: action index set.
- `T::Array{Float64,3}`: transition array, `T[s,s′,a] = P(s′ | s,a)`.
- `R::Array{Float64,2}`: reward array, `R[s,a]` = expected immediate reward.
- `γ::Float64`: discount factor, `0 ≤ γ < 1`.
"""
mutable struct MyMDPProblemModel <: AbstractMDPModel
    𝒮::Array{Int64,1}
    𝒜::Array{Int64,1}
    T::Array{Float64,3}
    R::Array{Float64,2}
    γ::Float64
    MyMDPProblemModel() = new();
end

"""
    mutable struct MyRectangularGridWorldModel <: AbstractWorldModel

A rectangular grid world environment. Maps between `(x,y)` coordinates and state indices, holds the
action move vectors, and the sparse reward dictionary.

### Fields
- `nrows::Int`, `ncols::Int`: grid dimensions.
- `coordinates::Dict{Int,Tuple{Int,Int}}`: state index → `(x,y)`.
- `states::Dict{Tuple{Int,Int},Int}`: `(x,y)` → state index.
- `moves::Dict{Int,Tuple{Int,Int}}`: action → `(Δx,Δy)`.
- `rewards::Dict{Tuple{Int,Int},Float64}`: `(x,y)` → reward (non-default cells only).
"""
mutable struct MyRectangularGridWorldModel <: AbstractWorldModel
    nrows::Int
    ncols::Int
    coordinates::Dict{Int,Tuple{Int,Int}}
    states::Dict{Tuple{Int,Int},Int}
    moves::Dict{Int,Tuple{Int,Int}}
    rewards::Dict{Tuple{Int,Int},Float64}
    MyRectangularGridWorldModel() = new();
end

"""
    mutable struct MyValueIterationModel <: AbstractSolutionModel

Value-iteration solver settings.

### Fields
- `maxiterations::Int64`: iteration cap.
- `ϵ::Float64`: sup-norm convergence tolerance, `ϵ > 0`.
"""
mutable struct MyValueIterationModel <: AbstractSolutionModel
    maxiterations::Int64
    ϵ::Float64
    MyValueIterationModel() = new();
end

"""
    mutable struct MyPolicyIterationModel <: AbstractSolutionModel

Policy-iteration solver settings.

### Fields
- `maxiterations::Int64`: iteration cap on policy-improvement sweeps.
"""
mutable struct MyPolicyIterationModel <: AbstractSolutionModel
    maxiterations::Int64
    MyPolicyIterationModel() = new();
end

"""
    mutable struct MyValueFunctionPolicy

Container for a solved MDP: the problem and its optimal value function `V`. Recover the policy with
`policy(Q(problem, V))`.

### Fields
- `problem::MyMDPProblemModel`: the solved problem.
- `V::Array{Float64,1}`: value function, one entry per state.
"""
mutable struct MyValueFunctionPolicy
    problem::MyMDPProblemModel
    V::Array{Float64,1}
    MyValueFunctionPolicy() = new();
end
