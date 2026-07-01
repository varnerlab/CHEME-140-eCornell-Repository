"""
    build(modeltype::Type{T}, data::NamedTuple) -> T where {T <: AbstractUtilityFunction}

Builds a utility function model of type `T` from parameters in a `NamedTuple`. Works for the
`MyLinearUtilityFunction`, `MyCobbDouglasUtilityFunction`, and `MyLeontiefUtilityFunction` types.

### Arguments
- `modeltype::Type{T}`: the utility function type to build.
- `data::NamedTuple`: must contain the key `α::Array{Float64,1}`, the parameter vector.

### Returns
- `T`: a populated utility function model.
"""
function build(modeltype::Type{T}, data::NamedTuple)::T where {T <: AbstractUtilityFunction}

    # build an empty model, set the parameter vector, return -
    model = modeltype();
    model.α = data.α;
    return model;
end

"""
    build(modeltype::Type{MySimpleCobbDouglasChoiceProblem}, data::NamedTuple) -> MySimpleCobbDouglasChoiceProblem

Builds a budget-constrained Cobb-Douglas choice problem from parameters in a `NamedTuple`.

### Arguments
- `modeltype::Type{MySimpleCobbDouglasChoiceProblem}`: the problem type to build.
- `data::NamedTuple`: must contain the keys:
    - `α::Array{Float64,1}`: exponent (preference) vector with `∑ αᵢ = 1`.
    - `c::Array{Float64,1}`: unit price of each good.
    - `I::Float64`: total budget (income).
    - `bounds::Array{Float64,2}`: lower/upper bounds on each good, one row per good `[L U]`.
    - `initial::Array{Float64,1}`: initial guess passed to the numerical solver.

### Returns
- `MySimpleCobbDouglasChoiceProblem`: a populated choice problem.
"""
function build(modeltype::Type{MySimpleCobbDouglasChoiceProblem},
    data::NamedTuple)::MySimpleCobbDouglasChoiceProblem

    # build an empty model, populate the fields, return -
    model = modeltype();
    model.α = data.α;
    model.c = data.c;
    model.I = data.I;
    model.bounds = data.bounds;
    model.initial = data.initial;
    return model;
end

"""
    build(modeltype::Type{MyLogarithmicUtilityFunction}) -> MyLogarithmicUtilityFunction

Builds a logarithmic wealth-utility model `U(w) = ln(w)`. Takes no parameters.
"""
build(modeltype::Type{MyLogarithmicUtilityFunction}) = modeltype();

"""
    build(modeltype::Type{MyLinearWealthUtilityFunction}, data::NamedTuple) -> MyLinearWealthUtilityFunction

Builds a linear wealth-utility model `U(w) = a w + b`.

### Arguments
- `data::NamedTuple`: must contain the keys `a::Float64` (slope) and `b::Float64` (intercept).
"""
function build(modeltype::Type{MyLinearWealthUtilityFunction}, data::NamedTuple)::MyLinearWealthUtilityFunction
    model = modeltype();
    model.a = data.a;
    model.b = data.b;
    return model;
end

"""
    build(modeltype::Type{MyPowerUtilityFunction}, data::NamedTuple) -> MyPowerUtilityFunction

Builds a power wealth-utility model `U(w) = w^τ`.

### Arguments
- `data::NamedTuple`: must contain the key `τ::Float64`, the risk (curvature) parameter.
"""
function build(modeltype::Type{MyPowerUtilityFunction}, data::NamedTuple)::MyPowerUtilityFunction
    model = modeltype();
    model.τ = data.τ;
    return model;
end

"""
    build(modeltype::Type{MyLinearRandomUtilityModel}, data::NamedTuple) -> MyLinearRandomUtilityModel

Builds a multinomial logit discrete-choice model with a linear-in-features utility `Vⱼ = βᵀ xⱼ`.

### Arguments
- `data::NamedTuple`: must contain the keys `β::Array{Float64,1}` (feature weights) and
  `μ::Float64` (logit scale parameter).
"""
function build(modeltype::Type{MyLinearRandomUtilityModel}, data::NamedTuple)::MyLinearRandomUtilityModel
    model = modeltype();
    model.β = data.β;
    model.μ = data.μ;
    return model;
end
