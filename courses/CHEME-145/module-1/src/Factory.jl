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
