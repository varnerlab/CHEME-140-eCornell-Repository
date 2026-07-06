"""
    build(::Type{MyHiddenMarkovModel}, data::NamedTuple) -> MyHiddenMarkovModel

Build a Hidden Markov Model from keys `P, E, π₀`. Throws `ArgumentError` if the rows of `P` or `E`
or the vector `π₀` are not probability distributions, or if the dimensions are inconsistent.
"""
function build(::Type{MyHiddenMarkovModel}, data::NamedTuple)::MyHiddenMarkovModel

    # get data -
    P = data.P; E = data.E; π₀ = data.π₀;

    # validate -
    (size(P, 1) == size(P, 2)) || throw(ArgumentError("transition matrix P must be square"));
    (size(E, 1) == size(P, 1)) || throw(ArgumentError("emission matrix E must have one row per hidden state"));
    (length(π₀) == size(P, 1)) || throw(ArgumentError("π₀ must have one entry per hidden state"));
    (all(≥(0), P) && all(≥(0), E) && all(≥(0), π₀)) || throw(ArgumentError("all probabilities must be non-negative"));
    for i ∈ 1:size(P, 1)
        isapprox(sum(P[i, :]), 1.0, atol = 1e-8) || throw(ArgumentError("row $(i) of P must sum to 1"));
        isapprox(sum(E[i, :]), 1.0, atol = 1e-8) || throw(ArgumentError("row $(i) of E must sum to 1"));
    end
    isapprox(sum(π₀), 1.0, atol = 1e-8) || throw(ArgumentError("π₀ must sum to 1"));

    # build -
    model = MyHiddenMarkovModel();
    model.P = P; model.E = E; model.π₀ = π₀;
    return model;
end

"""
    build(::Type{MyBaumWelchModel}, data::NamedTuple) -> MyBaumWelchModel

Build Baum-Welch settings from keys `maxiterations, ϵ`.
"""
function build(::Type{MyBaumWelchModel}, data::NamedTuple)::MyBaumWelchModel
    model = MyBaumWelchModel();
    model.maxiterations = data.maxiterations;
    model.ϵ = data.ϵ;
    return model;
end
