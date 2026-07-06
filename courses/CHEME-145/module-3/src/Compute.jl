import Distributions: loglikelihood # we add a loglikelihood method for MyHiddenMarkovModel (Task 3)

"""
    simulate(hmm::MyHiddenMarkovModel, T::Int; N::Int = 1,
        rng::AbstractRNG = Random.default_rng()) -> Array{NamedTuple,1}

Generate `N` sequences of length `T` by forward sampling: draw s₁ ~ π₀, then oₜ ~ E[sₜ,:] and
sₜ₊₁ ~ P[sₜ,:]. Each element of the returned array is `(hidden, observed)`, both `Array{Int64,1}`.
"""
function simulate(hmm::MyHiddenMarkovModel, T::Int; N::Int = 1,
    rng::AbstractRNG = Random.default_rng())::Array{NamedTuple,1}

    # initialize: categorical distributions for the initial state, transitions, and emissions -
    d₀ = Categorical(hmm.π₀);
    dP = [Categorical(hmm.P[i, :]) for i ∈ 1:size(hmm.P, 1)];
    dE = [Categorical(hmm.E[i, :]) for i ∈ 1:size(hmm.E, 1)];

    # main loop: sample N sequences -
    sequences = Array{NamedTuple,1}();
    for _ ∈ 1:N
        hidden = Array{Int64,1}(undef, T);
        observed = Array{Int64,1}(undef, T);
        s = rand(rng, d₀); # initial hidden state
        for t ∈ 1:T
            hidden[t] = s;
            observed[t] = rand(rng, dE[s]); # emit an observation from the current state
            s = rand(rng, dP[s]);           # move to the next hidden state
        end
        push!(sequences, (hidden = hidden, observed = observed));
    end

    # return -
    return sequences;
end
