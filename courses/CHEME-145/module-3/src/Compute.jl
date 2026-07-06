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

# scaled forward pass. Returns (α̂, c) where α̂[t,:] is the normalized forward variable at time t
# and c[t] is its normalizer, so log P(o | ℋ) = Σₜ log c[t].
function _forward(hmm::MyHiddenMarkovModel, observed::Array{Int64,1})
    P, E, π₀ = hmm.P, hmm.E, hmm.π₀;
    S = size(P, 1); T = length(observed);
    α̂ = zeros(Float64, T, S); c = zeros(Float64, T);

    # t = 1 -
    for i ∈ 1:S
        α̂[1, i] = π₀[i]*E[i, observed[1]];
    end
    c[1] = sum(α̂[1, :]);
    c[1] > 0.0 || error("observation sequence has zero probability under the model");
    α̂[1, :] = α̂[1, :] ./ c[1];

    # recursion -
    for t ∈ 2:T
        for j ∈ 1:S
            a = 0.0;
            for i ∈ 1:S
                a += α̂[t-1, i]*P[i, j];
            end
            α̂[t, j] = a*E[j, observed[t]];
        end
        c[t] = sum(α̂[t, :]);
        c[t] > 0.0 || error("observation sequence has zero probability under the model");
        α̂[t, :] = α̂[t, :] ./ c[t];
    end

    # return -
    return (α̂ = α̂, c = c);
end

"""
    loglikelihood(hmm::MyHiddenMarkovModel, observed::Array{Int64,1}) -> Float64

Compute log P(o | ℋ) with the scaled forward algorithm (evaluation problem). Numerically stable
for long sequences: the log-likelihood is recovered from the per-step scaling constants.
"""
function loglikelihood(hmm::MyHiddenMarkovModel, observed::Array{Int64,1})::Float64
    (_, c) = _forward(hmm, observed);
    return sum(log.(c));
end

"""
    viterbi(hmm::MyHiddenMarkovModel, observed::Array{Int64,1}) -> Array{Int64,1}

Compute the most likely hidden state path given the observations (decoding problem). Works in
log-space; zero-probability transitions and emissions become -Inf and are never selected.
"""
function viterbi(hmm::MyHiddenMarkovModel, observed::Array{Int64,1})::Array{Int64,1}
    P, E, π₀ = hmm.P, hmm.E, hmm.π₀;
    S = size(P, 1); T = length(observed);
    logP = log.(P); logE = log.(E); logπ₀ = log.(π₀);
    δ = fill(-Inf, T, S);     # best log-probability of any path ending in state j at time t
    ψ = zeros(Int64, T, S);   # backpointer to the best predecessor state

    # t = 1 -
    for i ∈ 1:S
        δ[1, i] = logπ₀[i] + logE[i, observed[1]];
    end

    # recursion -
    for t ∈ 2:T
        for j ∈ 1:S
            best_i = 1; best_val = -Inf;
            for i ∈ 1:S
                v = δ[t-1, i] + logP[i, j];
                if (v > best_val)
                    best_val = v; best_i = i;
                end
            end
            δ[t, j] = best_val + logE[j, observed[t]];
            ψ[t, j] = best_i;
        end
    end

    # backtrace -
    path = zeros(Int64, T);
    path[T] = argmax(δ[T, :]);
    for t ∈ (T-1):-1:1
        path[t] = ψ[t+1, path[t+1]];
    end

    # return -
    return path;
end
