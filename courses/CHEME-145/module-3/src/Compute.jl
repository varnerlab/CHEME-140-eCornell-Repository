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

# all permutations of a vector (exhaustive; used for small state spaces only)
function _permutations(v::Array{Int64,1})::Array{Array{Int64,1},1}
    if (length(v) ≤ 1)
        return [v];
    end
    result = Array{Array{Int64,1},1}();
    for (i, x) ∈ enumerate(v)
        rest = vcat(v[1:(i-1)], v[(i+1):end]);
        for p ∈ _permutations(rest)
            push!(result, vcat([x], p));
        end
    end
    return result;
end

"""
    align_states(P̂, Ê, P, E) -> (σ, P, E)

Baum-Welch learns states up to a permutation of the labels. Search all |𝒮|! permutations σ for
the one minimizing ‖P̂[σ,σ] - P‖F + ‖Ê[σ,:] - E‖F, and return the permutation together with the
re-labeled learned matrices. Exhaustive search: intended for small state spaces (|𝒮| ≤ 6).
"""
function align_states(P̂::Array{Float64,2}, Ê::Array{Float64,2},
    P::Array{Float64,2}, E::Array{Float64,2})

    S = size(P, 1);
    best_σ = collect(1:S); best_err = Inf;
    for σ ∈ _permutations(collect(1:S))
        err = norm(P̂[σ, σ] - P) + norm(Ê[σ, :] - E);
        if (err < best_err)
            best_err = err; best_σ = σ;
        end
    end
    return (σ = best_σ, P = P̂[best_σ, best_σ], E = Ê[best_σ, :]);
end

# scaled backward pass. Each row is normalized to sum to 1; any per-time scaling works because the
# scaling factors cancel in the γ and ξ posterior ratios.
function _backward(hmm::MyHiddenMarkovModel, observed::Array{Int64,1})::Array{Float64,2}
    P, E = hmm.P, hmm.E;
    S = size(P, 1); T = length(observed);
    β̂ = zeros(Float64, T, S);
    β̂[T, :] .= 1.0/S;
    for t ∈ (T-1):-1:1
        for i ∈ 1:S
            b = 0.0;
            for j ∈ 1:S
                b += P[i, j]*E[j, observed[t+1]]*β̂[t+1, j];
            end
            β̂[t, i] = b;
        end
        β̂[t, :] = β̂[t, :] ./ sum(β̂[t, :]);
    end
    return β̂;
end

"""
    solve(model::MyBaumWelchModel, sequences::Array{Array{Int64,1},1};
        number_of_hidden_states::Int, number_of_observable_states::Int,
        rng::AbstractRNG = Random.default_rng()) -> MyBaumWelchSolution

Learn HMM parameters (P, E, π₀) from observation sequences with the Baum-Welch (EM) algorithm
(learning problem). Initializes with random row-stochastic matrices drawn from `rng`, then
alternates the E-step (γ and ξ posteriors from the scaled forward/backward variables) and M-step
(expected-count re-estimates) until |ℒₖ - ℒₖ₋₁| < ϵ or `maxiterations` is reached. Converges to a
local maximum of the likelihood; the result depends on the initialization.
"""
function solve(model::MyBaumWelchModel, sequences::Array{Array{Int64,1},1};
    number_of_hidden_states::Int, number_of_observable_states::Int,
    rng::AbstractRNG = Random.default_rng())::MyBaumWelchSolution

    # initialize -
    S = number_of_hidden_states; O = number_of_observable_states;
    N = length(sequences);
    Kmax = model.maxiterations; ϵ = model.ϵ;

    # random row-stochastic initial parameter guesses -
    P = rand(rng, S, S); P = P ./ sum(P, dims = 2);
    E = rand(rng, S, O); E = E ./ sum(E, dims = 2);
    π₀ = rand(rng, S); π₀ = π₀ ./ sum(π₀);

    loglikelihood_history = Array{Float64,1}();
    ℒ_prev = -Inf; k = 0; converged = false;
    ξₜ = zeros(Float64, S, S); # buffer for the transition posterior at one time step

    # main loop -
    while (converged == false)

        hmm_k = build(MyHiddenMarkovModel, (P = P, E = E, π₀ = π₀)); # current iterate

        # E-step accumulators -
        π₀_acc = zeros(Float64, S);        # Σₙ γ₁(i)
        ξ_acc = zeros(Float64, S, S);      # Σₙ Σ_{t<T} ξₜ(i,j)
        γ_trans_acc = zeros(Float64, S);   # Σₙ Σ_{t<T} γₜ(i)
        γ_emit_acc = zeros(Float64, S, O); # Σₙ Σₜ 1[oₜ = o]⋅γₜ(i)
        γ_total_acc = zeros(Float64, S);   # Σₙ Σₜ γₜ(i)
        ℒ = 0.0;

        # E-step: expected sufficient statistics for every sequence -
        for n ∈ 1:N
            o = sequences[n]; T = length(o);
            (α̂, c) = _forward(hmm_k, o);
            β̂ = _backward(hmm_k, o);
            ℒ += sum(log.(c)); # log-likelihood of sequence n under the current iterate

            # state posterior γₜ(i): scaling factors cancel in the row-normalized product -
            γ = α̂ .* β̂;
            γ = γ ./ sum(γ, dims = 2);

            π₀_acc .+= γ[1, :];
            γ_total_acc .+= vec(sum(γ, dims = 1));
            γ_trans_acc .+= vec(sum(γ[1:(T-1), :], dims = 1));
            for t ∈ 1:T
                γ_emit_acc[:, o[t]] .+= γ[t, :];
            end

            # transition posterior ξₜ(i,j), normalized over (i,j) at each t -
            for t ∈ 1:(T-1)
                for i ∈ 1:S, j ∈ 1:S
                    ξₜ[i, j] = α̂[t, i]*P[i, j]*E[j, o[t+1]]*β̂[t+1, j];
                end
                ξ_acc .+= ξₜ ./ sum(ξₜ);
            end
        end

        push!(loglikelihood_history, ℒ);

        # M-step: re-estimate parameters from expected counts (keep old row on zero γ-mass) -
        π₀ = π₀_acc ./ N;
        P_new = copy(P); E_new = copy(E);
        for i ∈ 1:S
            if (γ_trans_acc[i] > 0.0)
                P_new[i, :] = ξ_acc[i, :] ./ γ_trans_acc[i];
                P_new[i, :] = P_new[i, :] ./ sum(P_new[i, :]); # renormalize (numerical safety)
            end
            if (γ_total_acc[i] > 0.0)
                E_new[i, :] = γ_emit_acc[i, :] ./ γ_total_acc[i];
                E_new[i, :] = E_new[i, :] ./ sum(E_new[i, :]);
            end
        end
        P = P_new; E = E_new;

        # convergence check (k counts completed iterations, so increment before the K_max test) -
        k += 1;
        if (abs(ℒ - ℒ_prev) < ϵ || k ≥ Kmax)
            converged = true;
        end
        ℒ_prev = ℒ;
    end

    # package and return -
    solution = MyBaumWelchSolution();
    solution.P = P; solution.E = E; solution.π₀ = π₀;
    solution.loglikelihood_history = loglikelihood_history;
    solution.iterations = k;
    return solution;
end
