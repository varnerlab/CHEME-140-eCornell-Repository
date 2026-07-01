# ---------------------------------------------------------------------------------------------- #
# Multinomial logit discrete choice. A feature matrix X holds one row per alternative and one
# column per feature. The deterministic utility of each alternative is V = X β, and the logit
# choice probabilities are the scaled softmax of V.
# ---------------------------------------------------------------------------------------------- #
"""
    deterministic_utility(model::MyLinearRandomUtilityModel, X::Array{Float64,2}) -> Array{Float64,1}

Computes the deterministic utility `Vⱼ = βᵀ xⱼ` of each alternative, where `X` is an
(alternatives × features) feature matrix.

### Returns
- `Array{Float64,1}`: the deterministic utility of each alternative.
"""
function deterministic_utility(model::MyLinearRandomUtilityModel, X::Array{Float64,2})::Array{Float64,1}
    return X*model.β;
end

"""
    logit_choice_probabilities(model::MyLinearRandomUtilityModel, V::Array{Float64,1}) -> Array{Float64,1}

Computes the multinomial logit choice probabilities `Pⱼ = exp(μ Vⱼ) / ∑ₖ exp(μ Vₖ)` from the
deterministic-utility vector `V`. The maximum of `μ V` is subtracted before exponentiating for
numerical stability; this leaves the probabilities unchanged.

### Returns
- `Array{Float64,1}`: the choice probability of each alternative (sums to 1).
"""
function logit_choice_probabilities(model::MyLinearRandomUtilityModel, V::Array{Float64,1})::Array{Float64,1}
    z = model.μ .* V;
    z = z .- maximum(z); # numerical stability (does not change the probabilities)
    e = exp.(z);
    return e ./ sum(e);
end

"""
    simulate_choices(P::Array{Float64,1}, N::Int) -> Array{Float64,1}

Draws `N` choices from the categorical distribution with probabilities `P` and returns the
empirical share of each alternative. Uses inverse-CDF sampling.

### Returns
- `Array{Float64,1}`: the simulated share of each alternative (sums to 1).
"""
function simulate_choices(P::Array{Float64,1}, N::Int)::Array{Float64,1}
    cP = cumsum(P);
    counts = zeros(Int, length(P));
    for _ ∈ 1:N
        u = rand();
        k = findfirst(c -> c ≥ u, cP); # first alternative whose cumulative probability ≥ u
        counts[k] += 1;
    end
    return counts ./ N;
end
