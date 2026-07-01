# PRIVATE ------------------------------------------------------------------------------------- #
# one-step Bellman lookahead: R(s,a) + γ Σ_s′ T(s′|s,a) V(s′)
function _lookahead(problem::MyMDPProblemModel, V::Array{Float64,1}, s::Int, a::Int)::Float64
    𝒮, T, R, γ = problem.𝒮, problem.T, problem.R, problem.γ;
    return R[s,a] + γ*sum(T[s, s′, a]*V[s′] for s′ ∈ 𝒮);
end

function _policy_reward_vector(problem::MyMDPProblemModel, π::Array{Int64,1})::Array{Float64,1}
    return [problem.R[s, π[s]] for s ∈ problem.𝒮];
end

function _policy_transition_matrix(problem::MyMDPProblemModel, π::Array{Int64,1})::Array{Float64,2}
    𝒮, T = problem.𝒮, problem.T;
    n = length(𝒮);
    P = zeros(Float64, n, n);
    for s ∈ 𝒮
        for s′ ∈ 𝒮
            P[s, s′] = T[s, s′, π[s]];
        end
    end
    return P;
end
# ---------------------------------------------------------------------------------------------- #

# PUBLIC -------------------------------------------------------------------------------------- #
"""
    Q(problem::MyMDPProblemModel, V::Array{Float64,1}) -> Array{Float64,2}

Action-value function `Q[s,a] = R(s,a) + γ Σ_s′ T(s′|s,a) V(s′)`.
"""
function Q(problem::MyMDPProblemModel, V::Array{Float64,1})::Array{Float64,2}
    𝒮, 𝒜 = problem.𝒮, problem.𝒜;
    Qa = zeros(Float64, length(𝒮), length(𝒜));
    for s ∈ 𝒮
        for a ∈ 𝒜
            Qa[s,a] = _lookahead(problem, V, s, a);
        end
    end
    return Qa;
end

"""
    policy(Q_array::Array{Float64,2}) -> Array{Int64,1}

Greedy policy: `π(s) = argmax_a Q[s,a]`.
"""
function policy(Q_array::Array{Float64,2})::Array{Int64,1}
    return [argmax(Q_array[s, :]) for s ∈ 1:size(Q_array, 1)];
end

"""
    solve(model::MyValueIterationModel, problem::MyMDPProblemModel) -> MyValueFunctionPolicy

Value iteration: apply the Bellman optimality backup until `‖V′-V‖∞ ≤ ϵ` or the iteration cap.
"""
function solve(model::MyValueIterationModel, problem::MyMDPProblemModel)::MyValueFunctionPolicy
    𝒮, 𝒜 = problem.𝒮, problem.𝒜;
    ϵ, maxiterations = model.ϵ, model.maxiterations;

    V  = zeros(Float64, length(𝒮));
    V′ = similar(V);
    k = 1;
    converged = false;
    while (converged == false)
        Δ = 0.0;
        for s ∈ 𝒮
            V′[s] = maximum(_lookahead(problem, V, s, a) for a ∈ 𝒜);
            Δ = max(Δ, abs(V′[s] - V[s]));
        end
        copyto!(V, V′);
        k += 1;
        if (Δ ≤ ϵ || k > maxiterations)
            converged = true;
        end
    end

    solution = MyValueFunctionPolicy();
    solution.problem = problem;
    solution.V = V;
    return solution;
end

"""
    policy_evaluation(problem::MyMDPProblemModel, π::Array{Int64,1}) -> Array{Float64,1}

Exact policy evaluation: solve `(I - γ Pπ) V = Rπ`.
"""
function policy_evaluation(problem::MyMDPProblemModel, π::Array{Int64,1})::Array{Float64,1}
    γ = problem.γ;
    R_π = _policy_reward_vector(problem, π);
    P_π = _policy_transition_matrix(problem, π);
    return (I - γ*P_π) \ R_π;
end

"""
    iterative_policy_evaluation(problem, π; ϵ, maxiterations) -> Array{Float64,1}

Iterative policy evaluation: repeat the Bellman expectation backup until `‖V′-V‖∞ ≤ ϵ`.
"""
function iterative_policy_evaluation(problem::MyMDPProblemModel, π::Array{Int64,1};
    ϵ::Float64 = 1e-8, maxiterations::Int64 = 10_000)::Array{Float64,1}

    𝒮 = problem.𝒮;
    V  = zeros(Float64, length(𝒮));
    V′ = similar(V);
    k = 1;
    converged = false;
    while (converged == false)
        Δ = 0.0;
        for s ∈ 𝒮
            V′[s] = _lookahead(problem, V, s, π[s]);
            Δ = max(Δ, abs(V′[s] - V[s]));
        end
        copyto!(V, V′);
        k += 1;
        (Δ ≤ ϵ || k > maxiterations) && (converged = true);
    end
    return V;
end

"""
    solve(model::MyPolicyIterationModel, problem::MyMDPProblemModel) -> MyValueFunctionPolicy

Policy iteration: alternate exact policy evaluation and greedy policy improvement until the policy
is stable.
"""
function solve(model::MyPolicyIterationModel, problem::MyMDPProblemModel)::MyValueFunctionPolicy
    𝒮 = problem.𝒮;
    π = ones(Int64, length(𝒮));   # arbitrary initial policy: action 1 everywhere
    V = zeros(Float64, length(𝒮));
    k = 1;
    converged = false;
    while (converged == false)
        V = policy_evaluation(problem, π);       # evaluate
        π′ = policy(Q(problem, V));               # improve
        if (π′ == π || k > model.maxiterations)
            converged = true;
        end
        π = π′;
        k += 1;
    end

    solution = MyValueFunctionPolicy();
    solution.problem = problem;
    solution.V = V;
    return solution;
end
# ---------------------------------------------------------------------------------------------- #
