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

"""
    build_mdp(world::MyRectangularGridWorldModel, γ::Float64; step_reward, offgrid_penalty, absorbing)
        -> MyMDPProblemModel

Construct a deterministic-move MDP from a grid world. A move to a valid non-absorbing cell earns that
cell's reward (or `step_reward`); a move off the grid earns `offgrid_penalty` and self-loops.
Absorbing cells self-loop with zero reward, so their value-to-go is `V=0` (the terminal reward is
earned on the step INTO the cell, not while sitting in it).
"""
function build_mdp(world::MyRectangularGridWorldModel, γ::Float64;
    step_reward::Float64 = -1.0, offgrid_penalty::Float64 = -1000.0,
    absorbing::Set{Tuple{Int,Int}} = Set{Tuple{Int,Int}}())::MyMDPProblemModel

    nstates = world.nrows*world.ncols;
    nactions = length(world.moves);
    𝒮 = collect(1:nstates);
    𝒜 = collect(1:nactions);
    rewards = world.rewards;

    R = zeros(Float64, nstates, nactions);
    T = zeros(Float64, nstates, nstates, nactions);

    for a ∈ 𝒜
        Δ = world.moves[a];
        for s ∈ 𝒮
            current = world.coordinates[s];

            # absorbing cells: zero future reward, self-loop (so V(absorbing) = 0) -
            if (in(current, absorbing) == true)
                R[s, a] = 0.0;
                T[s, s, a] = 1.0;
                continue;
            end

            newpos = current .+ Δ;

            # reward (current is non-absorbing) -
            if (haskey(world.states, newpos) == true)
                R[s,a] = haskey(rewards, newpos) ? rewards[newpos] : step_reward;
            else
                R[s,a] = offgrid_penalty;
            end

            # transition (current is non-absorbing) -
            if (haskey(world.states, newpos) == true)
                s′ = world.states[newpos];
                T[s, s′, a] = 1.0;
            else
                T[s, s, a] = 1.0;   # off-grid -> self-loop
            end
        end
    end

    return build(MyMDPProblemModel, (𝒮=𝒮, 𝒜=𝒜, T=T, R=R, γ=γ));
end

"""
    build_inventory_mdp(; capacity, demand_pmf, price, order_cost, fixed_cost, holding_cost,
        stockout_penalty, γ) -> MyMDPProblemModel

Single-item stochastic inventory control. State `s = i+1` for on-hand `i ∈ 0:capacity`; action
`a = o+1` for order-up quantity `o ∈ 0:capacity`. Post-order stock `q = i+o` (feasible when
`q ≤ capacity`); random demand `d` has pmf `demand_pmf` over `0:(length-1)`; next on-hand is
`max(0, q-d)`. Reward = price·min(q,d) − order_cost·o − fixed_cost·1(o>0) − holding_cost·q −
stockout_penalty·max(0,d-q). Infeasible actions self-loop with reward −1e6.
"""
function build_inventory_mdp(; capacity::Int, demand_pmf::Vector{Float64}, price::Float64,
    order_cost::Float64, fixed_cost::Float64, holding_cost::Float64, stockout_penalty::Float64,
    γ::Float64)::MyMDPProblemModel

    nstates = capacity + 1;
    nactions = capacity + 1;
    𝒮 = collect(1:nstates);
    𝒜 = collect(1:nactions);
    Dmax = length(demand_pmf) - 1;

    R = fill(-1.0e6, nstates, nactions);   # infeasible default
    T = zeros(Float64, nstates, nstates, nactions);

    for s ∈ 𝒮
        i = s - 1;                          # on-hand
        for a ∈ 𝒜
            o = a - 1;                      # order quantity
            q = i + o;                      # post-order stock
            if (q > capacity)               # infeasible -> self-loop, keep penalty
                T[s, s, a] = 1.0;
                continue;
            end

            expected_reward = 0.0;
            for d ∈ 0:Dmax
                pd = demand_pmf[d + 1];
                sales = min(q, d);
                unmet = max(0, d - q);
                reward = price*sales - order_cost*o - fixed_cost*(o > 0 ? 1.0 : 0.0) -
                         holding_cost*q - stockout_penalty*unmet;
                expected_reward += pd*reward;

                s′ = max(0, q - d) + 1;     # next state index
                T[s, s′, a] += pd;
            end
            R[s, a] = expected_reward;
        end
    end

    return build(MyMDPProblemModel, (𝒮=𝒮, 𝒜=𝒜, T=T, R=R, γ=γ));
end
