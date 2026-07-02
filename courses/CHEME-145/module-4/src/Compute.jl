using Random
using Statistics

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
    build_mdp(world::MyRectangularGridWorldModel, γ::Float64; step_reward, offgrid_penalty, absorbing, slip)
        -> MyMDPProblemModel

Construct an MDP from a grid world. With probability `1-slip` the chosen action's move happens; with
probability `slip` a uniformly random one of the four moves happens instead (`slip = 0` ⇒ deterministic).
A move to a valid non-absorbing cell earns that cell's reward (or `step_reward`); a move off the grid
earns `offgrid_penalty` and self-loops. Rewards are the expected reward over the (possibly slipped)
move. Absorbing cells self-loop with zero reward, so their value-to-go is `V=0`.
"""
function build_mdp(world::MyRectangularGridWorldModel, γ::Float64;
    step_reward::Float64 = -1.0, offgrid_penalty::Float64 = -1000.0,
    absorbing::Set{Tuple{Int,Int}} = Set{Tuple{Int,Int}}(),
    slip::Float64 = 0.0)::MyMDPProblemModel

    nstates = world.nrows*world.ncols;
    nactions = length(world.moves);
    𝒮 = collect(1:nstates);
    𝒜 = collect(1:nactions);
    rewards = world.rewards;

    R = zeros(Float64, nstates, nactions);
    T = zeros(Float64, nstates, nstates, nactions);

    # probability that direction a′ is actually attempted when action a is chosen -
    attempt_prob(a, a′) = (a′ == a ? (1.0 - slip) : 0.0) + slip/nactions;

    for s ∈ 𝒮
        current = world.coordinates[s];

        # absorbing cells: zero future reward, self-loop on every action (so V(absorbing) = 0) -
        if (in(current, absorbing) == true)
            for a ∈ 𝒜
                R[s, a] = 0.0;
                T[s, s, a] = 1.0;
            end
            continue;
        end

        for a ∈ 𝒜
            for a′ ∈ 𝒜
                w = attempt_prob(a, a′);
                w == 0.0 && continue;
                Δ = world.moves[a′];
                newpos = current .+ Δ;
                if (haskey(world.states, newpos) == true)
                    r = haskey(rewards, newpos) ? rewards[newpos] : step_reward;
                    s′ = world.states[newpos];
                    R[s, a] += w*r;
                    T[s, s′, a] += w;
                else
                    R[s, a] += w*offgrid_penalty;   # off-grid: stay and pay the penalty
                    T[s, s, a] += w;
                end
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

"""
    build_replacement_mdp(; max_age, income0, income_decline, maint0, maint_slope, replace_cost, γ)
        -> MyMDPProblemModel

Equipment replacement (optimal stopping). State `s = a+1` for machine age `a ∈ 0:max_age`; action
1 = keep, 2 = replace. Keep: reward `(income0 - income_decline·a) - (maint0 + maint_slope·a)`, age
advances to `min(a+1, max_age)`. Replace: reward `-replace_cost + (income0 - maint0)`, age resets to 1.
"""
function build_replacement_mdp(; max_age::Int, income0::Float64, income_decline::Float64,
    maint0::Float64, maint_slope::Float64, replace_cost::Float64, γ::Float64)::MyMDPProblemModel

    nstates = max_age + 1;
    𝒮 = collect(1:nstates);
    𝒜 = [1, 2];   # 1 = keep, 2 = replace
    R = zeros(Float64, nstates, 2);
    T = zeros(Float64, nstates, nstates, 2);

    for s ∈ 𝒮
        a_age = s - 1;

        # keep -
        R[s, 1] = (income0 - income_decline*a_age) - (maint0 + maint_slope*a_age);
        next_keep = min(a_age + 1, max_age) + 1;
        T[s, next_keep, 1] = 1.0;

        # replace -
        R[s, 2] = -replace_cost + (income0 - maint0);
        T[s, 2, 2] = 1.0;   # next age = 1 -> state index 2
    end

    return build(MyMDPProblemModel, (𝒮=𝒮, 𝒜=𝒜, T=T, R=R, γ=γ));
end

"""
    simulate_return(problem, s0, π_fn, H, rng) -> Float64

Simulate one trajectory of length `H` from state `s0` under policy `π_fn` (a function `s -> a`) and
return the discounted return `Σ_t γ^t R(s_t, a_t)`. Next states are sampled from `T`.
"""
function simulate_return(problem::MyMDPProblemModel, s0::Int, π_fn::Function, H::Int,
    rng::AbstractRNG)::Float64
    𝒮, T, R, γ = problem.𝒮, problem.T, problem.R, problem.γ;
    s = s0;
    G = 0.0;
    discount = 1.0;
    for _ ∈ 1:H
        a = π_fn(s);
        G += discount*R[s, a];
        # sample s′ ~ T[s,:,a] -
        u = rand(rng);
        cumulative = 0.0;
        s′ = s;
        for j ∈ 𝒮
            cumulative += T[s, j, a];
            if (u ≤ cumulative)
                s′ = j;
                break;
            end
        end
        s = s′;
        discount *= γ;
    end
    return G;
end

"""
    rollout_value(problem, s0; π_fn, H, N, rng) -> Float64

Monte Carlo rollout estimate of the value of state `s0` under base policy `π_fn`: the mean discounted
return over `N` simulated trajectories of horizon `H`.
"""
function rollout_value(problem::MyMDPProblemModel, s0::Int; π_fn::Function, H::Int = 100,
    N::Int = 1000, rng::AbstractRNG = Random.default_rng())::Float64
    return mean(simulate_return(problem, s0, π_fn, H, rng) for _ ∈ 1:N);
end

# UCT-based Monte Carlo tree search for a finite MDP. Uses a random base policy for the rollout
# (simulation) step and returns the most-visited / highest-value action at the root.
"""
    mcts(problem, model, s0; rng) -> Int

Run Monte Carlo tree search from state `s0` and return the estimated best action. Selection uses the
UCT rule with exploration constant `model.c`; leaves are evaluated by a random-policy rollout of
horizon `model.horizon`.
"""
function mcts(problem::MyMDPProblemModel, model::MyMCTSModel, s0::Int;
    rng::AbstractRNG = Random.default_rng())::Int

    𝒜, T, R, γ = problem.𝒜, problem.T, problem.R, problem.γ;
    N = Dict{Tuple{Int,Int},Int}();     # visit counts N[(s,a)]
    Qsa = Dict{Tuple{Int,Int},Float64}(); # value estimates Q[(s,a)]
    Ns = Dict{Int,Int}();               # state visit counts

    randpolicy = s -> rand(rng, 𝒜);

    _sample(s, a) = begin
        u = rand(rng); c = 0.0; s′ = s;
        for j ∈ problem.𝒮
            c += T[s, j, a];
            if (u ≤ c); s′ = j; break; end
        end
        s′
    end

    # one simulation from state s to the given depth -
    function simulate(s, d)
        if (d ≤ 0)
            return 0.0;
        end
        # if s unexpanded, initialize its actions and return a rollout estimate -
        if (haskey(Ns, s) == false)
            Ns[s] = 0;
            for a ∈ 𝒜
                N[(s,a)] = 0;
                Qsa[(s,a)] = 0.0;
            end
            return simulate_return(problem, s, randpolicy, model.horizon, rng);
        end

        # UCT action selection -
        Ns[s] += 1;
        logNs = log(Ns[s] + 1);
        best_a = 𝒜[1]; best_val = -Inf;
        for a ∈ 𝒜
            bonus = model.c*sqrt(logNs/(N[(s,a)] + 1));
            val = Qsa[(s,a)] + bonus;
            if (val > best_val); best_val = val; best_a = a; end
        end
        a = best_a;

        s′ = _sample(s, a);
        q = R[s, a] + γ*simulate(s′, d - 1);

        N[(s,a)] += 1;
        Qsa[(s,a)] += (q - Qsa[(s,a)])/N[(s,a)];
        return q;
    end

    for _ ∈ 1:model.iterations
        simulate(s0, model.depth);
    end

    # return the action with the highest mean value at the root -
    best_a = 𝒜[1]; best_q = -Inf;
    for a ∈ 𝒜
        qval = get(Qsa, (s0,a), -Inf);
        if (qval > best_q); best_q = qval; best_a = a; end
    end
    return best_a;
end
