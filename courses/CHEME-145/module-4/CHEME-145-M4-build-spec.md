# CHEME-145 Module 4 (Markov Decision Processes) — Build Spec

*Drafted 2026-07-01. Companion to `courses/CHEME-145/CHEME-145-refactor-plan.md`. Scope, structure, and example domains validated with the author in a brainstorming session.*

**Status: Built** on branch `m4-mdp-buildout` — self-contained `src/` MDP library + tests, value-iteration demo + ungraded, policy-iteration demo + ungraded, the equipment-replacement graded solution + student notebook, and the optional Advanced rollout/MCTS demos + ungraded activities.

## Goal

Build out Module 4 (Markov Decision Processes) of CHEME-145 to the same standard as Module 1: a self-contained local `src/` code base plus a set of escalating-practice notebooks. The theory lecture already exists (`CHEME-145-M4-Introduction-MarkovDecisionProcess-Read-Pages.ipynb`) and is not rewritten here.

## Decisions (from brainstorming)

1. **Topic scope:** core module keeps the four exact-dynamic-programming topics already in the lecture (MDP formulation, policies & value functions, value iteration, policy iteration) plus the discrete-choice tie-back. Random rollout and MCTS are delivered as **separate, optional "Advanced" notebooks** for learners who want to go further — not added to the core lecture.
2. **Graded activities:** a **single graded capstone** for the module (Module 1 precedent), not one per topic.
3. **Advanced depth:** each advanced topic (rollout, MCTS) gets a **Watch-Demo plus an ungraded Codio activity**.
4. **Capstone domain:** **equipment replacement** (keep vs. replace), which instantiates the Rust (1987) bus-engine tie-back the lecture already introduces.
5. **Core demo domains:** **gridworld** for value iteration, **inventory/maintenance** for policy iteration.
6. **Advanced environment:** rollout and MCTS **reuse the gridworld** students already know.
7. **Code approach:** self-contained local `src/` mirroring Module 1's `build(Type, data::NamedTuple)` factory style — **not** the external `VLDataScienceMachineLearningPackage.jl` the old course used.
8. **Notebooks are executed** in the local Julia environment with real outputs embedded (Julia 1.12.6 confirmed available).

## Notation (aligned to the lecture)

All notebooks and code use the lecture's notation: state-value $V^{\pi}(s)$, action-value $Q^{\pi}(s,a)$, policy $\pi$, discount $\gamma\in[0,1)$, MDP tuple $(\mathcal{S},\mathcal{A},P,R,\gamma)$, transition array $T$, sup-norm $\lVert\cdot\rVert_{\infty}$ for value-iteration convergence, tolerance $\epsilon>0$. The old gridworld code used $U$ for the value function; this build standardizes on $V$. The `MyValueFunctionPolicy` solution struct stores the value function in a field named `V`.

## Deliverables

Naming follows Module 1: `CHEME-145-M4-Example-<Topic>-<DeliveryType>.ipynb`. Optional notebooks use an `-Advanced-` marker so they are visually distinct in the directory.

### Core — Value Iteration (gridworld)
1. `CHEME-145-M4-Example-GridWorld-ValueIteration-Watch-Demo.ipynb` — build $(\mathcal{S},\mathcal{A},T,R,\gamma)$ for a gridworld, run value iteration, watch $V$ and $\pi$ converge, visualize the optimal path. Doubles as the MDP-formulation demo (the tuple is constructed explicitly).
2. `CHEME-145-M4-Example-GridWorld-ValueIteration-Ungraded-Codio-Activity.ipynb` — students run and modify value iteration (change rewards, $\gamma$, tolerance), observe how the policy changes.

### Core — Policy Iteration (inventory/maintenance)
3. `CHEME-145-M4-Example-Inventory-PolicyIteration-Watch-Demo.ipynb` — small stochastic inventory MDP; exact policy evaluation via $(\mathbf{I}-\gamma\mathbf{P}^{\pi})^{-1}\mathbf{R}^{\pi}$ then greedy improvement; watch the policy converge; compare iteration count to value iteration.
4. `CHEME-145-M4-Example-Inventory-PolicyIteration-Ungraded-Codio-Activity.ipynb` — students implement policy evaluation + policy iteration on a provided inventory MDP and compare convergence to value iteration.

### Core — Graded capstone (equipment replacement)
5. `CHEME-145-M4-Example-EquipmentReplacement-MDP-Graded-Codio-Activity.ipynb` **+ `-Solution.ipynb`** — formulate a keep/replace MDP (state = machine age/condition), solve via **both** value iteration and policy iteration, confirm the two policies agree, and write a recommendation (the replacement-age threshold). Closes the Rust (1987) loop opened in the lecture.

### Advanced — optional (gridworld)
6. `CHEME-145-M4-Advanced-RandomRollout-Watch-Demo.ipynb` — estimate a state's value by simulating trajectories under a base policy; compare the rollout estimate to the exact $V^{\star}$ from value iteration; demonstrate one-step rollout policy improvement.
7. `CHEME-145-M4-Advanced-RandomRollout-Ungraded-Codio-Activity.ipynb` — students run rollout, vary the number of trajectories and horizon, and quantify the estimate's accuracy against $V^{\star}$.
8. `CHEME-145-M4-Advanced-MCTS-Watch-Demo.ipynb` — Monte Carlo tree search (selection via UCT, expansion, rollout simulation, backpropagation) from a root state; compare the chosen action to the value-iteration optimum.
9. `CHEME-145-M4-Advanced-MCTS-Ungraded-Codio-Activity.ipynb` — students run MCTS, vary the iteration budget and exploration constant, and observe convergence to the optimal action.

Optional: a one-line pointer in the lecture's closing to the advanced track.

## `src/` scaffolding (mirrors Module 1: Types → Factory → Compute)

### `Types.jl`
- `abstract type AbstractMDPModel end`
- `abstract type AbstractWorldModel end`
- `abstract type AbstractSolutionModel end`
- `MyMDPProblemModel <: AbstractMDPModel` — fields `𝒮::Array{Int64,1}`, `𝒜::Array{Int64,1}`, `T::Array{Float64,3}` (`T[s,s′,a]`), `R::Array{Float64,2}` (`R[s,a]`), `γ::Float64`.
- `MyRectangularGridWorldModel <: AbstractWorldModel` — `nrows::Int`, `ncols::Int`, `coordinates::Dict{Int,Tuple{Int,Int}}`, `states::Dict{Tuple{Int,Int},Int}`, `moves::Dict{Int,Tuple{Int,Int}}`, `rewards::Dict{Tuple{Int,Int},Float64}`.
- `MyValueIterationModel <: AbstractSolutionModel` — `maxiterations::Int64`, `ϵ::Float64`.
- `MyPolicyIterationModel <: AbstractSolutionModel` — `maxiterations::Int64`.
- `MyValueFunctionPolicy` — `problem::MyMDPProblemModel`, `V::Array{Float64,1}` (solution container).
- `MyMonteCarloRolloutModel <: AbstractSolutionModel` — rollout parameters (trajectories, horizon, base policy). *(advanced)*
- `MyMCTSModel <: AbstractSolutionModel` — iteration budget, exploration constant `c`, horizon. *(advanced)*

### `Factory.jl`
- `build(::Type{MyMDPProblemModel}, data::NamedTuple)` — populate `𝒮,𝒜,T,R,γ`.
- `build(::Type{MyRectangularGridWorldModel}, data::NamedTuple)` — assemble `coordinates`, `states`, `moves` (LRUD), `rewards` from `nrows,ncols,rewards`.
- `build(...)` for `MyValueIterationModel`, `MyPolicyIterationModel`, and the advanced models.

### `Compute.jl`
- `solve(model::MyValueIterationModel, problem::MyMDPProblemModel)::MyValueFunctionPolicy` — Bellman optimality backup, sup-norm convergence to tolerance `ϵ`.
- `solve(model::MyPolicyIterationModel, problem::MyMDPProblemModel)::MyValueFunctionPolicy` — exact policy evaluation ($(\mathbf{I}-\gamma\mathbf{P}^{\pi})^{-1}\mathbf{R}^{\pi}$) + greedy improvement until the policy is stable.
- `iterative_policy_evaluation(problem, π; ...)` — iterative Bellman-expectation backup (for the ungraded contrast).
- `Q(problem::MyMDPProblemModel, V::Array{Float64,1})::Array{Float64,2}` — action-value from $V$.
- `policy(Q::Array{Float64,2})::Array{Int64,1}` — greedy policy.
- Trajectory/simulation helpers; `rollout(...)` and MCTS routines for the advanced track.

### `Include.jl` and `Project.toml`
- `Include.jl` activates the local environment, instantiates on first run, loads packages, includes `src/`. Same Paul Tol color palette as Module 1.
- `Project.toml` deps: `Plots, Colors, PrettyTables, LinearAlgebra, Random, Statistics, Distributions, DataFrames`. (No JuMP/Ipopt — no continuous optimization in this module.)

## Example-domain specifics (proposed; numbers finalized during build for clean answers)

- **Gridworld:** rectangular grid sized for a clear policy-arrow visualization (~10×10 for the demo, configurable). Actions LRUD; deterministic moves (a stochastic "slip" variant is an optional ungraded extension). Rewards: charging station $>0$, lava pits $\ll 0$, step cost $-1$, off-grid penalty $\ll 0$; charging station and lava are absorbing.
- **Inventory (policy iteration):** state = on-hand stock $0\ldots S_{\max}$ (~10–12 states), action = order quantity up to capacity, stochastic demand (small categorical or Poisson), reward = sales revenue − holding − ordering − lost-sales penalty. Small enough that the exact linear solve is instant and policy iteration converges in a few iterations, making the VI-vs-PI iteration-count contrast concrete.
- **Equipment replacement (capstone):** state = machine age/condition $0\ldots N$, actions = {keep, replace}. Keep → age advances (with an age-increasing failure probability); replace → age resets to new at a replacement cost. Reward: keep → revenue(age) − maintenance(age); replace → revenue(new) − replacement cost. The optimal policy is a replacement-age threshold; VI and PI must agree on it.

## Build order (vertical slice first, then fan out)

- **Phase 0 — scaffolding:** `src/` (Types, Factory, Compute), `Include.jl`, `Project.toml`; instantiate and smoke-test in Julia (build a tiny MDP, run VI and PI, check `Q`/`policy`).
- **Phase 1 — value iteration:** demo + ungraded (gridworld). Validates the full authoring + execution pattern before fan-out.
- **Phase 2 — policy iteration:** demo + ungraded (inventory).
- **Phase 3 — graded capstone:** equipment replacement + solution.
- **Phase 4 — advanced rollout:** demo + ungraded (gridworld).
- **Phase 5 — advanced MCTS:** demo + ungraded (gridworld).

## Authoring standards (from CLAUDE.md)

Each notebook follows the Module 1 template: title + intro, three Learning Objectives (`* __[title]:__ …`), Theory, Setup (`include("Include.jl")`), worked sections in `let` blocks, Summary + exactly three Key Takeaways, Additional Resources. Direct concise language, no unnecessary adjectives. Mathematical notation: explicit norms and tolerances, all variables defined before first use, function domains/codomains specified. Content must be supported by the material.

## Verification

- Every notebook executes top-to-bottom in the local Julia environment with outputs embedded.
- VI and PI produce the same optimal policy on shared problems (regression check for the capstone).
- Advanced rollout/MCTS estimates are compared against the exact $V^{\star}$/optimal action from value iteration.

## Non-goals

- No rewrite of the existing theory lecture (only an optional one-line advanced-track pointer).
- No external `VLDataScienceMachineLearningPackage.jl` dependency.
- No continuous-state or function-approximation methods (deep RL, linear value-function approximation) — out of scope for this module.
