# CHEME-145 Module 4 (Markov Decision Processes) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the self-contained `src/` MDP library and the nine escalating-practice notebooks (plus one solution) for Module 4, all executed with real outputs.

**Architecture:** A local Julia project (`module-4/`) mirroring Module 1: `Include.jl` activates a local environment and loads `src/` (Types → Factory → Compute). The MDP algorithms (value iteration, policy iteration, `Q`, `policy`, rollout, MCTS) and three problem builders (gridworld, inventory, equipment replacement) live in `src/` and are validated by `test/runtests.jl`. Notebooks are thin presentation layers over the tested library, authored to the Module 1 template and executed via `jupyter nbconvert`.

**Tech Stack:** Julia 1.12.6; packages `Plots, Colors, PrettyTables, LinearAlgebra, Random, Statistics, Distributions, DataFrames`; Jupyter `nbconvert` 7.16.4 with the `julia-1.12` kernel; `Test` stdlib.

## Global Constraints

- **Working directory:** `courses/CHEME-145/module-4/` (all paths below are relative to this unless noted).
- **Julia version floor:** 1.12 (kernel `julia-1.12`).
- **No external VL package:** do NOT depend on `VLDataScienceMachineLearningPackage.jl` or `VLDecisionsPackage.jl`. All code is local `src/`.
- **Notation (verbatim, matches the lecture):** value function `V` (never `U`), action-value `Q`, policy `π`, discount `γ ∈ [0,1)`, states `𝒮`, actions `𝒜`, transition array `T[s,s′,a]`, reward `R[s,a]`, value-iteration convergence in sup-norm `‖·‖∞` with tolerance `ϵ > 0`.
- **Factory pattern:** every model type has an empty inner constructor `T() = new()` and is populated by a `build(::Type{T}, data::NamedTuple)::T` method (Module 1 style).
- **Naming:** core notebooks `CHEME-145-M4-Example-<Topic>-<DeliveryType>.ipynb`; optional notebooks `CHEME-145-M4-Advanced-<Topic>-<DeliveryType>.ipynb`. DeliveryType ∈ {`Watch-Demo`, `Ungraded-Codio-Activity`, `Graded-Codio-Activity`, `Graded-Codio-Activity-Solution`}.
- **Notebook template (every notebook):** title + intro paragraph; blockquoted **Learning Objectives** with exactly three `* __[title]:__ …` bullets; **Theory**; **Setup** (`include("Include.jl")`); worked sections wrapping mutating code in `let … end` blocks; **Summary** with exactly three **Key Takeaways** (`* **[title]:** …`); **Additional Resources**. Direct concise language, no unnecessary adjectives, all math variables defined before first use, all norms/tolerances explicit.
- **Color palette:** reuse Module 1's Paul Tol muted `colors` dict (defined in `Include.jl`).

---

## Notebook authoring & execution procedure (shared by all notebook tasks)

Every notebook task follows this procedure; it is not repeated per task.

1. **Create the `.ipynb`** at the exact path. Use nbformat 4 with this metadata:
   ```json
   {
     "cells": [ ... ],
     "metadata": {
       "kernelspec": {"display_name": "Julia 1.12", "language": "julia", "name": "julia-1.12"},
       "language_info": {"name": "julia", "version": "1.12.6"}
     },
     "nbformat": 4,
     "nbformat_minor": 5
   }
   ```
   Markdown cells: `{"cell_type":"markdown","metadata":{},"source":[ ... ]}`. Code cells: `{"cell_type":"code","metadata":{},"execution_count":null,"outputs":[],"source":[ ... ]}`. Author cells with empty `outputs` — execution fills them. (Escaping-safe alternative: create an empty notebook with `"cells": []`, then insert each cell with the NotebookEdit tool, which handles JSON encoding.)
2. **Execute in place:**
   ```bash
   jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.kernel_name=julia-1.12 \
     --ExecutePreprocessor.timeout=600 <path>.ipynb
   ```
   Expected: exit code 0, no `CellExecutionError`. (First run of the first notebook triggers `Pkg.instantiate()` inside `Include.jl` and may take several minutes.)
3. **Verify** the embedded outputs match the library's tested values (each task states the exact check).
4. **Commit** the executed notebook.

Markdown prose is written to the Module 1 template above; each task gives the ordered cell manifest (section headings + content bullets) and the complete Julia code for every code cell.

---

## File structure

**Create:**
- `Project.toml` — local environment deps.
- `Include.jl` — path setup, env activation/instantiation, package loads, color palette, `include` of `src/`.
- `src/Types.jl` — abstract types + concrete model structs.
- `src/Factory.jl` — `build(...)` methods + gridworld assembly.
- `src/Compute.jl` — VI/PI/`Q`/`policy`/`policy_evaluation`/`iterative_policy_evaluation`, `build_mdp` (gridworld→MDP), `build_inventory_mdp`, `build_replacement_mdp`, `rollout_value`, `mcts` and helpers.
- `test/runtests.jl` — hand-verified assertions for the library.
- Nine notebooks + one solution (paths in their tasks).

**Reference (do not modify):** `../module-1/Include.jl`, `../module-1/src/*.jl` (style), `CHEME-145-M4-Introduction-MarkovDecisionProcess-Read-Pages.ipynb` (lecture/notation), `CHEME-145-M4-build-spec.md`.

---

## Phase 0 — Scaffolding & library

### Task 0.1: Local environment (`Project.toml`, `Include.jl`)

**Files:**
- Create: `Project.toml`, `Include.jl`

**Interfaces:**
- Produces: a working `include("Include.jl")` that loads packages and (after Task 0.2–0.5) the `src/` files, plus a `colors::Dict{Int,RGB}` palette.

- [ ] **Step 1: Write `Project.toml`**

```toml
[deps]
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
```

- [ ] **Step 2: Write `Include.jl`** (the `include(src)` lines are added as those files are created in later tasks)

```julia
# setup paths -
const _ROOT = @__DIR__;
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# activate the local environment; instantiate it the first time (no Manifest.toml yet) -
using Pkg
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.add(["Plots", "Colors", "PrettyTables", "LinearAlgebra",
        "Random", "Statistics", "Distributions", "DataFrames"]);
    Pkg.instantiate();
end

# load external packages -
using Plots
using Colors
using PrettyTables
using LinearAlgebra
using Random
using Statistics
using Distributions
using DataFrames

# color palette (Paul Tol muted) -
colors = Dict{Int,RGB}();
colors[1] = colorant"#0077BB"; # blue
colors[2] = colorant"#33BBEE"; # cyan
colors[3] = colorant"#EE7733"; # orange
colors[4] = colorant"#CC3311"; # red
colors[5] = colorant"#009988"; # teal
colors[6] = colorant"#EE3377"; # magenta

# load my codes -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
```

- [ ] **Step 3: Create empty `src/` files so `Include.jl` loads** (they are filled in Tasks 0.2–0.4)

Run: `mkdir -p src test && touch src/Types.jl src/Factory.jl src/Compute.jl`

- [ ] **Step 4: Instantiate and smoke-test the environment**

Run: `julia --project=. -e 'include("Include.jl"); println("env OK: ", @isdefined(colors))'`
Expected: prints `env OK: true` (first run installs/precompiles packages — may take several minutes).

- [ ] **Step 5: Commit**

```bash
git add courses/CHEME-145/module-4/Project.toml courses/CHEME-145/module-4/Include.jl courses/CHEME-145/module-4/src
git commit -m "M4: local Julia environment scaffolding"
```

### Task 0.2: Model types (`src/Types.jl`)

**Files:**
- Modify: `src/Types.jl`

**Interfaces:**
- Produces: `AbstractMDPModel`, `AbstractWorldModel`, `AbstractSolutionModel`; `MyMDPProblemModel` (`𝒮::Array{Int64,1}`, `𝒜::Array{Int64,1}`, `T::Array{Float64,3}`, `R::Array{Float64,2}`, `γ::Float64`); `MyRectangularGridWorldModel` (`nrows,ncols::Int`, `coordinates::Dict{Int,Tuple{Int,Int}}`, `states::Dict{Tuple{Int,Int},Int}`, `moves::Dict{Int,Tuple{Int,Int}}`, `rewards::Dict{Tuple{Int,Int},Float64}`); `MyValueIterationModel` (`maxiterations::Int64`, `ϵ::Float64`); `MyPolicyIterationModel` (`maxiterations::Int64`); `MyValueFunctionPolicy` (`problem::MyMDPProblemModel`, `V::Array{Float64,1}`).

- [ ] **Step 1: Write `src/Types.jl`**

```julia
# ---------------------------------------------------------------------------------------------- #
# Abstract types. MDP problems, world models, and solution (algorithm) models.
# ---------------------------------------------------------------------------------------------- #
abstract type AbstractMDPModel end
abstract type AbstractWorldModel end
abstract type AbstractSolutionModel end

"""
    mutable struct MyMDPProblemModel <: AbstractMDPModel

A finite Markov decision process (𝒮, 𝒜, T, R, γ).

### Fields
- `𝒮::Array{Int64,1}`: state index set.
- `𝒜::Array{Int64,1}`: action index set.
- `T::Array{Float64,3}`: transition array, `T[s,s′,a] = P(s′ | s,a)`.
- `R::Array{Float64,2}`: reward array, `R[s,a]` = expected immediate reward.
- `γ::Float64`: discount factor, `0 ≤ γ < 1`.
"""
mutable struct MyMDPProblemModel <: AbstractMDPModel
    𝒮::Array{Int64,1}
    𝒜::Array{Int64,1}
    T::Array{Float64,3}
    R::Array{Float64,2}
    γ::Float64
    MyMDPProblemModel() = new();
end

"""
    mutable struct MyRectangularGridWorldModel <: AbstractWorldModel

A rectangular grid world environment. Maps between `(x,y)` coordinates and state indices, holds the
action move vectors, and the sparse reward dictionary.

### Fields
- `nrows::Int`, `ncols::Int`: grid dimensions.
- `coordinates::Dict{Int,Tuple{Int,Int}}`: state index → `(x,y)`.
- `states::Dict{Tuple{Int,Int},Int}`: `(x,y)` → state index.
- `moves::Dict{Int,Tuple{Int,Int}}`: action → `(Δx,Δy)`.
- `rewards::Dict{Tuple{Int,Int},Float64}`: `(x,y)` → reward (non-default cells only).
"""
mutable struct MyRectangularGridWorldModel <: AbstractWorldModel
    nrows::Int
    ncols::Int
    coordinates::Dict{Int,Tuple{Int,Int}}
    states::Dict{Tuple{Int,Int},Int}
    moves::Dict{Int,Tuple{Int,Int}}
    rewards::Dict{Tuple{Int,Int},Float64}
    MyRectangularGridWorldModel() = new();
end

"""
    mutable struct MyValueIterationModel <: AbstractSolutionModel

Value-iteration solver settings.

### Fields
- `maxiterations::Int64`: iteration cap.
- `ϵ::Float64`: sup-norm convergence tolerance, `ϵ > 0`.
"""
mutable struct MyValueIterationModel <: AbstractSolutionModel
    maxiterations::Int64
    ϵ::Float64
    MyValueIterationModel() = new();
end

"""
    mutable struct MyPolicyIterationModel <: AbstractSolutionModel

Policy-iteration solver settings.

### Fields
- `maxiterations::Int64`: iteration cap on policy-improvement sweeps.
"""
mutable struct MyPolicyIterationModel <: AbstractSolutionModel
    maxiterations::Int64
    MyPolicyIterationModel() = new();
end

"""
    mutable struct MyValueFunctionPolicy

Container for a solved MDP: the problem and its optimal value function `V`. Recover the policy with
`policy(Q(problem, V))`.

### Fields
- `problem::MyMDPProblemModel`: the solved problem.
- `V::Array{Float64,1}`: value function, one entry per state.
"""
mutable struct MyValueFunctionPolicy
    problem::MyMDPProblemModel
    V::Array{Float64,1}
    MyValueFunctionPolicy() = new();
end
```

- [ ] **Step 2: Add the `include` for Types in `Include.jl`** — already present from Task 0.1 Step 2; confirm `include(joinpath(_PATH_TO_SRC, "Types.jl"))` is there.

- [ ] **Step 3: Load-test**

Run: `julia --project=. -e 'include("src/Types.jl"); m = MyMDPProblemModel(); println(isa(m, AbstractMDPModel))'`
Expected: prints `true`.

- [ ] **Step 4: Commit**

```bash
git add courses/CHEME-145/module-4/src/Types.jl
git commit -m "M4: MDP, gridworld, and solver model types"
```

### Task 0.3: Factory (`src/Factory.jl`)

**Files:**
- Modify: `src/Factory.jl`

**Interfaces:**
- Consumes: all types from Task 0.2.
- Produces: `build(::Type{MyMDPProblemModel}, ::NamedTuple)`, `build(::Type{MyValueIterationModel}, ::NamedTuple)` (keys `maxiterations`, `ϵ`), `build(::Type{MyPolicyIterationModel}, ::NamedTuple)` (key `maxiterations`), `build(::Type{MyRectangularGridWorldModel}, ::NamedTuple)` (keys `nrows`, `ncols`, `rewards`) with action order 1=left,2=right,3=down,4=up.

- [ ] **Step 1: Write `src/Factory.jl`**

```julia
"""
    build(::Type{MyMDPProblemModel}, data::NamedTuple) -> MyMDPProblemModel

Build a finite MDP from keys `𝒮, 𝒜, T, R, γ`.
"""
function build(::Type{MyMDPProblemModel}, data::NamedTuple)::MyMDPProblemModel
    model = MyMDPProblemModel();
    model.𝒮 = data.𝒮;
    model.𝒜 = data.𝒜;
    model.T = data.T;
    model.R = data.R;
    model.γ = data.γ;
    return model;
end

"""
    build(::Type{MyValueIterationModel}, data::NamedTuple) -> MyValueIterationModel

Build value-iteration settings from keys `maxiterations, ϵ`.
"""
function build(::Type{MyValueIterationModel}, data::NamedTuple)::MyValueIterationModel
    model = MyValueIterationModel();
    model.maxiterations = data.maxiterations;
    model.ϵ = data.ϵ;
    return model;
end

"""
    build(::Type{MyPolicyIterationModel}, data::NamedTuple) -> MyPolicyIterationModel

Build policy-iteration settings from key `maxiterations`.
"""
function build(::Type{MyPolicyIterationModel}, data::NamedTuple)::MyPolicyIterationModel
    model = MyPolicyIterationModel();
    model.maxiterations = data.maxiterations;
    return model;
end

"""
    build(::Type{MyRectangularGridWorldModel}, data::NamedTuple) -> MyRectangularGridWorldModel

Build a grid world from keys `nrows, ncols, rewards`. Actions are 1=left, 2=right, 3=down, 4=up.
"""
function build(::Type{MyRectangularGridWorldModel}, data::NamedTuple)::MyRectangularGridWorldModel

    # unpack -
    nrows = data.nrows;
    ncols = data.ncols;
    rewards = data.rewards;

    # coordinate <-> state maps -
    coordinates = Dict{Int,Tuple{Int,Int}}();
    states = Dict{Tuple{Int,Int},Int}();
    position_index = 1;
    for x ∈ 1:ncols
        for y ∈ 1:nrows
            coordinate = (x,y);
            coordinates[position_index] = coordinate;
            states[coordinate] = position_index;
            position_index += 1;
        end
    end

    # action move vectors (Δx,Δy) -
    moves = Dict{Int,Tuple{Int,Int}}();
    moves[1] = (-1, 0);  # left
    moves[2] = ( 1, 0);  # right
    moves[3] = ( 0,-1);  # down
    moves[4] = ( 0, 1);  # up

    # build -
    model = MyRectangularGridWorldModel();
    model.nrows = nrows;
    model.ncols = ncols;
    model.coordinates = coordinates;
    model.states = states;
    model.moves = moves;
    model.rewards = rewards;
    return model;
end
```

- [ ] **Step 2: Load-test the gridworld builder**

Run:
```bash
julia --project=. -e 'include("src/Types.jl"); include("src/Factory.jl");
w = build(MyRectangularGridWorldModel, (nrows=3, ncols=3, rewards=Dict{Tuple{Int,Int},Float64}()));
println(length(w.coordinates), " ", w.states[(1,1)], " ", w.moves[2])'
```
Expected: prints `9 1 (1, 0)`.

- [ ] **Step 3: Commit**

```bash
git add courses/CHEME-145/module-4/src/Factory.jl
git commit -m "M4: build() factory for MDP, solvers, and grid world"
```

### Task 0.4: Core algorithms (`src/Compute.jl`) + tests

**Files:**
- Modify: `src/Compute.jl`
- Create: `test/runtests.jl`

**Interfaces:**
- Consumes: types (0.2), `build` (0.3).
- Produces:
  - `solve(::MyValueIterationModel, ::MyMDPProblemModel)::MyValueFunctionPolicy`
  - `solve(::MyPolicyIterationModel, ::MyMDPProblemModel)::MyValueFunctionPolicy`
  - `Q(problem::MyMDPProblemModel, V::Array{Float64,1})::Array{Float64,2}`
  - `policy(Q_array::Array{Float64,2})::Array{Int64,1}`
  - `policy_evaluation(problem::MyMDPProblemModel, π::Array{Int64,1})::Array{Float64,1}` (exact linear solve)
  - `iterative_policy_evaluation(problem::MyMDPProblemModel, π::Array{Int64,1}; ϵ, maxiterations)::Array{Float64,1}`

- [ ] **Step 1: Write the failing test `test/runtests.jl`**

```julia
using Test
include(joinpath(@__DIR__, "..", "src", "Types.jl"));
include(joinpath(@__DIR__, "..", "src", "Factory.jl"));
include(joinpath(@__DIR__, "..", "src", "Compute.jl"));

@testset "core MDP algorithms" begin
    # tiny 2-state MDP. state 2 is absorbing with per-step reward 1.
    # action 1 = stay, action 2 = move. From state 1, "move" -> state 2.
    𝒮 = [1, 2];
    𝒜 = [1, 2];
    T = zeros(Float64, 2, 2, 2);
    T[1,1,1] = 1.0;   # stay in 1
    T[2,2,1] = 1.0;   # 2 absorbing
    T[1,2,2] = 1.0;   # move 1 -> 2
    T[2,2,2] = 1.0;   # 2 absorbing
    R = zeros(Float64, 2, 2);
    R[2,1] = 1.0; R[2,2] = 1.0;   # reward only in state 2
    γ = 0.9;
    problem = build(MyMDPProblemModel, (𝒮=𝒮, 𝒜=𝒜, T=T, R=R, γ=γ));

    # hand solution: V*(2)=1/(1-γ)=10, V*(1)=γ*V*(2)=9, π*(1)=2 (move).
    vi = build(MyValueIterationModel, (maxiterations=10_000, ϵ=1e-10));
    sol_vi = solve(vi, problem);
    @test isapprox(sol_vi.V[2], 10.0; atol=1e-6);
    @test isapprox(sol_vi.V[1],  9.0; atol=1e-6);
    π_vi = policy(Q(problem, sol_vi.V));
    @test π_vi[1] == 2;

    # policy iteration must agree with value iteration.
    pit = build(MyPolicyIterationModel, (maxiterations=100,));
    sol_pi = solve(pit, problem);
    @test isapprox(sol_pi.V[1], 9.0; atol=1e-6);
    @test isapprox(sol_pi.V[2], 10.0; atol=1e-6);
    @test policy(Q(problem, sol_pi.V)) == π_vi;

    # exact vs iterative policy evaluation agree for the greedy policy.
    Vexact = policy_evaluation(problem, π_vi);
    Viter  = iterative_policy_evaluation(problem, π_vi; ϵ=1e-12, maxiterations=100_000);
    @test isapprox(Vexact, Viter; atol=1e-4);
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. test/runtests.jl`
Expected: FAIL / error — `solve` (and friends) not defined.

- [ ] **Step 3: Write `src/Compute.jl`**

```julia
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `julia --project=. test/runtests.jl`
Expected: PASS — `Test Summary: | Pass  Total` with all passing.

- [ ] **Step 5: Commit**

```bash
git add courses/CHEME-145/module-4/src/Compute.jl courses/CHEME-145/module-4/test/runtests.jl
git commit -m "M4: value iteration, policy iteration, Q/policy + tests"
```

### Task 0.5: Grid-world MDP builder (`build_mdp`)

**Files:**
- Modify: `src/Compute.jl` (append), `test/runtests.jl` (append a testset)

**Interfaces:**
- Consumes: `MyRectangularGridWorldModel`, `build(::Type{MyMDPProblemModel}, …)`.
- Produces: `build_mdp(world::MyRectangularGridWorldModel, γ::Float64; step_reward::Float64=-1.0, offgrid_penalty::Float64=-1000.0, absorbing::Set{Tuple{Int,Int}}=Set{Tuple{Int,Int}}())::MyMDPProblemModel`.

- [ ] **Step 1: Append the failing testset to `test/runtests.jl`**

```julia
@testset "grid world MDP" begin
    rewards = Dict{Tuple{Int,Int},Float64}();
    rewards[(3,3)] = 100.0;    # goal
    rewards[(1,3)] = -100.0;   # hazard
    world = build(MyRectangularGridWorldModel, (nrows=3, ncols=3, rewards=rewards));
    absorbing = Set(keys(rewards));
    mdp = build_mdp(world, 0.95; step_reward=-1.0, offgrid_penalty=-1000.0, absorbing=absorbing);

    # transitions are proper distributions -
    for a ∈ mdp.𝒜, s ∈ mdp.𝒮
        @test isapprox(sum(mdp.T[s, :, a]), 1.0; atol=1e-9);
    end

    # from the cell just below the goal (3,2), moving up (action 4) must reach the goal and earn +100 -
    s = world.states[(3,2)];
    @test mdp.R[s, 4] == 100.0;
    @test mdp.T[s, world.states[(3,3)], 4] == 1.0;

    # absorbing cells self-loop with zero reward (value-to-go 0) -
    for cell ∈ absorbing
        sc = world.states[cell];
        for a ∈ mdp.𝒜
            @test mdp.R[sc, a] == 0.0;
            @test mdp.T[sc, sc, a] == 1.0;
        end
    end

    # value iteration policy sends (3,2) up into the goal; absorbing goal has V=0 -
    sol = solve(build(MyValueIterationModel, (maxiterations=10_000, ϵ=1e-8)), mdp);
    @test policy(Q(mdp, sol.V))[s] == 4;
    @test isapprox(sol.V[world.states[(3,3)]], 0.0; atol=1e-9);
end
```

- [ ] **Step 2: Run to verify it fails**

Run: `julia --project=. test/runtests.jl`
Expected: FAIL — `build_mdp` not defined.

- [ ] **Step 3: Append `build_mdp` to `src/Compute.jl`**

```julia
"""
    build_mdp(world::MyRectangularGridWorldModel, γ::Float64; step_reward, offgrid_penalty, absorbing)
        -> MyMDPProblemModel

Construct a deterministic-move MDP from a grid world. A move to a valid non-absorbing cell earns that
cell's reward (or `step_reward`); a move off the grid earns `offgrid_penalty` and self-loops. Absorbing
cells self-loop with zero reward, so their value-to-go is `V=0` (the terminal reward is earned on the
step INTO the cell, not while sitting in it).
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `julia --project=. test/runtests.jl`
Expected: PASS (both testsets).

- [ ] **Step 5: Commit**

```bash
git add courses/CHEME-145/module-4/src/Compute.jl courses/CHEME-145/module-4/test/runtests.jl
git commit -m "M4: grid-world -> MDP builder + tests"
```

---

## Phase 1 — Value iteration notebooks (grid world)

Shared grid world for both notebooks (10×10, γ=0.95): goal `(8,8)=+100`; lava `{(3,3),(5,5),(7,4),(2,7),(6,8)} = -100`; `step_reward=-1`; `offgrid_penalty=-1000`; goal and lava absorbing. All values built via `build_mdp`.

### Task 1.1: Value-iteration Watch-Demo

**Files:**
- Create: `CHEME-145-M4-Example-GridWorld-ValueIteration-Watch-Demo.ipynb`

**Cell manifest** (author prose to the template; code cells verbatim below):

1. **[md] Title + intro** — "Demo: Value Iteration on a Grid World." Intro: build an MDP for grid navigation and compute the optimal policy by value iteration. Three Learning Objectives: (a) __Formulate a grid-world MDP:__ build `(𝒮,𝒜,T,R,γ)` with `build_mdp`; (b) __Run value iteration:__ apply the Bellman optimality backup to convergence in `‖·‖∞`; (c) __Extract and visualize the policy:__ `π(s)=argmax_a Q(s,a)` and trace the optimal path.
2. **[md] Theory** — Bellman optimality operator, value-iteration update, sup-norm convergence, contraction modulus `γ` (mirror the lecture's Value Iteration section).
3. **[md] Setup** — standard include text.
4. **[code]**
   ```julia
   include("Include.jl");
   ```
5. **[md] Build the grid world** — describe grid, goal, lava, absorbing set.
6. **[code]**
   ```julia
   number_of_rows = 10;
   number_of_cols = 10;
   γ = 0.95;

   rewards = Dict{Tuple{Int,Int},Float64}();
   rewards[(8,8)] = 100.0;                       # charging station (goal)
   for cell ∈ [(3,3), (5,5), (7,4), (2,7), (6,8)]
       rewards[cell] = -100.0;                   # lava pits (hazards)
   end
   absorbing = Set(keys(rewards));

   world = build(MyRectangularGridWorldModel,
       (nrows = number_of_rows, ncols = number_of_cols, rewards = rewards));
   ```
7. **[md] Assemble the MDP** — R[s,a], T[s,s′,a] via `build_mdp`.
8. **[code]**
   ```julia
   mdp = build_mdp(world, γ; step_reward = -1.0, offgrid_penalty = -1000.0, absorbing = absorbing);
   (length(mdp.𝒮), length(mdp.𝒜))
   ```
9. **[md] Run value iteration** — describe solver settings and the sup-norm stopping rule.
10. **[code]**
    ```julia
    solution = let
        vi = build(MyValueIterationModel, (maxiterations = 1_000, ϵ = 1e-6));
        solution = solve(vi, mdp);
        println("‖V‖∞ = ", round(maximum(abs.(solution.V)), digits = 3));
        solution
    end;
    ```
11. **[md] Extract the policy** — `Q` then `policy`.
12. **[code]**
    ```julia
    my_Q = Q(mdp, solution.V);
    my_π = policy(my_Q);
    ```
13. **[md] Visualize the value function** — heatmap of `V` over the grid.
14. **[code]**
    ```julia
    let
        Vgrid = [ solution.V[world.states[(x,y)]] for y ∈ 1:number_of_rows, x ∈ 1:number_of_cols ];
        heatmap(1:number_of_cols, 1:number_of_rows, Vgrid, c = :viridis, aspect_ratio = :equal,
            bg = "floralwhite", background_color_outside = "white", framestyle = :box,
            xlabel = "x", ylabel = "y", title = "Optimal value V*(s)")
    end
    ```
15. **[md] Visualize the optimal path** — trace `π` from a start state to the goal (port the old path-drawing loop, using `my_π`, `world`, `absorbing`).
16. **[code]** — path plot:
    ```julia
    let
        startstate = (1,1);
        p = plot(bg = "floralwhite", background_color_outside = "white", framestyle = :box,
            aspect_ratio = :equal, xlabel = "x", ylabel = "y", title = "Optimal path");

        # draw the trajectory under π -
        s = world.states[startstate];
        visited = Set{Tuple{Int,Int}}([startstate]);
        done = false;
        while (done == false)
            current = world.coordinates[s];
            Δ = world.moves[my_π[s]];
            newpos = current .+ Δ;
            plot!([current[1], newpos[1]], [current[2], newpos[2]], label = "", arrow = true, lw = 2, c = colors[1]);
            if (in(newpos, absorbing) == true || in(newpos, visited) == true || haskey(world.states, newpos) == false)
                done = true;
            else
                s = world.states[newpos];
                push!(visited, newpos);
            end
        end

        # mark goal (green) and lava (red) -
        for (cell, r) ∈ rewards
            scatter!([cell[1]], [cell[2]], label = "", c = (r > 0 ? colors[5] : colors[4]), ms = 7);
        end
        scatter!([startstate[1]], [startstate[2]], label = "start", c = colors[3], ms = 7);
        current()
    end
    ```
17. **[md] Summary + 3 Key Takeaways** — MDP formulation via `build_mdp`; value iteration converges via the sup-norm contraction; policy is greedy w.r.t. `Q*`.
18. **[md] Additional Resources** — Bellman (1957); Sutton & Barto (2018); Kochenderfer et al. (2022).

- [ ] **Step 1:** Create the notebook per the manifest (shared authoring procedure).
- [ ] **Step 2:** Execute (shared procedure). Expected: exit 0, no errors; heatmap and path plots render.
- [ ] **Step 3: Verify** — the printed state/action counts are `(100, 4)`; the traced path terminates at the goal `(8,8)`; the policy matches the library:
  Run: `julia --project=. -e 'include("Include.jl"); rewards=Dict{Tuple{Int,Int},Float64}((8,8)=>100.0); for c in [(3,3),(5,5),(7,4),(2,7),(6,8)]; rewards[c]=-100.0; end; w=build(MyRectangularGridWorldModel,(nrows=10,ncols=10,rewards=rewards)); m=build_mdp(w,0.95;absorbing=Set(keys(rewards))); s=solve(build(MyValueIterationModel,(maxiterations=1000,ϵ=1e-6)),m); println(policy(Q(m,s.V))[w.states[(8,7)]])'`
  Expected: prints `4` (from just below the goal, move up).
- [ ] **Step 4: Commit**
  ```bash
  git add courses/CHEME-145/module-4/CHEME-145-M4-Example-GridWorld-ValueIteration-Watch-Demo.ipynb
  git commit -m "M4: value-iteration grid-world Watch-Demo"
  ```

### Task 1.2: Value-iteration Ungraded Codio Activity

**Files:**
- Create: `CHEME-145-M4-Example-GridWorld-ValueIteration-Ungraded-Codio-Activity.ipynb`

**Cell manifest:** same skeleton as 1.1 (build world → MDP → value iteration → policy → visualize), rewritten as a student activity: prose invites edits and adds two explicit experiments.

- Learning Objectives framed as "you will": formulate the MDP, run value iteration, and interpret how `γ` and the reward structure change the policy.
- After the base solve/visualize (reuse the 1.1 code cells with `startstate = (1,1)`), add:
  - **[md] Experiment 1 — discounting.** Re-solve for `γ ∈ {0.5, 0.9, 0.99}` and compare the path length / value near the goal.
    **[code]**
    ```julia
    for γ_try ∈ (0.5, 0.9, 0.99)
        m_try = build_mdp(world, γ_try; step_reward = -1.0, offgrid_penalty = -1000.0, absorbing = absorbing);
        s_try = solve(build(MyValueIterationModel, (maxiterations = 1_000, ϵ = 1e-6)), m_try);
        println("γ = $(γ_try):  V*(start) = ", round(s_try.V[world.states[(1,1)]], digits = 3));
    end
    ```
  - **[md] Experiment 2 — reward shaping.** Add a lava cell (or move the goal), rebuild, re-solve, and observe the policy change (student edits `rewards`).
    **[code]**
    ```julia
    let
        rewards2 = copy(rewards);
        rewards2[(4,4)] = -100.0;                       # add a hazard (students can change this)
        absorbing2 = Set(keys(rewards2));
        world2 = build(MyRectangularGridWorldModel, (nrows = number_of_rows, ncols = number_of_cols, rewards = rewards2));
        mdp2 = build_mdp(world2, γ; step_reward = -1.0, offgrid_penalty = -1000.0, absorbing = absorbing2);
        sol2 = solve(build(MyValueIterationModel, (maxiterations = 1_000, ϵ = 1e-6)), mdp2);
        println("added hazard at (4,4);  V*(start) = ", round(sol2.V[world2.states[(1,1)]], digits = 3));
    end
    ```
- **[md] Summary + 3 Key Takeaways** — value iteration is reusable across reward layouts; larger `γ` yields more far-sighted policies; the greedy policy adapts to hazards.

- [ ] **Step 1:** Create the notebook per the manifest.
- [ ] **Step 2:** Execute (shared procedure). Expected: exit 0; all three `γ` values print; both experiments run.
- [ ] **Step 3: Verify** — `V*(start)` is larger (less negative) for larger `γ`; adding the `(4,4)` hazard changes `V*(start)`.
- [ ] **Step 4: Commit**
  ```bash
  git add courses/CHEME-145/module-4/CHEME-145-M4-Example-GridWorld-ValueIteration-Ungraded-Codio-Activity.ipynb
  git commit -m "M4: value-iteration grid-world Ungraded Codio activity"
  ```

---

## Phase 2 — Policy iteration notebooks (inventory)

### Task 2.0: Inventory MDP builder (`build_inventory_mdp`) + test

**Files:**
- Modify: `src/Compute.jl` (append), `test/runtests.jl` (append)

**Interfaces:**
- Produces: `build_inventory_mdp(; capacity::Int, demand_pmf::Vector{Float64}, price::Float64, order_cost::Float64, fixed_cost::Float64, holding_cost::Float64, stockout_penalty::Float64, γ::Float64)::MyMDPProblemModel`. States `s = i+1` for on-hand `i ∈ 0:capacity`; actions `a = o+1` for order `o ∈ 0:capacity`; infeasible actions (`i+o > capacity`) self-loop with a large negative reward. Demand `d ∈ 0:(length(demand_pmf)-1)`.

- [ ] **Step 1: Append the failing testset**

```julia
@testset "inventory MDP: VI == PI" begin
    mdp = build_inventory_mdp(; capacity = 8, demand_pmf = [0.05,0.2,0.5,0.2,0.05],
        price = 10.0, order_cost = 3.0, fixed_cost = 2.0, holding_cost = 1.0,
        stockout_penalty = 8.0, γ = 0.95);

    # rows of T are distributions -
    for a ∈ mdp.𝒜, s ∈ mdp.𝒮
        @test isapprox(sum(mdp.T[s, :, a]), 1.0; atol=1e-9);
    end

    sol_vi = solve(build(MyValueIterationModel, (maxiterations = 10_000, ϵ = 1e-9)), mdp);
    sol_pi = solve(build(MyPolicyIterationModel, (maxiterations = 100,)), mdp);
    @test policy(Q(mdp, sol_vi.V)) == policy(Q(mdp, sol_pi.V));
    @test isapprox(sol_vi.V, sol_pi.V; atol=1e-4);
end
```

- [ ] **Step 2: Run to verify it fails** — `julia --project=. test/runtests.jl` → FAIL (`build_inventory_mdp` not defined).

- [ ] **Step 3: Append `build_inventory_mdp` to `src/Compute.jl`**

```julia
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
```

- [ ] **Step 4: Run to verify it passes** — `julia --project=. test/runtests.jl` → PASS.
- [ ] **Step 5: Commit**
  ```bash
  git add courses/CHEME-145/module-4/src/Compute.jl courses/CHEME-145/module-4/test/runtests.jl
  git commit -m "M4: stochastic inventory MDP builder + VI==PI test"
  ```

### Task 2.1: Policy-iteration Watch-Demo

**Files:**
- Create: `CHEME-145-M4-Example-Inventory-PolicyIteration-Watch-Demo.ipynb`

**Cell manifest:**

1. **[md] Title + intro** — "Demo: Policy Iteration for Inventory Control." Three objectives: (a) __Build a stochastic MDP:__ `build_inventory_mdp`; (b) __Policy iteration:__ exact evaluation `(I-γPπ)⁻¹Rπ` + greedy improvement; (c) __Compare to value iteration:__ same optimum, different iteration counts.
2. **[md] Theory** — Bellman expectation equation, matrix policy-evaluation solve, policy-improvement theorem (mirror the lecture's Policy Iteration section).
3. **[md] Setup** → **[code]** `include("Include.jl");`
4. **[md] Build the inventory MDP** → **[code]**
   ```julia
   mdp = build_inventory_mdp(; capacity = 8, demand_pmf = [0.05, 0.2, 0.5, 0.2, 0.05],
       price = 10.0, order_cost = 3.0, fixed_cost = 2.0, holding_cost = 1.0,
       stockout_penalty = 8.0, γ = 0.95);
   (length(mdp.𝒮), length(mdp.𝒜))
   ```
5. **[md] Solve by policy iteration** → **[code]**
   ```julia
   sol_pi = solve(build(MyPolicyIterationModel, (maxiterations = 100,)), mdp);
   π_pi = policy(Q(mdp, sol_pi.V));
   ```
6. **[md] Read the policy as an order rule** → **[code]** (order quantity per on-hand level)
   ```julia
   let
       rows = [ (i, π_pi[i+1] - 1, round(sol_pi.V[i+1], digits = 2)) for i ∈ 0:8 ];
       pretty_table(rows; header = ["on-hand i", "order o = π(i)", "V(i)"]);
   end
   ```
7. **[md] Cross-check with value iteration** → **[code]**
   ```julia
   let
       sol_vi = solve(build(MyValueIterationModel, (maxiterations = 10_000, ϵ = 1e-9)), mdp);
       π_vi = policy(Q(mdp, sol_vi.V));
       println("policies agree: ", π_vi == π_pi);
       println("max |V_vi - V_pi| = ", round(maximum(abs.(sol_vi.V .- sol_pi.V)), digits = 6));
   end
   ```
8. **[md] Base-stock structure** — plot order-up-to level `i + π(i)` vs on-hand `i` → **[code]**
   ```julia
   let
       order_up_to = [ (i) + (π_pi[i+1] - 1) for i ∈ 0:8 ];
       plot(0:8, order_up_to, seriestype = :steppost, lw = 2, c = colors[1], label = "order-up-to S",
           bg = "floralwhite", background_color_outside = "white", framestyle = :box,
           xlabel = "on-hand inventory i", ylabel = "post-order stock i + π(i)")
   end
   ```
9. **[md] Summary + 3 Key Takeaways** — exact policy evaluation is a linear solve; policy improvement is greedy w.r.t. `Q`; VI and PI reach the same policy.
10. **[md] Additional Resources** — Puterman (1994); Bertsekas, *Dynamic Programming and Optimal Control*; Sutton & Barto (2018).

- [ ] **Step 1:** Create per manifest. **Step 2:** Execute (shared procedure). **Step 3: Verify** — the printed "policies agree: true" and `max |V_vi - V_pi|` ≈ 0. **Step 4: Commit** `"M4: policy-iteration inventory Watch-Demo"`.

### Task 2.2: Policy-iteration Ungraded Codio Activity

**Files:**
- Create: `CHEME-145-M4-Example-Inventory-PolicyIteration-Ungraded-Codio-Activity.ipynb`

**Cell manifest:** student version. Objectives framed as "you will": build the inventory MDP, implement the policy-iteration loop step by step using `policy_evaluation` and `policy`/`Q`, and compare its iteration count to value iteration.

- Setup + build MDP (reuse Task 2.1 build cell).
- **[md] Step through policy iteration manually** → **[code]** (students see each improvement sweep)
  ```julia
  let
      π = ones(Int64, length(mdp.𝒮));   # start: order 0 everywhere
      for k ∈ 1:20
          V = policy_evaluation(mdp, π);
          π′ = policy(Q(mdp, V));
          println("sweep $(k):  changed actions = ", count(π′ .!= π));
          (π′ == π) && (println("converged after $(k) sweep(s)"); break);
          π = π′;
      end
  end
  ```
- **[md] Compare iteration counts: policy iteration vs. value iteration.** Policy iteration converges in a few expensive evaluate/improve cycles. Value iteration takes many cheap sweeps for its *value function* to converge, even though its *greedy policy* often becomes optimal much earlier. Count all three. → **[code]**
  ```julia
  let
      ϵ = 1e-9;
      π_star = policy(Q(mdp, solve(build(MyPolicyIterationModel, (maxiterations = 100,)), mdp).V));

      # policy iteration: number of evaluate/improve cycles until the policy is stable
      π = ones(Int64, length(mdp.𝒮)); pi_cycles = 0;
      while true
          Vπ = policy_evaluation(mdp, π);
          π′ = policy(Q(mdp, Vπ));
          pi_cycles += 1;
          (π′ == π) && break;
          π = π′;
      end

      # value iteration: sweeps until the value function converges, and the first sweep whose greedy policy is optimal
      V = zeros(Float64, length(mdp.𝒮));
      first_match = -1; vi_sweeps = 0; converged = false;
      while (converged == false && vi_sweeps < 5_000)
          Vnew = [ maximum(mdp.R[s,a] + mdp.γ*sum(mdp.T[s,s′,a]*V[s′] for s′ ∈ mdp.𝒮) for a ∈ mdp.𝒜) for s ∈ mdp.𝒮 ];
          vi_sweeps += 1;
          (policy(Q(mdp, Vnew)) == π_star && first_match < 0) && (first_match = vi_sweeps);
          Δ = maximum(abs.(Vnew .- V));
          V = Vnew;
          (Δ ≤ ϵ) && (converged = true);
      end

      println("policy iteration: converged in $(pi_cycles) evaluate/improve cycles");
      println("value iteration:  value function converged in $(vi_sweeps) sweeps (‖ΔV‖∞ ≤ $(ϵ))");
      println("value iteration:  greedy policy already optimal at sweep $(first_match)");
  end
  ```
- **[md] Experiment** — students vary `stockout_penalty` or `demand_pmf`, rebuild, and re-solve; observe the order-up-to level shift.
- **[md] Summary + 3 Key Takeaways** — (1) policy iteration reaches the optimum in a few expensive evaluate/improve cycles (each a full policy evaluation); (2) value iteration needs many cheap sweeps for its value function to converge, even though its greedy policy often becomes optimal much earlier; (3) both converge to the same optimal policy.

- [ ] **Step 1:** Create per manifest. **Step 2:** Execute. **Step 3: Verify** — the manual PI loop prints "converged after k cycle(s)" with small `k` (≈3); the comparison cell prints PI cycles ≈ 3, VI value-convergence sweeps ≈ 447, and VI greedy-optimal at sweep ≈ 2 (VI's value-convergence sweeps ≫ PI's cycles, while VI's greedy policy locks in early). **Step 4: Commit** `"M4: policy-iteration inventory Ungraded Codio activity"`.

---

## Phase 3 — Graded capstone (equipment replacement)

### Task 3.0: Equipment-replacement MDP builder (`build_replacement_mdp`) + test

**Files:**
- Modify: `src/Compute.jl` (append), `test/runtests.jl` (append)

**Interfaces:**
- Produces: `build_replacement_mdp(; max_age::Int, income0::Float64, income_decline::Float64, maint0::Float64, maint_slope::Float64, replace_cost::Float64, γ::Float64)::MyMDPProblemModel`. State `s = a+1` for age `a ∈ 0:max_age`; action 1 = keep, 2 = replace. Keep: reward `= (income0 - income_decline·a) - (maint0 + maint_slope·a)`, next age `min(a+1, max_age)`. Replace: reward `= -replace_cost + (income0 - maint0)`, next age `1`.

- [ ] **Step 1: Append the failing testset**

```julia
@testset "equipment replacement: threshold policy, VI == PI" begin
    mdp = build_replacement_mdp(; max_age = 10, income0 = 100.0, income_decline = 5.0,
        maint0 = 2.0, maint_slope = 5.0, replace_cost = 60.0, γ = 0.9);

    for a ∈ mdp.𝒜, s ∈ mdp.𝒮
        @test isapprox(sum(mdp.T[s, :, a]), 1.0; atol=1e-9);
    end

    sol_vi = solve(build(MyValueIterationModel, (maxiterations = 10_000, ϵ = 1e-9)), mdp);
    sol_pi = solve(build(MyPolicyIterationModel, (maxiterations = 100,)), mdp);
    π_vi = policy(Q(mdp, sol_vi.V));
    @test π_vi == policy(Q(mdp, sol_pi.V));            # VI == PI

    # monotone threshold: once "replace" (action 2) is chosen, it stays chosen for all older ages -
    replace_ages = [ a for a ∈ 0:10 if π_vi[a+1] == 2 ];
    @test !isempty(replace_ages);                     # replacement is used somewhere
    a_star = minimum(replace_ages);
    @test all(π_vi[a+1] == 2 for a ∈ a_star:10);      # monotone
end
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`build_replacement_mdp` not defined).

- [ ] **Step 3: Append `build_replacement_mdp` to `src/Compute.jl`**

```julia
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
```

- [ ] **Step 4: Run to verify it passes** — PASS.
- [ ] **Step 5: Commit** `"M4: equipment-replacement MDP builder + threshold test"`.

### Task 3.1: Graded Solution notebook (full solution)

**Files:**
- Create: `CHEME-145-M4-Example-EquipmentReplacement-MDP-Graded-Codio-Activity-Solution.ipynb`

**Cell manifest** (complete, working solution — the answer key):

1. **[md] Title + intro** — "Graded Activity (Solution): Equipment Replacement as an MDP." Frame the keep/replace decision as optimal stopping; note the Rust (1987) tie-back. Three objectives: (a) __Formulate the replacement MDP__; (b) __Solve via value iteration and policy iteration__ and confirm agreement; (c) __Recommend a replacement-age threshold__.
2. **[md] Theory / problem statement** — state = age, actions keep/replace, reward and dynamics as in `build_replacement_mdp`; discount `γ`.
3. **[md] Setup** → **[code]** `include("Include.jl");`
4. **[md] Build the MDP** → **[code]**
   ```julia
   mdp = build_replacement_mdp(; max_age = 10, income0 = 100.0, income_decline = 5.0,
       maint0 = 2.0, maint_slope = 5.0, replace_cost = 60.0, γ = 0.9);
   ```
5. **[md] Solve by value iteration** → **[code]**
   ```julia
   sol_vi = solve(build(MyValueIterationModel, (maxiterations = 10_000, ϵ = 1e-9)), mdp);
   π_vi = policy(Q(mdp, sol_vi.V));
   ```
6. **[md] Solve by policy iteration** → **[code]**
   ```julia
   sol_pi = solve(build(MyPolicyIterationModel, (maxiterations = 100,)), mdp);
   π_pi = policy(Q(mdp, sol_pi.V));
   println("policies agree: ", π_vi == π_pi);
   ```
7. **[md] The replacement threshold** → **[code]**
   ```julia
   a_star = minimum([ a for a ∈ 0:10 if π_vi[a+1] == 2 ]);
   println("recommended policy: replace when age ≥ ", a_star);
   ```
8. **[md] Policy + value table** → **[code]**
   ```julia
   let
       rows = [ (a, π_vi[a+1] == 1 ? "keep" : "replace", round(sol_vi.V[a+1], digits = 2)) for a ∈ 0:10 ];
       pretty_table(rows; header = ["age a", "π*(a)", "V*(a)"]);
   end
   ```
9. **[md] Visualize the value function and threshold** → **[code]**
   ```julia
   let
       plot(0:10, sol_vi.V, lw = 2, c = colors[1], marker = :circle, label = "V*(a)",
           bg = "floralwhite", background_color_outside = "white", framestyle = :box,
           xlabel = "machine age a", ylabel = "optimal value V*(a)");
       vline!([a_star], lw = 2, ls = :dash, c = colors[4], label = "replace threshold a* = $(a_star)")
   end
   ```
10. **[md] Recommendation (written)** — one paragraph: replace at age `a*`; VI and PI agree; connect to Rust (1987) optimal stopping.
11. **[md] Summary + 3 Key Takeaways** — replacement is an optimal-stopping MDP; VI and PI give the same threshold; the threshold is the actionable recommendation.
12. **[md] Additional Resources** — Rust (1987); Puterman (1994); Sutton & Barto (2018).

- [ ] **Step 1:** Create per manifest. **Step 2:** Execute. **Step 3: Verify** — "policies agree: true"; `a_star` matches `test/runtests.jl` (recompute via the same one-liner pattern). **Step 4: Commit** `"M4: equipment-replacement Graded Codio SOLUTION"`.

### Task 3.2: Graded student notebook (blanks)

**Files:**
- Create: `CHEME-145-M4-Example-EquipmentReplacement-MDP-Graded-Codio-Activity.ipynb`

Derived from the solution (Task 3.1): identical prose/structure, but the three assessed code cells have the student-completed lines replaced by `# TODO` stubs that raise until filled, and expected answers are removed from prose.

**Assessed blanks (everything else identical to the solution, executed to show setup outputs):**
- **Blank 1 — build the MDP** (cell 4): give the `build_replacement_mdp` call with the numeric keyword arguments replaced by `# TODO: set ...` and `error("complete build_replacement_mdp(...) call")` so an unedited run fails clearly.
- **Blank 2 — solve both ways** (cells 5–6): students write the two `solve(...)` calls and the `policy(Q(...))` extractions; stub with `error("compute π_vi and π_pi")`.
- **Blank 3 — extract the threshold** (cell 7): students write the `a_star = minimum(...)` comprehension; stub with `error("compute the replacement threshold a_star")`.

Add an ungraded-style **[md] instructions** block near the top listing the three tasks and the expected deliverable (the threshold age + a one-paragraph recommendation). Keep the Summary/Takeaways but phrased as prompts.

- [ ] **Step 1:** Create the student notebook (copy the solution JSON, replace the three assessed cells with stubs, strip answer values from prose).
- [ ] **Step 2:** Execute with `--allow-errors` so setup cells run and the intended stub error is captured (a student notebook is not expected to run clean):
  ```bash
  jupyter nbconvert --to notebook --execute --inplace --allow-errors \
    --ExecutePreprocessor.kernel_name=julia-1.12 --ExecutePreprocessor.timeout=600 \
    CHEME-145-M4-Example-EquipmentReplacement-MDP-Graded-Codio-Activity.ipynb
  ```
  Expected: exit 0; the stub cells show the intended `error(...)` message; non-stub cells (setup) run clean.
- [ ] **Step 3: Verify** — diff the student notebook against the solution: only the three assessed code cells and answer-bearing prose differ.
- [ ] **Step 4: Commit** `"M4: equipment-replacement Graded Codio activity (student)"`.

---

## Phase 4 — Advanced: Random rollout (grid world)

### Task 4.0: Rollout (`rollout_value`) + test

**Files:**
- Modify: `src/Compute.jl` (append), `test/runtests.jl` (append)

**Interfaces:**
- Produces:
  - `simulate_return(problem::MyMDPProblemModel, s0::Int, π_fn::Function, H::Int, rng::AbstractRNG)::Float64`
  - `rollout_value(problem::MyMDPProblemModel, s0::Int; π_fn::Function, H::Int=100, N::Int=1000, rng::AbstractRNG=Random.default_rng())::Float64`
- Consumes: `MyMDPProblemModel`, `Random`, `Statistics`.

- [ ] **Step 1: Append the failing testset** — rollout under the optimal policy approximates `V*`.

```julia
@testset "random rollout approximates V* under π*" begin
    # small grid world with a known optimal value from value iteration -
    rewards = Dict{Tuple{Int,Int},Float64}((5,5)=>100.0, (2,2)=>-100.0);
    world = build(MyRectangularGridWorldModel, (nrows=5, ncols=5, rewards=rewards));
    absorbing = Set(keys(rewards));
    mdp = build_mdp(world, 0.95; step_reward=-1.0, offgrid_penalty=-1000.0, absorbing=absorbing);
    sol = solve(build(MyValueIterationModel, (maxiterations=10_000, ϵ=1e-9)), mdp);
    π_star = policy(Q(mdp, sol.V));

    s0 = world.states[(1,1)];
    rng = Random.MersenneTwister(42);
    est = rollout_value(mdp, s0; π_fn = s -> π_star[s], H = 200, N = 4_000, rng = rng);
    @test isapprox(est, sol.V[s0]; rtol = 0.05);       # within 5%
end
```

- [ ] **Step 2: Run to verify it fails** — FAIL (`rollout_value` not defined).

- [ ] **Step 3: Append rollout code to `src/Compute.jl`**

```julia
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
```

- [ ] **Step 4: Run to verify it passes** — PASS.
- [ ] **Step 5: Commit** `"M4: Monte Carlo rollout + test"`.

### Task 4.1: Random-rollout Watch-Demo

**Files:**
- Create: `CHEME-145-M4-Advanced-RandomRollout-Watch-Demo.ipynb`

**Cell manifest:**

1. **[md] Title + intro** — "Advanced (optional): Estimating Value by Random Rollout." Motivate: exact DP needs the full model; rollout estimates value by simulation. Three objectives: (a) __Simulate returns__ with `simulate_return`; (b) __Estimate a state's value__ with `rollout_value` and compare to `V*`; (c) __Rollout policy improvement__ via one-step lookahead using rollout estimates.
2. **[md] Theory** — rollout as Monte Carlo estimate of `V^π`; base policy; horizon `H`; sample count `N`; standard error `∝ 1/√N`.
3. **[md] Setup** → **[code]** `include("Include.jl");`
4. **[md] Build the grid world + exact solution (reference)** → **[code]** (reuse the 5×5 world from the test, plus VI for `V*` and `π*`).
   ```julia
   rewards = Dict{Tuple{Int,Int},Float64}((5,5)=>100.0, (2,2)=>-100.0);
   world = build(MyRectangularGridWorldModel, (nrows=5, ncols=5, rewards=rewards));
   absorbing = Set(keys(rewards));
   mdp = build_mdp(world, 0.95; step_reward=-1.0, offgrid_penalty=-1000.0, absorbing=absorbing);
   sol = solve(build(MyValueIterationModel, (maxiterations=10_000, ϵ=1e-9)), mdp);
   π_star = policy(Q(mdp, sol.V));
   ```
5. **[md] Rollout under the optimal policy** → **[code]**
   ```julia
   let
       s0 = world.states[(1,1)];
       rng = Random.MersenneTwister(1);
       est = rollout_value(mdp, s0; π_fn = s -> π_star[s], H = 200, N = 5_000, rng = rng);
       println("rollout estimate = ", round(est, digits = 3), "   exact V* = ", round(sol.V[s0], digits = 3));
   end
   ```
6. **[md] Accuracy vs. number of trajectories** → **[code]** (plot estimate vs `N` converging to `V*`)
   ```julia
   let
       s0 = world.states[(1,1)];
       Ns = [50, 100, 250, 500, 1_000, 2_500, 5_000];
       ests = map(Ns) do n
           rollout_value(mdp, s0; π_fn = s -> π_star[s], H = 200, N = n, rng = Random.MersenneTwister(7));
       end
       plot(Ns, ests, lw = 2, marker = :circle, c = colors[1], label = "rollout estimate", xscale = :log10,
           bg = "floralwhite", background_color_outside = "white", framestyle = :box,
           xlabel = "trajectories N", ylabel = "estimated V(start)");
       hline!([sol.V[s0]], lw = 2, ls = :dash, c = colors[4], label = "exact V*")
   end
   ```
7. **[md] Rollout policy improvement** — one-step lookahead: for a random base policy, pick the action maximizing `R(s,a) + γ·rollout_value(next)` → **[code]** (demonstrate on one state; compare chosen action to `π*`).
   ```julia
   let
       rng = Random.MersenneTwister(11);
       base = s -> rand(rng, mdp.𝒜);                 # random base policy
       s = world.states[(1,1)];
       # one-step lookahead using rollout of the base policy from each successor -
       scores = map(mdp.𝒜) do a
           s′ = argmax(mdp.T[s, :, a]);                # deterministic grid successor
           mdp.R[s, a] + mdp.γ*rollout_value(mdp, s′; π_fn = base, H = 100, N = 500, rng = rng);
       end
       println("rollout-improved action = ", argmax(scores), "   optimal action π*(s) = ", π_star[s]);
   end
   ```
8. **[md] Summary + 3 Key Takeaways** — rollout estimates value by simulation; accuracy improves with `N`; one-step rollout improvement recovers good actions without solving the full MDP.
9. **[md] Additional Resources** — Sutton & Barto (2018), ch. 8; Kochenderfer et al. (2022); Bertsekas, *Rollout, Policy Iteration, and Distributed RL* (2020).

- [ ] **Step 1:** Create per manifest. **Step 2:** Execute. **Step 3: Verify** — rollout estimate within ~5% of `V*`; the estimate-vs-`N` curve approaches the exact line. **Step 4: Commit** `"M4: random-rollout Advanced Watch-Demo"`.

### Task 4.2: Random-rollout Ungraded Codio Activity

**Files:**
- Create: `CHEME-145-M4-Advanced-RandomRollout-Ungraded-Codio-Activity.ipynb`

**Cell manifest:** student version of 4.1. Objectives: run rollouts, quantify accuracy vs `V*`, and study the effect of horizon `H` and trajectory count `N`.

- Setup + build world/MDP + VI reference (reuse 4.1 cells).
- **[md] Experiment — horizon sensitivity** → **[code]**
  ```julia
  let
      s0 = world.states[(1,1)];
      for H ∈ (10, 25, 50, 100, 200)
          est = rollout_value(mdp, s0; π_fn = s -> π_star[s], H = H, N = 2_000, rng = Random.MersenneTwister(3));
          println("H = $(H):  estimate = ", round(est, digits = 3), "   (exact ", round(sol.V[s0], digits = 3), ")");
      end
  end
  ```
- **[md] Experiment — estimator variance** → **[code]** (repeat the estimate with several seeds, report spread)
  ```julia
  let
      s0 = world.states[(1,1)];
      ests = [ rollout_value(mdp, s0; π_fn = s -> π_star[s], H = 200, N = 1_000, rng = Random.MersenneTwister(seed)) for seed ∈ 1:10 ];
      println("mean = ", round(mean(ests), digits = 3), "   std = ", round(std(ests), digits = 3));
  end
  ```
- **[md] Summary + 3 Key Takeaways** — too-short horizons bias the estimate low; variance shrinks with `N`; rollout trades exactness for a model-light estimate.

- [ ] **Step 1:** Create. **Step 2:** Execute. **Step 3: Verify** — larger `H` moves the estimate toward `V*`; reported `std` shrinks relative to a smaller-`N` run. **Step 4: Commit** `"M4: random-rollout Advanced Ungraded Codio activity"`.

---

## Phase 5 — Advanced: Monte Carlo Tree Search (grid world)

### Task 5.0: MCTS (`mcts`) + test

**Files:**
- Modify: `src/Compute.jl` (append), `src/Types.jl` (append `MyMCTSModel`), `src/Factory.jl` (append its `build`), `test/runtests.jl` (append)

**Interfaces:**
- Produces:
  - `MyMCTSModel <: AbstractSolutionModel` with fields `iterations::Int64`, `c::Float64` (exploration constant), `horizon::Int64`, `depth::Int64`.
  - `build(::Type{MyMCTSModel}, data::NamedTuple)` (keys `iterations, c, horizon, depth`).
  - `mcts(problem::MyMDPProblemModel, model::MyMCTSModel, s0::Int; rng::AbstractRNG=Random.default_rng())::Int` — returns the best action at the root.
- Consumes: `simulate_return` (Task 4.0), `Random`.

- [ ] **Step 1: Append `MyMCTSModel` to `src/Types.jl`**

```julia
"""
    mutable struct MyMCTSModel <: AbstractSolutionModel

Monte Carlo tree search settings.

### Fields
- `iterations::Int64`: number of tree-building iterations.
- `c::Float64`: UCT exploration constant.
- `horizon::Int64`: rollout horizon for the simulation step.
- `depth::Int64`: maximum tree depth expanded before rollout.
"""
mutable struct MyMCTSModel <: AbstractSolutionModel
    iterations::Int64
    c::Float64
    horizon::Int64
    depth::Int64
    MyMCTSModel() = new();
end
```

- [ ] **Step 2: Append its `build` to `src/Factory.jl`**

```julia
"""
    build(::Type{MyMCTSModel}, data::NamedTuple) -> MyMCTSModel

Build MCTS settings from keys `iterations, c, horizon, depth`.
"""
function build(::Type{MyMCTSModel}, data::NamedTuple)::MyMCTSModel
    model = MyMCTSModel();
    model.iterations = data.iterations;
    model.c = data.c;
    model.horizon = data.horizon;
    model.depth = data.depth;
    return model;
end
```

- [ ] **Step 3: Append the failing testset** — MCTS root action equals the VI-optimal action.

```julia
@testset "MCTS recovers the optimal root action" begin
    rewards = Dict{Tuple{Int,Int},Float64}((5,5)=>100.0, (2,2)=>-100.0);
    world = build(MyRectangularGridWorldModel, (nrows=5, ncols=5, rewards=rewards));
    absorbing = Set(keys(rewards));
    mdp = build_mdp(world, 0.95; step_reward=-1.0, offgrid_penalty=-1000.0, absorbing=absorbing);
    π_star = policy(Q(mdp, solve(build(MyValueIterationModel, (maxiterations=10_000, ϵ=1e-9)), mdp).V));

    s0 = world.states[(4,5)];   # one step left of the goal (5,5): optimal action is right (2)
    model = build(MyMCTSModel, (iterations = 4_000, c = 50.0, horizon = 100, depth = 20));
    a = mcts(mdp, model, s0; rng = Random.MersenneTwister(99));
    @test a == π_star[s0];
end
```

- [ ] **Step 4: Run to verify it fails** — FAIL (`mcts` not defined).

- [ ] **Step 5: Append MCTS to `src/Compute.jl`**

```julia
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
```

- [ ] **Step 6: Run to verify it passes** — PASS. (If flaky, raise `iterations` or tune `c`; the test uses a fixed seed.)
- [ ] **Step 7: Commit** `"M4: MCTS (UCT) + test"`.

### Task 5.1: MCTS Watch-Demo

**Files:**
- Create: `CHEME-145-M4-Advanced-MCTS-Watch-Demo.ipynb`

**Cell manifest:**

1. **[md] Title + intro** — "Advanced (optional): Monte Carlo Tree Search." Motivate: search a tree of futures using rollouts + UCT, no full sweep of the state space. Three objectives: (a) __The four MCTS steps__ (selection, expansion, simulation, backpropagation); (b) __UCT__ balances exploration/exploitation; (c) __Compare the root action to the value-iteration optimum__.
2. **[md] Theory** — the four phases; UCT formula `Q(s,a) + c·√(ln N(s)/N(s,a))`; rollout evaluation.
3. **[md] Setup** → **[code]** `include("Include.jl");`
4. **[md] Build the grid world + VI reference** → **[code]** (reuse the 5×5 world; compute `π_star`).
5. **[md] Run MCTS from a state** → **[code]**
   ```julia
   let
       s0 = world.states[(4,5)];
       model = build(MyMCTSModel, (iterations = 5_000, c = 50.0, horizon = 100, depth = 20));
       a = mcts(mdp, model, s0; rng = Random.MersenneTwister(2));
       println("MCTS action = ", a, "   optimal action π*(s) = ", π_star[s0]);
   end
   ```
6. **[md] Convergence vs. iteration budget** → **[code]** (fraction of correct root actions vs iterations)
   ```julia
   let
       s0 = world.states[(4,5)];
       for iters ∈ (200, 500, 1_000, 2_500, 5_000)
           hits = count(1:20) do trial
               model = build(MyMCTSModel, (iterations = iters, c = 50.0, horizon = 100, depth = 20));
               mcts(mdp, model, s0; rng = Random.MersenneTwister(trial)) == π_star[s0];
           end
           println("iterations = $(iters):  correct root action in ", hits, "/20 runs");
       end
   end
   ```
7. **[md] Summary + 3 Key Takeaways** — MCTS builds a partial tree with rollouts; UCT trades exploration vs exploitation; more iterations converge to the optimal action.
8. **[md] Additional Resources** — Kocsis & Szepesvári (2006, UCT); Browne et al. (2012, MCTS survey); Sutton & Barto (2018).

- [ ] **Step 1:** Create. **Step 2:** Execute. **Step 3: Verify** — MCTS action equals `π*(s)`; the "correct in k/20" count rises with iterations. **Step 4: Commit** `"M4: MCTS Advanced Watch-Demo"`.

### Task 5.2: MCTS Ungraded Codio Activity

**Files:**
- Create: `CHEME-145-M4-Advanced-MCTS-Ungraded-Codio-Activity.ipynb`

**Cell manifest:** student version. Objectives: run MCTS, vary the exploration constant `c` and iteration budget, and observe convergence to the optimal action.

- Setup + world + VI reference (reuse 5.1 cells).
- **[md] Experiment — exploration constant** → **[code]**
  ```julia
  let
      s0 = world.states[(4,5)];
      for c_try ∈ (1.0, 10.0, 50.0, 200.0)
          hits = count(1:20) do trial
              model = build(MyMCTSModel, (iterations = 2_000, c = c_try, horizon = 100, depth = 20));
              mcts(mdp, model, s0; rng = Random.MersenneTwister(trial)) == π_star[s0];
          end
          println("c = $(c_try):  correct in ", hits, "/20 runs");
      end
  end
  ```
- **[md] Experiment — a harder start state** → **[code]** (a state farther from the goal; more iterations needed).
  ```julia
  let
      s0 = world.states[(1,1)];
      for iters ∈ (1_000, 5_000, 20_000)
          model = build(MyMCTSModel, (iterations = iters, c = 50.0, horizon = 200, depth = 30));
          a = mcts(mdp, model, s0; rng = Random.MersenneTwister(5));
          println("iterations = $(iters):  MCTS action = ", a, "   optimal = ", π_star[s0]);
      end
  end
  ```
- **[md] Summary + 3 Key Takeaways** — `c` controls exploration; harder states need a larger budget; MCTS approximates the optimum without full DP.

- [ ] **Step 1:** Create. **Step 2:** Execute. **Step 3: Verify** — a moderate `c` gives the most correct runs; the far start state converges to `π*` as iterations grow. **Step 4: Commit** `"M4: MCTS Advanced Ungraded Codio activity"`.

---

## Phase 6 — Wrap-up

### Task 6.1: Lecture pointer + spec/plan status

**Files:**
- Modify: `CHEME-145-M4-Introduction-MarkovDecisionProcess-Read-Pages.ipynb` (append one line to the closing/summary), `CHEME-145-M4-build-spec.md` (mark built)

- [ ] **Step 1:** In the lecture's Summary (before Additional Resources), add one sentence: exact dynamic programming (value/policy iteration) becomes intractable for very large state spaces; the optional Advanced notebooks cover two approximate planners — random rollout and Monte Carlo tree search.
- [ ] **Step 2:** Re-execute the lecture notebook (shared procedure) to confirm it still runs clean (it has no code, so this is a formatting check) — or skip if markdown-only.
- [ ] **Step 3: Commit** `"M4: lecture pointer to the Advanced (rollout/MCTS) track"`.

### Task 6.2: Full-suite verification

- [ ] **Step 1:** Run the library test suite: `julia --project=. test/runtests.jl` → all testsets PASS.
- [ ] **Step 2:** Re-execute every notebook headless (shared procedure; student graded notebook with `--allow-errors`). Confirm each exits 0.
- [ ] **Step 3:** `git status` clean; the module directory contains `Project.toml`, `Manifest.toml`, `Include.jl`, `src/{Types,Factory,Compute}.jl`, `test/runtests.jl`, and the 10 notebooks.

---

## Self-Review

**Spec coverage:**
- Self-contained local `src/` (no VL package) → Tasks 0.1–0.5, 2.0, 3.0, 4.0, 5.0. ✓
- Notation `V/Q/π/γ/‖·‖∞` → enforced in Global Constraints and all code. ✓
- Value-iteration Demo + Ungraded (gridworld) → Tasks 1.1, 1.2. ✓
- Policy-iteration Demo + Ungraded (inventory) → Tasks 2.1, 2.2. ✓
- Graded capstone + Solution (equipment replacement, VI+PI, recommendation) → Tasks 3.1, 3.2. ✓
- Advanced rollout Demo + Ungraded (gridworld) → Tasks 4.1, 4.2. ✓
- Advanced MCTS Demo + Ungraded (gridworld) → Tasks 5.1, 5.2. ✓
- Notebooks executed with real outputs → shared procedure + per-task execute steps. ✓
- VI == PI regression + rollout≈V* + MCTS==VI checks → Tasks 0.4, 2.0, 3.0, 4.0, 5.0. ✓
- Optional lecture pointer → Task 6.1. ✓

**Placeholder scan:** the only `# TODO`/`error(...)` stubs are the *intended* graded blanks in Task 3.2 (a deliverable requirement, not a plan gap); every library function is fully implemented in-plan.

**Type consistency:** `MyMDPProblemModel`, `MyRectangularGridWorldModel`, `MyValueIterationModel`, `MyPolicyIterationModel`, `MyValueFunctionPolicy`, `MyMCTSModel`; functions `build`, `build_mdp`, `build_inventory_mdp`, `build_replacement_mdp`, `solve`, `Q`, `policy`, `policy_evaluation`, `iterative_policy_evaluation`, `simulate_return`, `rollout_value`, `mcts` — names/signatures identical across defining tasks and notebook call sites. Solution value field is `V` everywhere; the policy is always recovered via `policy(Q(problem, V))`.
