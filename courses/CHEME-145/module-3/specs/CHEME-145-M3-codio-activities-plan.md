# CHEME-145 Module 3 Codio Activities — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the module-3 HMM `src/` library plus three notebooks: a Baum-Welch ungraded Codio activity, a dishonest-casino graded Solution, and its student version.

**Architecture:** Self-contained module-3 `src/` (Types → Factory → Compute) in the module-4 style, exposing `build`/`solve` plus `simulate`, `loglikelihood` (scaled forward), `viterbi`, and `align_states`. Notebooks are authored by Python `nbformat` builder scripts and executed headless with `nbconvert`. The student graded notebook is generated from the executed Solution by swapping three assessed cells for `error("TODO: ...")` stubs.

**Tech Stack:** Julia 1.12 (kernel `julia-1.12`), Distributions.jl, Plots.jl, PrettyTables.jl v3, Python 3 + `nbformat` + `jupyter nbconvert`.

**Spec:** `courses/CHEME-145/module-3/specs/CHEME-145-M3-codio-activities-spec.md` (approved 2026-07-06).

## Global Constraints

- Working directory for all commands: `/Users/jdv27/Desktop/julia_work/CHEME-140-eCornell-Repository/courses/CHEME-145/module-3` (call it `$M3`).
- Builder scripts live in `$BUILDERS` = `/private/tmp/claude-502/-Users-jdv27-Desktop-julia-work-CHEME-140-eCornell-Repository-courses-CHEME-145/e4aaeeba-b158-40ca-bb46-2199071ded89/scratchpad/m3-builders` (throwaway; any temp dir works — they are NOT committed, matching module-4).
- Keep `JSON = "0.21"` in `Project.toml` `[compat]`. Never commit `Manifest.toml` (gitignored repo-wide).
- Do NOT modify `CHEME-145-M3-Example-HiddenMarkovModels-Watch-Demo.ipynb` or `CHEME-145-M3-Lecture-HiddenMarkovModel-Read-Page.ipynb`.
- All randomness through explicit `Xoshiro(seed)` RNGs. Canonical seeds: data `Xoshiro(2026)`, fits `Xoshiro(11)`, tests `Xoshiro(42)`/`Xoshiro(7)`/`Xoshiro(123)`. If a numerical acceptance check fails for a seed, change that seed, re-run, and re-verify — outputs must be deterministic for whatever seed ships.
- Learning Objectives and Key Takeaways: exactly 3 items each, format `* __Title:__ description`, direct/simple/concise language, no unnecessary adjectives, all claims supported by notebook content (CLAUDE.md standards). Norms always explicit (`$\|\cdot\|_{F}$`).
- PrettyTables v3: use `column_labels =` (not `header=`) and pass a `Matrix` (not a `Vector{Tuple}`).
- Notebook execution command (from `$M3`): `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=julia-1.12 --ExecutePreprocessor.timeout=600 <nb>` — add `--allow-errors` ONLY for the student graded notebook.
- Julia test command (from `$M3`): `julia --project=. test/runtests.jl`.
- Commit messages: `M3: <what>` + blank line + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Environment, Include.jl, Types.jl, Factory.jl

**Files:**
- Modify: `$M3/Project.toml`
- Modify: `$M3/Include.jl`
- Create: `$M3/src/Types.jl`
- Create: `$M3/src/Factory.jl`
- Create: `$M3/test/runtests.jl`

**Interfaces:**
- Produces: `MyHiddenMarkovModel` (fields `P::Array{Float64,2}`, `E::Array{Float64,2}`, `π₀::Array{Float64,1}`), `MyBaumWelchModel` (fields `maxiterations::Int64`, `ϵ::Float64`), `MyBaumWelchSolution` (fields `P::Array{Float64,2}`, `E::Array{Float64,2}`, `π₀::Array{Float64,1}`, `loglikelihood_history::Array{Float64,1}`, `iterations::Int64`), `build(::Type{T}, data::NamedTuple)` for the first two. All later tasks consume these.

- [ ] **Step 1: Preflight checks**

Run (each must succeed):
```bash
cd /Users/jdv27/Desktop/julia_work/CHEME-140-eCornell-Repository/courses/CHEME-145/module-3
julia --version                      # expect 1.12.x
jupyter kernelspec list | grep julia-1.12
python3 -c "import nbformat; print(nbformat.__version__)"
ls Manifest.toml                     # if MISSING: julia --project=. -e 'include("Include.jl")' to bootstrap (adds VLDataScienceMachineLearningPackage by URL)
mkdir -p /private/tmp/claude-502/-Users-jdv27-Desktop-julia-work-CHEME-140-eCornell-Repository-courses-CHEME-145/e4aaeeba-b158-40ca-bb46-2199071ded89/scratchpad/m3-builders
```

- [ ] **Step 2: Add Random and Test to Project.toml [deps]**

In `$M3/Project.toml`, add to `[deps]` (keep every existing line, including the `[compat]` `JSON = "0.21"` pin):
```toml
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
```
Then run: `julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'` — expect no errors.

- [ ] **Step 3: Write the failing test file**

Create `$M3/test/runtests.jl` (Compute.jl include is added in Task 2):
```julia
using Test
using LinearAlgebra
using Random
using Distributions

include(joinpath(@__DIR__, "..", "src", "Types.jl"));
include(joinpath(@__DIR__, "..", "src", "Factory.jl"));

@testset "factory validation" begin
    P = [0.9 0.1; 0.2 0.8];
    E = [0.7 0.3; 0.1 0.9];
    π₀ = [0.5, 0.5];
    hmm = build(MyHiddenMarkovModel, (P = P, E = E, π₀ = π₀));
    @test hmm.P == P && hmm.E == E && hmm.π₀ == π₀;

    @test_throws ArgumentError build(MyHiddenMarkovModel, (P = [0.9 0.2; 0.2 0.8], E = E, π₀ = π₀)); # row of P does not sum to 1
    @test_throws ArgumentError build(MyHiddenMarkovModel, (P = P, E = [1.1 -0.1; 0.1 0.9], π₀ = π₀)); # negative entry in E
    @test_throws ArgumentError build(MyHiddenMarkovModel, (P = P, E = E, π₀ = [0.5, 0.25, 0.25]));   # π₀ wrong length
    @test_throws ArgumentError build(MyHiddenMarkovModel, (P = P, E = E, π₀ = [0.7, 0.7]));           # π₀ does not sum to 1

    solver = build(MyBaumWelchModel, (maxiterations = 10, ϵ = 1e-3));
    @test solver.maxiterations == 10 && solver.ϵ == 1e-3;
end
```

- [ ] **Step 4: Run test to verify it fails**

Run: `julia --project=. test/runtests.jl`
Expected: FAIL — `SystemError: opening file .../src/Types.jl` (file does not exist).

- [ ] **Step 5: Create src/Types.jl**

```julia
# ---------------------------------------------------------------------------------------------- #
# Abstract types. Hidden Markov models and solution (algorithm) models.
# ---------------------------------------------------------------------------------------------- #
abstract type AbstractHiddenMarkovModel end
abstract type AbstractSolutionModel end

"""
    mutable struct MyHiddenMarkovModel <: AbstractHiddenMarkovModel

A discrete Hidden Markov Model ℋ = (P, E, π₀).

### Fields
- `P::Array{Float64,2}`: transition matrix, `P[i,j] = P(sₜ₊₁ = j | sₜ = i)`. Rows sum to 1.
- `E::Array{Float64,2}`: emission matrix, `E[i,o] = P(oₜ = o | sₜ = i)`. Rows sum to 1.
- `π₀::Array{Float64,1}`: initial hidden-state distribution. Sums to 1.
"""
mutable struct MyHiddenMarkovModel <: AbstractHiddenMarkovModel
    P::Array{Float64,2}
    E::Array{Float64,2}
    π₀::Array{Float64,1}
    MyHiddenMarkovModel() = new();
end

"""
    mutable struct MyBaumWelchModel <: AbstractSolutionModel

Settings for the Baum-Welch (EM) learning algorithm.

### Fields
- `maxiterations::Int64`: maximum number of EM iterations K_max.
- `ϵ::Float64`: convergence tolerance on |ℒₖ - ℒₖ₋₁|.
"""
mutable struct MyBaumWelchModel <: AbstractSolutionModel
    maxiterations::Int64
    ϵ::Float64
    MyBaumWelchModel() = new();
end

"""
    mutable struct MyBaumWelchSolution

Learned parameters and convergence history returned by `solve(::MyBaumWelchModel, ...)`.

### Fields
- `P::Array{Float64,2}`: learned transition matrix.
- `E::Array{Float64,2}`: learned emission matrix.
- `π₀::Array{Float64,1}`: learned initial distribution.
- `loglikelihood_history::Array{Float64,1}`: total log-likelihood ℒₖ at each iteration.
- `iterations::Int64`: number of EM iterations performed.
"""
mutable struct MyBaumWelchSolution
    P::Array{Float64,2}
    E::Array{Float64,2}
    π₀::Array{Float64,1}
    loglikelihood_history::Array{Float64,1}
    iterations::Int64
    MyBaumWelchSolution() = new();
end
```

- [ ] **Step 6: Create src/Factory.jl**

```julia
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
```

- [ ] **Step 7: Run test to verify it passes**

Run: `julia --project=. test/runtests.jl`
Expected: PASS (`Test Summary: | Pass ... factory validation`).

- [ ] **Step 8: Rewrite Include.jl**

Replace the full contents of `$M3/Include.jl` with (module-4 pattern; keeps VLDataScienceMachineLearningPackage for the existing demo, adds `Random`, the Paul Tol palette, and the `src/` includes):
```julia
# setup paths -
const _ROOT = @__DIR__;
const _PATH_TO_SRC = joinpath(_ROOT, "src");

# activate the local environment; instantiate it the first time (no Manifest.toml yet) -
using Pkg
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.add(path="https://github.com/varnerlab/VLDataScienceMachineLearningPackage.jl.git");
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load the required packages -
using VLDataScienceMachineLearningPackage
using Distributions
using Plots
using Colors
using LinearAlgebra
using Statistics
using DataFrames
using PrettyTables
using Random

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
NOTE: `src/Compute.jl` does not exist until Task 2. Create it now as an empty file so Include.jl works:
```bash
touch src/Compute.jl
```
Sanity check (must print `MyHiddenMarkovModel` and no warnings/errors):
```bash
julia --project=. -e 'include("Include.jl"); println(typeof(build(MyHiddenMarkovModel, (P = [0.9 0.1; 0.2 0.8], E = [0.7 0.3; 0.1 0.9], π₀ = [0.5, 0.5]))))'
```
This also confirms our `build` wins over any name exported by VLDataScienceMachineLearningPackage (packages load before `src/`, so our definitions shadow silently).

- [ ] **Step 9: Commit**

```bash
git add Project.toml Include.jl src/Types.jl src/Factory.jl src/Compute.jl test/runtests.jl
git commit -m "M3: add HMM types, factory with validation, env + Include.jl src wiring

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: simulate (forward sampling)

**Files:**
- Modify: `$M3/src/Compute.jl`
- Modify: `$M3/test/runtests.jl`

**Interfaces:**
- Consumes: `MyHiddenMarkovModel`, `build` (Task 1).
- Produces: `simulate(hmm::MyHiddenMarkovModel, T::Int; N::Int = 1, rng::AbstractRNG = Random.default_rng())::Array{NamedTuple,1}` — each element `(hidden::Array{Int64,1}, observed::Array{Int64,1})`, both length `T`. Notebooks call `simulate(hmm, T; N = ..., rng = ...)` and take `.observed` / `.hidden`.

- [ ] **Step 1: Add the Compute.jl include and the failing testset**

In `$M3/test/runtests.jl`, after the `Factory.jl` include line, add:
```julia
include(joinpath(@__DIR__, "..", "src", "Compute.jl"));
```
At the end of the file, add:
```julia
@testset "simulate" begin
    P = [0.9 0.1; 0.2 0.8];
    E = [0.7 0.3; 0.1 0.9];
    π₀ = [0.5, 0.5];
    hmm = build(MyHiddenMarkovModel, (P = P, E = E, π₀ = π₀));

    sequences = simulate(hmm, 50; N = 4, rng = Xoshiro(123));
    @test length(sequences) == 4;
    @test all(length(s.hidden) == 50 && length(s.observed) == 50 for s in sequences);
    @test all(all(1 .≤ s.hidden .≤ 2) && all(1 .≤ s.observed .≤ 2) for s in sequences);

    # deterministic under a fixed seed -
    a = simulate(hmm, 50; N = 4, rng = Xoshiro(123));
    b = simulate(hmm, 50; N = 4, rng = Xoshiro(123));
    @test a == b;
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/runtests.jl`
Expected: FAIL — `UndefVarError: simulate not defined`.

- [ ] **Step 3: Implement simulate in src/Compute.jl**

Replace the (empty) `$M3/src/Compute.jl` with:
```julia
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/runtests.jl`
Expected: PASS (factory validation + simulate testsets).

- [ ] **Step 5: Commit**

```bash
git add src/Compute.jl test/runtests.jl
git commit -m "M3: add HMM forward-sampling simulate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Scaled forward algorithm + loglikelihood

**Files:**
- Modify: `$M3/src/Compute.jl`
- Modify: `$M3/test/runtests.jl`

**Interfaces:**
- Consumes: `MyHiddenMarkovModel` (Task 1).
- Produces: `_forward(hmm, observed::Array{Int64,1})` → `(α̂::Array{Float64,2}, c::Array{Float64,1})` (T×S scaled forward variables, T scaling constants; each row of `α̂` sums to 1); `loglikelihood(hmm::MyHiddenMarkovModel, observed::Array{Int64,1})::Float64` = `sum(log.(c))`, extending the `Distributions.loglikelihood` generic. Task 6 consumes `_forward`; notebooks consume `loglikelihood`.

- [ ] **Step 1: Add the failing testset**

Append to `$M3/test/runtests.jl`:
```julia
@testset "forward loglikelihood" begin
    P = [0.7 0.3; 0.4 0.6];
    E = [0.9 0.1; 0.2 0.8];
    π₀ = [0.6, 0.4];
    hmm = build(MyHiddenMarkovModel, (P = P, E = E, π₀ = π₀));
    o = [1, 2, 2, 1];

    # brute force: P(o) = Σ over all 2⁴ hidden paths of P(path, o) -
    total = 0.0;
    for path ∈ Iterators.product(fill(1:2, 4)...)
        p = π₀[path[1]]*E[path[1], o[1]];
        for t ∈ 2:4
            p *= P[path[t-1], path[t]]*E[path[t], o[t]];
        end
        total += p;
    end
    @test isapprox(loglikelihood(hmm, o), log(total); atol = 1e-12);

    # scaled forward variables: each row of α̂ sums to 1 -
    (α̂, c) = _forward(hmm, o);
    @test size(α̂) == (4, 2) && length(c) == 4;
    @test all(isapprox.(sum(α̂, dims = 2), 1.0; atol = 1e-12));
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/runtests.jl`
Expected: FAIL — `MethodError` / `UndefVarError: _forward not defined`.

- [ ] **Step 3: Implement _forward and loglikelihood**

Append to `$M3/src/Compute.jl`:
```julia
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/runtests.jl`
Expected: PASS (3 testsets).

- [ ] **Step 5: Commit**

```bash
git add src/Compute.jl test/runtests.jl
git commit -m "M3: add scaled forward algorithm and loglikelihood method

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Viterbi decoding

**Files:**
- Modify: `$M3/src/Compute.jl`
- Modify: `$M3/test/runtests.jl`

**Interfaces:**
- Consumes: `MyHiddenMarkovModel` (Task 1).
- Produces: `viterbi(hmm::MyHiddenMarkovModel, observed::Array{Int64,1})::Array{Int64,1}` — most likely hidden path, log-space with backtrace. The graded notebook consumes this.

- [ ] **Step 1: Add the failing testset**

Append to `$M3/test/runtests.jl`:
```julia
@testset "viterbi" begin
    P = [0.7 0.3; 0.4 0.6];
    E = [0.9 0.1; 0.2 0.8];
    π₀ = [0.6, 0.4];
    hmm = build(MyHiddenMarkovModel, (P = P, E = E, π₀ = π₀));
    o = [1, 2, 2, 1];

    # brute force: argmax over all 2⁴ hidden paths of P(path, o) -
    best_p = -Inf; best_path = Int64[];
    for path ∈ Iterators.product(fill(1:2, 4)...)
        p = π₀[path[1]]*E[path[1], o[1]];
        for t ∈ 2:4
            p *= P[path[t-1], path[t]]*E[path[t], o[t]];
        end
        if (p > best_p)
            best_p = p; best_path = collect(path);
        end
    end
    @test viterbi(hmm, o) == best_path;
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/runtests.jl`
Expected: FAIL — `UndefVarError: viterbi not defined`.

- [ ] **Step 3: Implement viterbi**

Append to `$M3/src/Compute.jl`:
```julia
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/runtests.jl`
Expected: PASS (4 testsets).

- [ ] **Step 5: Commit**

```bash
git add src/Compute.jl test/runtests.jl
git commit -m "M3: add log-space Viterbi decoding

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: align_states (label-switching helper)

**Files:**
- Modify: `$M3/src/Compute.jl`
- Modify: `$M3/test/runtests.jl`

**Interfaces:**
- Consumes: nothing new (pure matrix function; uses `norm` from LinearAlgebra).
- Produces: `align_states(P̂, Ê, P, E)` → NamedTuple `(σ::Array{Int64,1}, P::Array{Float64,2}, E::Array{Float64,2})`: the permutation of learned state labels minimizing `norm(P̂[σ,σ] - P) + norm(Ê[σ,:] - E)` (Frobenius), and the re-labeled matrices. Notebooks destructure positionally: `(σ, P̂, Ê) = align_states(...)`. Task 6's test consumes this.

- [ ] **Step 1: Add the failing testset**

Append to `$M3/test/runtests.jl`:
```julia
@testset "align_states" begin
    P = [0.8 0.15 0.05; 0.1 0.7 0.2; 0.25 0.25 0.5];
    E = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9];
    σ_true = [2, 3, 1];
    aligned = align_states(P[σ_true, σ_true], E[σ_true, :], P, E);
    @test isapprox(aligned.P, P; atol = 1e-12);
    @test isapprox(aligned.E, E; atol = 1e-12);
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/runtests.jl`
Expected: FAIL — `UndefVarError: align_states not defined`.

- [ ] **Step 3: Implement align_states**

Append to `$M3/src/Compute.jl`:
```julia
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/runtests.jl`
Expected: PASS (5 testsets).

- [ ] **Step 5: Commit**

```bash
git add src/Compute.jl test/runtests.jl
git commit -m "M3: add align_states permutation helper for label switching

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Baum-Welch solve

**Files:**
- Modify: `$M3/src/Compute.jl`
- Modify: `$M3/test/runtests.jl`

**Interfaces:**
- Consumes: `_forward` (Task 3), `build`, `MyBaumWelchModel`, `MyBaumWelchSolution` (Task 1), `simulate` (Task 2, test only), `align_states` (Task 5, test only).
- Produces: `_backward(hmm, observed)::Array{Float64,2}` (T×S scaled backward variables) and `solve(model::MyBaumWelchModel, sequences::Array{Array{Int64,1},1}; number_of_hidden_states::Int, number_of_observable_states::Int, rng::AbstractRNG = Random.default_rng())::MyBaumWelchSolution`. The ungraded notebook consumes `solve`.

- [ ] **Step 1: Add the failing testset**

Append to `$M3/test/runtests.jl`:
```julia
@testset "baum-welch" begin
    P_true = [0.9 0.1; 0.2 0.8];
    E_true = [0.95 0.05; 0.05 0.95];
    π₀_true = [0.5, 0.5];
    hmm_true = build(MyHiddenMarkovModel, (P = P_true, E = E_true, π₀ = π₀_true));

    sequences = [s.observed for s ∈ simulate(hmm_true, 100; N = 100, rng = Xoshiro(42))];
    solver = build(MyBaumWelchModel, (maxiterations = 200, ϵ = 1e-8));
    result = solve(solver, sequences;
        number_of_hidden_states = 2, number_of_observable_states = 2, rng = Xoshiro(7));

    # EM guarantee: the log-likelihood does not decrease -
    @test all(diff(result.loglikelihood_history) .≥ -1e-6);

    # learned parameters are row-stochastic -
    @test all(isapprox.(sum(result.P, dims = 2), 1.0; atol = 1e-8));
    @test all(isapprox.(sum(result.E, dims = 2), 1.0; atol = 1e-8));
    @test isapprox(sum(result.π₀), 1.0; atol = 1e-8);

    # recovers the true parameters after alignment (easy problem: near-deterministic emissions) -
    aligned = align_states(result.P, result.E, P_true, E_true);
    @test norm(aligned.P - P_true) < 0.15;
    @test norm(aligned.E - E_true) < 0.15;
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `julia --project=. test/runtests.jl`
Expected: FAIL — `UndefVarError: solve not defined`.

- [ ] **Step 3: Implement _backward and solve**

Append to `$M3/src/Compute.jl`:
```julia
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

        # convergence check -
        if (abs(ℒ - ℒ_prev) < ϵ || k ≥ Kmax)
            converged = true;
        end
        ℒ_prev = ℒ; k += 1;
    end

    # package and return -
    solution = MyBaumWelchSolution();
    solution.P = P; solution.E = E; solution.π₀ = π₀;
    solution.loglikelihood_history = loglikelihood_history;
    solution.iterations = k;
    return solution;
end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `julia --project=. test/runtests.jl`
Expected: PASS (6 testsets). If the recovery assertions fail, change the test seeds (`Xoshiro(42)`/`Xoshiro(7)`) once and re-run; keep whatever seed passes deterministically.

- [ ] **Step 5: Commit**

```bash
git add src/Compute.jl test/runtests.jl
git commit -m "M3: add multi-sequence Baum-Welch solve with scaled forward/backward

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Ungraded Baum-Welch notebook

**Files:**
- Create: `$BUILDERS/build_ungraded.py` (not committed)
- Create: `$M3/CHEME-145-M3-Example-HMM-BaumWelch-Ungraded-Codio-Activity.ipynb`

**Interfaces:**
- Consumes: everything from Tasks 1–6 via `Include.jl` (`build`, `simulate`, `solve`, `align_states`, `colors`, `Xoshiro`, `norm`, `pretty_table`).
- Produces: the executed ungraded notebook (later tasks do not depend on it).

- [ ] **Step 1: Write the builder script**

Create `$BUILDERS/build_ungraded.py` with exactly this content:
```python
import nbformat as nbf

NB_PATH = "/Users/jdv27/Desktop/julia_work/CHEME-140-eCornell-Repository/courses/CHEME-145/module-3/CHEME-145-M3-Example-HMM-BaumWelch-Ungraded-Codio-Activity.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Julia 1.12.6", "language": "julia", "name": "julia-1.12"}
nb.metadata["language_info"] = {"name": "julia", "version": "1.12.6"}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Activity: Learning Hidden Markov Model Parameters with Baum-Welch
In the Watch-Demo, we built a three-state Hidden Markov Model (HMM) with hidden mood states and observable emoji outputs, and simulated it with known parameters. This activity reverses the question: given only observation sequences generated by that model, can we recover the transition matrix $\mathbf{P}$ and the emission matrix $\mathbf{E}$? We generate synthetic observations from the demo model, fit an HMM with the Baum-Welch algorithm from a random initialization, compare the learned parameters to the truth, and measure how data size and emission noise change the quality of the fit.

> __Learning Objectives:__
>
> After completing this activity, students will be able to:
> * __Generate synthetic observation data from a known HMM:__ use forward sampling to produce observation sequences from the demo's three-state mood model, keeping only the observations for fitting.
> * __Fit HMM parameters with Baum-Welch:__ run the expectation-maximization iteration from a random initialization and verify that the log-likelihood $\mathcal{L}_{k}$ does not decrease between iterations.
> * __Evaluate parameter recovery:__ align the learned states to the true states, measure the errors $\|\hat{\mathbf{P}}-\mathbf{P}\|_{F}$ and $\|\hat{\mathbf{E}}-\mathbf{E}\|_{F}$, and quantify how the number of training sequences and the emission noise change these errors.

Let's get started.
___
""")

md(r"""
## Theory refresher: the learning problem
The Baum-Welch algorithm solves the learning problem: given observation sequences $\mathbf{o}^{(1)},\ldots,\mathbf{o}^{(N)}$ and the sizes of the state spaces, estimate the parameters $(\mathbf{P},\mathbf{E},\pi_{0})$ that best explain the data. It is an instance of the expectation-maximization (EM) algorithm:
* __E-step:__ with the current parameters, compute the state posterior $\gamma_{t}(i)$ and the transition posterior $\xi_{t}(i,j)$ for every sequence from the forward variables $\alpha_{t}(i)$ and backward variables $\beta_{t}(i)$.
* __M-step:__ re-estimate $\pi_{0}$, $\mathbf{P}$, and $\mathbf{E}$ from the expected state occupancies and transition counts.

Each iteration does not decrease the log-likelihood $\mathcal{L}_{k}$, and the iteration stops when $|\mathcal{L}_{k}-\mathcal{L}_{k-1}|<\epsilon$ or $k\geq{K}_{\text{max}}$. Baum-Welch converges to a local maximum of the likelihood; the result depends on the initialization. For the full development of $\alpha$, $\beta$, $\gamma$, and $\xi$, see the lecture notebook `CHEME-145-M3-Lecture-HiddenMarkovModel-Read-Page.ipynb`.
___
""")

md(r"""
## Setup, Data, and Prerequisites
First, we set up the computational environment by including the `Include.jl` file. The `Include.jl` file sets paths, loads required external packages, and includes the module's local source files (`src/Types.jl`, `src/Factory.jl`, `src/Compute.jl`) holding the types and algorithms used in this activity.
""")

code(r"""
include(joinpath(@__DIR__, "Include.jl")); # include the Include.jl file
""")

md(r"""
## Task 1: Build the true model
We use the same three-state mood model as the Watch-Demo. The hidden states are $\mathcal{S}\equiv\left\{\text{happy},\text{neutral},\text{sad}\right\}$, and the observable outputs are the emojis $\mathcal{O}\equiv\left\{😄,😐,😞\right\}$. The transition matrix $\mathbf{P}\in\mathbb{R}^{3\times{3}}$ and the emission matrix $\mathbf{E}\in\mathbb{R}^{3\times{3}}$ take the demo values, and we use a uniform initial distribution $\pi_{0}=(1/3,1/3,1/3)$. The `build(...)` factory method checks that the rows of $\mathbf{P}$ and $\mathbf{E}$ and the vector $\pi_{0}$ are valid probability distributions.
""")

code(r"""
number_of_hidden_states = 3; # how many hidden states do we have?
number_of_observable_states = 3; # how many observable states do we have?

P_true = [
    0.05 0.95 0.0 ; # moves for state 1 = happy
    0.6 0.2 0.2 ; # moves for state 2 = neutral
    0.0 0.3 0.7 ; # moves for state 3 = sad
];

E_true = [
    0.90 0.05 0.05 ; # 1 happy (but sometimes we see other faces)
    0.05 0.90 0.05 ; # 2 neutral (but sometimes we see other faces)
    0.05 0.05 0.90 ; # 3 sad (but sometimes we see other faces)
];

π₀_true = [1/3, 1/3, 1/3]; # uniform initial state distribution

hmm_true = build(MyHiddenMarkovModel, (P = P_true, E = E_true, π₀ = π₀_true));
""")

md(r"""
## Task 2: Generate synthetic observations
We generate `number_of_sequences = 50` observation sequences, each of length `sequence_length = 200`, by forward sampling from the true model with the `simulate(...)` method. The sampler returns both the hidden states and the observations for each sequence, but we keep __only the observations__: the fitter never sees the hidden states. We seed the random number generator so the notebook reproduces exactly.
""")

code(r"""
number_of_sequences = 50; # how many observation sequences do we generate?
sequence_length = 200; # length T of each sequence
rng_data = Xoshiro(2026); # seeded random number generator (reproducibility)

training_data = simulate(hmm_true, sequence_length; N = number_of_sequences, rng = rng_data);
observation_sequences = [sequence.observed for sequence ∈ training_data]; # keep only the observations
""")

md(r"""
Let's look at the first 20 observations of the first sequence as emojis:
""")

code(r"""
observable_emoji_map = Dict{Int,String}(1 => "😄", 2 => "😐", 3 => "😞");
[observable_emoji_map[o] for o ∈ observation_sequences[1][1:20]] |> v -> join(v, " ") |> println;
""")

md(r"""
## Task 3: Fit the model with Baum-Welch
We build a `MyBaumWelchModel` solver with `maxiterations = 500` and tolerance $\epsilon=10^{-6}$, and call `solve(...)` on the observation sequences. The solver initializes $(\mathbf{P},\mathbf{E},\pi_{0})$ as random row-stochastic matrices drawn from a seeded random number generator, then alternates the E-step and M-step until $|\mathcal{L}_{k}-\mathcal{L}_{k-1}|<\epsilon$ or the iteration limit is reached.
""")

code(r"""
rng_fit = Xoshiro(11); # seeds the random initialization
solver = build(MyBaumWelchModel, (maxiterations = 500, ϵ = 1e-6));
result = solve(solver, observation_sequences;
    number_of_hidden_states = number_of_hidden_states,
    number_of_observable_states = number_of_observable_states, rng = rng_fit);
println("Converged after $(result.iterations) iterations. Final log-likelihood ℒ = $(round(result.loglikelihood_history[end], digits = 2))");
""")

md(r"""
### Check: Does the log-likelihood increase at every iteration?
The EM guarantee from the lecture states that each Baum-Welch iteration does not decrease the log-likelihood $\mathcal{L}_{k}$. Let's verify this on the fitted history using the [@assert macro](https://docs.julialang.org/en/v1/base/base/#Base.@assert), then plot $\mathcal{L}_{k}$ versus the iteration count $k$:
""")

code(r"""
let
    ℒ = result.loglikelihood_history;
    @assert all(diff(ℒ) .≥ -1e-6) # non-decreasing (up to numerical roundoff)
    plot(1:length(ℒ), ℒ, lw = 2, c = colors[1], label = "ℒₖ",
        xlabel = "Baum-Welch iteration k", ylabel = "log-likelihood ℒₖ")
end
""")

md(r"""
## Task 4: Compare the learned model to the truth
Baum-Welch never sees the state labels, so the learned states come back in arbitrary order: the fitter may call "happy" state 3. Before comparing to the truth, we align the learned states to the true states with the `align_states(...)` method, which searches all permutations of the state labels for the one minimizing $\|\hat{\mathbf{P}}-\mathbf{P}\|_{F}+\|\hat{\mathbf{E}}-\mathbf{E}\|_{F}$, where $\|\cdot\|_{F}$ denotes the Frobenius norm.
""")

code(r"""
(σ, P̂, Ê) = align_states(result.P, result.E, P_true, E_true);
println("state alignment σ = $(σ)");
println("‖P̂ - P‖F = $(round(norm(P̂ - P_true), digits = 4))");
println("‖Ê - E‖F = $(round(norm(Ê - E_true), digits = 4))");
""")

md(r"""
Let's put the aligned learned parameters next to the true values. Each row of the tables below is a hidden state; learned values wear a hat:
""")

code(r"""
let
    S = number_of_hidden_states;
    state_names = ["happy", "neutral", "sad"];

    data_P = Matrix{Any}(undef, S, 1 + 2*S);
    data_E = Matrix{Any}(undef, S, 1 + 2*S);
    for i ∈ 1:S
        data_P[i, 1] = state_names[i];
        data_E[i, 1] = state_names[i];
        for j ∈ 1:S
            data_P[i, 1 + j] = round(P̂[i, j], digits = 3);
            data_P[i, 1 + S + j] = round(P_true[i, j], digits = 3);
            data_E[i, 1 + j] = round(Ê[i, j], digits = 3);
            data_E[i, 1 + S + j] = round(E_true[i, j], digits = 3);
        end
    end
    pretty_table(data_P; column_labels = ["state", "p̂ᵢ₁", "p̂ᵢ₂", "p̂ᵢ₃", "pᵢ₁", "pᵢ₂", "pᵢ₃"]);
    pretty_table(data_E; column_labels = ["state", "êᵢ₁", "êᵢ₂", "êᵢ₃", "eᵢ₁", "eᵢ₂", "eᵢ₃"]);
end
""")

md(r"""
## Experiment 1: Data size
How much data does Baum-Welch need? Holding the model and the sequence length $T=200$ fixed, we regenerate a fresh training set with $N\in\{2,10,50,200\}$ sequences, refit, realign, and print the parameter errors. More data gives better estimates of the expected transition and emission counts, so both errors shrink as $N$ grows.

> __Try it yourself:__ change the values in `N_try` below (or change `sequence_length` above and re-run the notebook) and watch how the errors respond.
""")

code(r"""
let
    for N_try ∈ (2, 10, 50, 200)
        sequences_try = [s.observed for s ∈ simulate(hmm_true, sequence_length; N = N_try, rng = Xoshiro(2026))];
        result_try = solve(solver, sequences_try;
            number_of_hidden_states = number_of_hidden_states,
            number_of_observable_states = number_of_observable_states, rng = Xoshiro(11));
        (_, P̂_try, Ê_try) = align_states(result_try.P, result_try.E, P_true, E_true);
        println("N = $(lpad(N_try, 3)):  ‖P̂ - P‖F = $(round(norm(P̂_try - P_true), digits = 4)),  ‖Ê - E‖F = $(round(norm(Ê_try - E_true), digits = 4))");
    end
end
""")

md(r"""
## Experiment 2: Emission noise
What happens when the observations carry less information about the hidden state? We rebuild the true model with emission diagonal $d\in\{0.9,0.7,0.5\}$ and off-diagonal entries $(1-d)/2$, regenerate $N=50$ training sequences from each model, refit, and print the parameter errors. As $d$ falls toward $1/3$, the emission rows approach the uniform distribution, the observations say less about the hidden mood, and recovery degrades. This is what "hidden" means: the harder the states are to see through the observations, the harder they are to learn.

> __Try it yourself:__ set $d = 0.4$ and compare the errors. What do you expect at exactly $d = 1/3$?
""")

code(r"""
let
    for d ∈ (0.9, 0.7, 0.5)
        off = (1 - d)/2;
        E_noisy = [
            d off off ;
            off d off ;
            off off d ;
        ];
        hmm_noisy = build(MyHiddenMarkovModel, (P = P_true, E = E_noisy, π₀ = π₀_true));
        sequences_noisy = [s.observed for s ∈ simulate(hmm_noisy, sequence_length; N = 50, rng = Xoshiro(2026))];
        result_noisy = solve(solver, sequences_noisy;
            number_of_hidden_states = number_of_hidden_states,
            number_of_observable_states = number_of_observable_states, rng = Xoshiro(11));
        (_, P̂_noisy, Ê_noisy) = align_states(result_noisy.P, result_noisy.E, P_true, E_noisy);
        println("d = $(d):  ‖P̂ - P‖F = $(round(norm(P̂_noisy - P_true), digits = 4)),  ‖Ê - E‖F = $(round(norm(Ê_noisy - E_noisy), digits = 4))");
    end
end
""")

md(r"""
___
## Summary
In this activity, we generated synthetic observation sequences from the three-state mood model of the Watch-Demo and used the Baum-Welch algorithm to recover the transition matrix $\mathbf{P}$ and the emission matrix $\mathbf{E}$ from the observations alone.

> __Key Takeaways:__
>
> * __Baum-Welch learns HMM parameters from observations alone:__ starting from a random initialization, the EM iteration recovered $\mathbf{P}$ and $\mathbf{E}$ from emoji sequences without access to the hidden mood states, up to a permutation of the state labels.
> * __The log-likelihood does not decrease between iterations:__ the fitted history $\mathcal{L}_{k}$ is non-decreasing, consistent with the EM convergence guarantee from the lecture, and the iteration stops at a local maximum of the likelihood.
> * __Recovery quality depends on data size and emission noise:__ the errors $\|\hat{\mathbf{P}}-\mathbf{P}\|_{F}$ and $\|\hat{\mathbf{E}}-\mathbf{E}\|_{F}$ fall as the number of training sequences grows (Experiment 1) and rise as the emission matrix approaches the uniform distribution (Experiment 2).

Together with the evaluation and decoding problems, the learning problem completes the three classic HMM problems from the lecture.

### Additional Resources
* Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.
* Lecture notebook: `CHEME-145-M3-Lecture-HiddenMarkovModel-Read-Page.ipynb` — full development of the forward, Viterbi, and Baum-Welch algorithms.
""")

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
```

- [ ] **Step 2: Build and execute the notebook**

```bash
python3 $BUILDERS/build_ungraded.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=julia-1.12 --ExecutePreprocessor.timeout=600 CHEME-145-M3-Example-HMM-BaumWelch-Ungraded-Codio-Activity.ipynb
```
Expected: nbconvert exits 0 (the in-notebook `@assert` on monotonicity acts as a gate — an error fails the run).

- [ ] **Step 3: Verify the executed output trends**

```bash
python3 - <<'EOF'
import json, re
nb = json.load(open('CHEME-145-M3-Example-HMM-BaumWelch-Ungraded-Codio-Activity.ipynb'))
text = "\n".join("".join(o.get('text', [])) for c in nb['cells'] if c['cell_type'] == 'code'
                 for o in c.get('outputs', []) if o.get('output_type') == 'stream')
print(text)
exp1 = re.findall(r'N =\s+(\d+):\s+‖P̂ - P‖F = ([\d.]+),\s+‖Ê - E‖F = ([\d.]+)', text)
exp2 = re.findall(r'd = ([\d.]+):\s+‖P̂ - P‖F = ([\d.]+)', text)
assert len(exp1) == 4 and len(exp2) == 3, "missing experiment output"
assert float(exp1[0][1]) > float(exp1[-1][1]) and float(exp1[0][2]) > float(exp1[-1][2]), "Exp1: error must shrink from N=2 to N=200"
assert float(exp2[0][1]) < float(exp2[-1][1]), "Exp2: error must grow from d=0.9 to d=0.5"
print("TRENDS OK")
EOF
```
Expected: printed outputs plus `TRENDS OK`. Also eyeball: alignment `σ` printed, table values near truth, iteration count < 500. If a trend fails, change the experiment seed(s) in the builder (e.g., `Xoshiro(2026)` → another year), rebuild, re-execute, re-verify.

- [ ] **Step 4: Quality pass**

Re-read every markdown cell in the executed notebook against CLAUDE.md: 3 LOs / 3 KTs in `* __Title:__` format, no unsupported claims, explicit norms, no spelling/grammar errors. Fix in the builder (not the .ipynb), rebuild, re-execute if anything changes.

- [ ] **Step 5: Commit**

```bash
git add CHEME-145-M3-Example-HMM-BaumWelch-Ungraded-Codio-Activity.ipynb
git commit -m "M3: add Baum-Welch ungraded Codio activity

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Graded Solution notebook (dishonest casino)

**Files:**
- Create: `$BUILDERS/build_graded_solution.py` (not committed)
- Create: `$M3/CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb`

**Interfaces:**
- Consumes: `build`, `simulate`, `loglikelihood`, `viterbi`, `colors`, `Xoshiro`, `mean` via `Include.jl`.
- Produces: the executed Solution notebook; Task 9 reads it to generate the student version. Cell order (0-based) after building: 0 title, 1 instructions, 2 story, 3 setup-md, 4 setup-code, 5 task1-md, **6 task1-code (assessed)**, 7 simulate-md, 8 simulate-code, 9 task2-md, **10 task2-code (assessed)**, 11 task3-md, **12 task3-code (assessed)**, 13 viz-md, 14 viz-code, **15 report-md (swapped)**, 16 summary-md.

- [ ] **Step 1: Write the builder script**

Create `$BUILDERS/build_graded_solution.py` with exactly this content:
```python
import nbformat as nbf

NB_PATH = "/Users/jdv27/Desktop/julia_work/CHEME-140-eCornell-Repository/courses/CHEME-145/module-3/CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Julia 1.12.6", "language": "julia", "name": "julia-1.12"}
nb.metadata["language_info"] = {"name": "julia", "version": "1.12.6"}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Graded Codio Activity: The Dishonest Casino
A casino offers a dice game. Unknown to the players, the casino switches between a fair die and a loaded die: which die is in play is a hidden state, and the rolls are the observations. In this activity, you will model the dishonest casino as a two-state Hidden Markov Model, score the observed rolls with the forward algorithm (the evaluation problem), and recover which die was in use with the Viterbi algorithm (the decoding problem). For the full development of both algorithms, see the lecture notebook `CHEME-145-M3-Lecture-HiddenMarkovModel-Read-Page.ipynb`.
""")

md(r"""
## Instructions
Complete the three tasks below by editing the indicated code cells, then run the notebook top to bottom.

1. __Task 1 (Build the casino HMM):__ in the cell under *Task 1: Build the casino HMM*, fill in the transition matrix `P`, the emission matrix `E`, and the initial distribution `π₀` from the story below, and build the model with `build(MyHiddenMarkovModel, ...)`.
2. __Task 2 (Score the evidence):__ in the cell under *Task 2: Score the evidence*, compute `ℓ_casino`, the log-likelihood of the observed rolls under the casino model, and `ℓ_fair`, the log-likelihood of the same rolls if every roll came from a single fair die.
3. __Task 3 (Decode with Viterbi):__ in the cell under *Task 3: Decode with Viterbi*, compute the most likely hidden state path and its accuracy against the true die sequence.

__Deliverable.__ In the *Report* section at the end of the notebook, state which model better explains the rolls (cite the log-likelihood gap $\ell_{\text{casino}}-\ell_{\text{fair}}$) and how reliably Viterbi recovers the loaded stretches (cite the decoding accuracy).
___
""")

md(r"""
## The dishonest casino
The casino has two dice. The __fair die__ shows each face $1,\ldots,6$ with probability $1/6$. The __loaded die__ shows a six with probability $1/2$ and each other face with probability $1/10$. After each roll, the casino may secretly swap the die: from the fair die, it switches to the loaded die with probability $0.05$; from the loaded die, it switches back to the fair die with probability $0.10$. The game starts with the fair die.

As a Hidden Markov Model: the hidden states are $\mathcal{S}\equiv\left\{1=\text{fair},2=\text{loaded}\right\}$, the observables are the die faces $\mathcal{O}\equiv\left\{1,\ldots,6\right\}$, and the model $\mathcal{H}=(\mathbf{P},\mathbf{E},\pi_{0})$ is

$$
\mathbf{P} = \begin{bmatrix}0.95 & 0.05\\ 0.10 & 0.90\end{bmatrix},\qquad
\mathbf{E} = \begin{bmatrix}1/6 & 1/6 & 1/6 & 1/6 & 1/6 & 1/6\\ 1/10 & 1/10 & 1/10 & 1/10 & 1/10 & 1/2\end{bmatrix},\qquad
\pi_{0} = (1, 0)
$$

where row $i$ of $\mathbf{P}$ holds the switching probabilities out of state $i$, row $i$ of $\mathbf{E}$ holds the face probabilities of die $i$, and $\pi_{0}$ says the game starts with the fair die.
___
""")

md(r"""
## Setup, Data, and Prerequisites
First, we set up the computational environment by including the `Include.jl` file. The `Include.jl` file sets paths, loads required external packages, and includes the module's local source files (`src/Types.jl`, `src/Factory.jl`, `src/Compute.jl`) holding the types and algorithms used in this activity.
""")

code(r"""
include(joinpath(@__DIR__, "Include.jl")); # include the Include.jl file
""")

md(r"""
## Task 1: Build the casino HMM
Fill in `P` (a $2\times{2}$ matrix), `E` (a $2\times{6}$ matrix), and `π₀` (a length-2 vector) from the story above, then build the model. The `build(...)` factory method throws an `ArgumentError` if any row is not a valid probability distribution, so a successful build is a first check on your answer.
""")

code(r"""
P = [
    0.95 0.05 ; # fair: stay fair 0.95, switch to loaded 0.05
    0.10 0.90 ; # loaded: switch to fair 0.10, stay loaded 0.90
];

E = [
    1/6 1/6 1/6 1/6 1/6 1/6 ; # fair die: each face with probability 1/6
    1/10 1/10 1/10 1/10 1/10 1/2 ; # loaded die: six with probability 1/2
];

π₀ = [1.0, 0.0]; # the game starts with the fair die

hmm = build(MyHiddenMarkovModel, (P = P, E = E, π₀ = π₀));
""")

md(r"""
## Simulate the casino
We simulate $T = 1{,}000$ rolls from the casino model with a seeded random number generator, so every run of the notebook produces the same game. The `simulate(...)` method returns both the hidden die sequence and the observed rolls. We keep the hidden sequence __only__ to score the decoder in Task 3 — a player at the table sees just the rolls.
""")

code(r"""
T = 1_000; # number of rolls
game = simulate(hmm, T; N = 1, rng = Xoshiro(2026))[1];
rolls = game.observed; # what a player sees: the sequence of die faces
true_die = game.hidden; # which die was actually in play (used only for scoring)
println("first 40 rolls: ", join(rolls[1:40], " "));
println("fraction of rolls that are sixes: ", round(count(==(6), rolls)/T, digits = 3));
""")

md(r"""
## Task 2: Score the evidence
The forward algorithm solves the evaluation problem: it computes the log-likelihood $\log P(\mathbf{o}\mid\mathcal{H})$ of an observation sequence under a model. Compute:
* `ℓ_casino`: the log-likelihood of `rolls` under the casino model, using `loglikelihood(hmm, rolls)`.
* `ℓ_fair`: the log-likelihood of `rolls` if every roll came from a single fair die. Independent rolls each have probability $1/6$, so $\ell_{\text{fair}} = T\log(1/6)$ — no forward algorithm needed.

If the casino really is switching dice, the casino model should explain the rolls better: $\ell_{\text{casino}}>\ell_{\text{fair}}$.
""")

code(r"""
ℓ_casino = loglikelihood(hmm, rolls); # log P(rolls | casino model), forward algorithm
ℓ_fair = T*log(1/6); # log-likelihood under a single fair die
println("ℓ_casino = $(round(ℓ_casino, digits = 2)), ℓ_fair = $(round(ℓ_fair, digits = 2)), gap = $(round(ℓ_casino - ℓ_fair, digits = 2))");
@assert ℓ_casino > ℓ_fair # the two-state model explains the rolls better
""")

md(r"""
## Task 3: Decode with Viterbi
The Viterbi algorithm solves the decoding problem: it computes the most likely hidden state path given the observations and the model. Compute the decoded path with `viterbi(hmm, rolls)`, then compute the decoding accuracy: the fraction of time steps where the decoded die matches `true_die`.
""")

code(r"""
decoded_die = viterbi(hmm, rolls); # most likely hidden path given the rolls
accuracy = mean(decoded_die .== true_die); # fraction of rolls where the decoder is right
println("decoding accuracy = $(round(accuracy, digits = 3))");
""")

md(r"""
## Visualize the decoding
The step plot below shows the true die and the Viterbi decode (offset upward for visibility) for the first 400 rolls. Viterbi catches the sustained loaded stretches; brief switches are hard to detect because a short run of rolls carries little evidence.
""")

code(r"""
let
    window = 1:400;
    plot(window, true_die[window], seriestype = :steppost, lw = 2, c = colors[1], label = "true die",
        yticks = ([1, 2], ["fair", "loaded"]), xlabel = "roll index t", ylims = (0.8, 2.35), legend = :topleft);
    plot!(window, decoded_die[window] .+ 0.08, seriestype = :steppost, lw = 2, c = colors[3], label = "Viterbi decode")
end
""")

md(r"""
## Report
In one or two sentences: which model better explains the rolls (cite the log-likelihood gap), and how reliably does Viterbi recover the loaded stretches (cite the decoding accuracy)?

__Model answer:__ The casino model explains the rolls better: $\ell_{\text{casino}}$ exceeds $\ell_{\text{fair}}$ by GAP_PLACEHOLDER nats, so the observed roll sequence is more probable under a switching model than under a single fair die. Viterbi recovers the die in play at ACC_PLACEHOLDER of the rolls, catching the sustained loaded stretches while missing brief switches.
""")

md(r"""
___
## Summary
In this activity, we modeled the dishonest casino as a two-state Hidden Markov Model, scored the observed rolls with the forward algorithm, and recovered the hidden die sequence with the Viterbi algorithm.

> __Key Takeaways:__
>
> * __The forward algorithm scores models against evidence:__ the log-likelihood of the rolls under the switching casino model exceeds the single-fair-die benchmark $T\log(1/6)$, so the rolls are evidence that the casino is switching dice.
> * __Viterbi decodes the hidden state path:__ the most likely path recovers the die in play at most time steps, catching sustained loaded stretches while missing brief switches.
> * __Hidden state inference needs only the model and the observations:__ both computations used the rolls and $\mathcal{H}=(\mathbf{P},\mathbf{E},\pi_{0})$ alone; the true die sequence was used only to score the decoder.

The evaluation and decoding problems, together with the learning problem from the Baum-Welch activity, complete the three classic HMM problems from the lecture.

### Additional Resources
* Durbin, R., Eddy, S. R., Krogh, A., & Mitchison, G. (1998). *Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids*. Cambridge University Press — the dishonest casino example.
* Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.
* Lecture notebook: `CHEME-145-M3-Lecture-HiddenMarkovModel-Read-Page.ipynb`.
""")

nb["cells"] = cells
nbf.write(nb, NB_PATH)
print(f"wrote {NB_PATH} with {len(cells)} cells")
```

- [ ] **Step 2: Build, execute, and read the numbers**

```bash
python3 $BUILDERS/build_graded_solution.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=julia-1.12 --ExecutePreprocessor.timeout=600 CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb
python3 -c "
import json
nb = json.load(open('CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb'))
for c in nb['cells']:
    for o in c.get('outputs', []):
        if o.get('output_type') == 'stream': print(''.join(o['text']))
"
```
Expected: nbconvert exits 0 (the `@assert ℓ_casino > ℓ_fair` gates the gap). Read the printed `gap = ...` and `decoding accuracy = ...`. **Check: accuracy ∈ [0.70, 0.95].** If outside the range, change the simulation seed in the builder (`Xoshiro(2026)` → another value), rebuild, re-execute, re-check.

- [ ] **Step 3: Fill the Report placeholders with the executed numbers**

In `$BUILDERS/build_graded_solution.py`, replace `GAP_PLACEHOLDER` with the printed gap rounded to the nearest integer (e.g., `about 45`) and `ACC_PLACEHOLDER` with the printed accuracy as a percentage (e.g., `about 88%`). Then rebuild and re-execute:
```bash
python3 $BUILDERS/build_graded_solution.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=julia-1.12 --ExecutePreprocessor.timeout=600 CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb
grep -c PLACEHOLDER CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb
```
Expected: grep prints `0` (exit code 1 — no placeholders remain).

- [ ] **Step 4: Quality pass**

Re-read all markdown against CLAUDE.md standards (3 KTs, `* __Title:__` format, direct language, claims supported). Fix in the builder, rebuild, re-execute if anything changes.

- [ ] **Step 5: Commit**

```bash
git add CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb
git commit -m "M3: add dishonest-casino graded Codio activity solution

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Student graded notebook

**Files:**
- Create: `$BUILDERS/make_student.py` (not committed)
- Create: `$M3/CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity.ipynb`
- Modify: `$M3/specs/CHEME-145-M3-codio-activities-spec.md` (acceptance wording)

**Interfaces:**
- Consumes: the executed Solution notebook (Task 8) and its cell markers: code cells whose source starts with `P = [` (task 1), `ℓ_casino = loglikelihood` (task 2), `decoded_die = viterbi` (task 3), and the markdown cell containing `## Report`.
- Produces: the student notebook, executed with `--allow-errors`.

- [ ] **Step 1: Write the generator script**

Create `$BUILDERS/make_student.py` with exactly this content:
```python
import nbformat as nbf

SRC = "/Users/jdv27/Desktop/julia_work/CHEME-140-eCornell-Repository/courses/CHEME-145/module-3/CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb"
DST = "/Users/jdv27/Desktop/julia_work/CHEME-140-eCornell-Repository/courses/CHEME-145/module-3/CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity.ipynb"

STUB_1 = '''# TODO: fill in the transition matrix P (2×2), the emission matrix E (2×6), and the
# initial distribution π₀ (length 2) from the story above, then delete the error(...)
# line so the cell runs.
error("TODO: build the casino HMM");

P = nothing; # TODO: 2×2 transition matrix, rows = (fair, loaded)
E = nothing; # TODO: 2×6 emission matrix, rows = (fair, loaded), columns = faces 1–6
π₀ = nothing; # TODO: length-2 initial distribution; the game starts with the fair die

hmm = build(MyHiddenMarkovModel, (P = P, E = E, π₀ = π₀));'''

STUB_2 = '''# TODO: compute ℓ_casino, the log-likelihood of rolls under the casino model
# (use loglikelihood(hmm, rolls)), and ℓ_fair, the log-likelihood under a single
# fair die, then delete the error(...) line so the cell runs.
error("TODO: score the evidence");

ℓ_casino = nothing; # TODO: forward-algorithm log-likelihood under the casino model
ℓ_fair = nothing; # TODO: log-likelihood of T independent fair-die rolls

println("ℓ_casino = $(round(ℓ_casino, digits = 2)), ℓ_fair = $(round(ℓ_fair, digits = 2)), gap = $(round(ℓ_casino - ℓ_fair, digits = 2))");'''

STUB_3 = '''# TODO: compute decoded_die, the most likely hidden path (use viterbi(hmm, rolls)),
# and accuracy, the fraction of rolls where decoded_die matches true_die, then
# delete the error(...) line so the cell runs.
error("TODO: decode with Viterbi");

decoded_die = nothing; # TODO: Viterbi path
accuracy = nothing; # TODO: mean agreement with true_die

println("decoding accuracy = $(round(accuracy, digits = 3))");'''

REPORT = '''## Report
Edit this cell. In one or two sentences: which model better explains the rolls (cite the log-likelihood gap), and how reliably does Viterbi recover the loaded stretches (cite the decoding accuracy)?

*Your answer here.*'''

nb = nbf.read(SRC, as_version=4)
replaced = []
for cell in nb.cells:
    src = cell.source
    if cell.cell_type == "code" and src.startswith("P = ["):
        cell.source = STUB_1; replaced.append("task1")
    elif cell.cell_type == "code" and src.startswith("ℓ_casino = loglikelihood"):
        cell.source = STUB_2; replaced.append("task2")
    elif cell.cell_type == "code" and src.startswith("decoded_die = viterbi"):
        cell.source = STUB_3; replaced.append("task3")
    elif cell.cell_type == "markdown" and src.startswith("## Report"):
        cell.source = REPORT; replaced.append("report")
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

assert sorted(replaced) == ["report", "task1", "task2", "task3"], f"expected 4 swaps, got {replaced}"
nbf.write(nb, DST)
print(f"wrote {DST}; swapped: {replaced}")
```

- [ ] **Step 2: Generate and execute the student notebook**

```bash
python3 $BUILDERS/make_student.py
jupyter nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.kernel_name=julia-1.12 --ExecutePreprocessor.timeout=600 CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity.ipynb
```
Expected: both commands exit 0.

- [ ] **Step 3: Verify the error pattern**

```bash
python3 - <<'EOF'
import json
nb = json.load(open('CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity.ipynb'))
errs, clean = [], []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code': continue
    (errs if any(o.get('output_type') == 'error' for o in c.get('outputs', [])) else clean).append(i)
print('error cells:', errs, ' clean cells:', clean)
srcs = [''.join(nb['cells'][i]['source']) for i in errs]
assert any('TODO: build the casino HMM' in s for s in srcs)
assert any('TODO: score the evidence' in s for s in srcs)
assert any('TODO: decode with Viterbi' in s for s in srcs)
assert 4 in clean, "setup cell (index 4) must run clean"
print("STUDENT NB OK: stubs error, setup clean; remaining errors are downstream of the stubs (module-4 precedent)")
EOF
```
Expected: the three stub cells and their downstream dependents (simulate cell, viz cell) error; the setup cell is clean; `STUDENT NB OK` printed.

- [ ] **Step 4: Align the spec acceptance wording**

In `$M3/specs/CHEME-145-M3-codio-activities-spec.md`, replace the acceptance line:
```
2. All three notebooks execute headless end-to-end; the student notebook errors only at the three TODO stubs.
```
with:
```
2. All three notebooks execute headless end-to-end; in the student notebook, the three TODO stubs error by design, and the only other errors are in downstream cells that consume stub results (module-4 precedent).
```

- [ ] **Step 5: Final full verification and commit**

```bash
julia --project=. test/runtests.jl        # expect: all 6 testsets PASS
git status --short                        # expect: only the student notebook + spec edit (Manifest.toml untracked/ignored)
git add CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity.ipynb specs/CHEME-145-M3-codio-activities-spec.md
git commit -m "M3: add dishonest-casino graded student notebook, align spec acceptance wording

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Acceptance checklist (from the spec)

- [ ] `test/runtests.jl` passes in the module-3 environment (Tasks 1–6).
- [ ] All three notebooks execute headless end-to-end (Tasks 7–9); student stubs + their downstream dependents are the only errors.
- [ ] Ungraded output: non-decreasing log-likelihood (in-notebook `@assert`), Exp 1 error shrinks N=2→200, Exp 2 error grows d=0.9→0.5 (Task 7 Step 3).
- [ ] Solution output: `ℓ_casino > ℓ_fair` (in-notebook `@assert`), accuracy ∈ [0.70, 0.95] (Task 8 Step 2).
- [ ] LOs/KTs: 3 items each, house format, CLAUDE.md language standards (Tasks 7–8 quality passes).
- [ ] Demo and lecture notebooks untouched (`git status` in Task 9 Step 5).
