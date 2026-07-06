# CHEME-145 Module 3 (Hidden Markov Models) — Codio Activities Build Spec

*Drafted 2026-07-06. Scope, structure, and example domains validated with the author in a brainstorming session. Companion to the module-4 build spec (`courses/CHEME-145/module-4/specs/CHEME-145-M4-build-spec.md`), whose conventions this build follows.*

**Status: Approved, not yet built.**

## Goal

Add two Codio activities to Module 3 (Hidden Markov Models), which currently has a lecture Read-Page (`CHEME-145-M3-Lecture-HiddenMarkovModel-Read-Page.ipynb`) and a Watch-Demo (`CHEME-145-M3-Example-HiddenMarkovModels-Watch-Demo.ipynb`):

1. An **ungraded** activity: fit an HMM with Baum-Welch to synthetic data generated from the Watch-Demo's mood/emoji model.
2. A **graded** activity (Solution + student pair): the dishonest casino — a self-contained story assessing the evaluation (forward) and decoding (Viterbi) problems.

Together with the demo (simulation problem) and the ungraded activity (learning problem), the module then exercises all three classic HMM problems from the lecture.

## Decisions (from brainstorming)

1. **Scope:** design both activities together; build the ungraded activity first, then the graded pair.
2. **Code home:** a self-contained module-3 `src/` (Types.jl → Factory.jl → Compute.jl) plus `test/runtests.jl`, mirroring module-4. Both new notebooks call `src/`; the existing demo and lecture notebooks are not modified.
3. **API style:** module-4 solver-model pattern — `build(MyHiddenMarkovModel, (P = ..., E = ..., π₀ = ...))`, `solve(build(MyBaumWelchModel, (maxiterations = ..., ϵ = ...)), sequences; ...)`. `viterbi(...)` and `loglikelihood(...)` are plain functions of the model.
4. **Ungraded experiments:** (a) data size, (b) emission noise. No separate initialization/local-maxima or label-switching experiments; label switching is handled quietly in the walkthrough with a permutation-alignment helper.
5. **Graded story:** dishonest casino (Durbin et al.): 2 hidden states (fair/loaded die), 6 observables (die faces).
6. **Graded assessed tasks (TODO stubs):** build P/E/π₀; forward log-likelihood vs the all-fair benchmark; Viterbi decode + accuracy. No Baum-Welch in the graded activity.
7. **Theory sections are light:** a short refresher only (a few sentences plus at most the key update equations), pointing to the lecture Read-Page for the full α, β, γ, ξ development.
8. **Reproducibility:** all randomness through explicitly seeded RNGs so notebooks reproduce exactly on Codio.

## Notation (aligned to the lecture)

Hidden states $s\in\mathcal{S}$, observables $o\in\mathcal{O}$, transition matrix $\mathbf{P}$ with entries $p_{ij}$, emission matrix $\mathbf{E}$ with entries $e_{i,o}$, initial distribution $\pi_0$, model $\mathcal{H}=(\mathbf{P},\mathbf{E},\pi_0)$. Forward variables $\alpha_t(i)$, backward variables $\beta_t(i)$, posteriors $\gamma_t(i)$ and $\xi_t(i,j)$, log-likelihood $\mathcal{L}_k$, convergence tolerance $\epsilon>0$, maximum iterations $K_{\text{max}}$. Frobenius norm $\lVert\cdot\rVert_{F}$ for parameter-error comparisons (matches the demo's stationary-distribution check).

## Deliverables

1. `src/Types.jl`, `src/Factory.jl`, `src/Compute.jl` — HMM library (below).
2. `test/runtests.jl` — unit tests (below).
3. `Include.jl` — extended to include the `src/` files after package loads; existing package list (including `VLDataScienceMachineLearningPackage`, which the demo uses) is kept.
4. `Project.toml` — add `Random` (stdlib) to `[deps]`; keep the `JSON = "0.21"` compat pin.
5. `CHEME-145-M3-Example-HMM-BaumWelch-Ungraded-Codio-Activity.ipynb`
6. `CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity-Solution.ipynb`
7. `CHEME-145-M3-Example-DishonestCasino-HMM-Graded-Codio-Activity.ipynb` (student version, generated from the Solution by stubbing the three assessed cells).

## `src/` scaffolding (mirrors module-4: Types → Factory → Compute)

### `Types.jl`
- `abstract type AbstractHiddenMarkovModel end`
- `abstract type AbstractSolutionModel end`
- `MyHiddenMarkovModel <: AbstractHiddenMarkovModel` — mutable struct, empty constructor (module-4 style). Fields: `P::Array{Float64,2}` (|𝒮|×|𝒮|), `E::Array{Float64,2}` (|𝒮|×|𝒪|), `π₀::Array{Float64,1}` (length |𝒮|).
- `MyBaumWelchModel <: AbstractSolutionModel` — fields `maxiterations::Int64`, `ϵ::Float64`.
- `MyBaumWelchSolution` — solution container. Fields: `P::Array{Float64,2}`, `E::Array{Float64,2}`, `π₀::Array{Float64,1}` (learned parameters), `loglikelihood_history::Array{Float64,1}` (ℒₖ per iteration), `iterations::Int64`.

### `Factory.jl`
- `build(::Type{MyHiddenMarkovModel}, data::NamedTuple)` — keys `P, E, π₀`. Validates: each row of `P` and `E` sums to 1 within tolerance, `π₀` sums to 1 within tolerance, all entries non-negative, `size(P,1) == size(P,2) == size(E,1) == length(π₀)`. Throws `ArgumentError` with a descriptive message on violation.
- `build(::Type{MyBaumWelchModel}, data::NamedTuple)` — keys `maxiterations, ϵ`.

### `Compute.jl`
- `simulate(hmm::MyHiddenMarkovModel, T::Int; N::Int = 1, rng::AbstractRNG = Random.default_rng())` → `Array{NamedTuple,1}` of length `N`, each element `(hidden::Array{Int64,1}, observed::Array{Int64,1})` of length `T`. Forward sampling: draw $s_1\sim\pi_0$, then $s_{t+1}\sim\mathbf{P}[s_t,:]$ and $o_t\sim\mathbf{E}[s_t,:]$. Named `simulate` (not `sample`) to avoid colliding with the `sample` exported via `Distributions`.
- `loglikelihood(hmm::MyHiddenMarkovModel, observed::Array{Int64,1})::Float64` — implemented as a **method extending the `Distributions.loglikelihood` generic** (no name shadowing). Scaled forward algorithm: per-step normalization constants $c_t$; $\log P(\mathbf{o}\mid\mathcal{H}) = \sum_t \log c_t$. Handles $T \geq 1{,}000$ without underflow.
- `viterbi(hmm::MyHiddenMarkovModel, observed::Array{Int64,1})::Array{Int64,1}` — most likely hidden path, computed in log-space with backtrace.
- `solve(model::MyBaumWelchModel, sequences::Array{Array{Int64,1},1}; number_of_hidden_states::Int, number_of_observable_states::Int, rng::AbstractRNG = Random.default_rng())::MyBaumWelchSolution` — multi-sequence Baum-Welch exactly as the lecture algorithm: scaled forward/backward per sequence, γ and ξ posteriors, M-step updates for $\pi_0$, $\mathbf{P}$, $\mathbf{E}$, convergence when $|\mathcal{L}_k-\mathcal{L}_{k-1}|<\epsilon$ or $k\geq K_{\text{max}}$. Initialization: random row-stochastic matrices from `rng` (uniform draws, rows normalized). Guard: if a state accumulates zero γ-mass in the M-step, keep its previous row (avoid NaN).
- `align_states(P̂::Array{Float64,2}, Ê::Array{Float64,2}, P::Array{Float64,2}, E::Array{Float64,2})` → `(σ, P̂_aligned, Ê_aligned)`: the permutation σ of learned state labels minimizing $\lVert\hat{\mathbf{P}}_\sigma-\mathbf{P}\rVert_{F}+\lVert\hat{\mathbf{E}}_\sigma-\mathbf{E}\rVert_{F}$ over all $|\mathcal{S}|!$ permutations (exhaustive; fine for 2–3 states; permutations generated by a small internal helper, no new dependency).
- Internal helpers `_forward` / `_backward` return the scaled variables and scaling constants shared by `loglikelihood` and `solve`.

### `test/runtests.jl`
Arrangement mirrors module-4's `test/`. Tests:
- Factory validation: bad rows / negative entries / dimension mismatch throw.
- `simulate`: correct lengths, values in range, deterministic under a fixed seed.
- Forward: `loglikelihood` matches brute-force enumeration over all hidden paths on a tiny model (e.g., |𝒮| = 2, T = 4), to ~1e-12.
- Viterbi: matches brute-force argmax over all hidden paths on the same tiny model.
- Baum-Welch: `loglikelihood_history` is non-decreasing; on easy synthetic data (near-deterministic emissions, generous sample size, fixed seed) recovers parameters after `align_states` within a loose tolerance.
- `align_states`: recovers a known permutation applied to a reference model.

## Notebook 1 — `CHEME-145-M3-Example-HMM-BaumWelch-Ungraded-Codio-Activity.ipynb`

Module-4 ungraded shape: fully runnable walkthrough (no stubs), experiments at the end. Framing: the Watch-Demo simulated an HMM whose parameters were known; this activity asks the reverse question — given only observation sequences, recover the parameters.

1. **Title + intro + Learning Objectives** — `# Activity: Learning HMM Parameters with Baum-Welch`; 3 objectives in the `* __Title:__ description` house format.
2. **Theory refresher** — short: the learning problem, one-paragraph E-step/M-step summary with the γ/ξ definitions, pointer to the lecture Read-Page for the full development.
3. **Setup** — `include("Include.jl")`.
4. **Task 1: Build the true model** — the Watch-Demo's matrices: $\mathbf{P} = \begin{bmatrix}0.05&0.95&0.0\\0.6&0.2&0.2\\0.0&0.3&0.7\end{bmatrix}$, $\mathbf{E} = \begin{bmatrix}0.90&0.05&0.05\\0.05&0.90&0.05\\0.05&0.05&0.90\end{bmatrix}$, uniform $\pi_0 = (1/3,1/3,1/3)$; `hmm_true = build(MyHiddenMarkovModel, ...)`. Mood/emoji story carried over from the demo.
5. **Task 2: Generate synthetic observations** — `simulate(hmm_true, 200; N = 50, rng)`; keep only `observed` for fitting (that is the point: the fitter never sees the hidden states).
6. **Task 3: Fit with Baum-Welch** — solver `(maxiterations = 500, ϵ = 1e-6)`, random init from a fixed seed; plot `loglikelihood_history` (monotone increase — EM guarantee from the lecture).
7. **Task 4: Compare to truth** — `align_states`, side-by-side PrettyTables of $\hat{\mathbf{P}}$ vs $\mathbf{P}$ and $\hat{\mathbf{E}}$ vs $\mathbf{E}$, print $\lVert\hat{\mathbf{P}}-\mathbf{P}\rVert_{F}$ and $\lVert\hat{\mathbf{E}}-\mathbf{E}\rVert_{F}$. One-sentence note on why alignment is needed (state labels are arbitrary to the fitter).
8. **Experiment 1: Data size** — refit for $N\in\{2,10,50,200\}$ sequences ($T=200$ fixed); print/plot both error norms vs $N$; students change $N$ or $T$.
9. **Experiment 2: Emission noise** — refit ($N=50$, $T=200$) with the emission diagonal $d\in\{0.9,0.7,0.5\}$ (off-diagonal $(1-d)/2$); observe recovery degrade as observations carry less information about the hidden state. Note connecting to what "hidden" means.
10. **Summary + Key Takeaways (3) + Additional Resources.**

Seeds are chosen at build time so the walkthrough converges cleanly and both experiment trends hold in the executed output.

## Notebook 2 — Graded pair (dishonest casino)

Module-4 graded shape: Instructions, story, Setup, alternating given/TODO cells, closing visualization, Summary. Solution written and executed first; student version generated by replacing the three assessed cells with `error("TODO: ...")` stubs (answers stripped, `nothing` placeholders in call skeletons), executed with `--allow-errors`.

**Model (Durbin et al. parameterization):** hidden states {1 = fair, 2 = loaded}; observables = die faces 1–6.
$$\mathbf{P} = \begin{bmatrix}0.95 & 0.05\\ 0.10 & 0.90\end{bmatrix},\qquad
\mathbf{E} = \begin{bmatrix}1/6 & 1/6 & 1/6 & 1/6 & 1/6 & 1/6\\ 0.1 & 0.1 & 0.1 & 0.1 & 0.1 & 0.5\end{bmatrix},\qquad
\pi_0 = (1, 0)$$

1. **Title + Instructions** — module-4 graded instructions block (complete the TODO cells, delete the `error(...)` lines, run top to bottom).
2. **The dishonest casino** — story + parameters above; brief refresher sentence tying the two assessed tasks to the lecture's evaluation and decoding problems, pointer to the Read-Page.
3. **Setup** — `include("Include.jl")`.
4. **TODO 1: Build the model** — fill in `P`, `E`, `π₀` from the story; `build(MyHiddenMarkovModel, ...)` call skeleton with `nothing` placeholders.
5. **Given: Simulate the casino** — fixed seed; `simulate(hmm, 1_000)`; show the first ~40 rolls; keep the true hidden sequence for scoring.
6. **TODO 2: Score the evidence** — compute `ℓ_casino = loglikelihood(hmm, rolls)` and the analytical all-fair benchmark `ℓ_fair = T * log(1/6)`; confirm `ℓ_casino > ℓ_fair` (the two-state model explains the rolls better).
7. **TODO 3: Decode + accuracy** — `path = viterbi(hmm, rolls)`; `accuracy = mean(path .== hidden_truth)`.
8. **Given: Visualize the decoding** — step plot of true vs decoded regime over time (Paul Tol palette per house style); shows loaded stretches caught and brief visits missed.
9. **Summary + Key Takeaways (3) + Additional Resources.**

Seed chosen at build time so the executed Solution shows decode accuracy in roughly the 0.80–0.95 range (not trivially 1.0) and a clear log-likelihood gap.

## Build and execution workflow

- Notebooks authored with Python `nbformat` builder scripts (raw triple-quoted sources; `ensure_ascii=False` when dumping JSON), per the module-4 workflow.
- Executed headless: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=julia-1.12 --ExecutePreprocessor.timeout=600 <nb>`; student graded version additionally `--allow-errors`.
- `Manifest.toml` stays untracked (repo-wide gitignore); the `JSON = "0.21"` compat pin keeps the fresh-environment kernel working on Codio.

## Acceptance checks

1. `test/runtests.jl` passes in the module-3 environment.
2. All three notebooks execute headless end-to-end; the student notebook errors only at the three TODO stubs.
3. Ungraded executed output: log-likelihood history non-decreasing; Experiment 1 error norms decrease as $N$ grows; Experiment 2 error norms increase as the emission diagonal decreases.
4. Solution executed output: `ℓ_casino > ℓ_fair`; decode accuracy printed in [0.70, 0.95].
5. Learning Objectives and Key Takeaways: exactly 3 items each, house format, direct/simple/concise language, content supported by the notebook (CLAUDE.md standards).
6. Existing demo and lecture notebooks unmodified.
