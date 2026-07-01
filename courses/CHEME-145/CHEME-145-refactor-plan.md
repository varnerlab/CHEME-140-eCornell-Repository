# CHEME-145 — Decision Systems: Refactor Plan

*Drafted 2026-06-30. Status: topic/lecture brainstorm — not yet validated against an existing week-by-week template. Module 1 drafting underway: the theory notebook and the first two escalating-practice demos are built (see Module 1 below).*

## Course Context

- One of 6 courses in a "Data Science and Basic Algorithms"-type sequence, delivered as part of a distance-learning Master's degree.
- Audience: professional students with a STEM background.
- Format (standard across the 6-course sequence, same as the eCornell CHEME-5800/5820 courses): each module = **theory lecture(s) → escalating practice examples → one graded example**.
- Example style: both toy examples (e.g., apples-and-oranges) and finance examples are fair game per module — no need to force one consistent storyline across all four modules.

## Module Overview

1. Classical Decision Systems
2. Markov Models
3. Hidden Markov Models
4. Markov Decision Processes

**Connective/capstone idea:** dynamic discrete choice models (e.g., Rust 1987 bus-engine replacement) are literally an MDP estimated as a structural discrete-choice model — they tie Module 1 (discrete choice) and Module 4 (MDPs) together. Worth surfacing explicitly, e.g., as a closing discussion in Module 4, so the course reads as one toolkit rather than four disconnected topics.

---

## Module 1 — Classical Decision Systems

**Lectures**
1. Preferences & utility functions — axioms (completeness, transitivity), ordinal vs. cardinal utility
2. Consumer choice & indifference curves — budget constraint, MRS, constrained optimization
3. Decision-making under risk — expected utility, von Neumann–Morgenstern axioms, risk aversion, Arrow-Pratt, certainty equivalent
4. Discrete choice models — random utility models, Luce's choice axiom, multinomial/conditional logit (McFadden), IIA

**Escalating practice**
- Demo (built): `module-1/CHEME-145-M1-IndifferenceCurves-Watch-Demo.ipynb` — compute and plot indifference curves (level sets) for linear, Cobb-Douglas, and Leontief utilities
- Demo (built): `module-1/CHEME-145-M1-CobbDouglas-Utility-Watch-Demo.ipynb` — budget-constrained Cobb-Douglas utility maximization; closed-form and numerical (JuMP/Ipopt) solutions, first-order/MRS checks, and the budget-line vs. indifference-curve tangency plot
- Demo (built): `module-1/CHEME-145-M1-ArrowPratt-Watch-Demo.ipynb` — Arrow-Pratt absolute/relative risk aversion A(w), R(w) for logarithmic/linear/power utilities, derivatives via automatic differentiation (ForwardDiff)
- Demo (built): `module-1/CHEME-145-M1-Insurance-Watch-Demo.ipynb` — insurance decision: expected value vs. expected utility, certainty equivalent CE = U⁻¹(E[U]), risk premium π = E[W] − CE, and the maximum premium a risk-averse agent will pay
- Ungraded notebook (todo): explore a discrete-choice dataset

The four demos are backed by self-contained local scaffolding in `module-1/` (`Include.jl`, `Project.toml`, `src/`): bundle utility types + `build`/`solve`/`indifference`/`budget`, plus scalar wealth utilities (log/linear/power) with `evaluate`/`inverse`, Arrow-Pratt `absolute_risk_aversion`/`relative_risk_aversion` (ForwardDiff), and `expected_utility`/`certainty_equivalent` — not the old `VLDecisionsPackage.jl`. Theory for all four Module 1 lectures is consolidated in `module-1/CHEME-145-M1-Introduction-ClassicalDecisionSystems-Read-Pages.ipynb`.

**Graded example**
- Estimate and interpret a multinomial logit on a choice dataset (mode choice, brand choice, or investment choice); report marginal effects/elasticities

---

## Module 2 — Markov Models

**Lectures**
1. Markov chains & the Markov property — states, transition matrices, transition diagrams
2. Multi-step transitions & the Chapman–Kolmogorov equation
3. Classifying states & stationary distributions — recurrence, periodicity, irreducibility, ergodicity
4. Absorbing chains — fundamental matrix, absorption probabilities, expected time to absorption

**Escalating practice**
- Demo: simulate a transition matrix, animate convergence to the stationary distribution (PageRank is a strong visual)
- Ungraded notebook: given a transition matrix, classify states and solve for the stationary distribution by hand and in code

**Graded example**
- Model a real process as a Markov chain (credit-rating migration, customer churn, or equipment-state degradation) and answer a concrete steady-state question (expected time to default, expected revenue)

---

## Module 3 — Hidden Markov Models

**Lectures**
1. From Markov chains to HMMs — motivation (partial observability), components (states, observations, A, B, π)
2. Evaluation & decoding — the forward algorithm, the Viterbi algorithm
3. Learning HMM parameters — Baum-Welch as EM specialized to HMMs (forward-backward for the E-step, re-estimation for the M-step)
4. *(optional)* Practical considerations — numerical stability (log-space), choosing the number of hidden states, sensitivity to initialization

**Escalating practice**
- Demo: the classic "occasionally dishonest casino" — generate data from a known HMM, run Viterbi to recover hidden states
- Ungraded notebook: run forward/Viterbi by hand on a small HMM; watch Baum-Welch parameters converge over iterations on synthetic data

**Graded example**
- Fit an HMM (via Baum-Welch) to recover hidden regimes — market regime detection (bull/bear) from a return series, or equipment-fault detection from sensor data

---

## Module 4 — Markov Decision Processes

**Lectures**
1. From Markov chains to MDPs — adding actions and rewards, the (S, A, P, R, γ) tuple, discounting
2. Policies & value functions — V(s), Q(s,a), the Bellman expectation equation
3. Value iteration — Bellman optimality equation, contraction argument, convergence
4. Policy iteration — policy evaluation + policy improvement, policy improvement theorem, comparison to value iteration

**Escalating practice**
- Demo: value iteration on a small gridworld, watching the value function and policy converge
- Ungraded notebook: implement policy evaluation and policy iteration on a provided MDP (inventory/maintenance problem), compare convergence speed to value iteration

**Graded example**
- Formulate a new domain-specific problem as an MDP from scratch (equipment replacement, inventory, or portfolio rebalancing), solve via both value iteration and policy iteration, and write up a recommendation. Natural place to circle back to Module 1's discrete-choice framing as a course capstone.

---

## Open Items / Next Steps

- [ ] Validate the 4-lectures-per-module pacing (16 lectures total) against the week-by-week template used by the other 5 courses in the sequence
- [ ] Pick concrete datasets for each graded example
- [ ] Decide whether Module 3's 4th (practical-considerations) lecture is in-scope or cut
- [ ] Draft lecture scripts / notebooks, starting wherever makes sense
- [ ] Write rubrics for the four graded examples
