using Test
using LinearAlgebra
using Random
using Distributions

include(joinpath(@__DIR__, "..", "src", "Types.jl"));
include(joinpath(@__DIR__, "..", "src", "Factory.jl"));
include(joinpath(@__DIR__, "..", "src", "Compute.jl"));

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

@testset "align_states" begin
    P = [0.8 0.15 0.05; 0.1 0.7 0.2; 0.25 0.25 0.5];
    E = [0.9 0.05 0.05; 0.05 0.9 0.05; 0.05 0.05 0.9];
    σ_true = [2, 3, 1];
    aligned = align_states(P[σ_true, σ_true], E[σ_true, :], P, E);
    @test isapprox(aligned.P, P; atol = 1e-12);
    @test isapprox(aligned.E, E; atol = 1e-12);
end

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

    # iteration cap: with ϵ = 0 the ϵ-criterion cannot fire (except on an exact tie), so the loop
    # must stop at maxiterations and never overrun it -
    short_sequences = [s.observed for s ∈ simulate(hmm_true, 20; N = 5, rng = Xoshiro(11))];
    cap_solver = build(MyBaumWelchModel, (maxiterations = 3, ϵ = 0.0));
    cap_result = solve(cap_solver, short_sequences;
        number_of_hidden_states = 2, number_of_observable_states = 2, rng = Xoshiro(3));
    @test cap_result.iterations ≤ 3;
    @test length(cap_result.loglikelihood_history) ≤ 3;

    # long sequence: the scaled forward algorithm stays finite at T ≥ 1,000 (no underflow) -
    @test isfinite(loglikelihood(hmm_true, simulate(hmm_true, 1_000; N = 1, rng = Xoshiro(99))[1].observed));
end
