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
