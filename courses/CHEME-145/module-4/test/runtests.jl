using Test
using LinearAlgebra
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
