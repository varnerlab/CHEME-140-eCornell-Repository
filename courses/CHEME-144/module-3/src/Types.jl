"""
    mutable struct MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem;

A model for the minimum variance portfolio allocation problem.

### Fields
- `w::Array{Float64,1}`: Optimal weights of the assets in the portfolio.
- `ḡ::Array{Float64,1}`: Expected growth rate vector of the assets.
- `Σ̂::Array{Float64,2}`: Covariance matrix of the asset returns.
"""
mutable struct MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem;
    
    w::Array{Float64,1} # optimal weights
    ḡ::Array{Float64,1} # expected growth rate of the optimal portfolio
    Σ̂::Array{Float64,2} # covariance matrix of the optimal portfolio
    R::Float64 # target return (not used in min-var problem)

    # constructor -
    MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem() = new();
end