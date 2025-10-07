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

"""
    mutable struct MyMarkowitzRiskyAssetOnlyPortfiolioChoiceProblem <: AbstractStochasticChoiceProblem

The `MyMarkowitzRiskyAssetOnlyPortfiolioChoiceProblem` mutable struct represents a [Minimum Variance portfolio problem](https://en.wikipedia.org/wiki/Modern_portfolio_theory) with risky assets only.

### Required fields
- `Σ::Array{Float64,2}`: The covariance matrix of the risky asset Returns
- `μ::Array{Float64,1}`: The expected returns of the risky assets
- `bounds::Array{Float64,2}`: The bounds on the risky asset weights
- `R::Float64`: The desired return of the portfolio
- `initial::Array{Float64,1}`: The initial portfolio weights    
"""
mutable struct MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem 

    # data -
    Σ::Array{Float64,2}
    μ::Array{Float64,1}
    bounds::Array{Float64,2}
    R::Float64
    initial::Array{Float64,1}

    # constructor
    MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem() = new();
end