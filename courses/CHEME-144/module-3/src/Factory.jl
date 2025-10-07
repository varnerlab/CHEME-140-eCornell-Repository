function build(modeltype::Type{MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem}, 
    data::NamedTuple)

    # get stuff from data -
    w = data.w;
    R = data.R;
    ḡ = data.ḡ;
    Σ̂ = data.Σ̂;


    model = modeltype();
    model.w = w;
    model.R = R;
    model.ḡ = ḡ;
    model.Σ̂ = Σ̂;

    return model;
end

function build(modeltype::Type{MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem}, 
    data::NamedTuple)::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem

    # get stuff from data -
    Σ = data.Σ;
    μ = data.μ;
    bounds = data.bounds;
    R = data.R;
    initial = data.initial;

    model = modeltype();
    model.Σ = Σ;
    model.μ = μ;
    model.bounds = bounds;
    model.R = R;
    model.initial = initial;

    return model;
end