function _objective_function(w::Array{Float64,1}, ḡ::Array{Float64,1}, 
    Σ̂::Array{Float64,2}, R::Float64, μ::Float64, ρ::Float64)
    
    # compue the objective function, and the penalty terms -
    f = w'*(Σ̂*w) - μ*(ḡ'*w - R) + ρ*(sum(w) - 1.0)^2 + ρ*sum(min.(0.0, w).^2);
    return f;
end


function compute(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem; 
    verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, β::Float64 = 0.99, 
    μ::Float64 = 0.0, ρ::Float64 = 100.0)

    # initialize -
    has_converged = false;

    # unpack the model parameters -
    w = model.w;
    ḡ = model.ḡ;
    Σ̂ = model.Σ̂;
    R = model.R;

    # initialize parameters for simulated annealing -
    T = T₀;
    current_w = w;
    current_f = _objective_function(current_w, ḡ, Σ̂, R, μ, ρ);
    
    # best solution found so far -
    w_best = current_w;
    f_best = current_f;

    for _ in 1:K
       
        # generate a new candidate solution -
        candidate_w = current_w + β * randn(length(w));
        candidate_f = _objective_function(candidate_w, ḡ, Σ̂, R, μ, ρ);

        # accept or reject the candidate solution -
        if candidate_f < current_f || rand() < exp((current_f - candidate_f) / T)
            current_w = candidate_w;
            current_f = candidate_f;
        end

        # Compute the diff between current and best solution found so far
        if (current_f < f_best)
            w_best = current_w;
            f_best = current_f;
        end

        # update the temperature -
        T *= β;
    end

    # update the model with the optimal weights -
    model.w = w_best;

    # return the model -
    return model;
end