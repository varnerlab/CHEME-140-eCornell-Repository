function _objective_function(w::Array{Float64,1}, ḡ::Array{Float64,1}, 
    Σ̂::Array{Float64,2}, R::Float64, μ::Float64, ρ::Float64)
    
    # compue the objective function, and the penalty terms -
    f = w'*(Σ̂*w) - μ*(ḡ'*w - R) + ρ*(sum(w) - 1.0)^2 + ρ*sum(min.(0.0, w).^2);
    return f;
end


function compute(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem; 
    verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, T₁::Float64 = 0.1, 
    α::Float64 = 0.99, β::Float64 = 0.01, τ::Float64 = 10.0,
    μ::Float64 = 100.0, ρ::Float64 = 100.0)

    # initialize -
    has_converged = false;

    # unpack the model parameters -
    w = model.w;
    ḡ = model.ḡ;
    Σ̂ = model.Σ̂;
    R = model.R;

    # initialize parameters for simulated annealing -
    T = T₀; # initial T -
    current_w = w;
    current_f = _objective_function(current_w, ḡ, Σ̂, R, μ, ρ);
    
    # best solution found so far -
    w_best = current_w;
    f_best = current_f;
    KL = K;

    while has_converged == false
    
        accepted_counter = 0; 
        for _ in 1:KL
       
            # generate a new candidate solution -
            candidate_w = current_w + β * randn(length(w));
            candidate_f = _objective_function(candidate_w, ḡ, Σ̂, R, μ, ρ);

            # accept or reject the candidate solution -
            if candidate_f < current_f || rand() < exp((current_f - candidate_f) / T)
                current_w = candidate_w;
                current_f = candidate_f;

                # we've accepeted this move, update the counter -
                accepted_counter += 1;
            end
        
            # Compute the diff between current and best solution found so far
            if (current_f < f_best)
                w_best = current_w;
                f_best = current_f;
            end
        end

        # update KL -
        fraction_accepted = accepted_counter/KL; # what is the fraction of accepted moves
        
        # Case 1: we are accepting alot, so decrease the number of iterations
        if (fraction_accepted > 0.8)
            KL = ceil(Int, 0.75*KL);
        end

        # Case 2: not accepting many moves, so increase the number of iteratons
        if (fraction_accepted < 0.2)
            KL = ceil(Int, 1.5*KL);
        end

        # update penalty parameters and T -
        μ *= τ*μ;
        ρ *= τ*ρ;

        if (T ≤ T₁)
            has_converged = true;
        else
            T *= (α*T); # Not done yet, so decrease the T -
        end
    end

    # update the model with the optimal weights -
    model.w = w_best;

    # return the model -
    return model;
end