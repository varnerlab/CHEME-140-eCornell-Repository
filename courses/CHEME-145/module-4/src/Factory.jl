"""
    build(::Type{MyMDPProblemModel}, data::NamedTuple) -> MyMDPProblemModel

Build a finite MDP from keys `𝒮, 𝒜, T, R, γ`.
"""
function build(::Type{MyMDPProblemModel}, data::NamedTuple)::MyMDPProblemModel
    model = MyMDPProblemModel();
    model.𝒮 = data.𝒮;
    model.𝒜 = data.𝒜;
    model.T = data.T;
    model.R = data.R;
    model.γ = data.γ;
    return model;
end

"""
    build(::Type{MyValueIterationModel}, data::NamedTuple) -> MyValueIterationModel

Build value-iteration settings from keys `maxiterations, ϵ`.
"""
function build(::Type{MyValueIterationModel}, data::NamedTuple)::MyValueIterationModel
    model = MyValueIterationModel();
    model.maxiterations = data.maxiterations;
    model.ϵ = data.ϵ;
    return model;
end

"""
    build(::Type{MyPolicyIterationModel}, data::NamedTuple) -> MyPolicyIterationModel

Build policy-iteration settings from key `maxiterations`.
"""
function build(::Type{MyPolicyIterationModel}, data::NamedTuple)::MyPolicyIterationModel
    model = MyPolicyIterationModel();
    model.maxiterations = data.maxiterations;
    return model;
end

"""
    build(::Type{MyRectangularGridWorldModel}, data::NamedTuple) -> MyRectangularGridWorldModel

Build a grid world from keys `nrows, ncols, rewards`. Actions are 1=left, 2=right, 3=down, 4=up.
"""
function build(::Type{MyRectangularGridWorldModel}, data::NamedTuple)::MyRectangularGridWorldModel

    # unpack -
    nrows = data.nrows;
    ncols = data.ncols;
    rewards = data.rewards;

    # coordinate <-> state maps -
    coordinates = Dict{Int,Tuple{Int,Int}}();
    states = Dict{Tuple{Int,Int},Int}();
    position_index = 1;
    for x ∈ 1:ncols
        for y ∈ 1:nrows
            coordinate = (x,y);
            coordinates[position_index] = coordinate;
            states[coordinate] = position_index;
            position_index += 1;
        end
    end

    # action move vectors (Δx,Δy) -
    moves = Dict{Int,Tuple{Int,Int}}();
    moves[1] = (-1, 0);  # left
    moves[2] = ( 1, 0);  # right
    moves[3] = ( 0,-1);  # down
    moves[4] = ( 0, 1);  # up

    # build -
    model = MyRectangularGridWorldModel();
    model.nrows = nrows;
    model.ncols = ncols;
    model.coordinates = coordinates;
    model.states = states;
    model.moves = moves;
    model.rewards = rewards;
    return model;
end
