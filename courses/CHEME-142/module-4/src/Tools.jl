# load coefficient tables (path relative to this file, so server.jl and Include.jl both work) -
const ATOMIC_WEIGHTS = JSON.parsefile(joinpath(@__DIR__, "..", "data", "atomic-weights.json"));
const ANTOINE_COEFFICIENTS = JSON.parsefile(joinpath(@__DIR__, "..", "data", "antoine-coefficients.json"));

"""
    _parse_formula(formula::AbstractString)::Dict{String,Int64}

Parse a chemical formula of element symbols with optional integer counts,
e.g. "C6H12O6". No parentheses. Throws `ArgumentError` on anything else.
"""
function _parse_formula(formula::AbstractString)::Dict{String,Int64}
    (isempty(formula)) && throw(ArgumentError("Empty formula"));
    counts = Dict{String,Int64}();
    matchedlength = 0;
    for m in eachmatch(r"([A-Z][a-z]?)(\d*)", formula)
        symbol = m.captures[1];
        (haskey(ATOMIC_WEIGHTS, symbol)) || throw(ArgumentError("Unknown element: $(symbol)"));
        count = isempty(m.captures[2]) ? 1 : parse(Int64, m.captures[2]);
        counts[symbol] = get(counts, symbol, 0) + count;
        matchedlength += length(m.match);
    end
    (matchedlength == length(formula)) || throw(ArgumentError("Unable to parse formula: $(formula)"));
    return counts;
end

function _tool_molecular_weight(arguments::Dict{String,Any})::String
    (haskey(arguments, "formula")) || throw(ArgumentError("Missing required argument: formula"));
    formula = arguments["formula"];
    counts = _parse_formula(formula);
    M = sum(count * ATOMIC_WEIGHTS[symbol] for (symbol, count) in counts);
    return JSON.json(Dict("formula" => formula, "molar_mass" => round(M, digits = 3), "units" => "g/mol"));
end

function _tool_antoine_vapor_pressure(arguments::Dict{String,Any})::String
    (haskey(arguments, "species")) || throw(ArgumentError("Missing required argument: species"));
    (haskey(arguments, "T")) || throw(ArgumentError("Missing required argument: T"));
    species = lowercase(arguments["species"]);
    (haskey(ANTOINE_COEFFICIENTS, species)) || throw(ArgumentError(
        "Unknown species: $(species). Known species: $(join(sort(collect(keys(ANTOINE_COEFFICIENTS))), ", "))"));
    p = ANTOINE_COEFFICIENTS[species];
    T = arguments["T"];
    (p["Tmin"] <= T <= p["Tmax"]) || throw(ArgumentError(
        "T = $(T) K is outside the valid range [$(p["Tmin"]), $(p["Tmax"])] K for $(species)"));
    Psat = 10.0^(p["A"] - p["B"] / (T + p["C"]));
    return JSON.json(Dict("species" => species, "T" => T, "Psat" => round(Psat, sigdigits = 5), "units" => "bar"));
end

function _tool_ideal_gas_solve(arguments::Dict{String,Any})::String
    R = 8.314; # J mol⁻¹ K⁻¹
    variables = ("P", "V", "n", "T");
    provided = [v for v in variables if haskey(arguments, v)];
    (length(provided) == 3) || throw(ArgumentError("Provide exactly three of P, V, n, T; got $(length(provided))"));
    unknown = only(v for v in variables if !(v in provided));
    P = get(arguments, "P", nothing); V = get(arguments, "V", nothing);
    n = get(arguments, "n", nothing); T = get(arguments, "T", nothing);
    value = (unknown == "P") ? n * R * T / V :
            (unknown == "V") ? n * R * T / P :
            (unknown == "n") ? P * V / (R * T) : P * V / (n * R);
    units = Dict("P" => "Pa", "V" => "m^3", "n" => "mol", "T" => "K")[unknown];
    return JSON.json(Dict("variable" => unknown, "value" => round(value, sigdigits = 6), "units" => units));
end

"""
    build_default_registry()::Dict{String,MCPTool}

The three ChemE tools this module's server ships with.
"""
function build_default_registry()::Dict{String,MCPTool}
    registry = Dict{String,MCPTool}();
    register!(registry, build(MCPTool, (
        name = "molecular_weight",
        description = "Compute the molar mass (g/mol) of a chemical formula, e.g. H2O or C6H12O6.",
        inputschema = Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}(
                "formula" => Dict{String,Any}("type" => "string", "description" => "Chemical formula, element symbols with optional integer counts")),
            "required" => ["formula"]),
        handler = _tool_molecular_weight)));
    register!(registry, build(MCPTool, (
        name = "antoine_vapor_pressure",
        description = "Saturation pressure (bar) of a named species at temperature T (K) from the Antoine equation.",
        inputschema = Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}(
                "species" => Dict{String,Any}("type" => "string", "description" => "Species name, e.g. water, acetone, ethanol, benzene"),
                "T" => Dict{String,Any}("type" => "number", "description" => "Temperature in K, inside the species' valid range")),
            "required" => ["species", "T"]),
        handler = _tool_antoine_vapor_pressure)));
    register!(registry, build(MCPTool, (
        name = "ideal_gas_solve",
        description = "Solve the ideal gas law PV = nRT for the one variable not provided (SI units: Pa, m^3, mol, K).",
        inputschema = Dict{String,Any}("type" => "object",
            "properties" => Dict{String,Any}(
                "P" => Dict{String,Any}("type" => "number", "description" => "Pressure in Pa"),
                "V" => Dict{String,Any}("type" => "number", "description" => "Volume in m^3"),
                "n" => Dict{String,Any}("type" => "number", "description" => "Amount in mol"),
                "T" => Dict{String,Any}("type" => "number", "description" => "Temperature in K")),
            "required" => []),
        handler = _tool_ideal_gas_solve)));
    return registry;
end
