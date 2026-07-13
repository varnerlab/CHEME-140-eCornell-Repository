include(joinpath(@__DIR__, "..", "Include.jl"));

@testset "formula parsing and molecular weight" begin
    @test _parse_formula("H2O") == Dict("H" => 2, "O" => 1)
    @test isapprox(JSON.parse(_tool_molecular_weight(Dict{String,Any}("formula" => "H2O")))["molar_mass"], 18.015; atol = 1e-3)
    @test isapprox(JSON.parse(_tool_molecular_weight(Dict{String,Any}("formula" => "C6H12O6")))["molar_mass"], 180.156; atol = 1e-3)
    @test_throws ArgumentError _parse_formula("Xx2")
    @test_throws ArgumentError _parse_formula("H2O)")
    @test_throws ArgumentError _tool_molecular_weight(Dict{String,Any}())
end

@testset "antoine vapor pressure" begin
    r = JSON.parse(_tool_antoine_vapor_pressure(Dict{String,Any}("species" => "water", "T" => 350.0)));
    @test isapprox(r["Psat"], 0.4178; rtol = 1e-2) # hand check: 10^(4.6543 - 1435.264/285.152)
    @test_throws ArgumentError _tool_antoine_vapor_pressure(Dict{String,Any}("species" => "water", "T" => 500.0))
    @test_throws ArgumentError _tool_antoine_vapor_pressure(Dict{String,Any}("species" => "octane", "T" => 300.0))
end

@testset "ideal gas solve" begin
    r = JSON.parse(_tool_ideal_gas_solve(Dict{String,Any}("n" => 1.0, "T" => 273.15, "V" => 0.022414)));
    @test r["variable"] == "P"
    @test isapprox(r["value"], 101320.0; rtol = 1e-3) # 1 mol at STP ≈ 1 atm
    @test_throws ArgumentError _tool_ideal_gas_solve(Dict{String,Any}("n" => 1.0, "T" => 273.15))
end

@testset "protocol: dispatch and errors" begin
    registry = build_default_registry();
    r = process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "initialize", "params" => Dict())));
    @test r["result"]["protocolVersion"] == PROTOCOL_VERSION
    @test process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "method" => "notifications/initialized"))) === nothing
    @test process_message(registry, "this is not JSON")["error"]["code"] == -32700
    @test process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "resources/list")))["error"]["code"] == -32601
    r = process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "id" => 3, "method" => "tools/call",
        "params" => Dict("name" => "gibbs_energy", "arguments" => Dict()))));
    @test r["error"]["code"] == -32602
    r = process_message(registry, JSON.json(Dict("jsonrpc" => "2.0", "id" => 4, "method" => "tools/call",
        "params" => Dict("name" => "antoine_vapor_pressure", "arguments" => Dict("species" => "water", "T" => 500.0)))));
    @test r["result"]["isError"] == true # execution failure is NOT a protocol error
end

@testset "protocol: in-process session (run_session)" begin
    registry = build_default_registry();
    messages = [
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 1, "method" => "initialize", "params" => Dict())),
        JSON.json(Dict("jsonrpc" => "2.0", "method" => "notifications/initialized")),
        JSON.json(Dict("jsonrpc" => "2.0", "id" => 2, "method" => "tools/list")),
    ];
    responses = run_session(registry, messages);
    @test length(responses) == 2 # the notification gets no response
    tools = JSON.parse(responses[2])["result"]["tools"];
    @test [t["name"] for t in tools] == ["antoine_vapor_pressure", "ideal_gas_solve", "molecular_weight"]
end

@testset "wire: real subprocess" begin
    command = `$(joinpath(Sys.BINDIR, "julia")) --project=$(_ROOT) $(_PATH_TO_SERVER)`;
    connection = connect(command);
    @test initialize!(connection)["result"]["serverInfo"]["name"] == SERVER_NAME
    @test length(listtools(connection)["result"]["tools"]) == 3
    r = calltool(connection, "molecular_weight", Dict("formula" => "C6H12O6"));
    @test isapprox(JSON.parse(r["result"]["content"][1]["text"])["molar_mass"], 180.156; atol = 1e-3)
    close(connection);
    @test process_exited(connection.process)
end
