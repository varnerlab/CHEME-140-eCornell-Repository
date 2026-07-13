"""
    connect(command::Cmd; verbose::Bool = false)::MCPServerConnection

Launch an MCP server subprocess and return a connection handle. With
`verbose = true`, every wire message is echoed: `→` outgoing, `←` incoming.
"""
function connect(command::Cmd; verbose::Bool = false)::MCPServerConnection
    connection = MCPServerConnection();
    connection.process = open(command, "r+");
    connection.requestid = 0;
    connection.verbose = verbose;
    return connection;
end

function send!(connection::MCPServerConnection, message::Dict)
    line = JSON.json(message);
    (connection.verbose) && println("→ ", line);
    println(connection.process, line); flush(connection.process);
end

function receive!(connection::MCPServerConnection)::Dict
    line = readline(connection.process);
    (isempty(line)) && error("Server closed the connection");
    (connection.verbose) && println("← ", line);
    return JSON.parse(line);
end

function request!(connection::MCPServerConnection, method::String, params::Dict)::Dict
    connection.requestid += 1;
    send!(connection, Dict("jsonrpc" => "2.0", "id" => connection.requestid,
        "method" => method, "params" => params));
    return receive!(connection);
end

"""
    initialize!(connection::MCPServerConnection)::Dict

Run the MCP initialization handshake: an `initialize` request followed by
the `notifications/initialized` notification (which gets no response).
"""
function initialize!(connection::MCPServerConnection)::Dict
    response = request!(connection, "initialize", Dict(
        "protocolVersion" => PROTOCOL_VERSION,
        "capabilities" => Dict{String,Any}(),
        "clientInfo" => Dict("name" => "cheme-142-notebook-client", "version" => "1.0.0")));
    send!(connection, Dict("jsonrpc" => "2.0", "method" => "notifications/initialized"));
    return response;
end

listtools(connection::MCPServerConnection)::Dict = request!(connection, "tools/list", Dict{String,Any}());

function calltool(connection::MCPServerConnection, name::String, arguments::Dict)::Dict
    return request!(connection, "tools/call", Dict("name" => name, "arguments" => arguments));
end

"""
    Base.close(connection::MCPServerConnection)

Close the server's stdin (EOF ends its dispatch loop) and wait for exit.
"""
function Base.close(connection::MCPServerConnection)
    close(connection.process.in);
    wait(connection.process);
    return nothing;
end
