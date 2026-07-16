# MCP Wire Capture

Real JSON-RPC messages captured from the executed `CHEME-142-M4-Example-MCP-ChemETools-Watch-Demo.ipynb` (verbose client mode). Each message is reproduced verbatim from the notebook's wire output, then pretty-printed here for readability; key order matches the wire. `→` is a message the client sent, `←` a message the server returned. Task 2.2 (`CHEME-142-M4-MCP-ProtocolMessages-Read-Pages.ipynb`) quotes these blocks.

Protocol constants: `protocolVersion = "2025-06-18"`; server `"cheme-142-m4-cheme-tools"` v`1.0.0`.

## Initialize handshake

The MCP session opens with an `initialize` request/response, followed by the `notifications/initialized` notification (a notification carries no `id` and receives no response).

**Client → Server**

```json
{
  "method": "initialize",
  "id": 1,
  "params": {
    "clientInfo": {
      "name": "cheme-142-notebook-client",
      "version": "1.0.0"
    },
    "protocolVersion": "2025-06-18",
    "capabilities": {}
  },
  "jsonrpc": "2.0"
}
```

**Server → Client**

```json
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "tools": {}
    },
    "serverInfo": {
      "name": "cheme-142-m4-cheme-tools",
      "version": "1.0.0"
    }
  }
}
```

**Client → Server**

```json
{
  "method": "notifications/initialized",
  "jsonrpc": "2.0"
}
```

## Tool discovery (`tools/list`)

The client requests the tool catalog; the server returns one descriptor per tool (`name`, `description`, `inputSchema`), sorted alphabetically by name.

**Client → Server**

```json
{
  "method": "tools/list",
  "id": 2,
  "params": {},
  "jsonrpc": "2.0"
}
```

**Server → Client**

```json
{
  "id": 2,
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "antoine_vapor_pressure",
        "inputSchema": {
          "properties": {
            "T": {
              "type": "number",
              "description": "Temperature in K, inside the species' valid range"
            },
            "species": {
              "type": "string",
              "description": "Species name, e.g. water, acetone, ethanol, benzene"
            }
          },
          "required": [
            "species",
            "T"
          ],
          "type": "object"
        },
        "description": "Saturation pressure (bar) of a named species at temperature T (K) from the Antoine equation."
      },
      {
        "name": "ideal_gas_solve",
        "inputSchema": {
          "properties": {
            "T": {
              "type": "number",
              "description": "Temperature in K"
            },
            "P": {
              "type": "number",
              "description": "Pressure in Pa"
            },
            "V": {
              "type": "number",
              "description": "Volume in m^3"
            },
            "n": {
              "type": "number",
              "description": "Amount in mol"
            }
          },
          "required": [],
          "type": "object"
        },
        "description": "Solve the ideal gas law PV = nRT for the one variable not provided (SI units: Pa, m^3, mol, K)."
      },
      {
        "name": "molecular_weight",
        "inputSchema": {
          "properties": {
            "formula": {
              "type": "string",
              "description": "Chemical formula, element symbols with optional integer counts"
            }
          },
          "required": [
            "formula"
          ],
          "type": "object"
        },
        "description": "Compute the molar mass (g/mol) of a chemical formula, e.g. H2O or C6H12O6."
      }
    ]
  }
}
```

## Tool call: `molecular_weight` (success)

A `tools/call` request naming the tool and its `arguments`; the result carries a `content` array and `isError` is `false`. The tool's text payload is itself a JSON string.

**Client → Server**

```json
{
  "method": "tools/call",
  "id": 3,
  "params": {
    "name": "molecular_weight",
    "arguments": {
      "formula": "C6H12O6"
    }
  },
  "jsonrpc": "2.0"
}
```

**Server → Client**

```json
{
  "id": 3,
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "text": "{\"units\":\"g/mol\",\"formula\":\"C6H12O6\",\"molar_mass\":180.156}",
        "type": "text"
      }
    ],
    "isError": false
  }
}
```

## Tool call: `antoine_vapor_pressure` (success)

A second successful `tools/call`, returning a saturation pressure the server reads from its Antoine coefficient table.

**Client → Server**

```json
{
  "method": "tools/call",
  "id": 4,
  "params": {
    "name": "antoine_vapor_pressure",
    "arguments": {
      "T": 320.0,
      "species": "acetone"
    }
  },
  "jsonrpc": "2.0"
}
```

**Server → Client**

```json
{
  "id": 4,
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "text": "{\"units\":\"bar\",\"T\":320.0,\"Psat\":0.7261,\"species\":\"acetone\"}",
        "type": "text"
      }
    ],
    "isError": false
  }
}
```

## Failure: unknown tool (protocol error)

The request names a tool the server does not have. The server returns a JSON-RPC error object with code `-32602` (invalid params) and no `result`.

**Client → Server**

```json
{
  "method": "tools/call",
  "id": 5,
  "params": {
    "name": "gibbs_energy",
    "arguments": {
      "species": "water"
    }
  },
  "jsonrpc": "2.0"
}
```

**Server → Client**

```json
{
  "error": {
    "message": "Unknown tool: gibbs_energy",
    "code": -32602
  },
  "id": 5,
  "jsonrpc": "2.0"
}
```

## Failure: out-of-range argument (tool execution error)

The request is valid and the tool runs, but the temperature is outside the species' valid range. The server returns a normal `result` with `isError` set to `true` and the failure text in `content`.

**Client → Server**

```json
{
  "method": "tools/call",
  "id": 6,
  "params": {
    "name": "antoine_vapor_pressure",
    "arguments": {
      "T": 500.0,
      "species": "water"
    }
  },
  "jsonrpc": "2.0"
}
```

**Server → Client**

```json
{
  "id": 6,
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "text": "ArgumentError: T = 500.0 K is outside the valid range [255.9, 373.0] K for water",
        "type": "text"
      }
    ],
    "isError": true
  }
}
```
