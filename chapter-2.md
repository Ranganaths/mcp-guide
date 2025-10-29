# Chapter 2: MCP Architecture

## 2.1 Architectural Overview {#architectural-overview}

MCP follows a **client-server architecture** with three primary components:

```
┌─────────────────────────────────────────────────────────┐
│                      MCP Host                           │
│ ┌───────────────────────────────────────────────────┐ │
│ │      AI Application (e.g., Claude Desktop)       │ │
│ │                                                   │ │
│ │ ┌──────────────┐ ┌──────────────┐                │ │
│ │ │ MCP Client 1 │ │ MCP Client 2 │ ...            │ │
│ │ └──────┬───────┘ └──────┬───────┘                │ │
│ └─────────┼──────────────────┼────────────────────┘ │
└────────────┼──────────────────┼───────────────────────┘
             │                  │
  ┌────────▼────────┐ ┌─────▼──────────┐
  │ MCP Server 1    │ │ MCP Server 2   │
  │ (e.g., GitHub)  │ │ (e.g., Slack)  │
  └─────────────────┘ └────────────────┘
```

## 2.2 Core Components {#core-components}

### 2.2.1 Host {#host}

**Definition**: The host is the application that runs the AI model and initiates connections to MCP servers.

**Characteristics**:
- Contains the LLM (Large Language Model)
- Manages user interactions
- Orchestrates MCP client connections
- Handles response generation

**Examples**:
- Claude Desktop
- Visual Studio Code with AI extensions
- Custom AI applications
- IDEs with AI capabilities

**Responsibilities**:

```python
# Pseudo-code for Host responsibilities
class MCPHost:
    def __init__(self):
        self.llm = LanguageModel()
        self.clients = []
    
    def add_mcp_server(self, server_config):
        """Initialize connection to MCP server"""
        client = MCPClient(server_config)
        client.connect()
        self.clients.append(client)
    
    def process_user_query(self, query):
        """Process user input with LLM and MCP context"""
        # Get available tools from all MCP servers
        available_tools = []
        for client in self.clients:
            available_tools.extend(client.list_tools())
        
        # Let LLM decide which tools to use
        response = self.llm.generate(
            query=query,
            tools=available_tools
        )
        
        # Execute tool calls via MCP clients
        if response.tool_calls:
            results = self.execute_tools(response.tool_calls)
            # Generate final response with tool results
            final_response = self.llm.generate(
                query=query,
                tool_results=results
            )
            return final_response
        
        return response
```

### 2.2.2 Client {#client}

**Definition**: The MCP client is a protocol implementation embedded within the host that maintains connections to MCP servers.

**Characteristics**:
- One client per server connection (1:1 relationship)
- Lightweight protocol handler
- Manages message serialization/deserialization
- Handles connection lifecycle

**Key Functions**:

```typescript
// TypeScript MCP Client Interface
interface MCPClient {
    // Connection management
    connect(transport: Transport): Promise<void>;
    disconnect(): Promise<void>;
    
    // Server capability discovery
    initialize(): Promise<ServerCapabilities>;
    listResources(): Promise<Resource[]>;
    listTools(): Promise<Tool[]>;
    listPrompts(): Promise<Prompt[]>;
    
    // Resource operations
    readResource(uri: string): Promise<ResourceContent>;
    subscribeToResource(uri: string): Promise<void>;
    
    // Tool operations
    callTool(name: string, arguments: any): Promise<ToolResult>;
    
    // Prompt operations
    getPrompt(name: string, arguments: any): Promise<PromptMessage[]>;
    
    // Sampling (for server-initiated LLM requests)
    createMessage(params: SamplingParams): Promise<SamplingResult>;
}
```

**Example Client Implementation**:

```python
# Python MCP Client Example
from mcp import Client
import asyncio

class MyMCPClient:
    def __init__(self, server_name: str):
        self.client = Client(server_name)
        self.resources = []
        self.tools = []
    
    async def initialize(self):
        """Connect and discover server capabilities"""
        await self.client.connect()
        
        # Discover what the server offers
        self.resources = await self.client.list_resources()
        self.tools = await self.client.list_tools()
        
        print(f"Connected to {self.client.server_name}")
        print(f"Available resources: {len(self.resources)}")
        print(f"Available tools: {len(self.tools)}")
    
    async def use_tool(self, tool_name: str, **kwargs):
        """Execute a tool on the server"""
        result = await self.client.call_tool(
            name=tool_name,
            arguments=kwargs
        )
        return result

# Usage
async def main():
    client = MyMCPClient("github-server")
    await client.initialize()
    
    # Use a tool
    issues = await client.use_tool(
        "get_issues",
        repository="anthropics/mcp",
        state="open"
    )
    print(f"Found {len(issues)} open issues")

asyncio.run(main())
```

### 2.2.3 Server {#server}

**Definition**: An MCP server exposes specific capabilities (resources, tools, prompts) to clients through the standardized protocol.

**Characteristics**:
- Independent processes
- Implement specific functionality
- Can serve multiple clients simultaneously
- Stateful or stateless depending on design

**Server Primitives** (what servers can expose):
1. **Resources**: Data and content
2. **Tools**: Executable functions
3. **Prompts**: Pre-defined instruction templates

**Example Server Implementation**:

```python
# Python MCP Server Example
from mcp.server import Server
from mcp.types import Tool, Resource, TextContent

class WeatherMCPServer(Server):
    def __init__(self):
        super().__init__("weather-server")
    
    async def list_tools(self):
        """Declare available tools"""
        return [
            Tool(
                name="get_current_weather",
                description="Get current weather for a location",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or coordinates"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units"
                        }
                    },
                    "required": ["location"]
                }
            ),
            Tool(
                name="get_forecast",
                description="Get weather forecast for next 7 days",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "days": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 7
                        }
                    },
                    "required": ["location"]
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: dict):
        """Handle tool execution"""
        if name == "get_current_weather":
            return await self._get_current_weather(
                arguments["location"],
                arguments.get("units", "celsius")
            )
        elif name == "get_forecast":
            return await self._get_forecast(
                arguments["location"],
                arguments.get("days", 7)
            )
    
    async def _get_current_weather(self, location: str, units: str):
        """Fetch current weather data"""
        # In reality, call weather API
        weather_data = {
            "location": location,
            "temperature": 22 if units == "celsius" else 72,
            "condition": "Partly cloudy",
            "humidity": 65,
            "wind_speed": 12
        }
        
        return {
            "content": [
                TextContent(
                    type="text",
                    text=f"Current weather in {location}: "
                        f"{weather_data['temperature']}°{'C' if units == 'celsius' else 'F'}, "
                        f"{weather_data['condition']}"
                )
            ]
        }
    
    async def list_resources(self):
        """Expose weather data as resources"""
        return [
            Resource(
                uri="weather://alerts",
                name="Weather Alerts",
                description="Active weather alerts and warnings",
                mimeType="application/json"
            )
        ]
    
    async def read_resource(self, uri: str):
        """Provide resource content"""
        if uri == "weather://alerts":
            alerts = [
                {"severity": "moderate", "event": "Thunderstorm Watch"},
                {"severity": "minor", "event": "Wind Advisory"}
            ]
            return {
                "contents": [
                    TextContent(
                        type="text",
                        text=str(alerts)
                    )
                ]
            }

# Run the server
if __name__ == "__main__":
    server = WeatherMCPServer()
    server.run()
```

## 2.3 Communication Layers {#communication-layers}

MCP architecture consists of two key layers:

### 2.3.1 Protocol Layer {#protocol-layer}

**Purpose**: Defines the message format and communication patterns.

**Key Features**:
- JSON-RPC 2.0 based messaging
- Request-response pattern
- Notification support (one-way messages)
- Error handling

**Message Structure**:

```json
// Request Example
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "get_current_weather",
        "arguments": {
            "location": "San Francisco",
            "units": "fahrenheit"
        }
    }
}

// Response Example
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Current weather in San Francisco: 68°F, Foggy"
            }
        ]
    }
}

// Error Example
{
    "jsonrpc": "2.0",
    "id": 1,
    "error": {
        "code": -32602,
        "message": "Invalid params",
        "data": {
            "details": "Location is required"
        }
    }
}
```

### 2.3.2 Transport Layer {#transport-layer}

**Purpose**: Defines how messages are transmitted between client and server.

MCP supports two primary transport mechanisms:

#### A. STDIO Transport (Standard Input/Output)

**Best For**: Local integrations where server runs on same machine as client.

**Characteristics**:
- Simple process-based communication
- Server runs as subprocess
- Client writes to stdin, reads from stdout
- Ideal for desktop applications

**Example Configuration**:

```json
{
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_TOKEN": "${GITHUB_TOKEN}"
            }
        },
        "filesystem": {
            "command": "python",
            "args": ["-m", "mcp_server_filesystem", "/Users/documents"],
            "transport": "stdio"
        }
    }
}
```

**STDIO Communication Flow**:

```
Client Process             Server Process
     │                          │
     │ spawn subprocess          │
     ├──────────────────────────────>│
     │                          │
     │ write to stdin           │
     │ {"method": "tools/list"} │
     ├──────────────────────────────>│
     │                          │
     │                          │ read from stdin
     │                          │ process request
     │                          │ write to stdout
     │                          │
     │ read from stdout         │
     │ {"result": [...]}        │
     │<──────────────────────────────┤
```

#### B. HTTP + SSE Transport (Server-Sent Events)

**Best For**: Remote integrations, cloud-based servers, networked deployments.

**Characteristics**:
- HTTP for client requests
- Server-Sent Events (SSE) for server-initiated messages
- Supports remote connections
- Ideal for web applications and distributed systems

**Example Configuration**:

```json
{
    "mcpServers": {
        "remote-database": {
            "url": "https://api.example.com/mcp",
            "transport": "http",
            "headers": {
                "Authorization": "Bearer ${API_TOKEN}"
            }
        }
    }
}
```

**HTTP+SSE Communication Flow**:

```
Client (Browser/App)       Server (Remote)
     │                          │
     │ POST /mcp/initialize     │
     ├──────────────────────────────>│
     │<──────────────────────────────┤
     │ 200 OK {capabilities}    │
     │                          │
     │ GET /mcp/events (SSE)    │
     ├──────────────────────────────>│
     │                          │
     │ event: resource_updated  │
     │<──────────────────────────────┤
     │                          │
     │ POST /mcp/tools/call     │
     ├──────────────────────────────>│
     │<──────────────────────────────┤
     │ 200 OK {result}          │
```

**Implementation Example**:

```python
# HTTP+SSE Server Example
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import asyncio

app = FastAPI()

class HTTPMCPServer:
    def __init__(self):
        self.clients = []

    @app.post("/mcp/initialize")
    async def initialize(self):
        """Handle initialization"""
        return {
            "protocolVersion": "1.0",
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": False
            },
            "serverInfo": {
                "name": "example-server",
                "version": "1.0.0"
            }
        }

    @app.get("/mcp/events")
    async def events(self):
        """Server-Sent Events endpoint"""
        async def event_generator():
            while True:
                # Send periodic updates
                yield {
                    "event": "resource_updated",
                    "data": {"uri": "data://stats", "timestamp": "2025-01-15T10:00:00Z"}
                }
                await asyncio.sleep(30)
        
        return EventSourceResponse(event_generator())

    @app.post("/mcp/tools/call")
    async def call_tool(self, request: dict):
        """Handle tool execution"""
        tool_name = request["name"]
        arguments = request["arguments"]
        
        # Execute tool logic
        result = await self.execute_tool(tool_name, arguments)
        return {"result": result}
```

## 2.4 Data Flow Architecture {#data-flow-architecture}

### Complete Request-Response Flow: {#complete-request-response-flow}

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │ "What's the weather in NYC?"
       │
┌──────▼──────────────────────────────┐
│            MCP Host                  │
│ ┌────────────────────────────┐     │
│ │   LLM analyzes query       │     │
│ │   Determines tools needed  │     │
│ └──────────┬─────────────────┘     │
│            │                       │
│ ┌──────────▼─────────────────┐     │
│ │      MCP Client            │     │
│ │   Prepares tool call       │     │
│ └──────────┬─────────────────┘     │
└─────────────┼──────────────────────┘
              │
              │ JSON-RPC Request
              │ {"method": "tools/call",
              │  "params": {"name": "get_weather"}}
              │
┌─────────────▼──────────────────────┐
│           MCP Server               │
│ ┌────────────────────────────┐   │
│ │    Receives request        │   │
│ │    Validates parameters    │   │
│ └──────────┬─────────────────┘   │
│            │                     │
│ ┌──────────▼─────────────────┐   │
│ │   Executes tool logic      │   │
│ │   Calls weather API        │   │
│ └──────────┬─────────────────┘   │
│            │                     │
│ ┌──────────▼─────────────────┐   │
│ │    Formats response        │   │
│ └──────────┬─────────────────┘   │
└─────────────┼──────────────────────┘
              │
              │ JSON-RPC Response
              │ {"result": {"temp": 68, ...}}
              │
┌─────────────▼──────────────────────┐
│            MCP Host                │
│ ┌────────────────────────────┐   │
│ │   MCP Client receives      │   │
│ │   Passes to LLM            │   │
│ └──────────┬─────────────────┘   │
│            │                     │
│ ┌──────────▼─────────────────┐   │
│ │   LLM generates response   │   │
│ │   "It's 68°F in NYC..."    │   │
│ └──────────┬─────────────────┘   │
└─────────────┼──────────────────────┘
              │
┌─────────────▼─────┐
│      User         │
│   Sees response   │
└───────────────────┘
```

## 2.5 Scalability and Performance Considerations {#scalability-and-performance-considerations}

### 2.5.1 Multiple Server Connections {#multiple-server-connections}

A single host can maintain connections to multiple MCP servers simultaneously:

```python
# Managing multiple MCP servers
class MCPHostManager:
    def __init__(self):
        self.servers = {}
    
    async def add_server(self, name: str, config: dict):
        """Add and initialize MCP server connection"""
        client = MCPClient(config)
        await client.connect()
        self.servers[name] = client
    
    async def query_all_servers(self, capability_type: str):
        """Get capabilities from all connected servers"""
        all_capabilities = {}
        
        for name, client in self.servers.items():
            if capability_type == "tools":
                all_capabilities[name] = await client.list_tools()
            elif capability_type == "resources":
                all_capabilities[name] = await client.list_resources()
        
        return all_capabilities
    
    async def intelligent_routing(self, user_query: str):
        """Route query to most appropriate server"""
        # Analyze query to determine which server(s) to use
        if "code" in user_query.lower():
            return await self.servers["github"].process(user_query)
        elif "weather" in user_query.lower():
            return await self.servers["weather"].process(user_query)
        else:
            # Query multiple servers in parallel
            results = await asyncio.gather(*[
                server.process(user_query)
                for server in self.servers.values()
            ])
            return results
```

### 2.5.2 Connection Pooling {#connection-pooling}

For high-performance scenarios:

```python
# Connection pool for MCP servers
from asyncio import Semaphore

class MCPConnectionPool:
    def __init__(self, server_config: dict, max_connections: int = 10):
        self.config = server_config
        self.semaphore = Semaphore(max_connections)
        self.connections = []
    
    async def acquire(self) -> MCPClient:
        """Get available connection from pool"""
        async with self.semaphore:
            if not self.connections:
                client = MCPClient(self.config)
                await client.connect()
                return client
            return self.connections.pop()
    
    async def release(self, client: MCPClient):
        """Return connection to pool"""
        self.connections.append(client)
```

## 2.6 Architecture Patterns {#architecture-patterns}

### Pattern 1: Direct Integration {#pattern-1-direct-integration}

```
Application → MCP Client → MCP Server → Data Source
```

**Use when**: Single application needs direct access to specific data sources.

### Pattern 2: Aggregator Pattern {#pattern-2-aggregator-pattern}

```
Application → MCP Client → MCP Aggregator Server
                              ├→ MCP Server A
                              ├→ MCP Server B
                              └→ MCP Server C
```

**Use when**: Need to combine data from multiple sources behind single interface.

### Pattern 3: Gateway Pattern {#pattern-3-gateway-pattern}

```
Applications → Load Balancer → MCP Gateway
   (many)                        ├→ MCP Server Pool A
                                 ├→ MCP Server Pool B
                                 └→ MCP Server Pool C
```

**Use when**: Enterprise deployment with multiple applications and servers.

## 2.7 Summary {#summary}

The MCP architecture provides:

- ✅ **Separation of Concerns**: Clear boundaries between host, client, and server
- ✅ **Flexibility**: Support for both local (STDIO) and remote (HTTP+SSE) deployments
- ✅ **Scalability**: Ability to connect multiple servers and handle concurrent requests
- ✅ **Standardization**: Consistent protocol across all implementations

Understanding this architecture is crucial for building robust MCP integrations.