# Chapter 3: MCP Specification with Examples

## 3.1 Protocol Foundation {#protocol-foundation}

MCP is built on **JSON-RPC 2.0**, a stateless, lightweight remote procedure call protocol.

### 3.1.1 Base Message Format {#base-message-format}

All MCP messages follow this structure:

```json
{
    "jsonrpc": "2.0",
    "id": <number | string | null>,
    "method": "<method_name>",
    "params": {<parameters>}
}
```

**Fields**:
- `jsonrpc`: Always "2.0"
- `id`: Unique identifier for request-response matching (null for notifications)
- `method`: The operation to perform
- `params`: Method-specific parameters

## 3.2 Server Primitives {#server-primitives}

MCP servers can expose three types of primitives:

### 3.2.1 Resources {#resources}

**Definition**: Resources represent data or content that can be read by the LLM.

**Characteristics**:
- URI-addressable
- Can be static or dynamic
- Support subscriptions for updates
- Include metadata (MIME type, description)

**Resource Schema**:

```typescript
interface Resource {
    uri: string;        // Unique identifier (e.g., "file:///path" or "db://table")
    name: string;       // Human-readable name
    description?: string; // Optional description
    mimeType?: string;  // Content type (e.g., "text/plain", "application/json")
}

interface ResourceContent {
    uri: string;
    mimeType?: string;
    text?: string;      // Text content
    blob?: string;      // Base64-encoded binary content
}
```

**Example: File System Resources**

```python
# MCP Server exposing filesystem as resources
from mcp.server import Server
from mcp.types import Resource, ResourceContent
import os

class FileSystemServer(Server):
    def __init__(self, root_path: str):
        super().__init__("filesystem")
        self.root_path = root_path
    
    async def list_resources(self) -> list[Resource]:
        """List all files in directory as resources"""
        resources = []
        
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.root_path)
                
                resources.append(Resource(
                    uri=f"file:///{rel_path}",
                    name=file,
                    description=f"File: {rel_path}",
                    mimeType=self._get_mime_type(file)
                ))
        
        return resources
    
    async def read_resource(self, uri: str) -> ResourceContent:
        """Read file content"""
        # Extract path from URI
        path = uri.replace("file:///", "")
        full_path = os.path.join(self.root_path, path)
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        return ResourceContent(
            uri=uri,
            mimeType=self._get_mime_type(path),
            text=content
        )
    
    def _get_mime_type(self, filename: str) -> str:
        """Determine MIME type from extension"""
        ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.py': 'text/x-python',
            '.js': 'text/javascript'
        }
        return mime_types.get(ext, 'application/octet-stream')
```

**Client Usage**:

```python
# Using resources from client
async def read_project_files(client: MCPClient):
    # List available resources
    resources = await client.list_resources()
    
    print(f"Found {len(resources)} files")
    
    # Read specific resource
    readme = await client.read_resource("file:///README.md")
    print(f"README content: {readme.text}")
    
    # Subscribe to changes
    await client.subscribe_resource("file:///config.json")
```

**Resource Update Notifications**:

```json
{
    "jsonrpc": "2.0",
    "method": "notifications/resources/updated",
    "params": {
        "uri": "file:///config.json"
    }
}
```

### 3.2.2 Tools {#tools}

**Definition**: Tools are functions that the LLM can invoke to perform actions or retrieve information.

**Characteristics**:
- Defined with JSON Schema for parameters
- Return structured results
- Can have side effects
- Support both synchronous and asynchronous execution

**Tool Schema**:

```typescript
interface Tool {
    name: string;       // Unique tool identifier
    description: string; // What the tool does
    inputSchema: {      // JSON Schema for parameters
        type: "object";
        properties: {[key: string]: any};
        required?: string[];
    };
}

interface ToolResult {
    content: Array<{
        type: "text" | "image" | "resource";
        text?: string;
        data?: string;
        mimeType?: string;
    }>;
    isError?: boolean;
}
```

**Example: Database Query Tool**

```python
from mcp.server import Server
from mcp.types import Tool, ToolResult, TextContent
import sqlite3

class DatabaseServer(Server):
    def __init__(self, db_path: str):
        super().__init__("database")
        self.db_path = db_path
    
    async def list_tools(self) -> list[Tool]:
        """Declare available tools"""
        return [
            Tool(
                name="execute_query",
                description="Execute a SQL SELECT query on the database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT statement to execute"
                        },
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of rows to return",
                            "default": 100
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_schema",
                description="Get database schema information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Specific table name (optional)"
                        }
                    }
                }
            ),
            Tool(
                name="insert_record",
                description="Insert a new record into a table",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "Table name"
                        },
                        "data": {
                            "type": "object",
                            "description": "Column-value pairs to insert"
                        }
                    },
                    "required": ["table", "data"]
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """Execute tool"""
        try:
            if name == "execute_query":
                return await self._execute_query(
                    arguments["query"],
                    arguments.get("limit", 100)
                )
            elif name == "get_schema":
                return await self._get_schema(
                    arguments.get("table_name")
                )
            elif name == "insert_record":
                return await self._insert_record(
                    arguments["table"],
                    arguments["data"]
                )
        except Exception as e:
            return ToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
    
    async def _execute_query(self, query: str, limit: int) -> ToolResult:
        """Execute SELECT query"""
        # Validate query is SELECT only
        if not query.strip().upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add LIMIT clause
        query = f"{query} LIMIT {limit}"
        cursor.execute(query)
        
        # Fetch results
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        # Format as table
        result_text = f"Columns: {', '.join(columns)}\\n\\n"
        for row in rows:
            result_text += " | ".join(str(val) for val in row) + "\\n"
        
        conn.close()
        
        return ToolResult(
            content=[TextContent(
                type="text",
                text=f"Query returned {len(rows)} rows:\\n\\n{result_text}"
            )]
        )
    
    async def _get_schema(self, table_name: str = None) -> ToolResult:
        """Get database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if table_name:
            # Get specific table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            schema_text = f"Schema for table '{table_name}':\\n\\n"
            for col in columns:
                schema_text += f"  {col[1]} {col[2]}"
                if col[3]:  # NOT NULL
                    schema_text += " NOT NULL"
                if col[5]:  # PRIMARY KEY
                    schema_text += " PRIMARY KEY"
                schema_text += "\\n"
        else:
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            schema_text = f"Database contains {len(tables)} tables:\\n\\n"
            for table in tables:
                schema_text += f"  - {table[0]}\\n"
        
        conn.close()
        
        return ToolResult(
            content=[TextContent(type="text", text=schema_text)]
        )
```

**Client Tool Usage**:

```python
# Using tools from client
async def query_database(client: MCPClient):
    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {[t.name for t in tools]}")
    
    # Get database schema
    schema = await client.call_tool("get_schema", {})
    print(schema.content[0].text)
    
    # Execute query
    result = await client.call_tool(
        "execute_query",
        {
            "query": "SELECT * FROM users WHERE active = 1",
            "limit": 10
        }
    )
    
    if not result.isError:
        print(result.content[0].text)
    else:
        print(f"Error: {result.content[0].text}")
```

**Tool Call Flow**:

```json
// Request
{
    "jsonrpc": "2.0",
    "id": 42,
    "method": "tools/call",
    "params": {
        "name": "execute_query",
        "arguments": {
            "query": "SELECT name, email FROM users LIMIT 5",
            "limit": 5
        }
    }
}

// Response
{
    "jsonrpc": "2.0",
    "id": 42,
    "result": {
        "content": [
            {
                "type": "text",
                "text": "Query returned 5 rows:\\n\\nColumns: name, email\\n\\nJohn Doe | john@example.com\\nJane Smith | jane@example.com\\n..."
            }
        ],
        "isError": false
    }
}
```

### 3.2.3 Prompts {#prompts}

**Definition**: Prompts are pre-defined, reusable templates that help structure LLM interactions.

**Characteristics**:
- Parameterizable templates
- Consistent formatting
- Can include context from resources
- Support for multi-turn conversations

**Prompt Schema**:

```typescript
interface Prompt {
    name: string;       // Unique prompt identifier
    description: string; // What the prompt does
    arguments?: Array<{ // Template parameters
        name: string;
        description: string;
        required?: boolean;
    }>;
}

interface PromptMessage {
    role: "user" | "assistant";
    content: {
        type: "text" | "image" | "resource";
        text?: string;
        data?: string;
        mimeType?: string;
    };
}
```

**Example: Code Review Prompts**

```python
from mcp.server import Server
from mcp.types import Prompt, PromptMessage, TextContent

class CodeReviewServer(Server):
    def __init__(self):
        super().__init__("code-review")
    
    async def list_prompts(self) -> list[Prompt]:
        """Declare available prompts"""
        return [
            Prompt(
                name="review_code",
                description="Comprehensive code review with best practices",
                arguments=[
                    {
                        "name": "code",
                        "description": "Code to review",
                        "required": True
                    },
                    {
                        "name": "language",
                        "description": "Programming language",
                        "required": True
                    },
                    {
                        "name": "focus_areas",
                        "description": "Specific areas to focus on (security, performance, style)",
                        "required": False
                    }
                ]
            ),
            Prompt(
                name="explain_code",
                description="Generate detailed code explanation",
                arguments=[
                    {
                        "name": "code",
                        "description": "Code to explain",
                        "required": True
                    },
                    {
                        "name": "audience",
                        "description": "Target audience (beginner, intermediate, expert)",
                        "required": False
                    }
                ]
            ),
            Prompt(
                name="suggest_tests",
                description="Suggest unit tests for code",
                arguments=[
                    {
                        "name": "code",
                        "description": "Code to test",
                        "required": True
                    },
                    {
                        "name": "test_framework",
                        "description": "Testing framework to use",
                        "required": False
                    }
                ]
            )
        ]
    
    async def get_prompt(self, name: str, arguments: dict) -> list[PromptMessage]:
        """Generate prompt messages"""
        if name == "review_code":
            return await self._review_code_prompt(arguments)
        elif name == "explain_code":
            return await self._explain_code_prompt(arguments)
        elif name == "suggest_tests":
            return await self._suggest_tests_prompt(arguments)
    
    async def _review_code_prompt(self, args: dict) -> list[PromptMessage]:
        """Generate code review prompt"""
        code = args["code"]
        language = args["language"]
        focus_areas = args.get("focus_areas", "all aspects")
        
        system_prompt = f"""You are an expert code reviewer specializing in {language}.
Review the following code focusing on: {focus_areas}.

Provide feedback on:
1. Code quality and maintainability
2. Potential bugs or issues
3. Performance considerations
4. Security vulnerabilities
5. Best practices and style
6. Suggested improvements

Be constructive and provide specific examples."""
        
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"{system_prompt}\\n\\n```{language}\\n{code}\\n```"
                )
            )
        ]
    
    async def _explain_code_prompt(self, args: dict) -> list[PromptMessage]:
        """Generate code explanation prompt"""
        code = args["code"]
        audience = args.get("audience", "intermediate")
        
        audience_guidance = {
            "beginner": "Use simple terms, explain basic concepts, avoid jargon",
            "intermediate": "Assume basic programming knowledge, explain design patterns",
            "expert": "Focus on advanced techniques, optimization, and architecture"
        }
        
        prompt_text = f"""Explain the following code for a {audience} developer.

{audience_guidance[audience]}

Explain:
1. What the code does (high-level purpose)
2. How it works (step-by-step breakdown)
3. Key concepts or patterns used
4. Any important details or gotchas

Code:

{code}
"""
        
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text)
            )
        ]
    
    async def _suggest_tests_prompt(self, args: dict) -> list[PromptMessage]:
        """Generate test suggestion prompt"""
        code = args["code"]
        framework = args.get("test_framework", "unittest")
        
        prompt_text = f"""Suggest comprehensive unit tests for the following code using {framework}.

Include tests for:
1. Happy path scenarios
2. Edge cases
3. Error conditions
4. Boundary values

Provide:
- Test case descriptions
- Sample test code
- Expected behaviors
- Mocking strategies if needed

Code to test:

{code}
"""
        
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=prompt_text)
            )
        ]
```

**Client Prompt Usage**:

```python
# Using prompts from client
async def use_code_review_prompt(client: MCPClient):
    # List available prompts
    prompts = await client.list_prompts()
    print(f"Available prompts: {[p.name for p in prompts]}")
    
    # Get specific prompt
    code_to_review = """
def process_payment(amount, card_number):
    # Process payment
    result = payment_gateway.charge(amount, card_number)
    return result
"""
    
    messages = await client.get_prompt(
        "review_code",
        {
            "code": code_to_review,
            "language": "python",
            "focus_areas": "security, error handling"
        }
    )
    
    # Messages are ready to send to LLM
    for msg in messages:
        print(f"{msg.role}: {msg.content.text}")
```

## 3.3 Client Primitives {#client-primitives}

### 3.3.1 Roots {#roots}

**Definition**: Roots represent the client's context boundaries - directories, projects, or data sources the client has access to.

**Purpose**: Allows servers to understand what resources the client can provide.

**Example**:

```json
{
    "jsonrpc": "2.0",
    "method": "notifications/roots/list_changed",
    "params": {
        "roots": [
            {
                "uri": "file:///Users/dev/project",
                "name": "My Project"
            },
            {
                "uri": "file:///Users/dev/documents",
                "name": "Documents"
            }
        ]
    }
}
```

### 3.3.2 Sampling {#sampling}

**Definition**: Sampling allows servers to request LLM completions from the client.

**Purpose**: Enables servers to use the LLM for processing without having direct access.

**Use Cases**:
- Content generation within server logic
- Decision-making in server workflows
- Text analysis before returning results

**Example**:

```python
# Server requesting sampling from client
async def analyze_sentiment(self, text: str) -> str:
    """Use client's LLM to analyze sentiment"""
    
    # Request LLM completion from client
    result = await self.request_sampling({
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": f"Analyze the sentiment of this text (positive/negative/neutral): {text}"
                }
            }
        ],
        "maxTokens": 10
    })
    
    return result.content
```

**Sampling Request**:

```json
{
    "jsonrpc": "2.0",
    "id": 15,
    "method": "sampling/createMessage",
    "params": {
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Summarize this in 3 words: The quick brown fox jumps over the lazy dog"
                }
            }
        ],
        "maxTokens": 10,
        "temperature": 0.7
    }
}
```

## 3.4 Complete MCP Message Catalog {#complete-mcp-message-catalog}

### 3.4.1 Initialization {#initialization}

```json
// Client → Server: Initialize connection
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "1.0",
        "capabilities": {
            "roots": true,
            "sampling": true
        },
        "clientInfo": {
            "name": "ExampleClient",
            "version": "1.0.0"
        }
    }
}

// Server → Client: Initialization response
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "1.0",
        "capabilities": {
            "resources": true,
            "tools": true,
            "prompts": true
        },
        "serverInfo": {
            "name": "ExampleServer",
            "version": "2.1.0"
        }
    }
}
```

### 3.4.2 Resources Methods {#resources-methods}

```json
// List resources
{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "resources/list"
}

// Read resource
{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "resources/read",
    "params": {
        "uri": "file:///path/to/resource"
    }
}

// Subscribe to resource updates
{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "resources/subscribe",
    "params": {
        "uri": "file:///path/to/resource"
    }
}

// Unsubscribe from resource
{
    "jsonrpc": "2.0",
    "id": 5,
    "method": "resources/unsubscribe",
    "params": {
        "uri": "file:///path/to/resource"
    }
}
```

### 3.4.3 Tools Methods {#tools-methods}

```json
// List tools
{
    "jsonrpc": "2.0",
    "id": 6,
    "method": "tools/list"
}

// Call tool
{
    "jsonrpc": "2.0",
    "id": 7,
    "method": "tools/call",
    "params": {
        "name": "tool_name",
        "arguments": {
            "param1": "value1",
            "param2": 123
        }
    }
}
```

### 3.4.4 Prompts Methods {#prompts-methods}

```json
// List prompts
{
    "jsonrpc": "2.0",
    "id": 8,
    "method": "prompts/list"
}

// Get prompt
{
    "jsonrpc": "2.0",
    "id": 9,
    "method": "prompts/get",
    "params": {
        "name": "prompt_name",
        "arguments": {
            "arg1": "value1"
        }
    }
}
```

### 3.4.5 Notifications {#notifications}

```json
// Resource updated (Server → Client)
{
    "jsonrpc": "2.0",
    "method": "notifications/resources/updated",
    "params": {
        "uri": "file:///path/to/resource"
    }
}

// Tool list changed (Server → Client)
{
    "jsonrpc": "2.0",
    "method": "notifications/tools/list_changed"
}

// Progress notification
{
    "jsonrpc": "2.0",
    "method": "notifications/progress",
    "params": {
        "progressToken": "token123",
        "progress": 50,
        "total": 100
    }
}
```

## 3.5 Complete Working Example {#complete-working-example}

Here's a complete example showing all primitives together:

```python
# complete_mcp_example.py
from mcp.server import Server
from mcp.types import Tool, Resource, Prompt, ToolResult, TextContent
import json

class CompleteMCPServer(Server):
    """Demonstration server with all primitive types"""
    
    def __init__(self):
        super().__init__("complete-example")
        self.data_store = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ]
        }
    
    # ===== RESOURCES =====
    async def list_resources(self) -> list[Resource]:
        return [
            Resource(
                uri="data://users",
                name="User Database",
                description="List of all users",
                mimeType="application/json"
            ),
            Resource(
                uri="data://stats",
                name="System Statistics",
                description="Current system stats",
                mimeType="application/json"
            )
        ]
    
    async def read_resource(self, uri: str):
        if uri == "data://users":
            return {
                "contents": [
                    TextContent(
                        type="text",
                        text=json.dumps(self.data_store["users"], indent=2)
                    )
                ]
            }
        elif uri == "data://stats":
            stats = {
                "total_users": len(self.data_store["users"]),
                "server_uptime": "2h 15m"
            }
            return {
                "contents": [
                    TextContent(
                        type="text",
                        text=json.dumps(stats, indent=2)
                    )
                ]
            }
    
    # ===== TOOLS =====
    async def list_tools(self) -> list[Tool]:
        return [
            Tool(
                name="add_user",
                description="Add a new user to the database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string", "format": "email"}
                    },
                    "required": ["name", "email"]
                }
            ),
            Tool(
                name="search_users",
                description="Search for users by name",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="delete_user",
                description="Delete a user by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "number"}
                    },
                    "required": ["user_id"]
                }
            )
        ]
    
    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        try:
            if name == "add_user":
                new_user = {
                    "id": len(self.data_store["users"]) + 1,
                    "name": arguments["name"],
                    "email": arguments["email"]
                }
                self.data_store["users"].append(new_user)
                
                # Notify clients that resource changed
                await self.notify_resource_updated("data://users")
                
                return ToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"User added successfully: {json.dumps(new_user)}"
                    )]
                )
            
            elif name == "search_users":
                query = arguments["query"].lower()
                results = [
                    u for u in self.data_store["users"]
                    if query in u["name"].lower()
                ]
                return ToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Found {len(results)} users:\\n{json.dumps(results, indent=2)}"
                    )]
                )
            
            elif name == "delete_user":
                user_id = arguments["user_id"]
                self.data_store["users"] = [
                    u for u in self.data_store["users"]
                    if u["id"] != user_id
                ]
                
                await self.notify_resource_updated("data://users")
                
                return ToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"User {user_id} deleted successfully"
                    )]
                )
        
        except Exception as e:
            return ToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )
    
    # ===== PROMPTS =====
    async def list_prompts(self) -> list[Prompt]:
        return [
            Prompt(
                name="user_report",
                description="Generate a user report",
                arguments=[
                    {"name": "format", "description": "Report format (summary/detailed)"}
                ]
            ),
            Prompt(
                name="welcome_email",
                description="Generate welcome email for new user",
                arguments=[
                    {"name": "user_name", "description": "Name of the user", "required": True}
                ]
            )
        ]
    
    async def get_prompt(self, name: str, arguments: dict):
        if name == "user_report":
            format_type = arguments.get("format", "summary")
            
            if format_type == "summary":
                prompt_text = f"""Generate a summary report of the user database.

Current users: {len(self.data_store['users'])}

Include:
- Total user count
- User growth trends
- Key statistics

Keep it concise (3-4 sentences)."""
            else:
                users_json = json.dumps(self.data_store["users"], indent=2)
                prompt_text = f"""Generate a detailed report of all users.

User Data:
{users_json}

Include:
- Individual user profiles
- Activity analysis
- Recommendations

Provide comprehensive analysis."""
            
            return [
                {"role": "user", "content": {"type": "text", "text": prompt_text}}
            ]
        
        elif name == "welcome_email":
            user_name = arguments["user_name"]
            prompt_text = f"""Write a warm, professional welcome email for our new user {user_name}.

Include:
- Friendly greeting
- Brief introduction to our service
- Next steps to get started
- Contact information for support

Tone: Professional but approachable
Length: 3-4 paragraphs"""
            
            return [
                {"role": "user", "content": {"type": "text", "text": prompt_text}}
            ]

# Run server
if __name__ == "__main__":
    server = CompleteMCPServer()
    server.run()
```

**Client Usage Example**:

```python
# client_example.py
from mcp import Client
import asyncio

async def demonstrate_all_features():
    # Initialize client
    client = Client("complete-example")
    await client.connect()
    
    print("=== MCP Complete Example ===\\n")
    
    # 1. LIST AND READ RESOURCES
    print("1. RESOURCES")
    resources = await client.list_resources()
    for res in resources:
        print(f"  - {res.name}: {res.uri}")
    
    # Read a resource
    users_data = await client.read_resource("data://users")
    print(f"\\nUsers Resource Content:\\n{users_data.contents[0].text}\\n")
    
    # 2. USE TOOLS
    print("2. TOOLS")
    tools = await client.list_tools()
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Add a user
    result = await client.call_tool("add_user", {
        "name": "Charlie",
        "email": "charlie@example.com"
    })
    print(f"\\nAdd User Result:\\n{result.content[0].text}\\n")
    
    # Search users
    search_result = await client.call_tool("search_users", {"query": "alice"})
    print(f"Search Result:\\n{search_result.content[0].text}\\n")
    
    # 3. USE PROMPTS
    print("3. PROMPTS")
    prompts = await client.list_prompts()
    for prompt in prompts:
        print(f"  - {prompt.name}: {prompt.description}")
    
    # Get a prompt
    report_prompt = await client.get_prompt("user_report", {"format": "summary"})
    print(f"\\nGenerated Prompt:\\n{report_prompt[0]['content']['text']}\\n")
    
    await client.disconnect()

# Run demonstration
asyncio.run(demonstrate_all_features())
```

## 3.6 Summary {#summary}

The MCP specification provides:

- ✅ **Three server primitives**: Resources (data), Tools (actions), Prompts (templates)
- ✅ **Two client primitives**: Roots (context), Sampling (LLM access)
- ✅ **JSON-RPC foundation**: Standardized message format
- ✅ **Flexible transport**: STDIO for local, HTTP+SSE for remote
- ✅ **Rich capabilities**: Discovery, subscriptions, notifications

This specification enables consistent, interoperable AI integrations across any platform.