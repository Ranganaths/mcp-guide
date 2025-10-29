# Chapter 4: MCP and LLM Integration

## 4.1 How LLMs Work with MCP {#how-llms-work-with-mcp}

Model Context Protocol transforms how Large Language Models interact with external systems. Instead of being limited to their training data, LLMs can now dynamically access real-time information and execute actions through MCP servers.

### 4.1.1 The Integration Flow {#the-integration-flow}

```
┌────────────────────────────────────────────────────────────┐
│                      User Query                           │
│           "What are my open GitHub issues?"               │
└────────────────┬───────────────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────────────┐
│                   LLM Analysis                            │
│ • Understands intent: Fetch GitHub issues                │
│ • Identifies required tool: get_github_issues            │
│ • Determines parameters: state="open"                    │
└────────────────┬───────────────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────────────┐
│                 MCP Client Layer                          │
│ • Formats tool call request                              │
│ • Routes to appropriate MCP server                       │
│ • Handles communication protocol                         │
└────────────────┬───────────────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────────────┐
│               MCP Server (GitHub)                         │
│ • Receives tool call                                     │
│ • Authenticates with GitHub API                          │
│ • Fetches open issues                                    │
│ • Returns structured data                                │
└────────────────┬───────────────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────────────┐
│             LLM Response Generation                       │
│ • Receives tool results                                  │
│ • Synthesizes natural language response                  │
│ • Formats for user presentation                          │
└────────────────┬───────────────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────────────┐
│                   User Response                           │
│ "You have 3 open issues:                                 │
│ 1. Bug: Login page crashes (Priority: High)              │
│ 2. Feature: Add dark mode (Priority: Medium)             │
│ 3. Documentation: Update README (Priority: Low)"         │
└────────────────────────────────────────────────────────────┘
```

## 4.2 Tool Discovery and Selection {#tool-discovery-and-selection}

One of the most powerful aspects of MCP is how LLMs can discover and intelligently select tools.

### 4.2.1 Tool Discovery Process {#tool-discovery-process}

```python
# Example: LLM-powered tool discovery
class MCPLLMIntegration:
    def __init__(self, llm_client, mcp_clients: list):
        self.llm = llm_client
        self.mcp_clients = mcp_clients
        self.available_tools = {}
    
    async def initialize(self):
        """Discover all available tools from MCP servers"""
        for client in self.mcp_clients:
            server_name = client.server_name
            tools = await client.list_tools()
            
            for tool in tools:
                # Store with server context
                self.available_tools[f"{server_name}.{tool.name}"] = {
                    "client": client,
                    "tool": tool,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
        
        print(f"Discovered {len(self.available_tools)} tools")
    
    def get_tool_descriptions_for_llm(self) -> list:
        """Format tools for LLM consumption"""
        tool_descriptions = []
        
        for tool_id, tool_info in self.available_tools.items():
            tool_descriptions.append({
                "type": "function",
                "function": {
                    "name": tool_id,
                    "description": tool_info["description"],
                    "parameters": tool_info["parameters"]
                }
            })
        
        return tool_descriptions
    
    async def process_query(self, user_query: str):
        """Process user query with tool-enabled LLM"""
        # Prepare tools for LLM
        tools = self.get_tool_descriptions_for_llm()
        
        # Initial LLM call with tool availability
        response = await self.llm.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            tools=tools,
            tool_choice="auto"  # Let LLM decide
        )
        
        # Check if LLM wants to use tools
        if response.tool_calls:
            return await self.execute_tool_calls(response, user_query)
        else:
            return response.content
    
    async def execute_tool_calls(self, llm_response, original_query: str):
        """Execute tools requested by LLM"""
        tool_results = []
        
        for tool_call in llm_response.tool_calls:
            tool_id = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # Get the appropriate MCP client
            tool_info = self.available_tools[tool_id]
            client = tool_info["client"]
            actual_tool_name = tool_info["tool"].name
            
            # Execute via MCP
            result = await client.call_tool(actual_tool_name, arguments)
            
            tool_results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_id,
                "content": result.content[0].text
            })
        
        # Second LLM call with tool results
        final_response = await self.llm.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "user", "content": original_query},
                llm_response,  # Original response with tool calls
                *tool_results  # Tool execution results
            ]
        )
        
        return final_response.content


# Usage Example
async def main():
    # Initialize MCP clients
    github_client = MCPClient("github-server")
    database_client = MCPClient("database-server")
    weather_client = MCPClient("weather-server")
    
    await github_client.connect()
    await database_client.connect()
    await weather_client.connect()
    
    # Initialize LLM integration
    integration = MCPLLMIntegration(
        llm_client=anthropic.Anthropic(),
        mcp_clients=[github_client, database_client, weather_client]
    )
    
    await integration.initialize()
    
    # Process queries
    queries = [
        "What's the weather in San Francisco?",
        "Show me my open GitHub issues",
        "How many users do we have in the database?"
    ]
    
    for query in queries:
        print(f"\\nQuery: {query}")
        response = await integration.process_query(query)
        print(f"Response: {response}")
```

### 4.2.2 Intelligent Tool Selection {#intelligent-tool-selection}

The LLM automatically determines which tools to use based on the user's query:

```python
# Example: Multi-tool query
class SmartToolSelector:
    """Demonstrates how LLMs intelligently select tools"""
    
    async def handle_complex_query(self, query: str):
        """
        Query: "Create a summary report of my GitHub activity
        and current weather, then email it to me"
        
        LLM will:
        1. Identify multiple tools needed
        2. Determine execution order
        3. Chain results together
        """
        
        # LLM analyzes query and plans execution:
        execution_plan = {
            "steps": [
                {
                    "tool": "github.get_user_activity",
                    "params": {"days": 7},
                    "purpose": "Get GitHub activity data"
                },
                {
                    "tool": "weather.get_current",
                    "params": {"location": "user_location"},
                    "purpose": "Get weather information"
                },
                {
                    "tool": "email.compose_and_send",
                    "params": {
                        "to": "user@example.com",
                        "subject": "Weekly Report",
                        "body": "{{github_summary}} + {{weather_info}}"
                    },
                    "purpose": "Send email with combined data",
                    "depends_on": ["step_1", "step_2"]
                }
            ]
        }
        
        # Execute plan
        results = await self.execute_plan(execution_plan)
        return results
    
    async def execute_plan(self, plan: dict):
        """Execute multi-step tool plan"""
        step_results = {}
        
        for i, step in enumerate(plan["steps"]):
            # Check dependencies
            if "depends_on" in step:
                for dep in step["depends_on"]:
                    if dep not in step_results:
                        raise Exception(f"Dependency {dep} not satisfied")
            
            # Inject previous results into parameters
            params = self.resolve_parameters(
                step["params"],
                step_results
            )
            
            # Execute tool
            result = await self.mcp_integration.call_tool(
                step["tool"],
                params
            )
            
            step_results[f"step_{i+1}"] = result
        
        return step_results
```

## 4.3 Context Management {#context-management}

MCP enables sophisticated context management for LLMs.

### 4.3.1 Dynamic Context Loading {#dynamic-context-loading}

```python
class ContextManager:
    """Manages context from MCP resources for LLM"""
    
    def __init__(self, mcp_clients: list, max_context_tokens: int = 100000):
        self.mcp_clients = mcp_clients
        self.max_context_tokens = max_context_tokens
    
    async def build_context_for_query(self, query: str) -> str:
        """
        Intelligently build context from MCP resources
        based on query relevance
        """
        # 1. Identify relevant resources
        relevant_resources = await self.identify_relevant_resources(query)
        
        # 2. Load and rank by relevance
        resource_contents = []
        for resource in relevant_resources:
            content = await self.load_resource(resource)
            relevance_score = self.calculate_relevance(query, content)
            
            resource_contents.append({
                "content": content,
                "score": relevance_score,
                "source": resource.uri
            })
        
        # 3. Sort by relevance
        resource_contents.sort(key=lambda x: x["score"], reverse=True)
        
        # 4. Build context within token limit
        context = self.build_context_string(resource_contents)
        
        return context
    
    async def identify_relevant_resources(self, query: str) -> list:
        """Identify which MCP resources are relevant to query"""
        all_resources = []
        
        for client in self.mcp_clients:
            resources = await client.list_resources()
            all_resources.extend([
                {"client": client, "resource": r}
                for r in resources
            ])
        
        # Use embeddings or keywords to filter
        relevant = []
        query_lower = query.lower()
        
        for item in all_resources:
            resource = item["resource"]
            # Simple relevance check (in production, use embeddings)
            if any(keyword in resource.description.lower()
                   for keyword in query_lower.split()):
                relevant.append(item)
        
        return relevant
    
    async def load_resource(self, resource_item: dict) -> str:
        """Load resource content from MCP server"""
        client = resource_item["client"]
        resource = resource_item["resource"]
        
        content = await client.read_resource(resource.uri)
        return content.contents[0].text
    
    def calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score (simplified)"""
        # In production, use embeddings similarity
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words) if query_words else 0
    
    def build_context_string(self, ranked_contents: list) -> str:
        """Build context string within token limits"""
        context_parts = []
        estimated_tokens = 0
        
        for item in ranked_contents:
            # Rough token estimate (4 chars ≈ 1 token)
            item_tokens = len(item["content"]) // 4
            
            if estimated_tokens + item_tokens > self.max_context_tokens:
                break
            
            context_parts.append(
                f"--- Source: {item['source']} ---\\n{item['content']}\\n"
            )
            estimated_tokens += item_tokens
        
        return "\\n".join(context_parts)


# Usage with LLM
async def context_aware_query(query: str):
    """Process query with dynamic context from MCP"""
    context_manager = ContextManager(mcp_clients)
    
    # Build relevant context
    context = await context_manager.build_context_for_query(query)
    
    # Send to LLM with context
    response = await llm.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {
                "role": "user",
                "content": f"""Context from connected systems:

{context}

User Query: {query}

Please answer based on the provided context."""
            }
        ]
    )
    
    return response.content
```

### 4.3.2 Streaming Context Updates {#streaming-context-updates}

MCP supports real-time context updates through subscriptions:

```python
class StreamingContextManager:
    """Manages streaming context updates from MCP"""
    
    def __init__(self):
        self.active_subscriptions = {}
        self.context_cache = {}
    
    async def subscribe_to_resources(self, resource_uris: list):
        """Subscribe to resources for real-time updates"""
        for uri in resource_uris:
            client = self.get_client_for_uri(uri)
            
            # Subscribe to resource
            await client.subscribe_resource(uri)
            
            # Set up update handler
            client.on_resource_updated(uri, self.handle_resource_update)
            
            self.active_subscriptions[uri] = client
    
    async def handle_resource_update(self, uri: str):
        """Handle resource update notification"""
        print(f"Resource updated: {uri}")
        
        # Reload resource
        client = self.active_subscriptions[uri]
        updated_content = await client.read_resource(uri)
        
        # Update cache
        self.context_cache[uri] = {
            "content": updated_content.contents[0].text,
            "timestamp": datetime.now()
        }
        
        # Notify LLM of context change
        await self.notify_context_changed(uri)
    
    async def notify_context_changed(self, uri: str):
        """Inform LLM that context has changed"""
        # Could trigger re-generation or update UI
        print(f"Context changed: {uri}")
        
        # Example: If in active conversation, notify user
        if self.has_active_conversation():
            await self.send_notification(
                f"Information about {uri} has been updated. "
                "Would you like me to refresh my analysis?"
            )
```

## 4.4 Advanced LLM Integration Patterns {#advanced-llm-integration-patterns}

### 4.4.1 Agentic Behavior with MCP {#agentic-behavior-with-mcp}

```python
class AgenticMCPAssistant:
    """
    Autonomous agent that uses MCP tools to accomplish goals
    """
    
    def __init__(self, llm, mcp_integration):
        self.llm = llm
        self.mcp = mcp_integration
        self.conversation_history = []
        self.max_iterations = 10
    
    async def accomplish_goal(self, goal: str):
        """
        Agent autonomously breaks down goal and uses tools
        """
        self.conversation_history = [
            {
                "role": "system",
                "content": """You are an autonomous AI assistant with access to tools via MCP.
Your goal is to accomplish the user's request by:
1. Breaking down the task into steps
2. Using available tools to gather information
3. Making decisions based on results
4. Iterating until the goal is achieved

Think step-by-step and explain your reasoning."""
            },
            {
                "role": "user",
                "content": goal
            }
        ]
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\\n--- Iteration {iteration} ---")
            
            # Get LLM decision
            response = await self.llm.chat.completions.create(
                model="claude-3-5-sonnet-20241022",
                messages=self.conversation_history,
                tools=self.mcp.get_tool_descriptions_for_llm()
            )
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') else None
            })
            
            # Check if done
            if not response.tool_calls:
                print("Goal accomplished!")
                return response.content
            
            # Execute tool calls
            print(f"Executing {len(response.tool_calls)} tools...")
            tool_results = await self.execute_tools(response.tool_calls)
            
            # Add results to history
            for result in tool_results:
                self.conversation_history.append(result)
        
        return "Max iterations reached. Goal may not be fully accomplished."
    
    async def execute_tools(self, tool_calls):
        """Execute tools and return results"""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            print(f"  - Calling {tool_name} with {arguments}")
            
            try:
                result = await self.mcp.call_tool(tool_name, arguments)
                
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result.content[0].text
                })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": f"Error: {str(e)}"
                })
        
        return results


# Example usage
async def demo_agentic_behavior():
    agent = AgenticMCPAssistant(llm, mcp_integration)
    
    goal = """
    Analyze my GitHub repository's recent issues and pull requests,
    create a summary report, and email it to my team.
    """
    
    result = await agent.accomplish_goal(goal)
    print(f"\\nFinal Result:\\n{result}")
    
    # The agent will autonomously:
    # 1. Call github.list_issues tool
    # 2. Call github.list_pull_requests tool
    # 3. Analyze the data
    # 4. Call report.generate tool with the analysis
    # 5. Call email.send tool with the report
```

### 4.4.2 Multi-Turn Conversations with Context {#multi-turn-conversations-with-context}

```python
class ConversationalMCPAssistant:
    """
    Maintains conversation context across multiple turns
    with MCP integration
    """
    
    def __init__(self, llm, mcp_integration):
        self.llm = llm
        self.mcp = mcp_integration
        self.conversation_history = []
        self.context_resources = set()
    
    async def add_user_message(self, message: str):
        """Add user message and process"""
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        return await self.generate_response()
    
    async def generate_response(self):
        """Generate response with MCP tool access"""
        # Include relevant context from MCP resources
        context = await self.build_current_context()
        
        # Prepare messages with context
        messages = [
            {"role": "system", "content": f"Relevant Context:\\n{context}"}
        ] + self.conversation_history
        
        # Generate response
        response = await self.llm.chat.completions.create(
            model="claude-3-5-sonnet-20241022",
            messages=messages,
            tools=self.mcp.get_tool_descriptions_for_llm()
        )
        
        # Handle tool calls if any
        if response.tool_calls:
            return await self.handle_tool_calls(response)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        return response.content
    
    async def build_current_context(self):
        """Build context from subscribed resources"""
        context_parts = []
        
        for uri in self.context_resources:
            client = self.mcp.get_client_for_uri(uri)
            resource = await client.read_resource(uri)
            context_parts.append(resource.contents[0].text)
        
        return "\\n---\\n".join(context_parts)
    
    async def subscribe_to_context(self, resource_uri: str):
        """Add resource to conversation context"""
        self.context_resources.add(resource_uri)
        client = self.mcp.get_client_for_uri(resource_uri)
        await client.subscribe_resource(resource_uri)


# Example conversation
async def demo_conversation():
    assistant = ConversationalMCPAssistant(llm, mcp_integration)
    
    # Subscribe to user's project context
    await assistant.subscribe_to_context("file:///project/README.md")
    await assistant.subscribe_to_context("github://repo/issues")
    
    # Multi-turn conversation
    print(await assistant.add_user_message(
        "What's the current state of my project?"
    ))
    
    print(await assistant.add_user_message(
        "Create a task to fix the most critical issue"
    ))
    
    print(await assistant.add_user_message(
        "What did I just ask you to do?"
    ))
    # Assistant remembers the conversation context
```

## 4.5 Performance Optimization {#performance-optimization}

### 4.5.1 Parallel Tool Execution {#parallel-tool-execution}

```python
class ParallelMCPExecutor:
    """Execute multiple MCP tools in parallel"""
    
    async def execute_parallel(self, tool_calls: list):
        """Execute independent tool calls concurrently"""
        # Analyze dependencies
        execution_graph = self.build_dependency_graph(tool_calls)
        
        results = {}
        
        # Execute in waves based on dependencies
        for wave in execution_graph:
            # All tools in a wave can run in parallel
            tasks = [
                self.execute_single_tool(tool_call)
                for tool_call in wave
            ]
            
            wave_results = await asyncio.gather(*tasks)
            
            for tool_call, result in zip(wave, wave_results):
                results[tool_call.id] = result
        
        return results
    
    def build_dependency_graph(self, tool_calls: list) -> list:
        """
        Group tool calls into waves where each wave can execute in parallel
        """
        # Simple implementation - in production, analyze parameter dependencies
        # For now, assume all can run in parallel
        return [tool_calls]
    
    async def execute_single_tool(self, tool_call):
        """Execute single tool call"""
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        return await self.mcp.call_tool(tool_name, arguments)


# Example: Parallel execution
async def fetch_multiple_sources():
    """Fetch data from multiple sources in parallel"""
    executor = ParallelMCPExecutor()
    
    tool_calls = [
        {"id": "1", "function": {"name": "github.get_issues", "arguments": "{}"}},
        {"id": "2", "function": {"name": "weather.get_current", "arguments": "{}"}},
        {"id": "3", "function": {"name": "database.get_stats", "arguments": "{}"}}
    ]
    
    # All execute concurrently
    results = await executor.execute_parallel(tool_calls)
    
    # Results available simultaneously
    print(f"Fetched {len(results)} results in parallel")
```

### 4.5.2 Caching and Memoization {#caching-and-memoization}

```python
class CachedMCPClient:
    """MCP client with intelligent caching"""
    
    def __init__(self, base_client):
        self.client = base_client
        self.cache = {}
        self.cache_ttl = 60  # seconds
    
    async def call_tool(self, name: str, arguments: dict):
        """Call tool with caching"""
        # Create cache key
        cache_key = self.create_cache_key(name, arguments)
        
        # Check cache
        if cache_key in self.cache:
            cached_entry = self.cache[cache_key]
            
            # Check if still valid
            if time.time() - cached_entry["timestamp"] < self.cache_ttl:
                print(f"Cache hit for {name}")
                return cached_entry["result"]
        
        # Execute tool
        result = await self.client.call_tool(name, arguments)
        
        # Cache result (only for idempotent tools)
        if self.is_cacheable(name):
            self.cache[cache_key] = {
                "result": result,
                "timestamp": time.time()
            }
        
        return result
    
    def create_cache_key(self, name: str, arguments: dict) -> str:
        """Create unique cache key"""
        args_str = json.dumps(arguments, sort_keys=True)
        return f"{name}:{args_str}"
    
    def is_cacheable(self, tool_name: str) -> bool:
        """Determine if tool results can be cached"""
        # Don't cache mutations or non-deterministic tools
        non_cacheable = ["create", "update", "delete", "random"]
        return not any(keyword in tool_name.lower() for keyword in non_cacheable)
```

## 4.6 Complete LLM + MCP Example {#complete-llm-mcp-example}

Here's a complete, production-ready example:

```python
# complete_llm_mcp_integration.py
import anthropic
from mcp import Client as MCPClient
import asyncio
import json

class ProductionMCPLLMAssistant:
    """
    Production-ready LLM assistant with MCP integration
    """
    
    def __init__(self, anthropic_api_key: str):
        self.llm = anthropic.Anthropic(api_key=anthropic_api_key)
        self.mcp_clients = {}
        self.conversation_history = []
    
    async def add_mcp_server(self, name: str, config: dict):
        """Add MCP server connection"""
        client = MCPClient(config)
        await client.connect()
        self.mcp_clients[name] = client
        print(f"Connected to MCP server: {name}")
    
    async def chat(self, user_message: str) -> str:
        """
        Process user message with full MCP integration
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Discover available tools
        available_tools = await self.get_all_tools()
        
        # Initial LLM call
        response = self.llm.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=self.conversation_history,
            tools=available_tools
        )
        
        # Process response
        while response.stop_reason == "tool_use":
            # Extract tool calls
            tool_uses = [
                block for block in response.content
                if block.type == "tool_use"
            ]
            
            # Execute tools
            tool_results = await self.execute_tool_uses(tool_uses)
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Add tool results to history
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })
            
            # Continue conversation with tool results
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                messages=self.conversation_history,
                tools=available_tools
            )
        
        # Extract final text response
        final_response = next(
            (block.text for block in response.content if hasattr(block, "text")),
            ""
        )
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": final_response
        })
        
        return final_response
    
    async def get_all_tools(self) -> list:
        """Get all tools from all MCP servers"""
        all_tools = []
        
        for server_name, client in self.mcp_clients.items():
            tools = await client.list_tools()
            
            for tool in tools:
                # Format for Anthropic API
                all_tools.append({
                    "name": f"{server_name}_{tool.name}",
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        
        return all_tools
    
    async def execute_tool_uses(self, tool_uses: list) -> list:
        """Execute tool use requests"""
        results = []
        
        for tool_use in tool_uses:
            # Parse server name and tool name
            full_name = tool_use.name
            server_name, tool_name = full_name.split("_", 1)
            
            # Get appropriate client
            client = self.mcp_clients.get(server_name)
            
            if not client:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": f"Error: Server {server_name} not found",
                    "is_error": True
                })
                continue
            
            try:
                # Execute tool via MCP
                result = await client.call_tool(tool_name, tool_use.input)
                
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result.content[0].text
                })
            
            except Exception as e:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": f"Error: {str(e)}",
                    "is_error": True
                })
        
        return results


# Demo usage
async def main():
    # Initialize assistant
    assistant = ProductionMCPLLMAssistant(
        anthropic_api_key="your-api-key"
    )
    
    # Add MCP servers
    await assistant.add_mcp_server("github", {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_TOKEN": "your-token"}
    })
    
    await assistant.add_mcp_server("filesystem", {
        "command": "python",
        "args": ["-m", "mcp_server_filesystem", "/Users/documents"]
    })
    
    # Interactive conversation
    print("MCP-Enabled AI Assistant Ready!")
    print("Type 'exit' to quit\\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
        
        response = await assistant.chat(user_input)
        print(f"\\nAssistant: {response}\\n")


if __name__ == "__main__":
    asyncio.run(main())
```

## 4.7 Summary {#summary}

LLM integration with MCP provides:

- ✅ **Dynamic Tool Access**: LLMs can discover and use tools at runtime
- ✅ **Intelligent Selection**: Models automatically choose appropriate tools
- ✅ **Context Management**: Real-time context from multiple sources
- ✅ **Agentic Behavior**: Autonomous goal accomplishment
- ✅ **Performance**: Parallel execution and caching
- ✅ **Conversational**: Multi-turn conversations with persistent context

This integration transforms LLMs from static knowledge bases into dynamic, action-capable assistants.