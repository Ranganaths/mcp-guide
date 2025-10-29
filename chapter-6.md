# Chapter 6: MCP Security, Gateway & Registry

## 6.1 MCP Security Overview {#mcp-security-overview}

Security is paramount when connecting AI systems to external data sources and tools. MCP provides comprehensive security mechanisms to ensure safe, controlled access.

### 6.1.1 Security Layers {#security-layers}

```
┌─────────────────────────────────────────────────┐
│              Application Security Layer         │
│              • User Authentication              │
│              • Session Management               │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│             MCP Authorization Layer             │
│             • OAuth 2.1 with PKCE              │
│             • Token Management                 │
│             • Scope-based Access Control       │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│            Transport Security Layer             │
│            • TLS/HTTPS Encryption               │
│            • Certificate Validation             │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│             Server Security Layer               │
│             • Input Validation                 │
│             • Rate Limiting                    │
│             • Audit Logging                    │
└─────────────────────────────────────────────────┘
```

## 6.2 Authentication and Authorization {#authentication-and-authorization}

### 6.2.1 OAuth 2.1 with PKCE {#oauth-2.1-with-pkce}

MCP uses OAuth 2.1 with Proof Key for Code Exchange (PKCE) as its standard authorization mechanism.

**OAuth 2.1 Flow with MCP:**

```
┌─────────────┐    ┌──────────────────┐
│ MCP Client  │    │ Authorization    │
│             │    │ Server           │
└──────┬──────┘    └────────┬─────────┘
       │                    │
       │ 1. Generate code_verifier & code_challenge
       │ code_verifier = random(43-128 chars)
       │ code_challenge = SHA256(code_verifier)
       │                    │
       │ 2. Authorization Request
       │ + client_id        │
       │ + code_challenge   │
       │ + code_challenge_method=S256
       ├────────────────────┼────────────────────>│
       │                    │
       │ 3. User authenticates & authorizes
       │                    │
       │ 4. Authorization Code
       │<───────────────────┼─────────────────────┤
       │                    │
       │ 5. Token Request   │
       │ + authorization_code
       │ + code_verifier    │
       ├────────────────────┼────────────────────>│
       │                    │
       │ 6. Verify code_challenge matches code_verifier
       │                    │
       │ 7. Access Token + Refresh Token
       │<───────────────────┼─────────────────────┤
```

**Implementation Example:**

```python
# oauth_mcp_client.py
import hashlib
import base64
import secrets
from urllib.parse import urlencode
import httpx

class SecureMCPClient:
    """
    MCP Client with OAuth 2.1 + PKCE authentication
    """
    
    def __init__(self, server_url: str, client_id: str):
        self.server_url = server_url
        self.client_id = client_id
        self.access_token = None
        self.refresh_token = None
    
    def generate_pkce_pair(self):
        """Generate PKCE code verifier and challenge"""
        # Generate code verifier (43-128 characters)
        code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
        
        # Generate code challenge
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        return code_verifier, code_challenge
    
    async def authenticate(self, authorization_server_url: str):
        """
        Perform OAuth 2.1 authentication flow
        """
        # Step 1: Generate PKCE pair
        code_verifier, code_challenge = self.generate_pkce_pair()
        
        # Step 2: Build authorization URL
        auth_params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256',
            'scope': 'mcp:read mcp:write mcp:tools',
            'redirect_uri': 'http://localhost:8080/callback'
        }
        
        auth_url = f"{authorization_server_url}/authorize?{urlencode(auth_params)}"
        
        print(f"Please visit: {auth_url}")
        print("Waiting for authorization...")
        
        # Step 3: Wait for authorization code (simplified)
        authorization_code = input("Enter authorization code: ")
        
        # Step 4: Exchange code for tokens
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                f"{authorization_server_url}/token",
                data={
                    'grant_type': 'authorization_code',
                    'code': authorization_code,
                    'client_id': self.client_id,
                    'code_verifier': code_verifier,
                    'redirect_uri': 'http://localhost:8080/callback'
                }
            )
            
            if token_response.status_code == 200:
                token_data = token_response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data.get('refresh_token')
                print("Authentication successful!")
            else:
                raise Exception(f"Authentication failed: {token_response.text}")
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call MCP tool with authentication"""
        if not self.access_token:
            raise Exception("Not authenticated. Call authenticate() first.")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.server_url}/mcp/tools/call",
                headers={
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json'
                },
                json={
                    'jsonrpc': '2.0',
                    'id': 1,
                    'method': 'tools/call',
                    'params': {
                        'name': tool_name,
                        'arguments': arguments
                    }
                }
            )
            
            if response.status_code == 401:
                # Token expired, try refresh
                await self.refresh_access_token(authorization_server_url)
                # Retry request
                return await self.call_tool(tool_name, arguments)
            
            return response.json()
```

### 6.2.2 Token-Based Security {#token-based-security}

```python
# secure_mcp_server.py
from mcp.server import Server
from typing import Optional
import jwt
import time

class SecureMCPServer(Server):
    """
    MCP Server with token validation
    """
    
    def __init__(self, name: str, jwt_secret: str):
        super().__init__(name)
        self.jwt_secret = jwt_secret
        self.token_blacklist = set()
    
    def validate_token(self, token: str) -> Optional[dict]:
        """
        Validate JWT access token
        """
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=['HS256']
            )
            
            # Check expiration
            if payload['exp'] < time.time():
                raise Exception("Token expired")
            
            # Check blacklist
            if payload['jti'] in self.token_blacklist:
                raise Exception("Token revoked")
            
            # Check audience (ensure token is for this server)
            if self.name not in payload.get('aud', []):
                raise Exception("Invalid audience")
            
            return payload
        
        except jwt.InvalidTokenError as e:
            print(f"Token validation failed: {e}")
            return None
    
    def check_scope(self, token_payload: dict, required_scope: str) -> bool:
        """Check if token has required scope"""
        token_scopes = token_payload.get('scope', '').split()
        return required_scope in token_scopes
    
    async def handle_request(self, request: dict, auth_header: str):
        """
        Handle MCP request with authentication
        """
        # Extract token
        if not auth_header or not auth_header.startswith('Bearer '):
            return {
                'jsonrpc': '2.0',
                'id': request.get('id'),
                'error': {
                    'code': -32001,
                    'message': 'Authentication required'
                }
            }
        
        token = auth_header.replace('Bearer ', '')
        
        # Validate token
        token_payload = self.validate_token(token)
        
        if not token_payload:
            return {
                'jsonrpc': '2.0',
                'id': request.get('id'),
                'error': {
                    'code': -32002,
                    'message': 'Invalid or expired token'
                }
            }
        
        # Check permissions based on method
        method = request.get('method')
        required_scope = self.get_required_scope(method)
        
        if not self.check_scope(token_payload, required_scope):
            return {
                'jsonrpc': '2.0',
                'id': request.get('id'),
                'error': {
                    'code': -32003,
                    'message': f'Insufficient permissions. Required: {required_scope}'
                }
            }
        
        # Process request
        return await self.process_request(request, token_payload)
    
    def get_required_scope(self, method: str) -> str:
        """Determine required scope for method"""
        if method.startswith('resources/'):
            return 'mcp:read'
        elif method.startswith('tools/'):
            return 'mcp:tools'
        else:
            return 'mcp:read'
```

## 6.3 Access Control and Permissions {#access-control-and-permissions}

### 6.3.1 Role-Based Access Control (RBAC) {#role-based-access-control-rbac}

```python
# rbac_mcp_server.py
from enum import Enum
from typing import Set, Dict

class Permission(Enum):
    READ_RESOURCES = "resources:read"
    WRITE_RESOURCES = "resources:write"
    EXECUTE_TOOLS = "tools:execute"
    MANAGE_PROMPTS = "prompts:manage"
    ADMIN = "admin:all"

class Role:
    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions

class RBACMCPServer(Server):
    """
    MCP Server with Role-Based Access Control
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        
        # Define roles
        self.roles = {
            'viewer': Role('viewer', {
                Permission.READ_RESOURCES
            }),
            'user': Role('user', {
                Permission.READ_RESOURCES,
                Permission.EXECUTE_TOOLS
            }),
            'admin': Role('admin', {
                Permission.READ_RESOURCES,
                Permission.WRITE_RESOURCES,
                Permission.EXECUTE_TOOLS,
                Permission.MANAGE_PROMPTS,
                Permission.ADMIN
            })
        }
        
        # User role assignments
        self.user_roles: Dict[str, str] = {}
    
    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")
        
        self.user_roles[user_id] = role_name
    
    def check_permission(self, user_id: str, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        user_role_name = self.user_roles.get(user_id)
        if not user_role_name:
            return False
        
        user_role = self.roles.get(user_role_name)
        if not user_role:
            return False
        
        return required_permission in user_role.permissions
```

### 6.3.2 Fine-Grained Access Control {#fine-grained-access-control}

```python
class AccessControlList:
    """
    Fine-grained access control for MCP resources and tools
    """
    
    def __init__(self):
        self.acl = {}  # resource/tool -> {user -> permissions}
    
    def grant_permission(self, resource_id: str, user_id: str, permissions: list):
        """Grant specific permissions to user for resource/tool"""
        if resource_id not in self.acl:
            self.acl[resource_id] = {}
        
        if user_id not in self.acl[resource_id]:
            self.acl[resource_id][user_id] = set()
        
        self.acl[resource_id][user_id].update(permissions)
    
    def check_permission(self, resource_id: str, user_id: str, permission: str) -> bool:
        """Check if user has permission for resource"""
        if resource_id not in self.acl:
            return False
        
        if user_id not in self.acl[resource_id]:
            return False
        
        return permission in self.acl[resource_id][user_id]

class FineGrainedMCPServer(Server):
    """MCP Server with fine-grained access control"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.acl = AccessControlList()
    
    async def call_tool(self, user_id: str, tool_name: str, arguments: dict):
        """Execute tool with fine-grained permission check"""
        # Check if user has execute permission for this specific tool
        if not self.acl.check_permission(f"tool:{tool_name}", user_id, "execute"):
            raise PermissionError(
                f"User {user_id} not authorized to execute tool {tool_name}"
            )
        
        # Check parameter-level permissions
        for param_name, param_value in arguments.items():
            if not self.check_parameter_permission(user_id, tool_name, param_name):
                raise PermissionError(
                    f"User {user_id} not authorized to use parameter {param_name}"
                )
        
        return await super().call_tool(tool_name, arguments)
```

## 6.4 MCP Gateway {#mcp-gateway}

An MCP Gateway acts as a centralized entry point for multiple MCP servers, providing routing, authentication, rate limiting, and monitoring.

### 6.4.1 Gateway Architecture {#gateway-architecture}

```
┌──────────────────────────────────────────────────────┐
│                    MCP Gateway                       │
│                                                      │
│ ┌────────────────────────────────────────────────┐ │
│ │         Authentication & Authorization         │ │
│ │         • Token Validation                     │ │
│ │         • User Identity                        │ │
│ └─────────────────┬──────────────────────────────┘ │
│                   │                                │
│ ┌─────────────────▼──────────────────────────────┐ │
│ │             Request Router                     │ │
│ │             • Server Discovery                 │ │
│ │             • Load Balancing                   │ │
│ └─────────────────┬──────────────────────────────┘ │
│                   │                                │
│ ┌─────────────────▼──────────────────────────────┐ │
│ │         Rate Limiting & Throttling             │ │
│ └─────────────────┬──────────────────────────────┘ │
│                   │                                │
│ ┌─────────────────▼──────────────────────────────┐ │
│ │          Logging & Monitoring                  │ │
│ └─────────────────┬──────────────────────────────┘ │
└────────────────────┼──────────────────────────────────┘
                     │
        ┌───────────┼───────────┐
        │           │           │
   ┌────▼────┐ ┌───▼────┐ ┌───▼────┐
   │  MCP    │ │  MCP   │ │  MCP   │
   │Server 1 │ │Server 2│ │Server 3│
   └─────────┘ └────────┘ └────────┘
```

### 6.4.2 Gateway Implementation {#gateway-implementation}

```python
# mcp_gateway.py
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
from typing import Optional, Dict
import time
from collections import defaultdict

class MCPGateway:
    """
    Centralized gateway for multiple MCP servers
    """
    
    def __init__(self):
        self.app = FastAPI(title="MCP Gateway")
        self.servers: Dict[str, str] = {}  # name -> URL mapping
        self.rate_limits: Dict[str, list] = defaultdict(list)
        self.setup_routes()
        self.setup_middleware()
    
    def setup_middleware(self):
        """Configure middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def register_server(self, name: str, url: str):
        """Register an MCP server"""
        self.servers[name] = url
        print(f"Registered MCP server: {name} at {url}")
    
    def setup_routes(self):
        """Setup gateway routes"""
        
        @self.app.post("/mcp/{server_name}/tools/call")
        async def call_tool(
            server_name: str,
            request: Request,
            authorization: Optional[str] = Header(None)
        ):
            """Route tool call to appropriate MCP server"""
            # 1. Authenticate
            user_id = await self.authenticate(authorization)
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # 2. Rate limiting
            if not self.check_rate_limit(user_id):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # 3. Get server URL
            if server_name not in self.servers:
                raise HTTPException(status_code=404, detail=f"Server {server_name} not found")
            
            server_url = self.servers[server_name]
            
            # 4. Forward request
            body = await request.json()
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{server_url}/mcp/tools/call",
                        json=body,
                        headers={"Authorization": authorization},
                        timeout=30.0
                    )
                    
                    # 5. Log request
                    await self.log_request(
                        user_id=user_id,
                        server=server_name,
                        method="tools/call",
                        status=response.status_code
                    )
                    
                    return response.json()
            
            except httpx.TimeoutException:
                raise HTTPException(status_code=504, detail="Server timeout")
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Server error: {str(e)}")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            # Check connectivity to all servers
            server_health = {}
            
            for name, url in self.servers.items():
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"{url}/health",
                            timeout=5.0
                        )
                        server_health[name] = response.status_code == 200
                except:
                    server_health[name] = False
            
            all_healthy = all(server_health.values())
            
            return {
                "status": "healthy" if all_healthy else "degraded",
                "servers": server_health
            }
    
    async def authenticate(self, authorization: Optional[str]) -> Optional[str]:
        """Validate authorization token and return user ID"""
        if not authorization or not authorization.startswith("Bearer "):
            return None
        
        token = authorization.replace("Bearer ", "")
        
        # Validate token (simplified)
        # In production, validate JWT properly
        try:
            # Decode token and extract user ID
            user_id = token  # In reality: decode JWT and extract user_id
            return user_id
        except:
            return None
    
    def check_rate_limit(self, user_id: str, limit: int = 100, window: int = 60) -> bool:
        """
        Check if user is within rate limit
        limit: requests per window (default: 100 requests)
        window: time window in seconds (default: 60 seconds)
        """
        now = time.time()
        
        # Clean old requests outside window
        self.rate_limits[user_id] = [
            req_time for req_time in self.rate_limits[user_id]
            if now - req_time < window
        ]
        
        # Check limit
        if len(self.rate_limits[user_id]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[user_id].append(now)
        return True
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the gateway"""
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


# Setup and run gateway
if __name__ == "__main__":
    gateway = MCPGateway()
    
    # Register MCP servers
    gateway.register_server("github", "http://localhost:8001")
    gateway.register_server("database", "http://localhost:8002")
    gateway.register_server("weather", "http://localhost:8003")
    
    # Run gateway
    gateway.run()
```

### 6.4.3 Load Balancing Gateway {#load-balancing-gateway}

```python
# load_balancing_gateway.py
import random
from typing import List

class LoadBalancingGateway(MCPGateway):
    """
    Gateway with load balancing across multiple server instances
    """
    
    def __init__(self):
        super().__init__()
        self.server_pools: Dict[str, List[str]] = {}  # server_name -> [URLs]
    
    def register_server_pool(self, name: str, urls: List[str]):
        """Register multiple instances of a server"""
        self.server_pools[name] = urls
        print(f"Registered server pool: {name} with {len(urls)} instances")
    
    def get_server_url(self, server_name: str, strategy: str = "round_robin") -> str:
        """
        Get server URL using load balancing strategy
        """
        if server_name not in self.server_pools:
            raise Exception(f"Server pool {server_name} not found")
        
        urls = self.server_pools[server_name]
        
        if strategy == "round_robin":
            # Simple round-robin (in production, track state)
            return urls[int(time.time()) % len(urls)]
        
        elif strategy == "random":
            return random.choice(urls)
        
        elif strategy == "least_connections":
            # Choose server with least active connections
            # In production, track connections per server
            return urls[0]
        
        else:
            return urls[0]


# Usage
gateway = LoadBalancingGateway()

# Register multiple instances of GitHub server
gateway.register_server_pool("github", [
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003"
])

# Requests will be load balanced across instances
```

## 6.5 MCP Registry {#mcp-registry}

An MCP Registry is a centralized directory for discovering available MCP servers and their capabilities.

### 6.5.1 Registry Architecture {#registry-architecture}

```
┌─────────────────────────────────────────────────────┐
│                   MCP Registry                      │
│                                                     │
│ ┌───────────────────────────────────────────────┐ │
│ │              Server Registration              │ │
│ │              • Server metadata                │ │
│ │              • Capabilities                   │ │
│ │              • Health status                  │ │
│ └───────────────────────────────────────────────┘ │
│                                                     │
│ ┌───────────────────────────────────────────────┐ │
│ │              Discovery Service                │ │
│ │              • Search by capability           │ │
│ │              • Filter by attributes           │ │
│ └───────────────────────────────────────────────┘ │
│                                                     │
│ ┌───────────────────────────────────────────────┐ │
│ │              Health Monitoring                │ │
│ │              • Periodic health checks         │ │
│ │              • Status tracking                │ │
│ └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 6.5.2 Registry Implementation {#registry-implementation}

```python
# mcp_registry.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import asyncio
from datetime import datetime

class ServerCapabilities(BaseModel):
    resources: bool = False
    tools: bool = False
    prompts: bool = False

class ServerMetadata(BaseModel):
    name: str
    url: str
    description: str
    version: str
    capabilities: ServerCapabilities
    tags: List[str] = []
    health_check_url: Optional[str] = None

class ServerRegistration(BaseModel):
    metadata: ServerMetadata
    registered_at: datetime
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # healthy, unhealthy, unknown

class MCPRegistry:
    """
    Registry for MCP server discovery
    """
    
    def __init__(self):
        self.app = FastAPI(title="MCP Registry")
        self.servers: Dict[str, ServerRegistration] = {}
        self.setup_routes()
        self.start_health_monitoring()
    
    def setup_routes(self):
        """Setup registry routes"""
        
        @self.app.post("/registry/register")
        async def register_server(metadata: ServerMetadata):
            """Register a new MCP server"""
            registration = ServerRegistration(
                metadata=metadata,
                registered_at=datetime.now()
            )
            
            self.servers[metadata.name] = registration
            
            # Perform initial health check
            await self.check_server_health(metadata.name)
            
            return {"status": "registered", "server": metadata.name}
        
        @self.app.get("/registry/discover")
        async def discover_servers(
            capability: Optional[str] = None,
            tag: Optional[str] = None,
            healthy_only: bool = True
        ):
            """Discover servers by capabilities or tags"""
            results = []
            
            for name, registration in self.servers.items():
                # Filter by health status
                if healthy_only and registration.health_status != "healthy":
                    continue
                
                # Filter by capability
                if capability:
                    caps = registration.metadata.capabilities
                    if capability == "resources" and not caps.resources:
                        continue
                    if capability == "tools" and not caps.tools:
                        continue
                    if capability == "prompts" and not caps.prompts:
                        continue
                
                # Filter by tag
                if tag and tag not in registration.metadata.tags:
                    continue
                
                results.append({
                    "name": registration.metadata.name,
                    "url": registration.metadata.url,
                    "description": registration.metadata.description,
                    "capabilities": registration.metadata.capabilities,
                    "health_status": registration.health_status,
                    "last_check": registration.last_health_check
                })
            
            return {"servers": results, "count": len(results)}
        
        @self.app.get("/registry/health")
        async def registry_health():
            """Get health status of all registered servers"""
            return {
                "total_servers": len(self.servers),
                "healthy_servers": sum(
                    1 for reg in self.servers.values()
                    if reg.health_status == "healthy"
                ),
                "servers": {
                    name: {
                        "status": reg.health_status,
                        "last_check": reg.last_health_check
                    }
                    for name, reg in self.servers.items()
                }
            }
    
    async def check_server_health(self, server_name: str):
        """Check health of a specific server"""
        if server_name not in self.servers:
            return
        
        registration = self.servers[server_name]
        health_url = registration.metadata.health_check_url or f"{registration.metadata.url}/health"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(health_url, timeout=5.0)
                
                if response.status_code == 200:
                    registration.health_status = "healthy"
                else:
                    registration.health_status = "unhealthy"
        
        except Exception:
            registration.health_status = "unhealthy"
        
        registration.last_health_check = datetime.now()
    
    def start_health_monitoring(self):
        """Start background health monitoring"""
        async def health_monitor():
            while True:
                # Check health of all servers every 30 seconds
                tasks = [
                    self.check_server_health(name)
                    for name in self.servers.keys()
                ]
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(30)
        
        # Start background task
        asyncio.create_task(health_monitor())


# Run registry
if __name__ == "__main__":
    registry = MCPRegistry()
    import uvicorn
    uvicorn.run(registry.app, host="0.0.0.0", port=8080)
```

### 6.5.3 Client-Side Discovery {#client-side-discovery}

```python
# mcp_discovery_client.py
import httpx
from typing import List, Optional

class MCPDiscoveryClient:
    """
    Client for discovering MCP servers via registry
    """
    
    def __init__(self, registry_url: str):
        self.registry_url = registry_url
    
    async def discover_servers(
        self,
        capability: Optional[str] = None,
        tag: Optional[str] = None,
        healthy_only: bool = True
    ) -> List[dict]:
        """Discover available MCP servers"""
        params = {}
        
        if capability:
            params["capability"] = capability
        if tag:
            params["tag"] = tag
        if healthy_only:
            params["healthy_only"] = healthy_only
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.registry_url}/registry/discover",
                params=params
            )
            
            if response.status_code == 200:
                return response.json()["servers"]
            else:
                raise Exception(f"Discovery failed: {response.text}")
    
    async def find_tools_servers(self) -> List[dict]:
        """Find servers that provide tools"""
        return await self.discover_servers(capability="tools")
    
    async def find_database_servers(self) -> List[dict]:
        """Find servers tagged as database servers"""
        return await self.discover_servers(tag="database")


# Usage
async def demo_discovery():
    client = MCPDiscoveryClient("http://registry.example.com")
    
    # Find all healthy servers with tool capabilities
    tool_servers = await client.find_tools_servers()
    print(f"Found {len(tool_servers)} tool servers")
    
    # Find database servers
    db_servers = await client.find_database_servers()
    print(f"Found {len(db_servers)} database servers")
    
    for server in tool_servers:
        print(f"Server: {server['name']} - {server['description']}")
```

## 6.6 Security Best Practices {#security-best-practices}

### 6.6.1 Comprehensive Security Checklist {#comprehensive-security-checklist}

**Authentication & Authorization:**
- ✅ Implement OAuth 2.1 with PKCE for secure authorization
- ✅ Use JWT tokens with proper expiration and refresh mechanisms
- ✅ Implement role-based access control (RBAC)
- ✅ Use fine-grained permissions for sensitive operations
- ✅ Validate tokens on every request

**Transport Security:**
- ✅ Always use HTTPS/TLS for communication
- ✅ Validate SSL certificates
- ✅ Implement certificate pinning for high-security environments
- ✅ Use strong cipher suites

**Input Validation:**
- ✅ Validate all input parameters
- ✅ Sanitize user inputs to prevent injection attacks
- ✅ Implement request size limits
- ✅ Use JSON schema validation

**Rate Limiting & Monitoring:**
- ✅ Implement rate limiting per user/IP
- ✅ Monitor for suspicious activity patterns
- ✅ Log all security-relevant events
- ✅ Set up alerting for security incidents

**Server Security:**
- ✅ Run servers with minimal privileges
- ✅ Regularly update dependencies
- ✅ Implement proper error handling (don't leak information)
- ✅ Use secure configuration management

## 6.7 Summary {#summary}

MCP security architecture provides:

- ✅ **Multi-layered Security**: Authentication, authorization, transport, and server security
- ✅ **OAuth 2.1 + PKCE**: Industry-standard secure authorization
- ✅ **Flexible Access Control**: RBAC and fine-grained permissions
- ✅ **Gateway Pattern**: Centralized security and routing
- ✅ **Service Discovery**: Registry for server discovery and health monitoring
- ✅ **Production Ready**: Rate limiting, monitoring, and best practices

This comprehensive security model ensures MCP deployments can safely operate in enterprise environments while maintaining flexibility and performance.