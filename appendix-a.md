# Appendix A: Quick Reference

## Common MCP Methods {#common-mcp-methods}

### Initialization
```
initialize(clientInfo, capabilities)
```

### Resources
```
resources/list()
resources/read(uri)
resources/subscribe(uri)
resources/unsubscribe(uri)
```

### Tools
```
tools/list()
tools/call(name, arguments)
```

### Prompts
```
prompts/list()
prompts/get(name, arguments)
```

### Sampling
```
sampling/createMessage(messages, maxTokens)
```

### Notifications
```
notifications/resources/updated(uri)
notifications/tools/list_changed()
notifications/progress(progressToken, progress, total)
```

## Error Codes {#error-codes}

| Code | Description |
|------|-------------|
| -32001 | Authentication required |
| -32002 | Invalid token |
| -32003 | Insufficient permissions |
| -32600 | Invalid request |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32603 | Internal error |

## Common Status Codes {#common-status-codes}

| Code | Status | Description |
|------|--------|-------------|
| 200 | Success | Request completed successfully |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not found | Resource or method not found |
| 429 | Rate limit exceeded | Too many requests |
| 500 | Server error | Internal server error |
| 502 | Bad gateway | Gateway error |
| 504 | Gateway timeout | Gateway timeout |

## Message Format Reference

### Basic Request
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "method_name",
    "params": {
        "parameter": "value"
    }
}
```

### Basic Response
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "data": "response_data"
    }
}
```

### Error Response
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "error": {
        "code": -32602,
        "message": "Invalid params",
        "data": {
            "details": "Additional error information"
        }
    }
}
```

## Transport Configuration Examples

### STDIO Transport
```json
{
    "command": "python",
    "args": ["-m", "mcp_server"],
    "transport": "stdio"
}
```

### HTTP Transport
```json
{
    "url": "https://api.example.com/mcp",
    "transport": "http",
    "headers": {
        "Authorization": "Bearer ${TOKEN}"
    }
}
```

## Security Headers

### Authentication
```
Authorization: Bearer <access_token>
```

### Content Type
```
Content-Type: application/json
```

### Common Request Headers
```
Accept: application/json
User-Agent: MCP-Client/1.0
X-Request-ID: unique-request-id
```

---

**End of Quick Reference**