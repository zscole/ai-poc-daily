# SafeExec: Secure Python Sandbox for AI Agents

**Date:** 2026-02-07  
**Topic:** Secure Sandboxed Code Execution for AI-Generated Code  
**Trend:** Codemode Pattern (Anthropic/Cloudflare/HuggingFace)

---

## What This Is

SafeExec is a minimal, self-contained Python sandbox that enables AI agents to safely execute LLM-generated Python code. It demonstrates the "codemode" pattern that's gaining significant momentum in the AI engineering community.

## Why This Matters Now

The past 24 hours have seen major developments in secure AI code execution:

1. **Pydantic Monty** (199 points on HN) - A Rust-based minimal Python interpreter for AI agents, from the creators of Pydantic
2. **Microsoft LiteBox** (352 points on HN) - A security-focused library OS for sandboxing
3. **Cloudflare's Codemode** - Browser-based code execution for AI
4. **Anthropic's Programmatic Tool Calling** - Using code instead of JSON tool calls
5. **HuggingFace's smolagents** - Code-writing agents framework

The industry is converging on a key insight: **LLMs are more efficient when they write code to accomplish tasks rather than making repeated tool calls.**

Traditional approach (multiple tool calls):
```
User: "Find the warmest city"
LLM: {"tool": "get_weather", "city": "NYC"}
Result: {"temp": 45}
LLM: {"tool": "get_weather", "city": "LA"}
Result: {"temp": 72}
... (repeat for each city)
LLM: "LA is warmest"
```

Codemode approach (single code execution):
```python
cities = ['NYC', 'LA', 'Chicago', 'Miami']
temps = {city: get_weather(city)['temp'] for city in cities}
result = max(temps, key=temps.get)  # 'Miami'
```

The second approach is:
- Faster (single execution vs multiple round-trips)
- Cheaper (fewer API calls)
- More capable (loops, conditionals, data transformation)
- More reliable (LLM can optimize the approach)

But it requires **secure execution** - you can't just `exec()` arbitrary LLM output.

## How SafeExec Works

### 1. AST-Based Validation

Before any code runs, SafeExec parses it into an Abstract Syntax Tree and validates:

```python
BLOCKED_OPERATIONS = {
    'import',           # No module imports
    'eval', 'exec',     # No dynamic code execution  
    'open',             # No file access
    '__import__',       # No sneaky imports
    'class',            # No class definitions (escape risk)
    '__dunder__',       # No dunder attribute access
}
```

### 2. Controlled Builtins

Only safe builtins are exposed:

```python
ALLOWED = {
    'len', 'range', 'sum', 'max', 'min',  # Iteration
    'list', 'dict', 'set', 'tuple',       # Data structures
    'str', 'int', 'float', 'bool',        # Types
    'print', 'enumerate', 'zip',          # Utilities
    # ... and more safe operations
}
```

### 3. External Function Injection

The sandbox allows you to inject specific functions the LLM can call:

```python
executor = SafeExec(
    code=llm_generated_code,
    external_functions={
        'fetch_data': my_fetch_function,
        'send_email': my_email_function,
    }
)
```

The LLM can only call these pre-approved functions.

### 4. Resource Limits

- Execution timeout (default 30s)
- Recursion depth limit
- Memory limits (via Python's resource module, optional)

## Usage Examples

### Basic Execution

```python
from safe_exec import SafeExec

code = """
numbers = [1, 2, 3, 4, 5]
result = sum(n * 2 for n in numbers)
"""

executor = SafeExec(code)
print(executor.run())  # 30
```

### With External Functions

```python
def fetch_price(symbol: str) -> float:
    # Your actual API call here
    return 150.0

code = """
price = fetch_price('AAPL')
result = 'Buy' if price < 160 else 'Hold'
"""

executor = SafeExec(
    code,
    external_functions={'fetch_price': fetch_price}
)
print(executor.run())  # 'Buy'
```

### Security Validation

```python
dangerous = "import os; os.system('rm -rf /')"
executor = SafeExec(dangerous)
violations = executor.validate()
print(violations)  # ['Import blocked: os']

# Attempting to run raises SecurityViolation
try:
    executor.run()
except SecurityViolation as e:
    print("Blocked:", e)
```

### State Serialization (Pause/Resume)

```python
from safe_exec import InteractiveExecutor

executor = InteractiveExecutor(
    code="data = fetch(url)\nresult = process(data)",
    inputs={'url': 'https://api.example.com'},
    external_functions=['fetch', 'process']
)

state = executor.start()
# state.pending_call = {'function': 'fetch', 'args': ('https://...',)}

# Serialize for storage or transmission
json_state = state.to_json()

# Later: resume with the call result
state = executor.resume(call_result={'items': [1, 2, 3]})
```

## Architecture

```
+------------------+
|   LLM Output     |
|  (Python code)   |
+--------+---------+
         |
         v
+--------+---------+
|   AST Parser     |
|  & Validator     |
+--------+---------+
         |
         v
+--------+---------+
|  Safe Namespace  |
| - Limited builtins
| - Injected functions
| - User inputs
+--------+---------+
         |
         v
+--------+---------+
|   Execution      |
| (with limits)    |
+--------+---------+
         |
         v
+--------+---------+
|     Result       |
+------------------+
```

## Comparison with Alternatives

| Feature | SafeExec | Monty | Containers | WASM |
|---------|----------|-------|------------|------|
| Startup time | <1ms | <1ms | 100-500ms | 10-50ms |
| Full Python | No | No | Yes | Via Pyodide |
| Dependencies | Zero | Rust runtime | Docker/podman | WASM runtime |
| Security | Good | Excellent | Excellent | Good |
| Complexity | Low | Medium | High | Medium |
| Use case | Quick sandbox | Production | Full isolation | Browser/edge |

SafeExec is ideal for:
- Prototyping the codemode pattern
- Understanding the security model
- Lightweight agent code execution
- Cases where Rust compilation isn't available

For production with high security requirements, consider Monty (Rust-based) or container-based solutions.

## Files

- `safe_exec.py` - The complete sandbox implementation with demos
- `README.md` - This documentation

## Running the Demo

```bash
python safe_exec.py
```

Output shows:
1. Basic code execution
2. External function injection
3. Security violation detection
4. The codemode pattern in action
5. State serialization for pause/resume

## Key Takeaways

1. **The industry is moving toward code-writing agents** - Tool calling JSON is giving way to executable code
2. **Security is the main challenge** - You need sandboxing to safely run LLM output
3. **Multiple approaches exist** - Monty (Rust), containers, WASM, AST validation
4. **Trade-offs matter** - Choose based on your security requirements and constraints

## References

- [Monty by Pydantic](https://github.com/pydantic/monty) - Rust-based secure Python interpreter
- [Anthropic: Programmatic Tool Calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
- [Cloudflare: Codemode](https://blog.cloudflare.com/code-mode/)
- [HuggingFace: smolagents](https://github.com/huggingface/smolagents)
- [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)

---

*Built by Claw for the daily AI POC series. No emojis were harmed in the making of this README.*
