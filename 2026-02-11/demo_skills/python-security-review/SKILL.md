---
name: python-security-review
description: Review Python code for common security vulnerabilities
---

# Python Security Code Review

## Overview
Review Python code for security vulnerabilities, focusing on the OWASP Top 10
and Python-specific issues.

## Critical Checks (Always Flag)

### 1. Injection Vulnerabilities
- SQL: Look for string formatting in queries
  BAD: `f"SELECT * FROM users WHERE id = {user_id}"`
  GOOD: `cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))`

- Command Injection: subprocess with shell=True
  BAD: `subprocess.run(f"ls {user_input}", shell=True)`
  GOOD: `subprocess.run(["ls", user_input])`

- SSTI: Direct user input in templates
  BAD: `render_template_string(user_input)`

### 2. Authentication Issues
- Hardcoded credentials
- Weak password hashing (MD5, SHA1 without salt)
- Missing rate limiting on auth endpoints
- JWT without expiration

### 3. Data Exposure
- Secrets in code or logs
- Debug mode in production
- Verbose error messages to users
- Sensitive data in URLs

### 4. Insecure Deserialization
- pickle.loads with untrusted data
- yaml.load without Loader=SafeLoader
- eval/exec on user input

### 5. Path Traversal
- User input in file paths without validation
- Missing path canonicalization

## Output Format
For each issue found:
```
[SEVERITY] Issue Title
Line: X
Code: `affected code snippet`
Risk: What could happen
Fix: How to fix it
```

Severity levels: CRITICAL, HIGH, MEDIUM, LOW, INFO

## False Positive Prevention
- Check if input is sanitized before flagging
- Look for middleware/decorators that might handle security
- Consider the context (internal tool vs public API)
