---
name: API Tester
version: 1.0.0
description: Test HTTP APIs with automatic request building and response validation
author: Example Author
license: MIT
tags:
  - api
  - testing
  - http
  - rest
triggers:
  - type: keyword
    value: test api
  - type: keyword
    value: http request
  - type: keyword
    value: api call
scripts:
  - name: request
    path: scripts/request.js
    runtime: node
    description: Make an HTTP request and display the response
    args:
      - name: method
        type: string
        required: false
        default: GET
        description: HTTP method (GET, POST, PUT, DELETE, PATCH)
      - name: url
        type: string
        required: true
        description: Target URL
      - name: data
        type: string
        required: false
        description: JSON data for request body
---

# API Tester Skill

This skill helps test HTTP APIs by making requests and validating responses.

## When to Use

Use this skill when the user wants to:
- Test an API endpoint
- Make HTTP requests
- Debug API responses
- Validate API behavior

## Supported Methods

- GET - Retrieve data
- POST - Create resources
- PUT - Update resources (full)
- PATCH - Update resources (partial)
- DELETE - Remove resources

## Usage Examples

Test a GET endpoint:
```
request --url https://api.example.com/users
```

Test a POST with data:
```
request --method POST --url https://api.example.com/users --data '{"name":"John"}'
```

## Response Analysis

The script outputs:
- HTTP status code and message
- Response headers
- Response body (formatted JSON if applicable)
- Timing information

## Security Notes

- Never send sensitive credentials in logs
- Use environment variables for API keys
- Sanitize URLs in output if they contain tokens
