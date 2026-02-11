---
name: json-schema-validator
description: Validate JSON objects against schemas with clear error reporting
---

# JSON Schema Validation

## Task Overview
Validate JSON objects against a schema and report all errors clearly.

## Core Rules
1. Check all required fields exist
2. Validate types match schema
3. Check string patterns (if specified)
4. Validate numeric ranges
5. Recursively validate nested objects

## Error Format
Report errors as:
- Path: field.subfield
- Issue: what's wrong
- Expected: what was expected
- Got: what was received

## Common Patterns
- Email: use regex /^[^@]+@[^@]+\.[^@]+$/
- Date: ISO 8601 format
- UUID: 8-4-4-4-12 hex pattern

## Edge Cases
- null vs missing field (different errors)
- Empty arrays vs missing arrays
- Extra fields (warn, don't error by default)

## Example
Input: {"name": 123, "email": "notanemail"}
Schema: {"name": "string", "email": "email"}

Output:
- Path: name | Issue: wrong type | Expected: string | Got: number
- Path: email | Issue: invalid format | Expected: email | Got: notanemail
