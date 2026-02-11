#!/usr/bin/env python3
"""
Practical examples of skill transfer.

These examples show real-world scenarios where skill transfer is valuable:
1. Code review skills
2. Data extraction skills
3. Domain-specific reasoning
"""

from skill_transfer import (
    SkillDocument,
    TestCase,
    SkillTransferEngine,
    ModelClient,
    SkillEvaluator,
)
from pathlib import Path


# Example 1: Code Review Skill
# Transfers expertise in reviewing Python code for security issues

CODE_REVIEW_SKILL = SkillDocument(
    name="python-security-review",
    description="Review Python code for common security vulnerabilities",
    content="""# Python Security Code Review

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
""",
    test_cases=[
        TestCase(
            input_prompt='''Review this code:
def login(username, password):
    query = f"SELECT * FROM users WHERE username = '{username}'"
    user = db.execute(query).fetchone()
    if user and user.password == password:
        return create_session(user)''',
            expected_contains=["SQL", "injection", "CRITICAL"],
            context="SQL injection vulnerability"
        ),
        TestCase(
            input_prompt='''Review this code:
import pickle
def load_config(data):
    return pickle.loads(base64.b64decode(data))''',
            expected_contains=["pickle", "deserialization", "untrusted"],
            context="Insecure deserialization"
        ),
        TestCase(
            input_prompt='''Review this code:
API_KEY = "sk-1234567890abcdef"
def call_api():
    return requests.get(URL, headers={"Authorization": API_KEY})''',
            expected_contains=["hardcoded", "secret", "credential"],
            context="Hardcoded secrets"
        ),
    ],
    metadata={"domain": "security", "language": "python"}
)


# Example 2: Data Extraction Skill
# Transfers expertise in extracting structured data from unstructured text

DATA_EXTRACTION_SKILL = SkillDocument(
    name="contact-extraction",
    description="Extract contact information from unstructured text",
    content="""# Contact Information Extraction

## Overview
Extract structured contact information from emails, messages, or documents.

## Fields to Extract

### Person
- name: Full name (prefer "First Last" format)
- title: Job title/role
- company: Organization name

### Contact Methods
- email: Email addresses
- phone: Phone numbers (normalize to +1-XXX-XXX-XXXX for US)
- linkedin: LinkedIn profile URLs
- twitter: Twitter/X handles

### Address
- street: Street address
- city: City name
- state: State/province
- postal: Postal/ZIP code
- country: Country (default: USA if US indicators present)

## Extraction Rules

### Names
- Prioritize email signatures over body mentions
- Handle "Dr.", "Mr.", "Ms." prefixes
- Split compound last names correctly (van der Berg, O'Connor)

### Phones
- Extract digits, ignore formatting
- Identify type: mobile, office, fax
- Handle extensions: "x123" or "ext. 123"

### Emails
- Standard regex: [a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}
- Watch for obfuscation: "john [at] company [dot] com"

## Output Format
```json
{
  "contacts": [
    {
      "name": "...",
      "email": "...",
      "phone": "...",
      "company": "...",
      "confidence": 0.95
    }
  ],
  "raw_mentions": ["...", "..."]
}
```

## Edge Cases
- Multiple contacts in one document: extract all
- Partial information: include with lower confidence
- Conflicting info: prefer most recent/prominent mention
""",
    test_cases=[
        TestCase(
            input_prompt='''Extract contacts from:
Hi,
Please reach out to John Smith at john.smith@acme.com or call 555-123-4567.

Best,
Jane Doe
Senior Engineer, TechCorp
jane@techcorp.io | (555) 987-6543''',
            expected_contains=["john.smith@acme.com", "jane@techcorp.io", "John Smith", "Jane Doe"],
            context="Multiple contacts extraction"
        ),
        TestCase(
            input_prompt='''Extract contacts from:
Contact: Dr. Maria van der Berg
Email: maria [at] university [dot] edu
Office: +1 (555) 234-5678 ext 123''',
            expected_contains=["van der Berg", "Maria", "5552345678", "123"],
            context="Complex name and obfuscated email"
        ),
    ],
    metadata={"domain": "nlp", "task": "ner"}
)


# Example 3: API Error Diagnosis Skill
# Transfers expertise in diagnosing API errors from logs

API_DIAGNOSIS_SKILL = SkillDocument(
    name="api-error-diagnosis",
    description="Diagnose API errors from logs and stack traces",
    content="""# API Error Diagnosis

## Overview
Analyze error logs and stack traces to identify root causes and suggest fixes.

## Common Error Patterns

### HTTP Status Codes
- 400: Bad Request - check input validation, required fields
- 401: Unauthorized - check auth headers, token expiration
- 403: Forbidden - check permissions, API key scope
- 404: Not Found - check URL, resource ID, API version
- 429: Rate Limited - implement backoff, check quotas
- 500: Server Error - check logs, report to provider
- 502/503/504: Gateway errors - retry with backoff

### Authentication Failures
- "Invalid token": Token expired or malformed
- "Missing authorization": Header not included
- "Insufficient scope": Token lacks required permissions

### Validation Errors
- "Required field missing": Check API docs for required params
- "Invalid format": Check data types, date formats
- "Value out of range": Check min/max constraints

### Connection Issues
- ETIMEDOUT: Server unreachable, firewall, DNS
- ECONNRESET: Connection dropped, server restart
- SSL errors: Certificate issues, TLS version

## Diagnosis Process

1. Identify error type (HTTP code, exception type)
2. Extract relevant context (request, headers, body)
3. Check common causes for that error type
4. Suggest specific fix with code example

## Output Format
```
Error: [Error type/code]
Likely Cause: [Most probable reason]
Evidence: [What in the logs supports this]
Fix: [Specific solution]
Code: [Example fix if applicable]
Prevention: [How to avoid in future]
```

## Red Flags
- Secrets in logs (mask them in output)
- PII exposure
- Repeated same errors (systemic issue)
""",
    test_cases=[
        TestCase(
            input_prompt='''Diagnose this error:
requests.exceptions.HTTPError: 401 Client Error: Unauthorized
Response: {"error": "invalid_token", "message": "Token has expired"}
Request headers: Authorization: Bearer eyJhbG...''',
            expected_contains=["expired", "token", "refresh", "401"],
            context="Token expiration"
        ),
        TestCase(
            input_prompt='''Diagnose this error:
ConnectionError: HTTPSConnectionPool(host='api.example.com', port=443):
Max retries exceeded (Caused by SSLError(SSLCertVerificationError(1,
'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed')))''',
            expected_contains=["SSL", "certificate", "verify"],
            context="SSL certificate error"
        ),
    ],
    metadata={"domain": "debugging", "focus": "apis"}
)


def run_skill_demo(skill: SkillDocument, output_dir: str = "./demo_skills"):
    """Run a demo of a skill without API calls."""
    print(f"\n{'='*60}")
    print(f"SKILL: {skill.name}")
    print(f"{'='*60}")
    print(f"\nDescription: {skill.description}")
    print(f"\nTest Cases: {len(skill.test_cases)}")

    for i, tc in enumerate(skill.test_cases, 1):
        print(f"\n  Test {i}: {tc.context}")
        print(f"    Input: {tc.input_prompt[:60]}...")
        print(f"    Expects: {tc.expected_contains[:3]}")

    # Save skill
    path = skill.save(Path(output_dir))
    print(f"\nSaved to: {path}")

    # Show skill content preview
    print(f"\nSkill Content Preview:")
    print("-" * 40)
    lines = skill.content.split("\n")[:20]
    print("\n".join(lines))
    if len(skill.content.split("\n")) > 20:
        print("...")
    print("-" * 40)


def main():
    """Demo all example skills."""
    import argparse

    parser = argparse.ArgumentParser(description="Skill Transfer Examples")
    parser.add_argument("--output", default="./demo_skills", help="Output directory")
    parser.add_argument("--skill", choices=["security", "contacts", "api", "all"],
                       default="all", help="Which skill to demo")
    args = parser.parse_args()

    skills = {
        "security": CODE_REVIEW_SKILL,
        "contacts": DATA_EXTRACTION_SKILL,
        "api": API_DIAGNOSIS_SKILL,
    }

    print("Skill Transfer Examples")
    print("=" * 60)
    print("\nThese pre-built skills demonstrate how domain expertise")
    print("can be encoded and transferred to cheaper models.")

    if args.skill == "all":
        for skill in skills.values():
            run_skill_demo(skill, args.output)
    else:
        run_skill_demo(skills[args.skill], args.output)

    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("""
To use these skills with actual models:

1. Install dependencies:
   pip install anthropic openai

2. Set API keys:
   export ANTHROPIC_API_KEY=sk-ant-...
   export OPENAI_API_KEY=sk-...

3. Evaluate a skill:
   from examples import CODE_REVIEW_SKILL
   from skill_transfer import SkillTransferEngine, ModelClient, SkillEvaluator

   # Evaluate on GPT-4o-mini
   client = ModelClient(provider="openai", model="gpt-4o-mini")
   evaluator = SkillEvaluator(client)
   result = evaluator.evaluate(CODE_REVIEW_SKILL)
   print(result)

4. Or use with local models (Ollama):
   client = ModelClient(
       provider="local",
       model="llama3.2",
       base_url="http://localhost:11434/v1"
   )
""")


if __name__ == "__main__":
    main()
