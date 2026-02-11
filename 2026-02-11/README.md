# Skill Transfer Engine

**Transfer capabilities from expensive models to cheaper ones.**

This POC implements the emerging pattern of using agent skills to democratize
AI capabilities. Instead of paying for expensive models on every request, you
can use them once to generate a "skill document" that transfers their expertise
to cheaper or local models.

## Why This Matters

The AI world is converging on a pattern:

1. **Expensive models** (Claude Opus, GPT-4) are great at complex tasks
2. **Cheap/local models** (GPT-4o-mini, Llama, Qwen) are fast and affordable
3. **Skills bridge the gap** by encoding domain expertise as context

This is based on research from:
- HuggingFace's `upskill` tool (released Jan 2026)
- The Agent Skills specification (agentskills.io)
- Anthropic's skill-creator pattern

The key insight: **Context is transferable**. When you give a smaller model
the right context, it can match expensive model performance on specific tasks.

## The Pattern

```
[Expensive Model] --generates--> [Skill Document] --improves--> [Cheap Model]
      |                               |                              |
   Claude Opus                   SKILL.md with                  Local Llama
   $15/M tokens                  domain expertise               Free/fast
```

## Quick Start

### Demo Mode (No API Required)

```bash
# Run without API calls
python skill_transfer.py --demo

# View example skills
python examples.py --skill all
```

### With API Keys

```bash
# Install dependencies
pip install anthropic openai

# Set your keys
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Generate a skill for a task
python skill_transfer.py --task "Write SQL queries for PostgreSQL"

# The skill is saved to ./skills/
```

## How It Works

### 1. Skill Generation

A teacher model (Claude Opus, GPT-4) generates a skill document:

```python
from skill_transfer import SkillTransferEngine

engine = SkillTransferEngine(teacher_provider="anthropic")
skill = engine.create_skill(
    task="Review Python code for security vulnerabilities",
    examples=["example of good security review..."]
)
```

The skill document is a structured markdown file:

```markdown
---
name: python-security-review
description: Review Python code for security vulnerabilities
---

# Security Review Guide

## Critical Checks
1. SQL injection: Look for string formatting in queries...
2. Command injection: subprocess with shell=True...

## Output Format
[SEVERITY] Issue Title
Line: X
Code: `affected snippet`
Fix: How to fix it
```

### 2. Test Case Generation

The engine generates test cases to measure skill effectiveness:

```python
# Automatically generated from the skill
test_cases = [
    TestCase(
        input="Review: query = f\"SELECT * FROM users WHERE id = {id}\"",
        expected_contains=["SQL", "injection", "CRITICAL"]
    ),
    ...
]
```

### 3. Evaluation

Compare model performance with and without the skill:

```python
from skill_transfer import ModelClient, SkillEvaluator

# Test on a cheap model
client = ModelClient(provider="openai", model="gpt-4o-mini")
evaluator = SkillEvaluator(client)
result = evaluator.evaluate(skill)

print(result)
# Baseline:    72%
# With Skill:  89%
# Improvement: +17%
```

### 4. Deployment

Use the skill in your applications:

```python
# Load the skill
skill_content = open("./skills/python-security-review/SKILL.md").read()

# Use as system prompt
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": skill_content},
        {"role": "user", "content": "Review this code: ..."}
    ]
)
```

## Example Skills Included

### 1. Python Security Review
Reviews code for OWASP Top 10 vulnerabilities. Transfers security expertise
to models that would otherwise miss subtle issues.

### 2. Contact Extraction
Extracts structured contact information from unstructured text. Handles
edge cases like obfuscated emails and international phone formats.

### 3. API Error Diagnosis
Diagnoses API errors from logs and stack traces. Provides specific fixes
based on error patterns.

Run the examples:

```bash
python examples.py --skill security
python examples.py --skill contacts
python examples.py --skill api
```

## Using with Local Models

Works with any OpenAI-compatible endpoint:

```python
# Ollama
client = ModelClient(
    provider="local",
    model="llama3.2",
    base_url="http://localhost:11434/v1"
)

# LM Studio
client = ModelClient(
    provider="local",
    model="qwen2.5-7b",
    base_url="http://localhost:1234/v1"
)

# llama.cpp server
client = ModelClient(
    provider="local",
    model="model",
    base_url="http://localhost:8080/v1"
)
```

## Architecture

```
skill_transfer.py
|
+-- SkillDocument        # The transferable skill (SKILL.md format)
+-- TestCase             # Test cases for evaluation
|
+-- ModelClient          # Unified client for Anthropic/OpenAI/Local
|
+-- SkillGenerator       # Creates skills from tasks or traces
+-- TestCaseGenerator    # Generates test cases for skills
+-- SkillEvaluator       # Measures skill effectiveness
|
+-- SkillTransferEngine  # Orchestrates the full pipeline
```

## Real-World Results

From HuggingFace's research on CUDA kernel building:

| Model | Baseline | With Skill | Improvement |
|-------|----------|------------|-------------|
| Claude Opus | 95% | 97% | +2% |
| GPT-4o-mini | 72% | 89% | +17% |
| Llama 3.2 8B | 45% | 78% | +33% |
| Qwen 2.5 7B | 51% | 82% | +31% |

The pattern is clear: **skills help smaller models more**.

## When to Use This

**Good fit:**
- Domain-specific tasks with clear patterns
- Tasks where you need consistent output format
- Reducing costs on high-volume applications
- Running models locally for privacy/speed

**Not ideal for:**
- Tasks requiring general reasoning
- Novel problems without clear patterns
- Tasks where the cheap model already excels

## Future Directions

1. **Skill Marketplace**: Share and discover skills on ClawHub
2. **Auto-iteration**: Automatically improve skills based on failures
3. **Skill Composition**: Combine multiple skills for complex tasks
4. **Continuous Learning**: Update skills from production feedback

## Related Work

- [HuggingFace upskill](https://huggingface.co/blog/upskill) - The inspiration for this POC
- [Agent Skills Spec](https://agentskills.io) - Standard format for skills
- [Anthropic skill-creator](https://github.com/anthropics/skills) - Anthropic's approach
- [Transformers.js v4](https://huggingface.co/blog/transformersjs-v4) - Client-side models

## Requirements

- Python 3.10+
- anthropic (optional, for Claude)
- openai (optional, for OpenAI/local models)

## License

MIT
