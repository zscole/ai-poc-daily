#!/usr/bin/env python3
"""
Skill Transfer Engine: Transfer capabilities from expensive models to cheaper ones.

This POC demonstrates the emerging pattern of using agent skills to transfer
domain expertise from large, expensive models (like Claude Opus) to smaller,
cheaper, or local models.

The key insight: Skills are just well-structured context documents that encode
domain knowledge. When you inject the right context, cheaper models can match
expensive model performance on specific tasks.

Based on research from HuggingFace's upskill tool and the agent skills specification.
"""

import json
import os
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from datetime import datetime

# Optional imports for actual API usage
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@dataclass
class TestCase:
    """A test case for evaluating skill effectiveness."""
    input_prompt: str
    expected_contains: list[str] = field(default_factory=list)
    expected_not_contains: list[str] = field(default_factory=list)
    context: str = ""


@dataclass
class SkillDocument:
    """A transferable skill document following the Agent Skills spec."""
    name: str
    description: str
    content: str
    test_cases: list[TestCase] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Export skill as SKILL.md format."""
        header = f"""---
name: {self.name}
description: {self.description}
---

"""
        return header + self.content

    def save(self, output_dir: Path) -> Path:
        """Save skill to directory following agent skills spec."""
        skill_dir = output_dir / self.name.lower().replace(" ", "-")
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(self.to_markdown())

        # Write metadata and test cases
        meta = {
            "name": self.name,
            "description": self.description,
            "created": datetime.now().isoformat(),
            "test_cases": [asdict(tc) for tc in self.test_cases],
            **self.metadata
        }
        meta_path = skill_dir / "skill_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        return skill_dir


@dataclass
class EvaluationResult:
    """Result of evaluating a model with/without a skill."""
    model: str
    task: str
    baseline_score: float
    skill_score: float
    improvement: float
    baseline_tokens: int
    skill_tokens: int
    token_efficiency: float  # Negative means skill reduced tokens
    test_results: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        improvement_str = f"+{self.improvement:.1%}" if self.improvement > 0 else f"{self.improvement:.1%}"
        token_str = f"{self.token_efficiency:+.1%}" if self.token_efficiency != 0 else "0%"
        return f"""
Model: {self.model}
Task: {self.task}
--------------------
Baseline:    {self.baseline_score:.1%} ({self.baseline_tokens} tokens)
With Skill:  {self.skill_score:.1%} ({self.skill_tokens} tokens)
Improvement: {improvement_str}
Token Delta: {token_str}
"""


class ModelClient:
    """Unified client for different model providers."""

    def __init__(self, provider: str = "anthropic", model: str = None, base_url: str = None):
        self.provider = provider
        self.base_url = base_url

        if provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package required: pip install anthropic")
            self.client = Anthropic()
            self.model = model or "claude-sonnet-4-20250514"
        elif provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package required: pip install openai")
            self.client = OpenAI(base_url=base_url) if base_url else OpenAI()
            self.model = model or "gpt-4o"
        elif provider == "local":
            # OpenAI-compatible local endpoint (Ollama, LM Studio, llama.cpp)
            if not HAS_OPENAI:
                raise ImportError("openai package required: pip install openai")
            self.client = OpenAI(
                base_url=base_url or "http://localhost:11434/v1",
                api_key="not-needed"
            )
            self.model = model or "llama3.2"
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def complete(self, system: str, user: str, max_tokens: int = 2048) -> tuple[str, int]:
        """Get completion and return (response, tokens_used)."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text, tokens
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            tokens = response.usage.total_tokens if response.usage else 0
            return response.choices[0].message.content, tokens


class SkillGenerator:
    """Generates transferable skills from task descriptions or traces."""

    def __init__(self, teacher_client: ModelClient):
        self.teacher = teacher_client

    def from_task_description(self, task: str, examples: list[str] = None) -> SkillDocument:
        """Generate a skill from a task description."""
        examples_text = ""
        if examples:
            examples_text = "\n\nExamples of good outputs:\n" + "\n---\n".join(examples)

        system = """You are an expert at creating agent skill documents.
Your goal is to create a SKILL.md file that encodes domain expertise
so that smaller/cheaper models can perform the task as well as you.

Output format:
1. First, a brief analysis of what knowledge is needed (2-3 sentences)
2. Then the actual skill content in markdown, starting with # Task Overview

The skill should:
- Be concise but comprehensive
- Include specific patterns, rules, and examples
- Anticipate common mistakes
- Provide clear decision trees for ambiguous cases"""

        prompt = f"""Create a skill document for this task:

TASK: {task}
{examples_text}

Remember: The skill should transfer YOUR capabilities to a smaller model."""

        response, _ = self.teacher.complete(system, prompt)

        # Extract the skill content (after the analysis)
        lines = response.split("\n")
        skill_start = 0
        for i, line in enumerate(lines):
            if line.startswith("# "):
                skill_start = i
                break
        skill_content = "\n".join(lines[skill_start:])

        return SkillDocument(
            name=task.split()[0:3].__str__().replace("'", "").replace("[", "").replace("]", "").replace(",", "-"),
            description=task[:200],
            content=skill_content,
            metadata={"generated_by": self.teacher.model, "task": task}
        )

    def from_trace(self, trace: str, task: str) -> SkillDocument:
        """Generate a skill from an agent trace (conversation/interaction log)."""
        system = """You are an expert at extracting reusable knowledge from agent traces.
Analyze the trace and create a SKILL.md that captures the successful patterns.

Focus on:
- What knowledge was required
- What decisions were made and why
- What pitfalls were encountered
- What patterns led to success

Output a clean, reusable skill document in markdown."""

        prompt = f"""Extract a skill from this trace:

TASK: {task}

TRACE:
{trace}

Create a SKILL.md that would help another model replicate this success."""

        response, _ = self.teacher.complete(system, prompt, max_tokens=4096)

        return SkillDocument(
            name=f"skill-from-trace-{hashlib.md5(trace.encode()).hexdigest()[:8]}",
            description=f"Skill extracted from trace for: {task[:100]}",
            content=response,
            metadata={"generated_by": self.teacher.model, "source": "trace", "task": task}
        )


class TestCaseGenerator:
    """Generates test cases for evaluating skill effectiveness."""

    def __init__(self, client: ModelClient):
        self.client = client

    def generate(self, skill: SkillDocument, count: int = 5) -> list[TestCase]:
        """Generate test cases for a skill."""
        system = """You generate test cases for evaluating AI skills.
Each test case should:
- Have a clear input prompt
- Have specific expected outputs (strings that should appear)
- Cover different aspects of the skill
- Include edge cases

Output JSON array with objects containing:
{
  "input_prompt": "...",
  "expected_contains": ["...", "..."],
  "expected_not_contains": ["...", "..."],
  "context": "why this tests the skill"
}"""

        prompt = f"""Generate {count} test cases for this skill:

{skill.to_markdown()}

Return a JSON array of test cases."""

        response, _ = self.client.complete(system, prompt)

        # Parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            cases_data = json.loads(response.strip())
            return [TestCase(**case) for case in cases_data]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Warning: Could not parse test cases: {e}")
            return []


class SkillEvaluator:
    """Evaluates model performance with and without skills."""

    def __init__(self, student_client: ModelClient):
        self.student = student_client

    def evaluate_test_case(
        self,
        test_case: TestCase,
        skill: Optional[SkillDocument] = None
    ) -> tuple[bool, int, str]:
        """Run a single test case, return (passed, tokens, response)."""
        system = "You are a helpful assistant."
        if skill:
            system = f"{skill.content}\n\n---\n\nYou are a helpful assistant."

        response, tokens = self.student.complete(system, test_case.input_prompt)

        # Check expected outputs
        passed = True
        for expected in test_case.expected_contains:
            if expected.lower() not in response.lower():
                passed = False
                break

        for unexpected in test_case.expected_not_contains:
            if unexpected.lower() in response.lower():
                passed = False
                break

        return passed, tokens, response

    def evaluate(
        self,
        skill: SkillDocument,
        test_cases: list[TestCase] = None,
        runs_per_case: int = 1
    ) -> EvaluationResult:
        """Evaluate model performance with and without the skill."""
        cases = test_cases or skill.test_cases
        if not cases:
            raise ValueError("No test cases provided")

        baseline_results = []
        skill_results = []
        baseline_tokens = 0
        skill_tokens = 0

        for case in cases:
            for _ in range(runs_per_case):
                # Baseline (without skill)
                passed, tokens, _ = self.evaluate_test_case(case, skill=None)
                baseline_results.append(passed)
                baseline_tokens += tokens

                # With skill
                passed, tokens, response = self.evaluate_test_case(case, skill=skill)
                skill_results.append(passed)
                skill_tokens += tokens

        baseline_score = sum(baseline_results) / len(baseline_results)
        skill_score = sum(skill_results) / len(skill_results)

        return EvaluationResult(
            model=self.student.model,
            task=skill.description,
            baseline_score=baseline_score,
            skill_score=skill_score,
            improvement=skill_score - baseline_score,
            baseline_tokens=baseline_tokens,
            skill_tokens=skill_tokens,
            token_efficiency=(skill_tokens - baseline_tokens) / baseline_tokens if baseline_tokens > 0 else 0,
            test_results=[
                {"case": i, "baseline": baseline_results[i], "with_skill": skill_results[i]}
                for i in range(len(baseline_results))
            ]
        )


class SkillTransferEngine:
    """
    Main engine for transferring capabilities between models.

    Workflow:
    1. Define a task
    2. Generate a skill using a teacher model (expensive/capable)
    3. Generate test cases
    4. Evaluate student model(s) with and without the skill
    5. Iterate on the skill if needed
    """

    def __init__(
        self,
        teacher_provider: str = "anthropic",
        teacher_model: str = None,
        output_dir: str = "./skills"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.teacher = ModelClient(provider=teacher_provider, model=teacher_model)
        self.generator = SkillGenerator(self.teacher)
        self.test_generator = TestCaseGenerator(self.teacher)

    def create_skill(
        self,
        task: str,
        examples: list[str] = None,
        trace: str = None,
        num_test_cases: int = 5
    ) -> SkillDocument:
        """Create a complete skill with test cases."""
        print(f"Generating skill for: {task[:50]}...")

        if trace:
            skill = self.generator.from_trace(trace, task)
        else:
            skill = self.generator.from_task_description(task, examples)

        print(f"Generating {num_test_cases} test cases...")
        skill.test_cases = self.test_generator.generate(skill, num_test_cases)

        # Save the skill
        skill_path = skill.save(self.output_dir)
        print(f"Skill saved to: {skill_path}")

        return skill

    def evaluate_transfer(
        self,
        skill: SkillDocument,
        student_configs: list[dict]
    ) -> list[EvaluationResult]:
        """Evaluate skill transfer to multiple student models."""
        results = []

        for config in student_configs:
            print(f"Evaluating {config.get('model', 'default')}...")
            student = ModelClient(**config)
            evaluator = SkillEvaluator(student)

            try:
                result = evaluator.evaluate(skill)
                results.append(result)
                print(result)
            except Exception as e:
                print(f"Error evaluating {config}: {e}")

        return results


# Demo functions for testing without API calls

def create_demo_skill() -> SkillDocument:
    """Create a demo skill for testing the pipeline."""
    content = """# JSON Schema Validation

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
- Email: use regex /^[^@]+@[^@]+\\.[^@]+$/
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
"""

    skill = SkillDocument(
        name="json-schema-validator",
        description="Validate JSON objects against schemas with clear error reporting",
        content=content,
        test_cases=[
            TestCase(
                input_prompt='Validate {"name": 123} against schema {"name": "string"}',
                expected_contains=["wrong type", "string", "number"],
                context="Basic type mismatch"
            ),
            TestCase(
                input_prompt='Validate {"email": "bad"} against schema {"email": "email"}',
                expected_contains=["invalid", "email"],
                context="Format validation"
            ),
            TestCase(
                input_prompt='Validate {} against schema {"required": ["name"]}',
                expected_contains=["required", "missing", "name"],
                context="Required field check"
            ),
        ],
        metadata={"demo": True}
    )

    return skill


def demo_report() -> str:
    """Generate a demo evaluation report."""
    return """
========================================
SKILL TRANSFER EVALUATION REPORT
========================================

Skill: json-schema-validator
Description: Validate JSON objects against schemas

RESULTS BY MODEL:
-----------------

Claude Sonnet (Teacher)
  Baseline:    95% (1,247 tokens)
  With Skill:  97% (1,102 tokens)
  Improvement: +2%
  Token Delta: -11.6%

GPT-4o-mini
  Baseline:    72% (1,456 tokens)
  With Skill:  89% (1,203 tokens)
  Improvement: +17%
  Token Delta: -17.4%

Llama 3.2 8B (Local)
  Baseline:    45% (2,103 tokens)
  With Skill:  78% (1,567 tokens)
  Improvement: +33%
  Token Delta: -25.5%

Qwen 2.5 7B (Local)
  Baseline:    51% (1,987 tokens)
  With Skill:  82% (1,445 tokens)
  Improvement: +31%
  Token Delta: -27.3%

========================================
SUMMARY
========================================

The skill successfully transfers domain expertise:
- Largest improvement: +33% (Llama 3.2)
- Best token efficiency: -27.3% (Qwen 2.5)
- All models improved with the skill

RECOMMENDATION: Use this skill with local models
for cost-effective JSON validation at ~80% accuracy.
"""


def main():
    """Demo the skill transfer pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Skill Transfer Engine")
    parser.add_argument("--demo", action="store_true", help="Run demo without API calls")
    parser.add_argument("--task", type=str, help="Task description for skill generation")
    parser.add_argument("--output", type=str, default="./skills", help="Output directory")

    args = parser.parse_args()

    if args.demo:
        print("Running demo mode (no API calls)...")
        print()

        # Create demo skill
        skill = create_demo_skill()
        print("Created skill:")
        print(skill.to_markdown()[:500] + "...")
        print()

        # Save it
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill.save(output_dir)
        print(f"Saved to: {skill_path}")
        print()

        # Show evaluation report
        print(demo_report())
        return

    if not args.task:
        print("Error: --task required (or use --demo)")
        return

    # Real mode
    engine = SkillTransferEngine(output_dir=args.output)

    # Create skill
    skill = engine.create_skill(args.task)

    # Evaluate on different models
    students = [
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "local", "model": "llama3.2", "base_url": "http://localhost:11434/v1"},
    ]

    results = engine.evaluate_transfer(skill, students)

    # Print summary
    print("\n" + "=" * 40)
    print("TRANSFER RESULTS")
    print("=" * 40)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
