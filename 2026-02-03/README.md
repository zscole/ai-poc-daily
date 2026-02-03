# Agent Skill Distillation Framework

**Date**: February 3, 2026
**Trend**: Skill-based agent architectures (OpenAI Codex App, HuggingFace Upskill)

## Why This Matters

The AI industry just hit an inflection point. Two major developments in the past week:

1. **OpenAI Codex App** (Feb 2, 2026) - Launched with a skills system that lets agents bundle instructions, resources, and scripts for reliable task execution. 658 points on HN.

2. **HuggingFace Upskill** (Jan 28, 2026) - Published research showing you can use Claude to generate skills that make smaller models perform like frontier models on specific tasks. 109 likes on HF blog.

The core insight: **Frontier models are expensive trial-and-error machines. Smaller models can match them on specific tasks if you give them the playbook.**

This POC demonstrates the mechanics of skill distillation - extracting transferable knowledge from agent traces.


## The Problem

Running Claude Opus 4.5 on every request is expensive. But smaller models fail on complex, domain-specific tasks because they lack procedural knowledge.

The traditional approach:
- Pay $$$ for every complex task
- OR accept lower accuracy with cheaper models

The skill distillation approach:
- Use frontier model ONCE to solve the task
- Extract the execution trace
- Distill into a structured skill file
- Apply skill to cheaper models forever


## How It Works

```
+------------------+     +-----------------+     +------------------+
|  FRONTIER MODEL  | --> |  TRACE CAPTURE  | --> | SKILL DISTILLER  |
|  (Claude Opus)   |     |  (7 steps,      |     |  - Patterns      |
|                  |     |   3,450 tokens) |     |  - Decisions     |
+------------------+     +-----------------+     |  - Instructions  |
                                                 +--------+---------+
                                                          |
                                                          v
                         +------------------+     +------------------+
                         |  CHEAPER MODEL   | <-- |   SKILL.md       |
                         |  (GLM-4-Flash)   |     |  (~500 tokens)   |
                         |  +36% accuracy   |     |                  |
                         +------------------+     +------------------+
```


## Results

Simulated evaluation based on Upskill paper findings:

| Model           | Baseline | + Skill | Improvement | Token Reduction |
|-----------------|----------|---------|-------------|-----------------|
| glm-4-flash     | 40.0%    | 76.0%   | +36.0%      | 48.0%           |
| mistral-small   | 45.0%    | 78.0%   | +33.0%      | 46.5%           |
| gpt-4o-mini     | 60.0%    | 84.0%   | +24.0%      | 42.0%           |
| llama-3.3-70b   | 65.0%    | 86.0%   | +21.0%      | 40.5%           |
| qwen-2.5-72b    | 70.0%    | 88.0%   | +18.0%      | 39.0%           |
| claude-sonnet-4 | 80.0%    | 92.0%   | +12.0%      | 36.0%           |
| gpt-4o          | 85.0%    | 94.0%   | +9.0%       | 34.5%           |
| claude-opus-4.5 | 95.0%    | 98.0%   | +3.0%       | 31.5%           |
| **AVERAGE**     | -        | -       | **+19.5%**  | **39.8%**       |

Key finding: **Weaker models benefit dramatically from skills** (+24-36% accuracy boost).


## Quick Start

```bash
# No dependencies required (Python 3.8+)
python3 skill_distillation.py
```


## Code Architecture

```
skill_distillation.py
|
+-- Data Structures
|   +-- TraceStep       # Single step in execution trace
|   +-- AgentTrace      # Complete trace from frontier model
|   +-- Skill           # Distilled skill (outputs SKILL.md)
|   +-- EvalResult      # Evaluation metrics
|
+-- TraceAnalyzer
|   +-- extract_patterns()      # Find recurring action sequences
|   +-- extract_key_decisions() # Identify critical reasoning
|
+-- SkillDistiller
|   +-- distill()               # Convert trace -> skill
|   +-- _extract_instructions() # Generate step-by-step guide
|   +-- _generate_test_cases()  # Create validation tests
|
+-- SkillEvaluator
    +-- simulate_evaluation()   # Test skill across models
    +-- print_results()         # Display comparison table
```


## Generated Skill Example

The POC generates a SKILL.md file following the Agent Skills spec:

```markdown
---
name: build-fused-layernorm-gelu
description: Skill for: Build a fused LayerNorm + GELU CUDA kernel...
---

# build-fused-layernorm-gelu

## Overview
Skill for building optimized CUDA kernels using HuggingFace kernel-builder.

## Instructions
1. Perform sequence: Analyze requirements -> Create project structure -> Configure build.toml
2. Perform sequence: Fix compilation error -> Create PyTorch bindings -> Test kernel

## Examples
### Primary Example
**Input:** Build a fused LayerNorm + GELU CUDA kernel optimized for NVIDIA H100
**Output:** Kernel validated: 2.3x speedup, numerical accuracy within 1e-5
```


## Key Insights

### 1. Skill Transfer Efficiency

The inverse relationship between model capability and skill benefit is profound:

- GLM-4-Flash: **+36%** accuracy boost (0.40 -> 0.76)
- Claude Opus: **+3%** boost (0.95 -> 0.98)

This means skills are most valuable for cost optimization, not capability expansion.


### 2. Economic Implications

Rough pricing comparison for 1,000 complex tasks:

| Approach | Cost | Accuracy |
|----------|------|----------|
| Claude Opus direct | ~$70 | 95% |
| GLM-4-Flash (no skill) | ~$0.50 | 40% |
| GLM-4-Flash + skill | ~$0.50 | 76% |
| Hybrid (1 Opus + 999 GLM) | ~$0.57 | 76% |

The hybrid approach gives you 76% accuracy at 1/120th the cost.


### 3. Skill Design Patterns

From analyzing the Upskill and Codex ecosystems, effective skills share:

- **Concrete examples**: Not just instructions, but input/output pairs
- **Error recovery**: Document common failures and fixes
- **Tool specifications**: Exact versions, configs, API patterns
- **Validation steps**: How to verify correct execution


## Connections to Current Trends

### OpenAI Codex App (Feb 2, 2026)

The Codex app introduces a skill library and skill creator interface. Notable features:
- Skills bundle instructions + resources + scripts
- Automations run skills on schedules
- Multi-agent orchestration with worktrees

This POC demonstrates the underlying mechanics of how skills are generated.


### HuggingFace Upskill (Jan 28, 2026)

The Upskill framework provides:
- `upskill generate` - Create skills from traces
- `upskill eval` - Benchmark skills across models
- Teacher-student evaluation pattern

This POC implements a minimal version of this pattern.


### Nano-vLLM (Feb 2, 2026)

A 1,200-line vLLM implementation that matches full vLLM performance. Shows that minimal, educational implementations can be production-grade. This POC follows that philosophy for skill distillation.


## Production Considerations

To use this in production:

1. **Capture real traces**: Instrument your frontier model calls to log execution traces

2. **Build a skill library**: Organize skills by domain (CUDA, deployment, testing, etc.)

3. **A/B test**: Compare skill-augmented smaller models against baseline

4. **Iterate on skills**: Use evaluation results to refine skill instructions

5. **Version control**: Track skill changes like code


## References

- OpenAI Codex App: https://openai.com/index/introducing-the-codex-app/
- HuggingFace Upskill: https://huggingface.co/blog/upskill
- Agent Skills Spec: https://agentskills.io
- Nano-vLLM: https://github.com/GeeeekExplorer/nano-vllm
- Neutree Blog (Nano-vLLM explainer): https://neutree.ai/blog/nano-vllm-part-1


## Files

- `skill_distillation.py` - Main implementation (~600 lines)
- `skill_artifacts.json` - Exported trace, skill, and evaluation results
- `README.md` - This file
