# Agent Skills Runtime

A TypeScript implementation of the [agentskills.io](https://agentskills.io) specification for discovering, validating, and executing Agent Skills.

## Why This Matters

Agent Skills are becoming the standard way to extend AI agent capabilities. With OpenAI releasing their [Skills Catalog for Codex](https://github.com/openai/skills) and Anthropic open-sourcing the specification, we are seeing a convergence on a universal format for packaging agent capabilities.

This POC implements a complete runtime that can:

- **Discover** skills from filesystem paths (recursive or flat)
- **Parse** SKILL.md files with YAML frontmatter
- **Validate** skill structure and dependencies
- **Match** skills to user queries using triggers, keywords, and fuzzy matching
- **Execute** skill scripts safely with timeout and argument validation
- **Generate** LLM context for skill injection into prompts

## The agentskills.io Standard

Agent Skills are folders containing:

```
my-skill/
  SKILL.md           # Instructions + metadata (YAML frontmatter)
  scripts/           # Executable scripts (node, python, shell, deno)
  assets/            # Data files, templates, references
```

The SKILL.md file uses YAML frontmatter for metadata:

```markdown
---
name: Code Generator
version: 1.0.0
description: Generates code snippets
tags: [code, generation]
triggers:
  - type: keyword
    value: generate code
scripts:
  - name: generate
    path: scripts/generate.js
    runtime: node
---

# Code Generator

Instructions for the agent on how to use this skill...
```

## Quick Start

```bash
# Install dependencies
npm install

# Run tests
npm test

# Run demo
npm start
```

## Usage

### Basic Runtime

```typescript
import { AgentSkillsRuntime } from 'agent-skills-runtime';

const runtime = new AgentSkillsRuntime({
  skillPaths: ['./skills', '~/.agent-skills'],
  allowedRuntimes: ['node', 'python', 'shell'],
  maxExecutionTime: 30000,
});

await runtime.initialize();

// Find relevant skills
const matches = runtime.findSkills('generate typescript code');

// Execute a skill script
const result = await runtime.executeScript(
  matches[0].skill,
  'generate',
  { language: 'typescript', description: 'hello world' }
);

console.log(result.output);
```

### Generate LLM Context

```typescript
// Full context for a specific skill
const context = runtime.generateContext(skill);

// Compact reference for system prompts
const reference = runtime.generateCompactReference();

// Catalog of all skills
const catalog = runtime.generateCatalog();
```

### Individual Components

```typescript
import { 
  SkillParser, 
  SkillDiscovery, 
  SkillExecutor, 
  SkillContextGenerator 
} from 'agent-skills-runtime';

// Parse a single skill
const parser = new SkillParser();
const skill = parser.parseSkill('./my-skill');
const validation = parser.validateSkill(skill);

// Discover skills
const discovery = new SkillDiscovery();
const skills = await discovery.discover({ 
  paths: ['./skills'], 
  recursive: true 
});

// Execute scripts
const executor = new SkillExecutor({ maxExecutionTime: 10000 });
const result = await executor.execute({
  skill,
  script: skill.scripts.get('main'),
  args: { input: 'test' },
  workdir: '/tmp',
});

// Generate context
const generator = new SkillContextGenerator();
const context = generator.generateContext(skill);
```

## Architecture

```
src/
  types.ts          # TypeScript interfaces
  parser.ts         # SKILL.md parsing and validation
  discovery.ts      # Skill discovery and matching
  executor.ts       # Safe script execution
  context.ts        # LLM context generation
  runtime.ts        # High-level API
  index.ts          # Public exports
```

## Skill Matching

Skills are matched against queries using:

1. **Explicit triggers** - Keywords, patterns, or intents defined in the skill
2. **Implicit triggers** - Skill name and tags
3. **Description matching** - Word overlap with description
4. **Fuzzy matching** - Partial matches for typo tolerance

Matches are scored and returned in relevance order.

## Script Execution

Scripts are executed in isolated child processes with:

- **Timeout enforcement** - Kills long-running scripts
- **Argument validation** - Checks required args and types
- **Runtime restrictions** - Only allowed runtimes can execute
- **Output capture** - Collects stdout/stderr

## Example Skills Included

### Code Generator

Generates boilerplate code for TypeScript, Python, Rust, Go, and Shell.

```bash
# Execute directly
node example-skills/code-generator/scripts/generate.js \
  --language typescript \
  --description "sort an array"
```

### API Tester

Makes HTTP requests and displays formatted responses.

```bash
# Execute directly
node example-skills/api-tester/scripts/request.js \
  --url https://httpbin.org/get
```

## Integration Points

This runtime is designed to integrate with:

- **OpenClaw** - Drop skills into ~/.openclaw/skills/
- **Claude Code / Codex** - Use the same skill format
- **Custom agents** - Import the runtime and inject context

## Comparison with Existing Solutions

| Feature | This Runtime | OpenAI Skills | Claude Skills |
|---------|-------------|---------------|---------------|
| Open source | Yes | Partial | Yes |
| TypeScript | Yes | N/A | N/A |
| Script execution | Yes | Via agent | Via agent |
| Standalone library | Yes | No | No |
| Context generation | Yes | N/A | N/A |

## Future Directions

- Semantic similarity matching using embeddings
- Skill dependency resolution
- Remote skill registries
- Skill versioning and updates
- Permission sandboxing

## References

- [agentskills.io Specification](https://agentskills.io)
- [OpenAI Skills Catalog](https://github.com/openai/skills)
- [Anthropic Skills Examples](https://github.com/anthropics/skills)

## License

MIT
