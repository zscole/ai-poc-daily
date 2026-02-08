/**
 * Agent Skills Runtime - Test Suite
 * Basic tests for all components
 */

import { SkillParser } from './parser.js';
import { SkillDiscovery } from './discovery.js';
import { SkillExecutor } from './executor.js';
import { SkillContextGenerator } from './context.js';
import { AgentSkillsRuntime } from './runtime.js';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdirSync, writeFileSync, rmSync, existsSync } from 'node:fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void | Promise<void>) {
  return (async () => {
    try {
      await fn();
      console.log(`[PASS] ${name}`);
      passed++;
    } catch (error) {
      console.log(`[FAIL] ${name}`);
      console.log(`       ${error instanceof Error ? error.message : error}`);
      failed++;
    }
  })();
}

function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}

function assertEqual<T>(actual: T, expected: T, message?: string) {
  if (actual !== expected) {
    throw new Error(message || `Expected ${expected}, got ${actual}`);
  }
}

// Setup test fixtures
const testDir = join(__dirname, '..', '.test-skills');

function setupTestSkills() {
  if (existsSync(testDir)) {
    rmSync(testDir, { recursive: true });
  }
  mkdirSync(testDir, { recursive: true });

  // Create test skill 1: code-generator
  const skill1Dir = join(testDir, 'code-generator');
  mkdirSync(skill1Dir);
  mkdirSync(join(skill1Dir, 'scripts'));
  
  writeFileSync(join(skill1Dir, 'SKILL.md'), `---
name: Code Generator
version: 1.0.0
description: Generates code snippets in various languages
tags:
  - code
  - generation
triggers:
  - type: keyword
    value: generate code
  - type: keyword
    value: write function
scripts:
  - name: generate
    path: scripts/generate.js
    runtime: node
    args:
      - name: language
        type: string
        required: true
      - name: description
        type: string
        required: true
---

# Code Generator Skill

This skill generates code snippets based on natural language descriptions.

## Usage

Ask the agent to generate code in a specific language by describing what you want.

## Examples

- "Generate a TypeScript function that sorts an array"
- "Write a Python script to read a CSV file"
`);

  writeFileSync(join(skill1Dir, 'scripts', 'generate.js'), `
const args = process.argv.slice(2);
const langIdx = args.indexOf('--language');
const descIdx = args.indexOf('--description');

const language = langIdx >= 0 ? args[langIdx + 1] : 'javascript';
const description = descIdx >= 0 ? args[descIdx + 1] : 'hello world';

console.log(\`// Generated \${language} code for: \${description}\`);
console.log(\`function example() { return "Hello, World!"; }\`);
`);

  // Create test skill 2: data-analyzer
  const skill2Dir = join(testDir, 'data-analyzer');
  mkdirSync(skill2Dir);
  mkdirSync(join(skill2Dir, 'templates'));
  
  writeFileSync(join(skill2Dir, 'SKILL.md'), `---
name: Data Analyzer
version: 2.0.0
description: Analyzes datasets and generates reports
tags:
  - data
  - analysis
  - reporting
triggers:
  - type: keyword
    value: analyze data
  - type: pattern
    value: ".*statistics.*"
---

# Data Analyzer

Provides tools for data analysis and reporting.

## Capabilities

- Statistical analysis
- Data visualization suggestions
- Report generation
`);

  writeFileSync(join(skill2Dir, 'templates', 'report.md'), `# Analysis Report

## Summary
{{summary}}

## Findings
{{findings}}
`);
}

function cleanupTestSkills() {
  if (existsSync(testDir)) {
    rmSync(testDir, { recursive: true });
  }
}

async function runTests() {
  console.log('========================================');
  console.log('Agent Skills Runtime - Test Suite');
  console.log('========================================');
  console.log('');

  setupTestSkills();

  // Parser tests
  console.log('--- Parser Tests ---');
  
  await test('Parser: parses skill with frontmatter', () => {
    const parser = new SkillParser();
    const skill = parser.parseSkill(join(testDir, 'code-generator'));
    assert(skill !== null, 'Skill should be parsed');
    assertEqual(skill!.manifest.name, 'Code Generator');
    assertEqual(skill!.manifest.version, '1.0.0');
    assert(skill!.instructions.includes('Code Generator Skill'), 'Instructions should contain content');
  });

  await test('Parser: discovers scripts', () => {
    const parser = new SkillParser();
    const skill = parser.parseSkill(join(testDir, 'code-generator'));
    assert(skill !== null, 'Skill should be parsed');
    assert(skill!.scripts.has('generate'), 'Should have generate script');
    assertEqual(skill!.scripts.get('generate')!.runtime, 'node');
  });

  await test('Parser: validates skill structure', () => {
    const parser = new SkillParser();
    const skill = parser.parseSkill(join(testDir, 'code-generator'));
    assert(skill !== null, 'Skill should be parsed');
    const validation = parser.validateSkill(skill!);
    assert(validation.valid, 'Skill should be valid');
  });

  // Discovery tests
  console.log('');
  console.log('--- Discovery Tests ---');

  await test('Discovery: finds skills in directory', async () => {
    const discovery = new SkillDiscovery();
    const skills = await discovery.discover({ paths: [testDir], recursive: true });
    assertEqual(skills.length, 2, 'Should find 2 skills');
  });

  await test('Discovery: matches skills by keyword', async () => {
    const discovery = new SkillDiscovery();
    await discovery.discover({ paths: [testDir], recursive: true });
    const matches = discovery.findSkills('generate code');
    assert(matches.length > 0, 'Should find matching skills');
    assertEqual(matches[0].skill.manifest.name, 'Code Generator');
  });

  await test('Discovery: matches skills by tag', async () => {
    const discovery = new SkillDiscovery();
    await discovery.discover({ paths: [testDir], recursive: true });
    const matches = discovery.findSkills('analysis');
    assert(matches.some(m => m.skill.manifest.name === 'Data Analyzer'), 'Should match Data Analyzer');
  });

  // Executor tests
  console.log('');
  console.log('--- Executor Tests ---');

  await test('Executor: validates script arguments', () => {
    const executor = new SkillExecutor();
    const script = {
      name: 'test',
      path: 'test.js',
      runtime: 'node' as const,
      args: [
        { name: 'required', type: 'string' as const, required: true },
        { name: 'optional', type: 'number' as const },
      ],
    };
    
    const valid = executor.validateArgs(script, { required: 'value' });
    assert(valid.valid, 'Should be valid with required arg');
    
    const invalid = executor.validateArgs(script, {});
    assert(!invalid.valid, 'Should be invalid without required arg');
  });

  await test('Executor: executes node script', async () => {
    const parser = new SkillParser();
    const executor = new SkillExecutor({ maxExecutionTime: 5000 });
    const skill = parser.parseSkill(join(testDir, 'code-generator'));
    assert(skill !== null, 'Skill should be parsed');
    
    const result = await executor.execute({
      skill: skill!,
      script: skill!.scripts.get('generate')!,
      args: { language: 'typescript', description: 'test function' },
      workdir: skill!.path,
    });
    
    assert(result.success, `Execution should succeed: ${result.error}`);
    assert(result.output?.includes('typescript'), 'Output should mention language');
  });

  // Context Generator tests
  console.log('');
  console.log('--- Context Generator Tests ---');

  await test('Context: generates skill context', () => {
    const parser = new SkillParser();
    const generator = new SkillContextGenerator();
    const skill = parser.parseSkill(join(testDir, 'code-generator'));
    
    const context = generator.generateContext(skill!);
    assert(context.includes('# Skill: Code Generator'), 'Should have skill header');
    assert(context.includes('## Instructions'), 'Should have instructions section');
    assert(context.includes('## Available Scripts'), 'Should list scripts');
  });

  await test('Context: generates compact reference', () => {
    const parser = new SkillParser();
    const generator = new SkillContextGenerator();
    const skills = [
      parser.parseSkill(join(testDir, 'code-generator'))!,
      parser.parseSkill(join(testDir, 'data-analyzer'))!,
    ];
    
    const compact = generator.generateCompactReference(skills);
    assert(compact.includes('<available_skills>'), 'Should have XML structure');
    assert(compact.includes('Code Generator'), 'Should include skill names');
  });

  // Runtime tests
  console.log('');
  console.log('--- Runtime Integration Tests ---');

  await test('Runtime: full initialization flow', async () => {
    const runtime = new AgentSkillsRuntime({
      skillPaths: [testDir],
    });
    
    await runtime.initialize();
    const stats = runtime.getStats();
    assertEqual(stats.skillCount, 2);
    assert(stats.initialized, 'Should be initialized');
  });

  await test('Runtime: find and execute skill', async () => {
    const runtime = new AgentSkillsRuntime({
      skillPaths: [testDir],
      maxExecutionTime: 5000,
    });
    
    await runtime.initialize();
    const matches = runtime.findSkills('generate code');
    assert(matches.length > 0, 'Should find skills');
    
    const result = await runtime.executeScript(
      matches[0].skill,
      'generate',
      { language: 'python', description: 'sort array' }
    );
    assert(result.success, `Script should execute: ${result.error}`);
  });

  // Cleanup
  cleanupTestSkills();

  // Summary
  console.log('');
  console.log('========================================');
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log('========================================');
  
  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(console.error);
