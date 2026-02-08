/**
 * Agent Skills Runtime - Demo
 * Demonstrates the full capabilities of the runtime
 */

import { AgentSkillsRuntime } from './runtime.js';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const exampleSkillsPath = join(__dirname, '..', 'example-skills');

async function main() {
  console.log('========================================');
  console.log('Agent Skills Runtime - Demo');
  console.log('========================================');
  console.log('');

  // Initialize runtime
  console.log('[1] Initializing runtime...');
  const runtime = new AgentSkillsRuntime({
    skillPaths: [exampleSkillsPath],
    allowedRuntimes: ['node', 'python', 'shell'],
    maxExecutionTime: 10000,
    sandboxed: true,
  });

  await runtime.initialize();
  
  const stats = runtime.getStats();
  console.log(`    Found ${stats.skillCount} skills in ${stats.skillPaths.length} paths`);
  console.log('');

  // List all skills
  console.log('[2] Discovered skills:');
  const skills = runtime.getAllSkills();
  for (const skill of skills) {
    console.log(`    - ${skill.manifest.name}: ${skill.manifest.description || 'No description'}`);
    if (skill.scripts.size > 0) {
      console.log(`      Scripts: ${Array.from(skill.scripts.keys()).join(', ')}`);
    }
    if (skill.manifest.tags) {
      console.log(`      Tags: ${skill.manifest.tags.join(', ')}`);
    }
  }
  console.log('');

  // Find skills for a query
  console.log('[3] Finding skills for query "generate code"...');
  const matches = runtime.findSkills('generate code');
  if (matches.length > 0) {
    for (const match of matches.slice(0, 3)) {
      console.log(`    - ${match.skill.manifest.name} (score: ${Math.round(match.score * 100)}%)`);
    }
  } else {
    console.log('    No matching skills found');
  }
  console.log('');

  // Generate context for a skill
  if (skills.length > 0) {
    console.log('[4] Generating LLM context for first skill...');
    const context = runtime.generateContext(skills[0]);
    console.log('--- Context Preview (first 500 chars) ---');
    console.log(context.slice(0, 500));
    console.log('...');
    console.log('--- End Preview ---');
    console.log('');
  }

  // Generate catalog
  console.log('[5] Generating skill catalog...');
  const catalog = runtime.generateCatalog();
  console.log('--- Catalog Preview ---');
  console.log(catalog.slice(0, 600));
  console.log('...');
  console.log('--- End Preview ---');
  console.log('');

  // Generate compact reference
  console.log('[6] Generating compact reference for system prompt...');
  const compact = runtime.generateCompactReference();
  console.log(compact);
  console.log('');

  // Execute a script if available
  const codeGenSkill = skills.find(s => s.scripts.has('generate'));
  if (codeGenSkill) {
    console.log('[7] Executing code generator script...');
    const result = await runtime.executeScript(codeGenSkill, 'generate', {
      language: 'typescript',
      description: 'hello world function',
    });
    
    if (result.success) {
      console.log('    Execution successful!');
      console.log('    Output:', result.output?.slice(0, 200) || '(no output)');
    } else {
      console.log('    Execution failed:', result.error);
    }
    console.log(`    Duration: ${result.duration}ms`);
  }

  console.log('');
  console.log('========================================');
  console.log('Demo complete!');
  console.log('========================================');
}

main().catch(console.error);
