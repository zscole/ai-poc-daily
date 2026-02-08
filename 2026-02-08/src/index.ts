/**
 * Agent Skills Runtime
 * A TypeScript implementation of the agentskills.io specification
 * 
 * This library provides:
 * - Skill discovery from filesystem paths
 * - SKILL.md parsing with YAML frontmatter support
 * - Skill matching based on triggers, keywords, and semantic similarity
 * - Safe script execution with timeout and sandboxing
 * - LLM context generation for skill injection
 */

export { AgentSkillsRuntime, type AgentSkillsRuntimeOptions } from './runtime.js';
export { SkillParser } from './parser.js';
export { SkillDiscovery } from './discovery.js';
export { SkillExecutor } from './executor.js';
export { SkillContextGenerator } from './context.js';

export type {
  Skill,
  SkillManifest,
  SkillScript,
  SkillAsset,
  SkillTrigger,
  SkillDependency,
  ScriptArg,
  SkillMatch,
  SkillExecutionContext,
  SkillExecutionResult,
  SkillDiscoveryOptions,
  SkillRuntimeConfig,
  ValidationResult,
  ValidationError,
  ValidationWarning,
} from './types.js';
