/**
 * Agent Skills Runtime - Type Definitions
 * Following the agentskills.io specification
 */

export interface SkillManifest {
  name: string;
  version: string;
  description: string;
  author?: string;
  license?: string;
  tags?: string[];
  dependencies?: SkillDependency[];
  triggers?: SkillTrigger[];
  scripts?: SkillScript[];
  assets?: SkillAsset[];
}

export interface SkillDependency {
  name: string;
  version?: string;
  optional?: boolean;
}

export interface SkillTrigger {
  type: 'keyword' | 'pattern' | 'intent';
  value: string;
  description?: string;
}

export interface SkillScript {
  name: string;
  path: string;
  runtime: 'node' | 'python' | 'shell' | 'deno';
  description?: string;
  args?: ScriptArg[];
}

export interface ScriptArg {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'file';
  required?: boolean;
  description?: string;
  default?: string | number | boolean;
}

export interface SkillAsset {
  name: string;
  path: string;
  type: 'data' | 'template' | 'reference';
  description?: string;
}

export interface Skill {
  id: string;
  path: string;
  manifest: SkillManifest;
  instructions: string;
  scripts: Map<string, SkillScript>;
  assets: Map<string, SkillAsset>;
}

export interface SkillMatch {
  skill: Skill;
  trigger: SkillTrigger;
  score: number;
}

export interface SkillExecutionContext {
  skill: Skill;
  script: SkillScript;
  args: Record<string, unknown>;
  workdir: string;
  env?: Record<string, string>;
}

export interface SkillExecutionResult {
  success: boolean;
  output?: string;
  error?: string;
  exitCode?: number;
  duration: number;
}

export interface SkillDiscoveryOptions {
  paths: string[];
  recursive?: boolean;
  validateManifests?: boolean;
}

export interface SkillRuntimeConfig {
  skillPaths: string[];
  allowedRuntimes: ('node' | 'python' | 'shell' | 'deno')[];
  maxExecutionTime: number;
  sandboxed: boolean;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  field: string;
  message: string;
  severity: 'error';
}

export interface ValidationWarning {
  field: string;
  message: string;
  severity: 'warning';
}
