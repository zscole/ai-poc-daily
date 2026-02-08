/**
 * Agent Skills Runtime - Main Runtime Class
 * High-level API for skill discovery, matching, and execution
 */

import { SkillParser } from './parser.js';
import { SkillDiscovery } from './discovery.js';
import { SkillExecutor } from './executor.js';
import { SkillContextGenerator } from './context.js';
import type {
  Skill,
  SkillMatch,
  SkillRuntimeConfig,
  SkillExecutionResult,
  ValidationResult,
} from './types.js';

export interface AgentSkillsRuntimeOptions {
  skillPaths: string[];
  allowedRuntimes?: ('node' | 'python' | 'shell' | 'deno')[];
  maxExecutionTime?: number;
  sandboxed?: boolean;
  autoDiscover?: boolean;
  validateOnLoad?: boolean;
}

export class AgentSkillsRuntime {
  private parser: SkillParser;
  private discovery: SkillDiscovery;
  private executor: SkillExecutor;
  private contextGenerator: SkillContextGenerator;
  private config: AgentSkillsRuntimeOptions;
  private initialized = false;

  constructor(options: AgentSkillsRuntimeOptions) {
    this.config = {
      autoDiscover: true,
      validateOnLoad: true,
      ...options,
    };

    this.parser = new SkillParser();
    this.discovery = new SkillDiscovery();
    this.executor = new SkillExecutor({
      skillPaths: options.skillPaths,
      allowedRuntimes: options.allowedRuntimes,
      maxExecutionTime: options.maxExecutionTime,
      sandboxed: options.sandboxed,
    });
    this.contextGenerator = new SkillContextGenerator();
  }

  /**
   * Initialize the runtime and discover skills
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    if (this.config.autoDiscover) {
      await this.discovery.discover({
        paths: this.config.skillPaths,
        recursive: true,
        validateManifests: this.config.validateOnLoad,
      });
    }

    this.initialized = true;
  }

  /**
   * Find skills relevant to a query or task
   */
  findSkills(query: string): SkillMatch[] {
    return this.discovery.findSkills(query);
  }

  /**
   * Get a specific skill by ID
   */
  getSkill(id: string): Skill | undefined {
    return this.discovery.getSkill(id);
  }

  /**
   * Get all available skills
   */
  getAllSkills(): Skill[] {
    return this.discovery.getAllSkills();
  }

  /**
   * Load a skill from a path
   */
  loadSkill(skillPath: string): Skill | null {
    return this.parser.parseSkill(skillPath);
  }

  /**
   * Validate a skill
   */
  validateSkill(skill: Skill): ValidationResult {
    return this.parser.validateSkill(skill);
  }

  /**
   * Execute a skill script
   */
  async executeScript(
    skill: Skill,
    scriptName: string,
    args: Record<string, unknown> = {},
    workdir?: string
  ): Promise<SkillExecutionResult> {
    const script = skill.scripts.get(scriptName);
    if (!script) {
      return {
        success: false,
        error: `Script '${scriptName}' not found in skill '${skill.manifest.name}'`,
        duration: 0,
      };
    }

    // Validate arguments
    const validation = this.executor.validateArgs(script, args);
    if (!validation.valid) {
      return {
        success: false,
        error: `Invalid arguments: ${validation.errors.join(', ')}`,
        duration: 0,
      };
    }

    return this.executor.execute({
      skill,
      script,
      args,
      workdir: workdir || skill.path,
    });
  }

  /**
   * Generate LLM context for a skill
   */
  generateContext(skill: Skill): string {
    return this.contextGenerator.generateContext(skill);
  }

  /**
   * Generate context for matched skills
   */
  generateMatchedContext(query: string, maxSkills = 3): string {
    const matches = this.findSkills(query);
    return this.contextGenerator.generateMatchedContext(matches, maxSkills);
  }

  /**
   * Generate a skill catalog
   */
  generateCatalog(): string {
    return this.contextGenerator.generateCatalog(this.getAllSkills());
  }

  /**
   * Generate compact skill reference for system prompts
   */
  generateCompactReference(): string {
    return this.contextGenerator.generateCompactReference(this.getAllSkills());
  }

  /**
   * Load asset content from a skill
   */
  loadAsset(skill: Skill, assetName: string): string | null {
    return this.contextGenerator.loadAssetContent(skill, assetName);
  }

  /**
   * Refresh skill discovery
   */
  async refresh(): Promise<void> {
    this.discovery.clearCache();
    this.initialized = false;
    await this.initialize();
  }

  /**
   * Add a skill path and rediscover
   */
  async addSkillPath(path: string): Promise<void> {
    this.config.skillPaths.push(path);
    await this.refresh();
  }

  /**
   * Get runtime statistics
   */
  getStats(): {
    skillCount: number;
    skillPaths: string[];
    initialized: boolean;
  } {
    return {
      skillCount: this.getAllSkills().length,
      skillPaths: this.config.skillPaths,
      initialized: this.initialized,
    };
  }
}

// Export for convenience
export { SkillParser } from './parser.js';
export { SkillDiscovery } from './discovery.js';
export { SkillExecutor } from './executor.js';
export { SkillContextGenerator } from './context.js';
export * from './types.js';
