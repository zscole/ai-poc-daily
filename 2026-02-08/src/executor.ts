/**
 * Agent Skills Runtime - Script Executor
 * Safely executes skill scripts in isolated environments
 */

import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import type { Skill, SkillScript, SkillExecutionContext, SkillExecutionResult, SkillRuntimeConfig } from './types.js';

export class SkillExecutor {
  private config: SkillRuntimeConfig;

  constructor(config: Partial<SkillRuntimeConfig> = {}) {
    this.config = {
      skillPaths: config.skillPaths || [],
      allowedRuntimes: config.allowedRuntimes || ['node', 'python', 'shell'],
      maxExecutionTime: config.maxExecutionTime || 30000,
      sandboxed: config.sandboxed ?? true,
    };
  }

  /**
   * Execute a skill script
   */
  async execute(context: SkillExecutionContext): Promise<SkillExecutionResult> {
    const startTime = Date.now();

    // Validate runtime is allowed
    if (!this.config.allowedRuntimes.includes(context.script.runtime)) {
      return {
        success: false,
        error: `Runtime '${context.script.runtime}' is not allowed`,
        duration: Date.now() - startTime,
      };
    }

    // Resolve script path
    const scriptPath = join(context.skill.path, context.script.path);
    if (!existsSync(scriptPath)) {
      return {
        success: false,
        error: `Script not found: ${scriptPath}`,
        duration: Date.now() - startTime,
      };
    }

    // Build command
    const { command, args } = this.buildCommand(context.script, scriptPath, context.args);

    // Execute
    try {
      const result = await this.spawnProcess(command, args, {
        cwd: context.workdir || context.skill.path,
        env: { ...process.env, ...context.env },
        timeout: this.config.maxExecutionTime,
      });

      return {
        ...result,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - startTime,
      };
    }
  }

  /**
   * Build the command and arguments for a script
   */
  private buildCommand(
    script: SkillScript,
    scriptPath: string,
    args: Record<string, unknown>
  ): { command: string; args: string[] } {
    const scriptArgs = this.buildScriptArgs(script, args);

    switch (script.runtime) {
      case 'node':
        return { command: 'node', args: [scriptPath, ...scriptArgs] };
      case 'python':
        return { command: 'python3', args: [scriptPath, ...scriptArgs] };
      case 'shell':
        return { command: 'bash', args: [scriptPath, ...scriptArgs] };
      case 'deno':
        return {
          command: 'deno',
          args: ['run', '--allow-read', '--allow-write', '--allow-net', scriptPath, ...scriptArgs],
        };
      default:
        throw new Error(`Unknown runtime: ${script.runtime}`);
    }
  }

  /**
   * Build script arguments from provided args
   */
  private buildScriptArgs(script: SkillScript, args: Record<string, unknown>): string[] {
    const result: string[] = [];

    // If script has defined args, use those
    if (script.args) {
      for (const arg of script.args) {
        const value = args[arg.name] ?? arg.default;
        if (value !== undefined) {
          result.push(`--${arg.name}`, String(value));
        } else if (arg.required) {
          throw new Error(`Required argument missing: ${arg.name}`);
        }
      }
    } else {
      // Pass all args as --key value pairs
      for (const [key, value] of Object.entries(args)) {
        if (value !== undefined && value !== null) {
          result.push(`--${key}`, String(value));
        }
      }
    }

    return result;
  }

  /**
   * Spawn a process and capture output
   */
  private spawnProcess(
    command: string,
    args: string[],
    options: { cwd: string; env: NodeJS.ProcessEnv; timeout: number }
  ): Promise<Omit<SkillExecutionResult, 'duration'>> {
    return new Promise((resolve) => {
      const proc = spawn(command, args, {
        cwd: options.cwd,
        env: options.env,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      let stdout = '';
      let stderr = '';

      proc.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      proc.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      const timeout = setTimeout(() => {
        proc.kill('SIGTERM');
        resolve({
          success: false,
          error: `Execution timed out after ${options.timeout}ms`,
          exitCode: -1,
        });
      }, options.timeout);

      proc.on('close', (code) => {
        clearTimeout(timeout);
        resolve({
          success: code === 0,
          output: stdout,
          error: stderr || undefined,
          exitCode: code ?? undefined,
        });
      });

      proc.on('error', (err) => {
        clearTimeout(timeout);
        resolve({
          success: false,
          error: err.message,
        });
      });
    });
  }

  /**
   * Validate script arguments before execution
   */
  validateArgs(script: SkillScript, args: Record<string, unknown>): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (!script.args) {
      return { valid: true, errors: [] };
    }

    for (const arg of script.args) {
      const value = args[arg.name];

      if (arg.required && value === undefined && arg.default === undefined) {
        errors.push(`Missing required argument: ${arg.name}`);
        continue;
      }

      if (value !== undefined) {
        switch (arg.type) {
          case 'number':
            if (typeof value !== 'number' && isNaN(Number(value))) {
              errors.push(`Argument '${arg.name}' must be a number`);
            }
            break;
          case 'boolean':
            if (typeof value !== 'boolean' && !['true', 'false', '0', '1'].includes(String(value))) {
              errors.push(`Argument '${arg.name}' must be a boolean`);
            }
            break;
          case 'file':
            if (typeof value === 'string' && !existsSync(value)) {
              errors.push(`File not found for argument '${arg.name}': ${value}`);
            }
            break;
        }
      }
    }

    return { valid: errors.length === 0, errors };
  }
}
