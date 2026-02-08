/**
 * Agent Skills Runtime - Skill Parser
 * Parses skill folders and SKILL.md files
 */

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, basename, dirname } from 'node:path';
import { parse as parseYaml } from 'yaml';
import type { Skill, SkillManifest, SkillScript, SkillAsset, ValidationResult, ValidationError, ValidationWarning } from './types.js';

export class SkillParser {
  /**
   * Parse a skill from a directory path
   */
  parseSkill(skillPath: string): Skill | null {
    const skillMdPath = join(skillPath, 'SKILL.md');
    
    if (!existsSync(skillMdPath)) {
      console.error(`No SKILL.md found at ${skillPath}`);
      return null;
    }

    const content = readFileSync(skillMdPath, 'utf-8');
    const { frontmatter, instructions } = this.parseSkillMd(content);
    
    const manifest = this.parseManifest(frontmatter, skillPath);
    if (!manifest) {
      return null;
    }

    const scripts = this.discoverScripts(skillPath, manifest);
    const assets = this.discoverAssets(skillPath, manifest);

    return {
      id: this.generateSkillId(manifest.name),
      path: skillPath,
      manifest,
      instructions,
      scripts,
      assets,
    };
  }

  /**
   * Parse SKILL.md content into frontmatter and instructions
   */
  private parseSkillMd(content: string): { frontmatter: Record<string, unknown>; instructions: string } {
    const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
    
    if (frontmatterMatch) {
      try {
        const frontmatter = parseYaml(frontmatterMatch[1]) as Record<string, unknown>;
        return {
          frontmatter,
          instructions: frontmatterMatch[2].trim(),
        };
      } catch {
        // Fall through to no-frontmatter case
      }
    }

    // No frontmatter - extract metadata from markdown headers
    const nameMatch = content.match(/^#\s+(.+)$/m);
    const descMatch = content.match(/^>\s*(.+)$/m) || content.match(/\n\n([^#\n].+)\n/);
    
    return {
      frontmatter: {
        name: nameMatch?.[1] || 'Unknown Skill',
        description: descMatch?.[1] || '',
      },
      instructions: content,
    };
  }

  /**
   * Parse and validate manifest from frontmatter
   */
  private parseManifest(frontmatter: Record<string, unknown>, skillPath: string): SkillManifest | null {
    const name = (frontmatter.name as string) || basename(skillPath);
    const version = (frontmatter.version as string) || '1.0.0';
    const description = (frontmatter.description as string) || '';

    return {
      name,
      version,
      description,
      author: frontmatter.author as string | undefined,
      license: frontmatter.license as string | undefined,
      tags: frontmatter.tags as string[] | undefined,
      dependencies: frontmatter.dependencies as SkillManifest['dependencies'],
      triggers: frontmatter.triggers as SkillManifest['triggers'],
      scripts: frontmatter.scripts as SkillManifest['scripts'],
      assets: frontmatter.assets as SkillManifest['assets'],
    };
  }

  /**
   * Discover scripts in the skill directory
   */
  private discoverScripts(skillPath: string, manifest: SkillManifest): Map<string, SkillScript> {
    const scripts = new Map<string, SkillScript>();

    // Add scripts from manifest
    if (manifest.scripts) {
      for (const script of manifest.scripts) {
        scripts.set(script.name, script);
      }
    }

    // Auto-discover scripts in scripts/ directory
    const scriptsDir = join(skillPath, 'scripts');
    if (existsSync(scriptsDir) && statSync(scriptsDir).isDirectory()) {
      const files = readdirSync(scriptsDir);
      for (const file of files) {
        const name = file.replace(/\.[^.]+$/, '');
        if (!scripts.has(name)) {
          const runtime = this.inferRuntime(file);
          if (runtime) {
            scripts.set(name, {
              name,
              path: join('scripts', file),
              runtime,
            });
          }
        }
      }
    }

    return scripts;
  }

  /**
   * Discover assets in the skill directory
   */
  private discoverAssets(skillPath: string, manifest: SkillManifest): Map<string, SkillAsset> {
    const assets = new Map<string, SkillAsset>();

    // Add assets from manifest
    if (manifest.assets) {
      for (const asset of manifest.assets) {
        assets.set(asset.name, asset);
      }
    }

    // Auto-discover assets in assets/ and data/ directories
    for (const dir of ['assets', 'data', 'templates', 'references']) {
      const assetDir = join(skillPath, dir);
      if (existsSync(assetDir) && statSync(assetDir).isDirectory()) {
        const files = readdirSync(assetDir);
        for (const file of files) {
          const name = file;
          if (!assets.has(name)) {
            assets.set(name, {
              name,
              path: join(dir, file),
              type: dir === 'templates' ? 'template' : dir === 'references' ? 'reference' : 'data',
            });
          }
        }
      }
    }

    return assets;
  }

  /**
   * Infer script runtime from file extension
   */
  private inferRuntime(filename: string): SkillScript['runtime'] | null {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'js':
      case 'mjs':
      case 'cjs':
        return 'node';
      case 'ts':
      case 'mts':
        return 'deno';
      case 'py':
        return 'python';
      case 'sh':
      case 'bash':
        return 'shell';
      default:
        return null;
    }
  }

  /**
   * Generate a URL-safe skill ID
   */
  private generateSkillId(name: string): string {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
  }

  /**
   * Validate a skill
   */
  validateSkill(skill: Skill): ValidationResult {
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];

    // Required fields
    if (!skill.manifest.name) {
      errors.push({ field: 'name', message: 'Skill name is required', severity: 'error' });
    }
    if (!skill.manifest.description) {
      warnings.push({ field: 'description', message: 'Skill description is recommended', severity: 'warning' });
    }
    if (!skill.instructions || skill.instructions.length < 50) {
      warnings.push({ field: 'instructions', message: 'Skill instructions should be detailed (50+ chars)', severity: 'warning' });
    }

    // Validate scripts exist
    for (const [name, script] of skill.scripts) {
      const scriptPath = join(skill.path, script.path);
      if (!existsSync(scriptPath)) {
        errors.push({ field: `scripts.${name}`, message: `Script file not found: ${script.path}`, severity: 'error' });
      }
    }

    // Validate assets exist
    for (const [name, asset] of skill.assets) {
      const assetPath = join(skill.path, asset.path);
      if (!existsSync(assetPath)) {
        warnings.push({ field: `assets.${name}`, message: `Asset file not found: ${asset.path}`, severity: 'warning' });
      }
    }

    // Validate triggers
    if (skill.manifest.triggers) {
      for (const trigger of skill.manifest.triggers) {
        if (!trigger.type || !trigger.value) {
          errors.push({ field: 'triggers', message: 'Trigger must have type and value', severity: 'error' });
        }
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }
}
