/**
 * Agent Skills Runtime - Context Generator
 * Generates LLM-ready context from skills
 */

import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import type { Skill, SkillMatch } from './types.js';

export class SkillContextGenerator {
  /**
   * Generate a context string for an LLM from a skill
   */
  generateContext(skill: Skill): string {
    const sections: string[] = [];

    // Header
    sections.push(`# Skill: ${skill.manifest.name}`);
    sections.push('');

    // Metadata
    if (skill.manifest.description) {
      sections.push(`> ${skill.manifest.description}`);
      sections.push('');
    }

    // Instructions
    sections.push('## Instructions');
    sections.push('');
    sections.push(skill.instructions);
    sections.push('');

    // Available scripts
    if (skill.scripts.size > 0) {
      sections.push('## Available Scripts');
      sections.push('');
      for (const [name, script] of skill.scripts) {
        sections.push(`### ${name}`);
        sections.push(`- Runtime: ${script.runtime}`);
        sections.push(`- Path: ${script.path}`);
        if (script.description) {
          sections.push(`- Description: ${script.description}`);
        }
        if (script.args && script.args.length > 0) {
          sections.push('- Arguments:');
          for (const arg of script.args) {
            const required = arg.required ? ' (required)' : '';
            const defaultVal = arg.default !== undefined ? ` [default: ${arg.default}]` : '';
            sections.push(`  - \`${arg.name}\` (${arg.type})${required}${defaultVal}: ${arg.description || ''}`);
          }
        }
        sections.push('');
      }
    }

    // Available assets
    if (skill.assets.size > 0) {
      sections.push('## Available Assets');
      sections.push('');
      for (const [name, asset] of skill.assets) {
        sections.push(`- **${name}** (${asset.type}): ${asset.path}`);
        if (asset.description) {
          sections.push(`  ${asset.description}`);
        }
      }
      sections.push('');
    }

    // Tags
    if (skill.manifest.tags && skill.manifest.tags.length > 0) {
      sections.push(`Tags: ${skill.manifest.tags.join(', ')}`);
      sections.push('');
    }

    return sections.join('\n');
  }

  /**
   * Generate context for multiple matched skills
   */
  generateMatchedContext(matches: SkillMatch[], maxSkills = 3): string {
    const sections: string[] = [];
    
    sections.push('# Available Skills for This Task');
    sections.push('');
    sections.push('The following skills have been identified as potentially relevant:');
    sections.push('');

    const topMatches = matches.slice(0, maxSkills);
    
    for (let i = 0; i < topMatches.length; i++) {
      const { skill, score } = topMatches[i];
      sections.push(`## ${i + 1}. ${skill.manifest.name} (relevance: ${Math.round(score * 100)}%)`);
      sections.push('');
      sections.push(skill.manifest.description || 'No description available');
      sections.push('');
      
      // Brief summary of capabilities
      if (skill.scripts.size > 0) {
        sections.push(`Scripts: ${Array.from(skill.scripts.keys()).join(', ')}`);
      }
      if (skill.manifest.tags && skill.manifest.tags.length > 0) {
        sections.push(`Tags: ${skill.manifest.tags.join(', ')}`);
      }
      sections.push('');
    }

    sections.push('---');
    sections.push('');
    sections.push('To use a skill, reference it by name and follow its instructions.');
    
    return sections.join('\n');
  }

  /**
   * Load and include asset content in context
   */
  loadAssetContent(skill: Skill, assetName: string): string | null {
    const asset = skill.assets.get(assetName);
    if (!asset) return null;

    const assetPath = join(skill.path, asset.path);
    if (!existsSync(assetPath)) return null;

    try {
      return readFileSync(assetPath, 'utf-8');
    } catch {
      return null;
    }
  }

  /**
   * Generate a skill catalog summary
   */
  generateCatalog(skills: Skill[]): string {
    const sections: string[] = [];
    
    sections.push('# Agent Skills Catalog');
    sections.push('');
    sections.push(`Total skills available: ${skills.length}`);
    sections.push('');

    // Group by tags
    const tagGroups = new Map<string, Skill[]>();
    const untagged: Skill[] = [];

    for (const skill of skills) {
      if (skill.manifest.tags && skill.manifest.tags.length > 0) {
        for (const tag of skill.manifest.tags) {
          const existing = tagGroups.get(tag) || [];
          existing.push(skill);
          tagGroups.set(tag, existing);
        }
      } else {
        untagged.push(skill);
      }
    }

    // Output by category
    for (const [tag, tagSkills] of Array.from(tagGroups.entries()).sort()) {
      sections.push(`## ${tag}`);
      sections.push('');
      for (const skill of tagSkills) {
        sections.push(`- **${skill.manifest.name}**: ${skill.manifest.description || 'No description'}`);
      }
      sections.push('');
    }

    if (untagged.length > 0) {
      sections.push('## Other');
      sections.push('');
      for (const skill of untagged) {
        sections.push(`- **${skill.manifest.name}**: ${skill.manifest.description || 'No description'}`);
      }
      sections.push('');
    }

    return sections.join('\n');
  }

  /**
   * Generate a compact skill reference for system prompts
   */
  generateCompactReference(skills: Skill[]): string {
    const lines: string[] = [];
    
    lines.push('<available_skills>');
    
    for (const skill of skills) {
      lines.push(`  <skill name="${skill.manifest.name}">`);
      lines.push(`    <description>${skill.manifest.description || ''}</description>`);
      if (skill.manifest.triggers && skill.manifest.triggers.length > 0) {
        const triggers = skill.manifest.triggers.map(t => t.value).join(', ');
        lines.push(`    <triggers>${triggers}</triggers>`);
      }
      lines.push(`    <path>${skill.path}</path>`);
      lines.push('  </skill>');
    }
    
    lines.push('</available_skills>');
    
    return lines.join('\n');
  }
}
