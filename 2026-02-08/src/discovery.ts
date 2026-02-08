/**
 * Agent Skills Runtime - Skill Discovery
 * Discovers and indexes skills from configured paths
 */

import { readdirSync, statSync, existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { SkillParser } from './parser.js';
import type { Skill, SkillMatch, SkillDiscoveryOptions, SkillTrigger } from './types.js';

export class SkillDiscovery {
  private parser: SkillParser;
  private skillCache: Map<string, Skill> = new Map();
  private triggerIndex: Map<string, { skill: Skill; trigger: SkillTrigger }[]> = new Map();

  constructor() {
    this.parser = new SkillParser();
  }

  /**
   * Discover skills from configured paths
   */
  async discover(options: SkillDiscoveryOptions): Promise<Skill[]> {
    const skills: Skill[] = [];

    for (const basePath of options.paths) {
      const resolvedPath = resolve(basePath);
      if (!existsSync(resolvedPath)) {
        console.warn(`Skill path does not exist: ${resolvedPath}`);
        continue;
      }

      const discovered = options.recursive
        ? this.discoverRecursive(resolvedPath)
        : this.discoverFlat(resolvedPath);

      for (const skill of discovered) {
        if (options.validateManifests) {
          const validation = this.parser.validateSkill(skill);
          if (!validation.valid) {
            console.warn(`Skill validation failed for ${skill.manifest.name}:`, validation.errors);
            continue;
          }
        }
        skills.push(skill);
        this.cacheSkill(skill);
      }
    }

    return skills;
  }

  /**
   * Discover skills in immediate subdirectories
   */
  private discoverFlat(basePath: string): Skill[] {
    const skills: Skill[] = [];
    
    // Check if basePath itself is a skill
    if (existsSync(join(basePath, 'SKILL.md'))) {
      const skill = this.parser.parseSkill(basePath);
      if (skill) skills.push(skill);
      return skills;
    }

    // Check subdirectories
    const entries = readdirSync(basePath);
    for (const entry of entries) {
      const entryPath = join(basePath, entry);
      if (statSync(entryPath).isDirectory()) {
        if (existsSync(join(entryPath, 'SKILL.md'))) {
          const skill = this.parser.parseSkill(entryPath);
          if (skill) skills.push(skill);
        }
      }
    }

    return skills;
  }

  /**
   * Recursively discover skills
   */
  private discoverRecursive(basePath: string, maxDepth = 5, currentDepth = 0): Skill[] {
    if (currentDepth > maxDepth) return [];
    
    const skills: Skill[] = [];

    // Check if current path is a skill
    if (existsSync(join(basePath, 'SKILL.md'))) {
      const skill = this.parser.parseSkill(basePath);
      if (skill) skills.push(skill);
      // Don't recurse into skill directories
      return skills;
    }

    // Recurse into subdirectories
    try {
      const entries = readdirSync(basePath);
      for (const entry of entries) {
        if (entry.startsWith('.')) continue; // Skip hidden
        const entryPath = join(basePath, entry);
        try {
          if (statSync(entryPath).isDirectory()) {
            skills.push(...this.discoverRecursive(entryPath, maxDepth, currentDepth + 1));
          }
        } catch {
          // Skip inaccessible entries
        }
      }
    } catch {
      // Skip inaccessible directories
    }

    return skills;
  }

  /**
   * Cache a skill and index its triggers
   */
  private cacheSkill(skill: Skill): void {
    this.skillCache.set(skill.id, skill);

    // Index triggers
    if (skill.manifest.triggers) {
      for (const trigger of skill.manifest.triggers) {
        const key = this.normalizeTriggerKey(trigger);
        const existing = this.triggerIndex.get(key) || [];
        existing.push({ skill, trigger });
        this.triggerIndex.set(key, existing);
      }
    }

    // Index by name and tags as implicit triggers
    this.indexImplicitTrigger(skill, skill.manifest.name);
    if (skill.manifest.tags) {
      for (const tag of skill.manifest.tags) {
        this.indexImplicitTrigger(skill, tag);
      }
    }
  }

  private indexImplicitTrigger(skill: Skill, keyword: string): void {
    const key = keyword.toLowerCase();
    const existing = this.triggerIndex.get(key) || [];
    existing.push({
      skill,
      trigger: { type: 'keyword', value: keyword },
    });
    this.triggerIndex.set(key, existing);
  }

  private normalizeTriggerKey(trigger: SkillTrigger): string {
    return trigger.value.toLowerCase();
  }

  /**
   * Find skills matching a query
   */
  findSkills(query: string): SkillMatch[] {
    const matches: SkillMatch[] = [];
    const queryLower = query.toLowerCase();
    const queryWords = queryLower.split(/\s+/);

    // Check trigger index
    for (const [key, entries] of this.triggerIndex) {
      let score = 0;
      
      // Exact match
      if (queryLower.includes(key)) {
        score = 1.0;
      }
      // Word match
      else if (queryWords.some(word => key.includes(word) || word.includes(key))) {
        score = 0.7;
      }
      // Partial match
      else if (this.fuzzyMatch(queryLower, key)) {
        score = 0.4;
      }

      if (score > 0) {
        for (const { skill, trigger } of entries) {
          // Boost score for explicit triggers
          const boostedScore = trigger.type === 'keyword' ? score * 0.9 : score;
          matches.push({ skill, trigger, score: boostedScore });
        }
      }
    }

    // Check skill descriptions
    for (const skill of this.skillCache.values()) {
      const descLower = skill.manifest.description.toLowerCase();
      if (queryWords.some(word => descLower.includes(word))) {
        const existingMatch = matches.find(m => m.skill.id === skill.id);
        if (!existingMatch) {
          matches.push({
            skill,
            trigger: { type: 'intent', value: query },
            score: 0.3,
          });
        }
      }
    }

    // Deduplicate and sort by score
    const seen = new Set<string>();
    return matches
      .filter(m => {
        if (seen.has(m.skill.id)) return false;
        seen.add(m.skill.id);
        return true;
      })
      .sort((a, b) => b.score - a.score);
  }

  /**
   * Simple fuzzy matching
   */
  private fuzzyMatch(query: string, target: string): boolean {
    let qi = 0;
    for (let ti = 0; ti < target.length && qi < query.length; ti++) {
      if (query[qi] === target[ti]) qi++;
    }
    return qi >= query.length * 0.6;
  }

  /**
   * Get a skill by ID
   */
  getSkill(id: string): Skill | undefined {
    return this.skillCache.get(id);
  }

  /**
   * Get all cached skills
   */
  getAllSkills(): Skill[] {
    return Array.from(this.skillCache.values());
  }

  /**
   * Clear the cache
   */
  clearCache(): void {
    this.skillCache.clear();
    this.triggerIndex.clear();
  }
}
