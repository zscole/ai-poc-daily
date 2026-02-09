# Agent Team Orchestrator

A proof-of-concept demonstrating the parallel agent orchestration pattern from Anthropic's Claude C Compiler (CCC) project.

## Background

On February 8, 2026, Anthropic published a detailed engineering blog post about building a C compiler entirely with Claude. The most remarkable aspect: 16 Claude agents worked in parallel on a shared codebase, producing a 100,000-line Rust compiler capable of building the Linux kernel.

Key stats from CCC:
- 2,000 Claude Code sessions over 2 weeks
- 2 billion input tokens, 140 million output tokens
- $20,000 in API costs
- Zero compiler-specific dependencies
- 99% pass rate on GCC torture test suite

This POC extracts and demonstrates the core orchestration patterns that made this possible.

## What This Demonstrates

### 1. File-Based Task Locking

Agents claim tasks by creating lock files atomically. This prevents two agents from working on the same task:

```
current_tasks/
  parse_if_statement.lock    <- Agent-01 owns this
  codegen_function.lock      <- Agent-03 owns this
```

Uses `os.O_CREAT | os.O_EXCL` for atomic file creation - if the file exists, the claim fails.

### 2. Parallel Agent Execution

Multiple agents run simultaneously, each:
- Claiming available tasks
- Working in isolated workspaces
- Pushing results to shared upstream
- Handling merge conflicts

### 3. Git-Based Synchronization

Each agent has its own workspace cloned from a shared upstream:

```
upstream.git/           <- Bare repo (shared state)
workspaces/
  agent-00/            <- Clone for agent 0
  agent-01/            <- Clone for agent 1
  ...
```

Agents pull before starting, push after completing, and handle merge conflicts.

### 4. Role Specialization

Different agents can have different roles:
- WORKER: General task execution
- REVIEWER: Code review and quality
- OPTIMIZER: Performance improvements
- DOCUMENTER: Documentation maintenance

### 5. Progress Tracking

The registry tracks all task states and provides real-time statistics.

## Running the Demo

```bash
python3 agent_team.py
```

Sample output:
```
============================================================
Agent Team Orchestrator Demo
Based on patterns from Anthropic's Claude C Compiler (CCC)
============================================================

[2026-02-09T03:03:05] Added task: task-a82b589c - Implement lexer for C tokens...
[2026-02-09T03:03:05] Starting orchestrator with 4 agents
[2026-02-09T03:03:05] Agent agent-02 starting task task-a82b589c: Implement lexer...
[2026-02-09T03:03:05] Agent agent-01 starting task task-3ea23626: Parse for/while...
...
[2026-02-09T03:03:08] Orchestration complete in 3.00s

Results Summary
============================================================
Duration: 3.00 seconds
Tasks completed: 15
Tasks failed: 1
Throughput: 5.01 tasks/second
```

## Architecture

```
AgentTeamOrchestrator
    |
    +-- TaskRegistry (file-based task management)
    |       tasks/           - Task definitions
    |       current_tasks/   - Lock files
    |       completed_tasks/ - Finished tasks
    |
    +-- WorkspaceManager (git synchronization)
    |       upstream.git/    - Shared bare repo
    |       workspaces/      - Per-agent clones
    |
    +-- Agent[] (parallel workers)
            Each runs agent_loop():
            1. Get available tasks
            2. Claim task (atomic lock)
            3. Sync workspace (git pull)
            4. Execute task
            5. Push changes (git push)
            6. Release lock
            7. Repeat
```

## Key Patterns from CCC

### Test-Driven Development for Agents

From the Anthropic blog:

> "Claude will work autonomously to solve whatever problem I give it. So it's important that the task verifier is nearly perfect, otherwise Claude will solve the wrong problem."

The test harness guides agent behavior - agents optimize for passing tests.

### Context Window Management

> "The test harness should not print thousands of useless bytes. At most, it should print a few lines of output and log all important information to a file."

Agents have limited context - design feedback to be concise.

### Parallelism via Task Independence

When tasks are independent, parallelization is trivial. When they're interdependent (like compiling the Linux kernel), you need:
- Oracle comparison (use known-good compiler to isolate failures)
- Delta debugging (binary search for problematic files)
- Automatic conflict resolution

## Production Considerations

This is a simplified demonstration. A production system would need:

1. **LLM Integration**: Replace `demo_task_handler` with actual LLM agent calls
2. **Persistent State**: Use a database instead of file-based registry
3. **Distributed Execution**: Run agents across multiple machines/containers
4. **Better Conflict Resolution**: Automatic merge conflict handling
5. **Cost Management**: Token budgets, rate limiting, model selection
6. **Observability**: Detailed logging, metrics, tracing

## Why This Matters

The CCC project demonstrates that:

1. AI can build complex systems (100k lines, compiles Linux kernel)
2. Parallel agent teams dramatically expand capability
3. The orchestration layer is critical - agents need structure
4. Test-driven development works for agents too
5. $20k in API costs can replace months of human work

This pattern will become standard for large-scale AI development projects.

## References

- [Building a C compiler with a team of parallel Claudes](https://anthropic.com/engineering/building-c-compiler) - Anthropic Engineering Blog
- [Claude's C Compiler Source](https://github.com/anthropics/claudes-c-compiler) - GitHub
- [CCC vs GCC Benchmark](https://harshanu.space/en/tech/ccc-vs-gcc/) - Independent Analysis

## License

MIT
