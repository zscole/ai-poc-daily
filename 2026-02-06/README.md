# Multi-Agent Orchestration Framework

A lightweight Python framework for coordinating multiple LLM agents working in parallel on shared tasks. Inspired by Claude Code's Agent Teams feature and the C compiler case study where 16 parallel Claude instances built a 100,000-line compiler.

**Date:** 2026-02-06

## Why This Matters

Yesterday (2026-02-05) saw two major releases that signal a clear direction for AI development:

1. **Claude Opus 4.6** introduced Agent Teams - the ability to orchestrate multiple Claude Code instances working together on shared codebases
2. **GPT-5.3-Codex** added interactive steering and demonstrated self-improvement capabilities (the model helped train and deploy itself)

The common thread: **multi-agent coordination is the next frontier**. Single-agent systems hit context limits, lose coherence on long tasks, and can't explore multiple approaches simultaneously. Agent teams solve this by:

- Parallelizing independent work across multiple context windows
- Enabling specialization (architecture, security, performance agents)
- Reducing "context rot" by keeping each agent focused
- Allowing agents to challenge and build on each other's work

This POC provides a production-ready foundation for building multi-agent systems.

## Key Features

- **Task Queue with File-Based Locking**: Prevents race conditions when multiple agents try to claim the same task
- **Parallel Execution**: Agents run concurrently via asyncio
- **Role Specialization**: Lead, Worker, Reviewer, and Specialist roles with different behaviors
- **Message Bus**: Agents can communicate and coordinate directly
- **Dependency Management**: Tasks can depend on other tasks
- **Pluggable LLM Backend**: Works with any LLM API (Anthropic, OpenAI, local models)

## Architecture

```
                    +------------------+
                    |    AgentTeam     |
                    |  (Orchestrator)  |
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
   +----v----+         +-----v-----+        +-----v-----+
   |  Lead   |         |  Worker   |        | Reviewer  |
   |  Agent  |         |  Agents   |        |  Agents   |
   +---------+         +-----------+        +-----------+
        |                    |                    |
        +--------------------+--------------------+
                             |
              +--------------+--------------+
              |                             |
       +------v------+              +-------v-------+
       |  TaskQueue  |              |  MessageBus   |
       | (file-lock) |              | (broadcast)   |
       +-------------+              +---------------+
```

## Installation

```bash
# No external dependencies for core framework
python agent_team.py

# For Anthropic API example
pip install anthropic
export ANTHROPIC_API_KEY=your_key
python example_anthropic.py
```

## Quick Start

```python
import asyncio
from pathlib import Path
from agent_team import AgentTeam, AgentRole

async def main():
    # Create workspace for shared state
    team = AgentTeam(Path("/tmp/my_team"))
    
    # Add agents with roles
    team.add_agent("lead-1", AgentRole.LEAD, my_llm_function)
    team.add_agent("worker-1", AgentRole.WORKER, my_llm_function)
    team.add_agent("worker-2", AgentRole.WORKER, my_llm_function)
    
    # Add tasks (with optional dependencies)
    t1 = team.add_task("Research the problem space")
    t2 = team.add_task("Implement solution A")
    t3 = team.add_task("Implement solution B")
    t4 = team.add_task("Compare and synthesize", dependencies=[t2.id, t3.id])
    
    # Run until completion
    status = await team.run(timeout=300)
    print(f"Completed: {status['completed']}/{status['total']}")

asyncio.run(main())
```

## Task Locking Mechanism

The key insight from the C compiler case study: when multiple agents work on a shared codebase, you need synchronization. This framework uses file-based locks:

```python
# Agent tries to claim a task
lock_path = locks_dir / f"{task_id}.lock"
try:
    # O_EXCL ensures atomic creation
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, agent_id.encode())
    os.close(fd)
    return True  # Lock acquired
except FileExistsError:
    return False  # Another agent got it first
```

This approach:
- Works across processes (not just threads)
- Survives agent crashes (locks can be cleaned up)
- Requires no external coordination service
- Matches the "git push" pattern used in the C compiler project

## Creating Specialized Agents

Subclass `Agent` to create domain-specific behavior:

```python
class SecurityAgent(Agent):
    async def execute_task(self, task: Task) -> str:
        prompt = f"""You are a security engineer.
        
Task: {task.description}

Focus on:
- Input validation vulnerabilities
- Authentication issues
- Data exposure risks

Identify specific issues with severity ratings."""
        
        return await asyncio.to_thread(self.llm_fn, prompt)
```

## Best Practices from the C Compiler Case Study

1. **Write high-quality tests**: Agents will optimize for whatever metric you give them. Make sure your verifiers are nearly perfect.

2. **Design for LLM constraints**:
   - Keep output concise to avoid context pollution
   - Include progress indicators
   - Use grep-friendly log formats (ERROR on same line as reason)
   - Pre-compute summary statistics

3. **Make parallelism easy**:
   - Break large tasks into independent subtasks
   - Use oracle comparisons to isolate failures
   - Each agent should be able to make progress independently

4. **Specialize agents**:
   - One agent for code quality
   - One agent for performance
   - One agent for documentation
   - Let the lead synthesize

## API Reference

### AgentTeam

```python
team = AgentTeam(workspace: Path)
team.add_agent(agent_id: str, role: AgentRole, llm_fn: Callable) -> Agent
team.add_task(description: str, dependencies: list[str] = None) -> Task
await team.run(timeout: float = None) -> dict
team.get_status() -> dict
```

### Agent

```python
agent = Agent(agent_id, role, task_queue, message_bus, llm_fn)
await agent.execute_task(task: Task) -> str  # Override this
await agent.run_loop(max_iterations: int = None)
agent.stop()
```

### TaskQueue

```python
queue = TaskQueue(workspace: Path)
queue.add_task(description: str, dependencies: list[str] = None) -> Task
queue.claim_task(agent_id: str) -> Optional[Task]
queue.complete_task(task_id: str, agent_id: str, result: str)
queue.fail_task(task_id: str, agent_id: str, error: str)
queue.get_status() -> dict
```

### MessageBus

```python
bus = MessageBus(workspace: Path)
bus.send(message: AgentMessage)
bus.receive(agent_id: str, since: float = 0) -> list[AgentMessage]
```

## Comparison with Claude Code Agent Teams

| Feature | This POC | Claude Code Agent Teams |
|---------|----------|------------------------|
| Parallel execution | Yes | Yes |
| Task locking | File-based | Git-based |
| Agent communication | MessageBus | Direct messaging |
| Split-pane display | No | Yes (tmux/iTerm2) |
| Lead delegation mode | Manual | Built-in |
| Plan approval workflow | Manual | Built-in |

This POC provides the core orchestration primitives. Claude Code Agent Teams adds UX features (split panes, interactive steering) on top of similar foundations.

## Connection to Today's News

**Claude Opus 4.6** (1908 HN points):
- 1M token context window
- Agent Teams feature
- Context compaction for longer tasks
- 76% on 8-needle MRCR v2 (vs 18.5% for previous models)

**GPT-5.3-Codex** (1281 HN points):
- First model that helped create itself
- State-of-the-art on SWE-Bench Pro
- Interactive steering while working
- High capability for cybersecurity (new classification)

**C Compiler Case Study** (517 HN points):
- 16 parallel Claude agents
- 2,000 Claude Code sessions
- $20,000 API cost
- 100,000-line Rust compiler that builds Linux 6.9

The trend is unmistakable: **2026 is the year of multi-agent systems**. This POC gives you the building blocks to start experimenting today.

## Files

- `agent_team.py` - Core framework (run directly for demo)
- `example_anthropic.py` - Real-world example with Anthropic API
- `README.md` - This file

## License

MIT

## References

- [Claude Opus 4.6 Announcement](https://www.anthropic.com/news/claude-opus-4-6)
- [GPT-5.3-Codex Announcement](https://openai.com/index/introducing-gpt-5-3-codex/)
- [Agent Teams Documentation](https://code.claude.com/docs/en/agent-teams)
- [Building a C Compiler with Agent Teams](https://www.anthropic.com/engineering/building-c-compiler)
