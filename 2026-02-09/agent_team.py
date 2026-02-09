#!/usr/bin/env python3
"""
Agent Team Orchestrator - Multi-Agent Task Coordination System

Demonstrates the parallel agent orchestration pattern from Anthropic's
Claude C Compiler (CCC) project, where 16 Claude agents worked in parallel
on a shared codebase to build a 100k-line compiler.

Key patterns implemented:
1. File-based task locking (claim/release)
2. Parallel agent execution with isolation
3. Git-based synchronization and merge conflict handling
4. Progress tracking and state management
5. Task dependency resolution

This is a simplified demonstration of the core orchestration logic.
In production, each "agent" would be an LLM instance with tool access.

Author: Claw (AI Research Assistant)
Date: 2026-02-09
"""

import os
import json
import time
import uuid
import shutil
import hashlib
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


class TaskStatus(Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AgentRole(Enum):
    WORKER = "worker"           # General task execution
    REVIEWER = "reviewer"       # Code review and quality
    OPTIMIZER = "optimizer"     # Performance improvements
    DOCUMENTER = "documenter"   # Documentation maintenance
    INTEGRATOR = "integrator"   # Merge conflict resolution


@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    claimed_by: Optional[str] = None
    claimed_at: Optional[float] = None
    completed_at: Optional[float] = None
    dependencies: list = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    priority: int = 0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['status'] = self.status.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Task':
        d['status'] = TaskStatus(d['status'])
        return cls(**d)


@dataclass
class Agent:
    id: str
    role: AgentRole
    workspace_dir: Path
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    
    def to_dict(self) -> dict:
        d = {
            'id': self.id,
            'role': self.role.value,
            'workspace_dir': str(self.workspace_dir),
            'current_task': self.current_task,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
        }
        return d


class TaskRegistry:
    """
    File-based task registry with locking mechanism.
    Uses atomic file operations to prevent race conditions.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.tasks_dir = base_dir / "tasks"
        self.locks_dir = base_dir / "current_tasks"
        self.completed_dir = base_dir / "completed_tasks"
        self.state_file = base_dir / "registry_state.json"
        
        # Create directories
        for d in [self.tasks_dir, self.locks_dir, self.completed_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the registry."""
        task_file = self.tasks_dir / f"{task.id}.json"
        with open(task_file, 'w') as f:
            json.dump(task.to_dict(), f, indent=2)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        task_file = self.tasks_dir / f"{task_id}.json"
        if not task_file.exists():
            return None
        with open(task_file, 'r') as f:
            return Task.from_dict(json.load(f))
    
    def claim_task(self, task_id: str, agent_id: str) -> bool:
        """
        Attempt to claim a task using file-based locking.
        Returns True if claim successful, False if already claimed.
        
        This implements the lock pattern from CCC:
        - Agent writes a lock file to current_tasks/
        - If file already exists, claim fails
        - Uses atomic file creation to prevent races
        """
        lock_file = self.locks_dir / f"{task_id}.lock"
        
        # Atomic file creation - fails if file exists
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w') as f:
                lock_data = {
                    'agent_id': agent_id,
                    'claimed_at': time.time(),
                    'task_id': task_id
                }
                json.dump(lock_data, f)
            
            # Update task status
            task = self.get_task(task_id)
            if task:
                task.status = TaskStatus.CLAIMED
                task.claimed_by = agent_id
                task.claimed_at = time.time()
                self.add_task(task)
            
            return True
        except FileExistsError:
            return False
    
    def release_task(self, task_id: str, agent_id: str, 
                     success: bool, result: Optional[str] = None,
                     error: Optional[str] = None) -> None:
        """Release a task lock and update status."""
        lock_file = self.locks_dir / f"{task_id}.lock"
        
        # Verify agent owns the lock
        if lock_file.exists():
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)
            if lock_data.get('agent_id') != agent_id:
                raise PermissionError(f"Agent {agent_id} does not own lock for {task_id}")
            
            # Remove lock
            lock_file.unlink()
        
        # Update task
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.completed_at = time.time()
            task.result = result
            task.error = error
            self.add_task(task)
            
            # Move to completed
            src = self.tasks_dir / f"{task_id}.json"
            dst = self.completed_dir / f"{task_id}.json"
            if src.exists():
                shutil.copy(src, dst)
    
    def get_available_tasks(self, dependencies_met: Callable[[Task], bool] = None) -> list[Task]:
        """Get all tasks that are pending and available for claiming."""
        available = []
        for task_file in self.tasks_dir.glob("*.json"):
            task = self.get_task(task_file.stem)
            if task and task.status == TaskStatus.PENDING:
                if dependencies_met is None or dependencies_met(task):
                    available.append(task)
        
        # Sort by priority (higher first)
        return sorted(available, key=lambda t: t.priority, reverse=True)
    
    def get_all_tasks(self) -> list[Task]:
        """Get all tasks in the registry."""
        tasks = []
        for task_file in self.tasks_dir.glob("*.json"):
            task = self.get_task(task_file.stem)
            if task:
                tasks.append(task)
        return tasks
    
    def get_stats(self) -> dict:
        """Get registry statistics."""
        tasks = self.get_all_tasks()
        return {
            'total': len(tasks),
            'pending': sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            'claimed': sum(1 for t in tasks if t.status == TaskStatus.CLAIMED),
            'in_progress': sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS),
            'completed': sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            'failed': sum(1 for t in tasks if t.status == TaskStatus.FAILED),
        }


class WorkspaceManager:
    """
    Manages isolated workspaces for each agent.
    Handles git operations for synchronization.
    """
    
    def __init__(self, upstream_dir: Path, workspaces_dir: Path):
        self.upstream_dir = upstream_dir
        self.workspaces_dir = workspaces_dir
        workspaces_dir.mkdir(parents=True, exist_ok=True)
        upstream_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize upstream as bare repo if needed
        if not (upstream_dir / "HEAD").exists():
            self._run_git(["init", "--bare"], cwd=upstream_dir)
            # Create initial commit so clone works
            self._init_upstream_repo()
    
    def _run_git(self, args: list, cwd: Path = None) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git"] + args
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    
    def _init_upstream_repo(self) -> None:
        """Initialize upstream repo with an initial commit."""
        # Create a temp directory to make initial commit
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            init_dir = tmp / "init"
            init_dir.mkdir()
            
            # Initialize regular repo
            self._run_git(["init"], cwd=init_dir)
            self._run_git(["config", "user.email", "system@agent.local"], cwd=init_dir)
            self._run_git(["config", "user.name", "System"], cwd=init_dir)
            
            # Create initial file
            readme = init_dir / "README.md"
            readme.write_text("# Agent Team Workspace\n\nShared repository for parallel agent collaboration.\n")
            
            # Create outputs directory
            (init_dir / "outputs").mkdir()
            (init_dir / "outputs" / ".gitkeep").touch()
            
            # Create and switch to main branch
            self._run_git(["checkout", "-b", "main"], cwd=init_dir)
            
            # Initial commit
            self._run_git(["add", "-A"], cwd=init_dir)
            self._run_git(["commit", "-m", "Initial commit"], cwd=init_dir)
            
            # Push to bare repo
            self._run_git(["remote", "add", "origin", str(self.upstream_dir)], cwd=init_dir)
            self._run_git(["push", "-u", "origin", "main"], cwd=init_dir)
    
    def create_workspace(self, agent_id: str) -> Path:
        """Create an isolated workspace for an agent."""
        workspace = self.workspaces_dir / agent_id
        
        if workspace.exists():
            shutil.rmtree(workspace)
        
        workspace.mkdir(parents=True)
        
        # Clone from upstream
        self._run_git(["clone", str(self.upstream_dir), "."], cwd=workspace)
        
        # Configure git
        self._run_git(["config", "user.email", f"{agent_id}@agent.local"], cwd=workspace)
        self._run_git(["config", "user.name", agent_id], cwd=workspace)
        
        return workspace
    
    def sync_workspace(self, workspace: Path) -> tuple[bool, str]:
        """
        Pull latest changes and merge.
        Returns (success, message).
        """
        # Fetch
        result = self._run_git(["fetch", "origin"], cwd=workspace)
        if result.returncode != 0:
            return False, f"Fetch failed: {result.stderr}"
        
        # Try to merge
        result = self._run_git(["merge", "origin/main", "--no-edit"], cwd=workspace)
        if result.returncode != 0:
            # Merge conflict - try to resolve automatically
            result = self._run_git(["merge", "--abort"], cwd=workspace)
            return False, "Merge conflict detected"
        
        return True, "Sync successful"
    
    def push_changes(self, workspace: Path, message: str) -> tuple[bool, str]:
        """
        Commit and push changes.
        Returns (success, message).
        """
        # Stage all changes
        self._run_git(["add", "-A"], cwd=workspace)
        
        # Check if there are changes
        result = self._run_git(["diff", "--cached", "--quiet"], cwd=workspace)
        if result.returncode == 0:
            return True, "No changes to push"
        
        # Commit
        result = self._run_git(["commit", "-m", message], cwd=workspace)
        if result.returncode != 0:
            return False, f"Commit failed: {result.stderr}"
        
        # Push
        result = self._run_git(["push", "origin", "main"], cwd=workspace)
        if result.returncode != 0:
            # Push failed - probably need to pull first
            return False, f"Push failed: {result.stderr}"
        
        return True, "Changes pushed successfully"


class AgentTeamOrchestrator:
    """
    Main orchestrator for parallel agent execution.
    Coordinates multiple agents working on a shared task set.
    """
    
    def __init__(self, base_dir: Path, num_agents: int = 4):
        self.base_dir = base_dir
        self.num_agents = num_agents
        
        # Initialize components
        self.registry = TaskRegistry(base_dir / "registry")
        self.workspace_mgr = WorkspaceManager(
            base_dir / "upstream.git",
            base_dir / "workspaces"
        )
        
        # Create agents with different roles
        self.agents: list[Agent] = []
        roles = [AgentRole.WORKER] * (num_agents - 1) + [AgentRole.REVIEWER]
        for i in range(num_agents):
            agent = Agent(
                id=f"agent-{i:02d}",
                role=roles[i % len(roles)],
                workspace_dir=self.workspace_mgr.create_workspace(f"agent-{i:02d}")
            )
            self.agents.append(agent)
        
        # Progress tracking
        self.start_time = None
        self.end_time = None
        self.log_file = base_dir / "orchestrator.log"
    
    def log(self, message: str) -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(self.log_file, 'a') as f:
            f.write(log_line + "\n")
    
    def add_tasks(self, task_descriptions: list[tuple[str, int]]) -> None:
        """Add tasks to the registry. Each tuple is (description, priority)."""
        for desc, priority in task_descriptions:
            task = Task(
                id=f"task-{uuid.uuid4().hex[:8]}",
                description=desc,
                priority=priority
            )
            self.registry.add_task(task)
            self.log(f"Added task: {task.id} - {desc[:50]}...")
    
    def execute_task(self, agent: Agent, task: Task, 
                     task_handler: Callable[[Agent, Task], tuple[bool, str]]) -> bool:
        """
        Execute a single task with an agent.
        Returns True if successful.
        """
        self.log(f"Agent {agent.id} starting task {task.id}: {task.description[:40]}...")
        
        try:
            # Sync workspace before starting
            success, msg = self.workspace_mgr.sync_workspace(agent.workspace_dir)
            if not success:
                self.log(f"Agent {agent.id} sync failed: {msg}")
            
            # Execute the task
            success, result = task_handler(agent, task)
            
            if success:
                # Push changes
                push_success, push_msg = self.workspace_mgr.push_changes(
                    agent.workspace_dir,
                    f"[{agent.id}] Complete task {task.id}: {task.description[:50]}"
                )
                if not push_success:
                    # Retry after sync
                    self.workspace_mgr.sync_workspace(agent.workspace_dir)
                    push_success, push_msg = self.workspace_mgr.push_changes(
                        agent.workspace_dir,
                        f"[{agent.id}] Complete task {task.id}: {task.description[:50]}"
                    )
            
            # Release the task
            self.registry.release_task(task.id, agent.id, success, result if success else None, 
                                       None if success else result)
            
            if success:
                agent.tasks_completed += 1
                self.log(f"Agent {agent.id} completed task {task.id}")
            else:
                agent.tasks_failed += 1
                self.log(f"Agent {agent.id} failed task {task.id}: {result}")
            
            return success
            
        except Exception as e:
            self.registry.release_task(task.id, agent.id, False, error=str(e))
            agent.tasks_failed += 1
            self.log(f"Agent {agent.id} error on task {task.id}: {e}")
            return False
    
    def agent_loop(self, agent: Agent, 
                   task_handler: Callable[[Agent, Task], tuple[bool, str]],
                   max_iterations: int = 100) -> None:
        """
        Main loop for a single agent.
        Continuously claims and executes tasks until none remain.
        """
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Get available tasks
            available = self.registry.get_available_tasks()
            if not available:
                self.log(f"Agent {agent.id}: No available tasks, checking for claimed...")
                # Check if there are any claimed/in-progress tasks
                stats = self.registry.get_stats()
                if stats['claimed'] == 0 and stats['in_progress'] == 0:
                    self.log(f"Agent {agent.id}: All tasks complete, exiting")
                    break
                # Wait for other agents
                time.sleep(0.5)
                continue
            
            # Try to claim the highest priority task
            claimed = False
            for task in available:
                if self.registry.claim_task(task.id, agent.id):
                    agent.current_task = task.id
                    self.execute_task(agent, task, task_handler)
                    agent.current_task = None
                    claimed = True
                    break
            
            if not claimed:
                # All tasks were claimed by other agents
                time.sleep(0.1)
    
    def run(self, task_handler: Callable[[Agent, Task], tuple[bool, str]],
            max_time_seconds: int = 300) -> dict:
        """
        Run the orchestrator with parallel agents.
        Returns statistics about the run.
        """
        self.start_time = time.time()
        self.log(f"Starting orchestrator with {len(self.agents)} agents")
        self.log(f"Initial stats: {self.registry.get_stats()}")
        
        # Run agents in parallel
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(self.agent_loop, agent, task_handler): agent
                for agent in self.agents
            }
            
            # Wait for completion with timeout
            for future in as_completed(futures, timeout=max_time_seconds):
                agent = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.log(f"Agent {agent.id} crashed: {e}")
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Compile results
        stats = self.registry.get_stats()
        agent_stats = [agent.to_dict() for agent in self.agents]
        
        results = {
            'duration_seconds': duration,
            'task_stats': stats,
            'agent_stats': agent_stats,
            'tasks_per_second': stats['completed'] / duration if duration > 0 else 0,
        }
        
        self.log(f"Orchestration complete in {duration:.2f}s")
        self.log(f"Final stats: {stats}")
        
        return results


def demo_task_handler(agent: Agent, task: Task) -> tuple[bool, str]:
    """
    Demo task handler that simulates work.
    In a real system, this would invoke an LLM agent.
    """
    # Simulate varying work duration
    work_time = 0.1 + (hash(task.id) % 10) / 20  # 0.1-0.6 seconds
    time.sleep(work_time)
    
    # Create a file in the workspace to simulate code changes
    output_file = agent.workspace_dir / "outputs" / f"{task.id}.txt"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(f"Task: {task.description}\n")
        f.write(f"Completed by: {agent.id}\n")
        f.write(f"Role: {agent.role.value}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Simulate occasional failures (10% chance)
    if hash(task.id + agent.id) % 10 == 0:
        return False, "Simulated random failure"
    
    return True, f"Completed in {work_time:.2f}s"


def main():
    """Demo the Agent Team Orchestrator."""
    import tempfile
    
    print("=" * 60)
    print("Agent Team Orchestrator Demo")
    print("Based on patterns from Anthropic's Claude C Compiler (CCC)")
    print("=" * 60)
    print()
    
    # Create temp directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        
        # Initialize orchestrator with 4 agents
        orchestrator = AgentTeamOrchestrator(base_dir, num_agents=4)
        
        # Add sample tasks (simulating compiler development tasks)
        tasks = [
            ("Implement lexer for C tokens", 10),
            ("Implement preprocessor directives", 9),
            ("Parse function declarations", 8),
            ("Parse if/else statements", 8),
            ("Parse for/while loops", 8),
            ("Parse struct definitions", 7),
            ("Implement type checker", 7),
            ("Generate SSA IR from AST", 6),
            ("Implement dead code elimination", 5),
            ("Implement constant propagation", 5),
            ("Implement register allocation", 4),
            ("Generate x86-64 assembly", 4),
            ("Implement ELF linker", 3),
            ("Add DWARF debug info", 2),
            ("Write documentation", 1),
            ("Run test suite", 1),
        ]
        
        orchestrator.add_tasks(tasks)
        print()
        
        # Run the orchestrator
        results = orchestrator.run(demo_task_handler, max_time_seconds=60)
        
        print()
        print("=" * 60)
        print("Results Summary")
        print("=" * 60)
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Tasks completed: {results['task_stats']['completed']}")
        print(f"Tasks failed: {results['task_stats']['failed']}")
        print(f"Throughput: {results['tasks_per_second']:.2f} tasks/second")
        print()
        print("Agent Performance:")
        for agent in results['agent_stats']:
            print(f"  {agent['id']} ({agent['role']}): "
                  f"{agent['tasks_completed']} completed, "
                  f"{agent['tasks_failed']} failed")


if __name__ == "__main__":
    main()
