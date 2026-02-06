#!/usr/bin/env python3
"""
Multi-Agent Orchestration Framework
====================================
A lightweight implementation of parallel LLM agent coordination,
inspired by Claude Code's Agent Teams and the C compiler case study.

This POC demonstrates:
- Task queue with file-based locking (prevents race conditions)
- Parallel agent execution with role specialization
- Shared state management via git-like synchronization
- Result synthesis from multiple agent outputs

Designed for developers who want to orchestrate multiple LLM instances
working on a shared codebase or problem space.

Author: Claw (OpenClaw Daily POC)
Date: 2026-02-06
"""

import asyncio
import json
import os
import hashlib
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any
from datetime import datetime
import fcntl
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')


class TaskStatus(Enum):
    PENDING = "pending"
    LOCKED = "locked"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRole(Enum):
    LEAD = "lead"           # Coordinates work, synthesizes results
    WORKER = "worker"       # Implements tasks
    REVIEWER = "reviewer"   # Reviews and critiques work
    SPECIALIST = "specialist"  # Domain-specific expertise


@dataclass
class Task:
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    recipient: str  # Use '*' for broadcast
    content: str
    timestamp: float = field(default_factory=time.time)
    message_type: str = "info"  # info, request, response, alert


class TaskQueue:
    """
    File-based task queue with locking.
    Prevents race conditions when multiple agents try to claim the same task.
    Based on the synchronization approach used in the C compiler case study.
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.tasks_file = workspace / "tasks.json"
        self.locks_dir = workspace / "locks"
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.tasks_file.exists():
            self._write_tasks([])
    
    def _get_lock_path(self, task_id: str) -> Path:
        return self.locks_dir / f"{task_id}.lock"
    
    def _acquire_lock(self, task_id: str, agent_id: str) -> bool:
        """Try to acquire exclusive lock on a task"""
        lock_path = self._get_lock_path(task_id)
        try:
            # Create lock file with agent ID
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, agent_id.encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False
    
    def _release_lock(self, task_id: str, agent_id: str) -> bool:
        """Release lock if owned by this agent"""
        lock_path = self._get_lock_path(task_id)
        try:
            with open(lock_path, 'r') as f:
                owner = f.read()
            if owner == agent_id:
                os.remove(lock_path)
                return True
            return False
        except FileNotFoundError:
            return True
    
    def _read_tasks(self) -> list[Task]:
        with open(self.tasks_file, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return [Task.from_dict(t) for t in data]
    
    def _write_tasks(self, tasks: list[Task]):
        with open(self.tasks_file, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump([t.to_dict() for t in tasks], f, indent=2)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def add_task(self, description: str, dependencies: list[str] = None) -> Task:
        """Add a new task to the queue"""
        task_id = hashlib.sha256(
            f"{description}{time.time()}".encode()
        ).hexdigest()[:12]
        
        task = Task(
            id=task_id,
            description=description,
            dependencies=dependencies or []
        )
        
        tasks = self._read_tasks()
        tasks.append(task)
        self._write_tasks(tasks)
        return task
    
    def claim_task(self, agent_id: str) -> Optional[Task]:
        """
        Claim the next available task.
        Returns None if no tasks available or all locked.
        """
        tasks = self._read_tasks()
        completed_ids = {t.id for t in tasks if t.status == TaskStatus.COMPLETED}
        
        for task in tasks:
            # Skip if not pending
            if task.status != TaskStatus.PENDING:
                continue
            
            # Skip if dependencies not met
            if not all(dep in completed_ids for dep in task.dependencies):
                continue
            
            # Try to acquire lock
            if self._acquire_lock(task.id, agent_id):
                task.status = TaskStatus.IN_PROGRESS
                task.assigned_to = agent_id
                self._write_tasks(tasks)
                return task
        
        return None
    
    def complete_task(self, task_id: str, agent_id: str, result: str):
        """Mark task as completed and release lock"""
        tasks = self._read_tasks()
        for task in tasks:
            if task.id == task_id and task.assigned_to == agent_id:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = time.time()
                self._write_tasks(tasks)
                self._release_lock(task_id, agent_id)
                return True
        return False
    
    def fail_task(self, task_id: str, agent_id: str, error: str):
        """Mark task as failed and release lock"""
        tasks = self._read_tasks()
        for task in tasks:
            if task.id == task_id and task.assigned_to == agent_id:
                task.status = TaskStatus.FAILED
                task.result = f"FAILED: {error}"
                self._write_tasks(tasks)
                self._release_lock(task_id, agent_id)
                return True
        return False
    
    def get_status(self) -> dict:
        """Get summary of task queue status"""
        tasks = self._read_tasks()
        return {
            'total': len(tasks),
            'pending': sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            'in_progress': sum(1 for t in tasks if t.status == TaskStatus.IN_PROGRESS),
            'completed': sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
            'failed': sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            'tasks': [t.to_dict() for t in tasks]
        }


class MessageBus:
    """
    Simple message passing between agents.
    Enables coordination and information sharing.
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.messages_file = workspace / "messages.json"
        if not self.messages_file.exists():
            self._write_messages([])
    
    def _read_messages(self) -> list[dict]:
        with open(self.messages_file, 'r') as f:
            return json.load(f)
    
    def _write_messages(self, messages: list[dict]):
        with open(self.messages_file, 'w') as f:
            json.dump(messages, f, indent=2)
    
    def send(self, message: AgentMessage):
        """Send a message"""
        messages = self._read_messages()
        messages.append(asdict(message))
        self._write_messages(messages)
    
    def receive(self, agent_id: str, since: float = 0) -> list[AgentMessage]:
        """Get messages for an agent since timestamp"""
        messages = self._read_messages()
        relevant = [
            AgentMessage(**m) for m in messages
            if m['timestamp'] > since
            and (m['recipient'] == agent_id or m['recipient'] == '*')
            and m['sender'] != agent_id
        ]
        return relevant


class Agent:
    """
    Base agent class with task execution and coordination capabilities.
    Subclass and override execute_task() to customize behavior.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        task_queue: TaskQueue,
        message_bus: MessageBus,
        llm_fn: Optional[Callable[[str], str]] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.task_queue = task_queue
        self.message_bus = message_bus
        self.llm_fn = llm_fn or self._default_llm
        self.logger = logging.getLogger(agent_id)
        self.last_message_check = 0
        self.running = False
    
    def _default_llm(self, prompt: str) -> str:
        """Placeholder LLM function - replace with actual API call"""
        return f"[{self.agent_id}] Processed: {prompt[:100]}..."
    
    async def execute_task(self, task: Task) -> str:
        """
        Execute a task. Override this in subclasses for custom behavior.
        """
        self.logger.info(f"Executing task: {task.description}")
        
        # Build context from messages
        messages = self.message_bus.receive(self.agent_id, self.last_message_check)
        self.last_message_check = time.time()
        
        context = ""
        if messages:
            context = "\n".join([f"[{m.sender}]: {m.content}" for m in messages])
            context = f"\n\nRecent team communications:\n{context}\n\n"
        
        prompt = f"""You are {self.agent_id}, a {self.role.value} agent.

Task: {task.description}
{context}
Execute this task and provide a clear, concise result."""

        result = await asyncio.to_thread(self.llm_fn, prompt)
        return result
    
    async def run_loop(self, max_iterations: int = None):
        """
        Main agent loop - claim and execute tasks until queue is empty.
        """
        self.running = True
        iterations = 0
        
        self.logger.info(f"Starting agent loop (role: {self.role.value})")
        
        while self.running:
            if max_iterations and iterations >= max_iterations:
                break
            
            # Check for messages
            messages = self.message_bus.receive(self.agent_id, self.last_message_check)
            self.last_message_check = time.time()
            
            for msg in messages:
                self.logger.info(f"Received message from {msg.sender}: {msg.content[:50]}...")
            
            # Try to claim a task
            task = self.task_queue.claim_task(self.agent_id)
            
            if task is None:
                # No tasks available, wait and retry
                await asyncio.sleep(1)
                
                # Check if all done
                status = self.task_queue.get_status()
                if status['pending'] == 0 and status['in_progress'] == 0:
                    self.logger.info("All tasks completed, stopping")
                    break
                continue
            
            self.logger.info(f"Claimed task {task.id}: {task.description[:50]}...")
            
            try:
                result = await self.execute_task(task)
                self.task_queue.complete_task(task.id, self.agent_id, result)
                self.logger.info(f"Completed task {task.id}")
                
                # Broadcast completion
                self.message_bus.send(AgentMessage(
                    sender=self.agent_id,
                    recipient='*',
                    content=f"Completed: {task.description[:50]}",
                    message_type='info'
                ))
                
            except Exception as e:
                self.logger.error(f"Task {task.id} failed: {e}")
                self.task_queue.fail_task(task.id, self.agent_id, str(e))
            
            iterations += 1
        
        self.running = False
    
    def stop(self):
        """Stop the agent loop"""
        self.running = False


class LeadAgent(Agent):
    """
    Lead agent that coordinates work and synthesizes results.
    Can spawn tasks dynamically based on problem decomposition.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = AgentRole.LEAD
    
    def decompose_problem(self, problem: str, num_subtasks: int = 4) -> list[str]:
        """
        Break a problem into subtasks.
        In production, use LLM to do intelligent decomposition.
        """
        prompt = f"""Break this problem into {num_subtasks} independent subtasks:

Problem: {problem}

Return a JSON list of task descriptions."""

        result = self.llm_fn(prompt)
        
        # Simplified parsing - in production, use proper JSON extraction
        # This is a fallback that creates generic subtasks
        tasks = [
            f"Subtask 1: Research and analysis for - {problem}",
            f"Subtask 2: Implementation approach for - {problem}",
            f"Subtask 3: Testing strategy for - {problem}",
            f"Subtask 4: Documentation for - {problem}",
        ]
        return tasks[:num_subtasks]
    
    def synthesize_results(self) -> str:
        """Combine results from completed tasks into final output"""
        status = self.task_queue.get_status()
        completed = [t for t in status['tasks'] if t['status'] == 'completed']
        
        results_text = "\n\n".join([
            f"### {t['description'][:50]}...\n{t['result']}"
            for t in completed
        ])
        
        prompt = f"""Synthesize these task results into a coherent final output:

{results_text}

Provide a clear, comprehensive summary."""

        return self.llm_fn(prompt)


class AgentTeam:
    """
    Orchestrates multiple agents working in parallel.
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        workspace.mkdir(parents=True, exist_ok=True)
        
        self.task_queue = TaskQueue(workspace)
        self.message_bus = MessageBus(workspace)
        self.agents: list[Agent] = []
        self.logger = logging.getLogger("AgentTeam")
    
    def add_agent(
        self,
        agent_id: str,
        role: AgentRole,
        llm_fn: Optional[Callable[[str], str]] = None
    ) -> Agent:
        """Add an agent to the team"""
        if role == AgentRole.LEAD:
            agent = LeadAgent(
                agent_id, role, self.task_queue, self.message_bus, llm_fn
            )
        else:
            agent = Agent(
                agent_id, role, self.task_queue, self.message_bus, llm_fn
            )
        self.agents.append(agent)
        self.logger.info(f"Added agent {agent_id} with role {role.value}")
        return agent
    
    def add_task(self, description: str, dependencies: list[str] = None) -> Task:
        """Add a task to the shared queue"""
        return self.task_queue.add_task(description, dependencies)
    
    async def run(self, timeout: float = None):
        """Run all agents in parallel until completion or timeout"""
        self.logger.info(f"Starting team with {len(self.agents)} agents")
        
        tasks = [agent.run_loop() for agent in self.agents]
        
        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=timeout
                )
            else:
                await asyncio.gather(*tasks)
        except asyncio.TimeoutError:
            self.logger.warning("Team execution timed out")
            for agent in self.agents:
                agent.stop()
        
        status = self.task_queue.get_status()
        self.logger.info(
            f"Team finished: {status['completed']}/{status['total']} tasks completed"
        )
        return status
    
    def get_status(self) -> dict:
        """Get current status of team and tasks"""
        return {
            'agents': [
                {'id': a.agent_id, 'role': a.role.value, 'running': a.running}
                for a in self.agents
            ],
            'tasks': self.task_queue.get_status()
        }


# -----------------------------------------------------------------------------
# Example: Simulated Multi-Agent Code Review
# -----------------------------------------------------------------------------

def mock_llm(prompt: str) -> str:
    """Mock LLM for demonstration - replace with actual API calls"""
    import random
    time.sleep(0.1)  # Simulate API latency
    
    if "Research" in prompt:
        return "Research findings: Identified key patterns and best practices."
    elif "Implementation" in prompt:
        return "Implementation: Proposed modular architecture with clean interfaces."
    elif "Testing" in prompt:
        return "Testing strategy: Unit tests, integration tests, and property-based testing."
    elif "Documentation" in prompt:
        return "Documentation: API reference, usage examples, and architecture overview."
    elif "Synthesize" in prompt:
        return "Final synthesis: Comprehensive solution combining all agent contributions."
    else:
        return f"Processed: {prompt[:50]}..."


async def demo():
    """Demonstrate multi-agent orchestration"""
    print("=" * 60)
    print("Multi-Agent Orchestration Demo")
    print("=" * 60)
    
    # Create workspace
    workspace = Path("/tmp/agent_team_demo")
    if workspace.exists():
        import shutil
        shutil.rmtree(workspace)
    
    # Initialize team
    team = AgentTeam(workspace)
    
    # Add agents with different roles
    team.add_agent("lead-1", AgentRole.LEAD, mock_llm)
    team.add_agent("worker-1", AgentRole.WORKER, mock_llm)
    team.add_agent("worker-2", AgentRole.WORKER, mock_llm)
    team.add_agent("reviewer-1", AgentRole.REVIEWER, mock_llm)
    
    # Add tasks (simulating a code refactoring project)
    t1 = team.add_task("Research existing codebase patterns")
    t2 = team.add_task("Implementation: Refactor authentication module")
    t3 = team.add_task("Implementation: Refactor database layer")
    t4 = team.add_task("Testing: Write test suite for auth module", [t2.id])
    t5 = team.add_task("Testing: Write test suite for db layer", [t3.id])
    t6 = team.add_task("Documentation: Update API docs", [t4.id, t5.id])
    
    print("\nTasks created:")
    for t in [t1, t2, t3, t4, t5, t6]:
        deps = f" (depends on: {t.dependencies})" if t.dependencies else ""
        print(f"  - {t.id}: {t.description}{deps}")
    
    print("\nRunning team...")
    
    # Run team with timeout
    status = await team.run(timeout=30)
    
    print("\n" + "=" * 60)
    print("Final Status")
    print("=" * 60)
    print(f"Total tasks: {status['total']}")
    print(f"Completed: {status['completed']}")
    print(f"Failed: {status['failed']}")
    
    print("\nTask Results:")
    for task in status['tasks']:
        status_icon = "[OK]" if task['status'] == 'completed' else "[  ]"
        print(f"  {status_icon} {task['description'][:40]}...")
        if task['result']:
            print(f"       Result: {task['result'][:60]}...")
    
    # Lead agent synthesizes results
    lead = team.agents[0]
    if isinstance(lead, LeadAgent):
        print("\n" + "=" * 60)
        print("Lead Agent Synthesis")
        print("=" * 60)
        synthesis = lead.synthesize_results()
        print(synthesis)
    
    return status


if __name__ == "__main__":
    asyncio.run(demo())
