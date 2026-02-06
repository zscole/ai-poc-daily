#!/usr/bin/env python3
"""
Real-world example: Multi-Agent Code Analysis with Anthropic API

This example demonstrates using the agent_team framework with actual
LLM API calls to analyze a codebase from multiple perspectives.

Requirements:
    pip install anthropic

Usage:
    export ANTHROPIC_API_KEY=your_key
    python example_anthropic.py
"""

import asyncio
import os
from pathlib import Path
from anthropic import Anthropic

from agent_team import (
    AgentTeam, AgentRole, LeadAgent, Agent, Task, AgentMessage
)


def create_anthropic_llm(
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024
):
    """Factory function to create an Anthropic-powered LLM function"""
    client = Anthropic()
    
    def llm_fn(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    return llm_fn


class ArchitectAgent(Agent):
    """Agent specialized in architecture review"""
    
    async def execute_task(self, task: Task) -> str:
        prompt = f"""You are a senior software architect reviewing code.

Task: {task.description}

Focus on:
- Overall architecture patterns
- Module dependencies and coupling
- Scalability concerns
- Design pattern usage

Provide specific, actionable feedback."""

        return await asyncio.to_thread(self.llm_fn, prompt)


class SecurityAgent(Agent):
    """Agent specialized in security analysis"""
    
    async def execute_task(self, task: Task) -> str:
        prompt = f"""You are a security engineer analyzing code.

Task: {task.description}

Focus on:
- Input validation vulnerabilities
- Authentication/authorization issues
- Data exposure risks
- Injection vulnerabilities
- Secure coding practices

Identify specific issues with severity ratings."""

        return await asyncio.to_thread(self.llm_fn, prompt)


class PerformanceAgent(Agent):
    """Agent specialized in performance analysis"""
    
    async def execute_task(self, task: Task) -> str:
        prompt = f"""You are a performance engineer analyzing code.

Task: {task.description}

Focus on:
- Algorithm complexity
- Memory usage patterns
- I/O bottlenecks
- Caching opportunities
- Concurrency issues

Provide specific optimization recommendations."""

        return await asyncio.to_thread(self.llm_fn, prompt)


async def analyze_codebase(code_path: str):
    """
    Run multi-agent analysis on a codebase.
    
    Creates a team with specialized agents that analyze code from
    different perspectives in parallel.
    """
    print("=" * 70)
    print("Multi-Agent Code Analysis")
    print("=" * 70)
    
    # Read the code
    with open(code_path, 'r') as f:
        code = f.read()
    
    code_snippet = code[:2000] + "..." if len(code) > 2000 else code
    
    # Create workspace
    workspace = Path("/tmp/code_analysis")
    if workspace.exists():
        import shutil
        shutil.rmtree(workspace)
    
    # Initialize team with Anthropic-powered agents
    team = AgentTeam(workspace)
    llm = create_anthropic_llm()
    
    # Add lead agent
    lead = team.add_agent("lead", AgentRole.LEAD, llm)
    
    # Add specialized agents with custom classes
    # (Using factory pattern to inject dependencies)
    architect = ArchitectAgent(
        "architect", AgentRole.SPECIALIST,
        team.task_queue, team.message_bus, llm
    )
    team.agents.append(architect)
    
    security = SecurityAgent(
        "security", AgentRole.SPECIALIST,
        team.task_queue, team.message_bus, llm
    )
    team.agents.append(security)
    
    performance = PerformanceAgent(
        "performance", AgentRole.SPECIALIST,
        team.task_queue, team.message_bus, llm
    )
    team.agents.append(performance)
    
    # Create analysis tasks
    team.add_task(f"Architecture Review:\n\n```python\n{code_snippet}\n```")
    team.add_task(f"Security Analysis:\n\n```python\n{code_snippet}\n```")
    team.add_task(f"Performance Analysis:\n\n```python\n{code_snippet}\n```")
    
    print(f"\nAnalyzing: {code_path}")
    print(f"Team size: {len(team.agents)} agents")
    print("Running parallel analysis...\n")
    
    # Run analysis
    status = await team.run(timeout=120)
    
    # Print results
    print("\n" + "=" * 70)
    print("Analysis Results")
    print("=" * 70)
    
    for task in status['tasks']:
        if task['status'] == 'completed':
            print(f"\n--- {task['description'][:50]} ---")
            print(task['result'])
            print()
    
    # Synthesize findings
    if isinstance(lead, LeadAgent):
        print("\n" + "=" * 70)
        print("Executive Summary (Lead Agent Synthesis)")
        print("=" * 70)
        print(lead.synthesize_results())
    
    return status


async def main():
    # Default: analyze the agent_team.py file itself
    code_path = Path(__file__).parent / "agent_team.py"
    
    if not code_path.exists():
        print(f"File not found: {code_path}")
        print("Creating a sample file to analyze...")
        
        sample = '''
def process_user_input(data):
    """Process user input and store in database"""
    query = f"INSERT INTO users VALUES ('{data['name']}', '{data['email']}')"
    db.execute(query)
    
    # Load all users into memory
    all_users = db.execute("SELECT * FROM users").fetchall()
    
    for user in all_users:
        send_notification(user)
    
    return {"status": "ok", "password": data.get("password")}
'''
        code_path = Path("/tmp/sample_code.py")
        code_path.write_text(sample)
    
    await analyze_codebase(str(code_path))


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set")
        print("Set it to run with real API calls")
        print("Running demo mode with mock responses...\n")
        
        # Run basic demo instead
        from agent_team import demo
        asyncio.run(demo())
    else:
        asyncio.run(main())
