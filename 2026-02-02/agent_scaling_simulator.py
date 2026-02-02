#!/usr/bin/env python3
"""
Agent Scaling Simulator - Demonstrating When Multi-Agent Systems Help (or Hurt)

Based on Google Research's "Towards a Science of Scaling Agent Systems" paper:
https://arxiv.org/abs/2512.08296

Key Findings Demonstrated:
1. Centralized coordination improves parallelizable tasks by ~80%
2. Sequential tasks DEGRADE with multi-agent (39-70% worse)
3. Independent agents amplify errors 17.2x; centralized contains to 4.4x
4. Tool-heavy tasks suffer from multi-agent coordination overhead

This POC simulates these effects to help developers choose the right architecture.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import json
import math

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


class TaskType(Enum):
    PARALLELIZABLE = "parallelizable"  # Independent subtasks (financial analysis)
    SEQUENTIAL = "sequential"          # Strict dependencies (planning)
    TOOL_HEAVY = "tool_heavy"          # Many tool calls (coding agents)
    HYBRID = "hybrid"                  # Mixed dependencies


class Architecture(Enum):
    SINGLE_AGENT = "single_agent"
    INDEPENDENT = "independent"        # Parallel, no communication
    CENTRALIZED = "centralized"        # Hub-and-spoke with orchestrator
    DECENTRALIZED = "decentralized"    # Peer-to-peer mesh
    HYBRID = "hybrid"                  # Hierarchical + peer coordination


@dataclass
class Task:
    """Represents a task with properties that affect agent performance."""
    name: str
    task_type: TaskType
    subtasks: int
    tool_count: int
    decomposability: float  # 0-1, how easily split into subtasks
    sequential_depth: int   # How many steps depend on previous steps
    base_difficulty: float  # 0-1


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    latency_ms: float
    errors: list[str]
    subtask_results: list[bool]


@dataclass
class SimulationResult:
    """Complete simulation result for an architecture."""
    architecture: Architecture
    task: Task
    accuracy: float
    latency_ms: float
    error_amplification: float
    coordination_overhead: float
    improvement_vs_single: float


class AgentSimulator:
    """
    Simulates agent architectures based on empirical findings from Google Research.
    
    The simulation models real-world effects observed in their 180-configuration study:
    - Coordination overhead
    - Error propagation
    - Task decomposition benefits/penalties
    """
    
    # Empirical parameters from the paper
    CENTRALIZED_PARALLELIZABLE_BOOST = 0.809  # +80.9% on parallelizable
    DECENTRALIZED_WEB_NAV_BOOST = 0.092       # +9.2% on navigation
    SEQUENTIAL_PENALTY_RANGE = (0.39, 0.70)   # 39-70% degradation
    
    ERROR_AMPLIFICATION = {
        Architecture.SINGLE_AGENT: 1.0,
        Architecture.INDEPENDENT: 17.2,
        Architecture.CENTRALIZED: 4.4,
        Architecture.DECENTRALIZED: 8.5,
        Architecture.HYBRID: 6.0,
    }
    
    BASE_SINGLE_AGENT_ACCURACY = 0.65  # Baseline ~65%
    
    def __init__(self, agent_count: int = 4, verbose: bool = True):
        self.agent_count = agent_count
        self.verbose = verbose
        
    def log(self, msg: str, color: str = Colors.END):
        if self.verbose:
            print(f"{color}{msg}{Colors.END}")
    
    def calculate_coordination_overhead(
        self, 
        architecture: Architecture, 
        task: Task
    ) -> float:
        """
        Calculate coordination overhead based on architecture and task properties.
        
        Key insight: Tool-heavy tasks suffer disproportionately from coordination.
        """
        base_overhead = {
            Architecture.SINGLE_AGENT: 0.0,
            Architecture.INDEPENDENT: 0.05,
            Architecture.CENTRALIZED: 0.15,
            Architecture.DECENTRALIZED: 0.25,
            Architecture.HYBRID: 0.20,
        }
        
        overhead = base_overhead[architecture]
        
        # Tool-coordination trade-off: more tools = more overhead
        tool_penalty = task.tool_count * 0.01 * (1 if architecture != Architecture.SINGLE_AGENT else 0)
        overhead += tool_penalty
        
        # Agent count scales overhead non-linearly
        if architecture != Architecture.SINGLE_AGENT:
            overhead *= math.log(self.agent_count + 1)
            
        return min(overhead, 0.5)  # Cap at 50% overhead
    
    def calculate_decomposition_benefit(
        self, 
        architecture: Architecture, 
        task: Task
    ) -> float:
        """
        Calculate benefit from task decomposition.
        
        Key insight: Only parallelizable tasks with high decomposability benefit.
        """
        if architecture == Architecture.SINGLE_AGENT:
            return 0.0
            
        if task.task_type == TaskType.SEQUENTIAL:
            # Sequential tasks are HURT by decomposition
            penalty = random.uniform(*self.SEQUENTIAL_PENALTY_RANGE)
            return -penalty
            
        if task.task_type == TaskType.PARALLELIZABLE:
            # Parallelizable tasks benefit, especially with centralized
            benefit = task.decomposability * 0.5
            if architecture == Architecture.CENTRALIZED:
                benefit *= (1 + self.CENTRALIZED_PARALLELIZABLE_BOOST)
            return benefit
            
        if task.task_type == TaskType.TOOL_HEAVY:
            # Tool-heavy tasks have diminishing returns
            return task.decomposability * 0.2 - task.tool_count * 0.02
            
        # Hybrid tasks get partial benefit
        return task.decomposability * 0.3
    
    def simulate_errors(
        self, 
        architecture: Architecture, 
        task: Task
    ) -> tuple[int, float]:
        """
        Simulate error occurrence and amplification.
        
        Key insight: Independent agents amplify errors 17.2x vs 4.4x for centralized.
        """
        # Base error rate per subtask
        base_error_rate = 0.1 + task.base_difficulty * 0.2
        
        errors = 0
        for _ in range(task.subtasks):
            if random.random() < base_error_rate:
                errors += 1
        
        # Amplification based on architecture
        amplification = self.ERROR_AMPLIFICATION[architecture]
        
        # Centralized systems can catch and correct some errors
        if architecture == Architecture.CENTRALIZED:
            errors = max(0, errors - int(errors * 0.3))  # 30% error recovery
        
        amplified_impact = errors * amplification / task.subtasks
        
        return errors, min(amplified_impact, 1.0)
    
    def simulate_latency(
        self, 
        architecture: Architecture, 
        task: Task
    ) -> float:
        """
        Simulate execution latency based on parallelization potential.
        """
        base_latency = task.subtasks * 100  # 100ms per subtask base
        
        if architecture == Architecture.SINGLE_AGENT:
            return base_latency  # Sequential execution
            
        if task.task_type == TaskType.SEQUENTIAL:
            # Can't parallelize sequential tasks
            return base_latency * 1.1  # Actually slower due to overhead
        
        # Parallelization factor
        parallel_factor = {
            Architecture.INDEPENDENT: self.agent_count,
            Architecture.CENTRALIZED: self.agent_count * 0.7,  # Orchestrator bottleneck
            Architecture.DECENTRALIZED: self.agent_count * 0.6,  # Communication overhead
            Architecture.HYBRID: self.agent_count * 0.8,
        }
        
        factor = parallel_factor.get(architecture, 1)
        parallelizable_portion = task.decomposability
        
        parallel_time = (base_latency * parallelizable_portion) / factor
        sequential_time = base_latency * (1 - parallelizable_portion)
        
        return parallel_time + sequential_time
    
    async def run_simulation(
        self, 
        architecture: Architecture, 
        task: Task
    ) -> SimulationResult:
        """Run a complete simulation for an architecture/task combination."""
        
        self.log(f"\n{'='*60}", Colors.CYAN)
        self.log(f"Simulating: {architecture.value.upper()}", Colors.BOLD)
        self.log(f"Task: {task.name} ({task.task_type.value})", Colors.BLUE)
        self.log(f"{'='*60}", Colors.CYAN)
        
        # Calculate components
        coordination_overhead = self.calculate_coordination_overhead(architecture, task)
        decomposition_benefit = self.calculate_decomposition_benefit(architecture, task)
        errors, error_impact = self.simulate_errors(architecture, task)
        latency = self.simulate_latency(architecture, task)
        
        self.log(f"  Coordination overhead: {coordination_overhead:.1%}")
        self.log(f"  Decomposition benefit: {decomposition_benefit:+.1%}")
        self.log(f"  Errors: {errors}, Impact: {error_impact:.1%}")
        self.log(f"  Latency: {latency:.0f}ms")
        
        # Calculate final accuracy
        base_accuracy = self.BASE_SINGLE_AGENT_ACCURACY
        
        # Apply modifiers
        accuracy = base_accuracy
        accuracy += decomposition_benefit
        accuracy -= coordination_overhead
        accuracy -= error_impact
        
        # Ensure bounds
        accuracy = max(0.0, min(1.0, accuracy))
        
        # Calculate improvement vs single agent baseline
        single_baseline = base_accuracy - self.simulate_errors(
            Architecture.SINGLE_AGENT, task
        )[1]
        improvement = (accuracy - single_baseline) / single_baseline if single_baseline > 0 else 0
        
        self.log(f"\n  {Colors.GREEN}Final Accuracy: {accuracy:.1%}{Colors.END}")
        self.log(f"  {Colors.YELLOW}vs Single Agent: {improvement:+.1%}{Colors.END}")
        
        return SimulationResult(
            architecture=architecture,
            task=task,
            accuracy=accuracy,
            latency_ms=latency,
            error_amplification=self.ERROR_AMPLIFICATION[architecture],
            coordination_overhead=coordination_overhead,
            improvement_vs_single=improvement,
        )


class ArchitecturePredictor:
    """
    Predicts optimal architecture based on task properties.
    
    Based on the paper's predictive model (R²=0.524) that correctly identifies
    the optimal coordination strategy for 87% of unseen configurations.
    """
    
    def predict(self, task: Task) -> tuple[Architecture, str]:
        """
        Predict the best architecture for a task.
        
        Returns: (recommended_architecture, reasoning)
        """
        # Rule 1: Sequential tasks should use single agent
        if task.task_type == TaskType.SEQUENTIAL or task.sequential_depth > 3:
            return (
                Architecture.SINGLE_AGENT,
                "Sequential dependencies prevent effective parallelization. "
                "Multi-agent coordination fragments the reasoning process."
            )
        
        # Rule 2: High decomposability + low tools = centralized
        if task.decomposability > 0.7 and task.tool_count < 5:
            return (
                Architecture.CENTRALIZED,
                "Highly parallelizable task with low tool density. "
                "Centralized coordination can achieve up to 80.9% improvement."
            )
        
        # Rule 3: Tool-heavy tasks suffer from multi-agent overhead
        if task.tool_count > 10:
            return (
                Architecture.SINGLE_AGENT,
                "Tool-heavy task would suffer from coordination overhead. "
                "The 'tool-coordination trade-off' makes single agent optimal."
            )
        
        # Rule 4: Web navigation benefits from decentralized
        if "web" in task.name.lower() or "browse" in task.name.lower():
            return (
                Architecture.DECENTRALIZED,
                "Web navigation tasks benefit from peer-to-peer coordination. "
                "Decentralized shows +9.2% improvement vs +0.2% for others."
            )
        
        # Rule 5: Hybrid tasks benefit from hybrid architecture
        if task.task_type == TaskType.HYBRID:
            return (
                Architecture.HYBRID,
                "Mixed sequential/parallel structure benefits from "
                "hierarchical oversight with flexible peer execution."
            )
        
        # Default: If moderately decomposable, use centralized
        if task.decomposability > 0.4:
            return (
                Architecture.CENTRALIZED,
                "Moderate decomposability suggests centralized coordination "
                "will provide benefits while containing error amplification to 4.4x."
            )
        
        return (
            Architecture.SINGLE_AGENT,
            "Low decomposability and task properties don't clearly favor "
            "multi-agent coordination. Single agent avoids overhead."
        )


def create_sample_tasks() -> list[Task]:
    """Create diverse sample tasks demonstrating different scenarios."""
    return [
        Task(
            name="Financial Analysis",
            task_type=TaskType.PARALLELIZABLE,
            subtasks=8,
            tool_count=3,
            decomposability=0.85,
            sequential_depth=1,
            base_difficulty=0.5,
        ),
        Task(
            name="Multi-step Planning",
            task_type=TaskType.SEQUENTIAL,
            subtasks=6,
            tool_count=2,
            decomposability=0.2,
            sequential_depth=6,
            base_difficulty=0.6,
        ),
        Task(
            name="Code Generation",
            task_type=TaskType.TOOL_HEAVY,
            subtasks=10,
            tool_count=16,
            decomposability=0.5,
            sequential_depth=3,
            base_difficulty=0.7,
        ),
        Task(
            name="Web Research",
            task_type=TaskType.PARALLELIZABLE,
            subtasks=12,
            tool_count=4,
            decomposability=0.75,
            sequential_depth=2,
            base_difficulty=0.4,
        ),
        Task(
            name="Document Processing",
            task_type=TaskType.HYBRID,
            subtasks=8,
            tool_count=6,
            decomposability=0.6,
            sequential_depth=3,
            base_difficulty=0.5,
        ),
    ]


async def run_full_comparison(task: Task, verbose: bool = True) -> dict:
    """Run all architectures against a task and compare results."""
    simulator = AgentSimulator(agent_count=4, verbose=verbose)
    predictor = ArchitecturePredictor()
    
    results = {}
    for arch in Architecture:
        result = await simulator.run_simulation(arch, task)
        results[arch] = result
    
    # Get prediction
    predicted_arch, reasoning = predictor.predict(task)
    
    # Find actual best
    best_arch = max(results.keys(), key=lambda a: results[a].accuracy)
    
    return {
        "task": task.name,
        "task_type": task.task_type.value,
        "results": {
            arch.value: {
                "accuracy": results[arch].accuracy,
                "latency_ms": results[arch].latency_ms,
                "error_amplification": results[arch].error_amplification,
                "improvement_vs_single": results[arch].improvement_vs_single,
            }
            for arch in Architecture
        },
        "prediction": {
            "recommended": predicted_arch.value,
            "reasoning": reasoning,
            "actual_best": best_arch.value,
            "prediction_correct": predicted_arch == best_arch,
        },
    }


def print_summary(all_results: list[dict]):
    """Print a summary comparison table."""
    print(f"\n{Colors.BOLD}{'='*80}")
    print("SIMULATION SUMMARY - Agent Scaling Principles in Action")
    print(f"{'='*80}{Colors.END}\n")
    
    # Table header
    print(f"{'Task':<25} {'Type':<15} {'Best Arch':<15} {'Predicted':<15} {'Match':<8}")
    print("-" * 80)
    
    correct = 0
    for result in all_results:
        match = "✓" if result["prediction"]["prediction_correct"] else "✗"
        if result["prediction"]["prediction_correct"]:
            correct += 1
        color = Colors.GREEN if match == "✓" else Colors.RED
        
        print(f"{result['task']:<25} {result['task_type']:<15} "
              f"{result['prediction']['actual_best']:<15} "
              f"{result['prediction']['recommended']:<15} "
              f"{color}{match}{Colors.END}")
    
    print("-" * 80)
    print(f"\n{Colors.BOLD}Prediction Accuracy: {correct}/{len(all_results)} "
          f"({100*correct/len(all_results):.0f}%){Colors.END}")
    
    # Key findings
    print(f"\n{Colors.CYAN}KEY FINDINGS FROM SIMULATION:{Colors.END}")
    print("""
    1. PARALLELIZABLE TASKS: Centralized coordination shows significant gains
       when tasks can be decomposed into independent subtasks.
    
    2. SEQUENTIAL TASKS: Multi-agent architectures consistently degrade 
       performance by 39-70%. Single agent is optimal.
    
    3. ERROR AMPLIFICATION: Independent agents amplify errors 17.2x vs
       4.4x for centralized. Architecture choice affects reliability.
    
    4. TOOL-HEAVY TASKS: Coordination overhead compounds with tool count.
       Single agent often optimal for high tool density.
    
    5. THE MYTH OF "MORE AGENTS": Adding agents isn't universally better.
       Match architecture to task properties for optimal results.
    """)


async def main():
    """Main entry point."""
    print(f"""{Colors.BOLD}
╔═══════════════════════════════════════════════════════════════════════════╗
║                    AGENT SCALING SIMULATOR                                ║
║                                                                           ║
║  Based on Google Research: "Towards a Science of Scaling Agent Systems"  ║
║  Paper: https://arxiv.org/abs/2512.08296                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
{Colors.END}""")
    
    tasks = create_sample_tasks()
    all_results = []
    
    for task in tasks:
        print(f"\n{Colors.HEADER}{'#'*60}")
        print(f"# TASK: {task.name}")
        print(f"{'#'*60}{Colors.END}")
        
        result = await run_full_comparison(task, verbose=True)
        all_results.append(result)
        
        print(f"\n{Colors.YELLOW}Predictor says: {result['prediction']['recommended'].upper()}")
        print(f"Reason: {result['prediction']['reasoning']}{Colors.END}")
    
    print_summary(all_results)
    
    # Save results
    with open("simulation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{Colors.GREEN}Results saved to simulation_results.json{Colors.END}")
    
    return all_results


if __name__ == "__main__":
    asyncio.run(main())
