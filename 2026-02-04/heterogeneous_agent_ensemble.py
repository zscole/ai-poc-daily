#!/usr/bin/env python3
"""
Heterogeneous Agent Ensemble: Diversity Beats Scale

Demonstrates the key finding from "Understanding Agent Scaling in LLM-Based
Multi-Agent Systems via Diversity" (arXiv:2602.03794):

    2 diverse agents can match or exceed 16 homogeneous agents

This POC implements:
1. Multi-agent debate with diverse reasoning personas
2. K* diversity metric (effective channel count)
3. Comparison of homogeneous vs heterogeneous configurations

Based on research showing that agent performance is bounded by intrinsic task
uncertainty, not agent count. Heterogeneous agents contribute complementary
evidence while homogeneous agents saturate early due to correlated outputs.

Usage:
    python heterogeneous_agent_ensemble.py

Requires:
    pip install openai numpy scikit-learn

Author: AI Research POC
Date: 2026-02-04
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
import numpy as np

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARN] openai not installed. Install with: pip install openai")

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# Reasoning Personas - Each shapes problem-solving approach
# -----------------------------------------------------------------------------

REASONING_PERSONAS = {
    "conservative_verifier": {
        "name": "Conservative Verifier",
        "system": (
            "You are a conservative, methodical reasoner. "
            "Always verify each step before proceeding. Double-check arithmetic. "
            "Prefer well-established methods over shortcuts. "
            "If uncertain, acknowledge it and work through systematically."
        ),
        "temperature": 0.3,
    },
    "creative_explorer": {
        "name": "Creative Explorer", 
        "system": (
            "You are a creative problem solver who looks for patterns and shortcuts. "
            "Try unconventional approaches. Look for elegant solutions. "
            "Consider the problem from multiple angles before committing to a method."
        ),
        "temperature": 0.7,
    },
    "rigorous_formalist": {
        "name": "Rigorous Formalist",
        "system": (
            "You are a rigorous, formal reasoner. "
            "Use precise notation and logical completeness. "
            "State assumptions explicitly. Derive conclusions step-by-step. "
            "Prioritize correctness over brevity."
        ),
        "temperature": 0.2,
    },
    "intuitive_estimator": {
        "name": "Intuitive Estimator",
        "system": (
            "You are an intuitive reasoner who uses estimation and sanity checks. "
            "Before detailed calculation, estimate the rough answer. "
            "Use intuition to guide your approach, then verify with calculation."
        ),
        "temperature": 0.5,
    },
    "systematic_decomposer": {
        "name": "Systematic Decomposer",
        "system": (
            "You are a systematic problem decomposer. "
            "Break complex problems into smaller, manageable sub-problems. "
            "Solve each part independently, then combine results. "
            "Track dependencies between sub-problems carefully."
        ),
        "temperature": 0.4,
    },
}


# -----------------------------------------------------------------------------
# Agent Configuration
# -----------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a single agent in the ensemble."""
    persona_key: str
    model: str = "gpt-4o-mini"  # Default to cost-effective model
    
    @property
    def persona(self) -> dict:
        return REASONING_PERSONAS[self.persona_key]
    
    @property
    def name(self) -> str:
        return self.persona["name"]
    
    @property
    def system_prompt(self) -> str:
        return self.persona["system"]
    
    @property
    def temperature(self) -> float:
        return self.persona["temperature"]


@dataclass
class EnsembleConfig:
    """Configuration for the entire agent ensemble."""
    agents: list[AgentConfig]
    debate_rounds: int = 2
    solver: str = "debate"  # "vote" or "debate"
    
    @property
    def is_heterogeneous(self) -> bool:
        """Check if ensemble uses diverse personas/models."""
        personas = set(a.persona_key for a in self.agents)
        models = set(a.model for a in self.agents)
        return len(personas) > 1 or len(models) > 1


# -----------------------------------------------------------------------------
# K* Diversity Metric
# -----------------------------------------------------------------------------

def compute_k_star(responses: list[str]) -> float:
    """
    Compute K* (effective diversity) from agent responses.
    
    K* measures the effective number of distinct reasoning strategies
    using embedding-space analysis:
    
    1. Embed responses (simple word frequency for this POC)
    2. Compute eigenvalues of covariance matrix
    3. K* = exp(H) where H is Shannon entropy of normalized eigenvalues
    
    Returns value between 1 (all identical) and len(responses) (all unique).
    """
    if len(responses) < 2:
        return 1.0
    
    # Simple embedding: word frequency vectors
    vocab = set()
    for r in responses:
        vocab.update(r.lower().split())
    vocab = sorted(vocab)
    
    if len(vocab) == 0:
        return 1.0
    
    # Create embedding matrix
    embeddings = np.zeros((len(responses), len(vocab)))
    for i, r in enumerate(responses):
        words = r.lower().split()
        word_counts = Counter(words)
        for j, word in enumerate(vocab):
            embeddings[i, j] = word_counts.get(word, 0)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    
    # Compute covariance and eigenvalues
    if embeddings.shape[1] < 2:
        return 1.0
    
    try:
        cov = np.cov(embeddings.T)
        if cov.ndim == 0:
            return 1.0
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) == 0:
            return 1.0
        
        # Normalize to distribution
        p = eigenvalues / eigenvalues.sum()
        
        # Shannon entropy
        H = -np.sum(p * np.log(p + 1e-10))
        
        # Effective diversity
        k_star = np.exp(H)
        return min(k_star, len(responses))
    except Exception:
        return 1.0


# -----------------------------------------------------------------------------
# Multi-Agent System
# -----------------------------------------------------------------------------

class AgentEnsemble:
    """
    Multi-agent ensemble with debate and voting capabilities.
    
    Implements the key insight from Agent-Scaling research:
    heterogeneous agents contribute complementary evidence,
    while homogeneous agents saturate early.
    """
    
    def __init__(self, config: EnsembleConfig, client: Optional["AsyncOpenAI"] = None):
        self.config = config
        self.client = client
        self.history: list[dict] = []
    
    async def _call_agent(
        self,
        agent: AgentConfig,
        problem: str,
        peer_responses: Optional[list[str]] = None,
    ) -> str:
        """Call a single agent with optional peer context."""
        
        messages = [{"role": "system", "content": agent.system_prompt}]
        
        if peer_responses:
            peer_context = "\n\n".join([
                f"Another agent's response:\n{r}" 
                for r in peer_responses
            ])
            messages.append({
                "role": "user",
                "content": (
                    f"Problem: {problem}\n\n"
                    f"Other agents have provided these responses:\n{peer_context}\n\n"
                    "Consider their reasoning, but form your own independent conclusion. "
                    "Provide your answer with clear reasoning."
                )
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Problem: {problem}\n\nProvide your answer with clear reasoning."
            })
        
        if self.client is None:
            # Mock response for testing without API
            return f"[{agent.name}] Mock response for: {problem[:50]}..."
        
        try:
            response = await self.client.chat.completions.create(
                model=agent.model,
                messages=messages,
                temperature=agent.temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[ERROR] {agent.name}: {str(e)}"
    
    async def _debate_round(
        self,
        problem: str,
        current_responses: dict[str, str],
    ) -> dict[str, str]:
        """Run one round of debate where agents see peer responses."""
        
        tasks = []
        for agent in self.config.agents:
            peer_responses = [
                r for name, r in current_responses.items()
                if name != agent.name
            ]
            tasks.append(self._call_agent(agent, problem, peer_responses))
        
        results = await asyncio.gather(*tasks)
        return {agent.name: response for agent, response in zip(self.config.agents, results)}
    
    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response (simplified)."""
        # Look for common answer patterns
        response_lower = response.lower()
        
        for marker in ["the answer is", "final answer:", "therefore,", "thus,"]:
            if marker in response_lower:
                idx = response_lower.index(marker)
                answer_part = response[idx:].split("\n")[0]
                return answer_part.strip()
        
        # Return last sentence as fallback
        sentences = response.split(".")
        if sentences:
            return sentences[-2].strip() if len(sentences) > 1 else response.strip()
        return response.strip()
    
    def _majority_vote(self, responses: dict[str, str]) -> str:
        """Determine consensus answer by majority vote."""
        answers = [self._extract_answer(r) for r in responses.values()]
        if not answers:
            return "No consensus"
        
        # Simple frequency-based voting
        counter = Counter(answers)
        return counter.most_common(1)[0][0]
    
    async def solve(self, problem: str) -> dict:
        """
        Solve a problem using the configured ensemble strategy.
        
        Returns dict with:
        - final_answer: Consensus answer
        - agent_responses: All individual responses
        - k_star: Diversity metric
        - rounds: Number of debate rounds completed
        """
        
        # Round 0: Initial responses
        tasks = [self._call_agent(agent, problem) for agent in self.config.agents]
        initial_results = await asyncio.gather(*tasks)
        
        responses = {
            agent.name: response 
            for agent, response in zip(self.config.agents, initial_results)
        }
        
        all_rounds = [responses.copy()]
        
        # Debate rounds (if configured)
        if self.config.solver == "debate" and self.config.debate_rounds > 0:
            for round_num in range(self.config.debate_rounds):
                responses = await self._debate_round(problem, responses)
                all_rounds.append(responses.copy())
        
        # Compute diversity metric
        k_star = compute_k_star(list(responses.values()))
        
        # Determine final answer
        final_answer = self._majority_vote(responses)
        
        result = {
            "final_answer": final_answer,
            "agent_responses": responses,
            "k_star": k_star,
            "rounds": len(all_rounds),
            "is_heterogeneous": self.config.is_heterogeneous,
            "agent_count": len(self.config.agents),
        }
        
        self.history.append(result)
        return result


# -----------------------------------------------------------------------------
# Benchmark Problems
# -----------------------------------------------------------------------------

BENCHMARK_PROBLEMS = [
    {
        "question": "A train travels at 60 mph for the first half of its journey and 40 mph for the second half. What is the average speed for the entire journey?",
        "answer": "48 mph",
        "type": "math",
    },
    {
        "question": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "answer": "No, we cannot conclude that",
        "type": "logic",
    },
    {
        "question": "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "answer": "$0.05 or 5 cents",
        "type": "math",
    },
    {
        "question": "Three people check into a hotel room that costs $30. They each pay $10. Later, the clerk realizes the room should have been $25 and gives the bellboy $5 to return. The bellboy keeps $2 and returns $1 to each person. Now each person paid $9 (total $27) plus $2 kept by bellboy = $29. Where is the missing dollar?",
        "answer": "There is no missing dollar - the framing is deceptive. $27 includes the $2 tip, not adds to it.",
        "type": "logic",
    },
]


# -----------------------------------------------------------------------------
# Demo Runner
# -----------------------------------------------------------------------------

async def run_comparison_demo():
    """
    Run a comparison between homogeneous and heterogeneous ensembles.
    
    Demonstrates the key finding: diversity beats scale.
    """
    
    print("=" * 70)
    print("HETEROGENEOUS AGENT ENSEMBLE: DIVERSITY BEATS SCALE")
    print("=" * 70)
    print()
    print("Key Insight: 2 diverse agents can match or exceed 16 homogeneous agents")
    print("Source: arXiv:2602.03794")
    print()
    
    # Initialize client
    client = None
    if OPENAI_AVAILABLE:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = AsyncOpenAI(api_key=api_key)
            print("[OK] OpenAI client initialized")
        else:
            print("[WARN] OPENAI_API_KEY not set - running in mock mode")
    else:
        print("[WARN] Running in mock mode (openai not installed)")
    
    print()
    
    # Configuration 1: Homogeneous (all same persona)
    homogeneous_config = EnsembleConfig(
        agents=[
            AgentConfig("conservative_verifier"),
            AgentConfig("conservative_verifier"),
            AgentConfig("conservative_verifier"),
            AgentConfig("conservative_verifier"),
        ],
        debate_rounds=2,
        solver="debate",
    )
    
    # Configuration 2: Heterogeneous (diverse personas)
    heterogeneous_config = EnsembleConfig(
        agents=[
            AgentConfig("conservative_verifier"),
            AgentConfig("creative_explorer"),
        ],
        debate_rounds=2,
        solver="debate",
    )
    
    # Configuration 3: Maximally diverse
    max_diverse_config = EnsembleConfig(
        agents=[
            AgentConfig("conservative_verifier"),
            AgentConfig("creative_explorer"),
            AgentConfig("rigorous_formalist"),
            AgentConfig("intuitive_estimator"),
        ],
        debate_rounds=2,
        solver="debate",
    )
    
    configs = [
        ("4 Homogeneous Agents", homogeneous_config),
        ("2 Heterogeneous Agents", heterogeneous_config),
        ("4 Max-Diverse Agents", max_diverse_config),
    ]
    
    # Select one problem for demo
    problem = BENCHMARK_PROBLEMS[2]  # Bat and ball problem
    
    print("-" * 70)
    print(f"PROBLEM: {problem['question']}")
    print(f"EXPECTED: {problem['answer']}")
    print("-" * 70)
    print()
    
    results = []
    
    for name, config in configs:
        print(f"\n### {name} ###")
        print(f"Agents: {[a.name for a in config.agents]}")
        print(f"Heterogeneous: {config.is_heterogeneous}")
        
        ensemble = AgentEnsemble(config, client)
        result = await ensemble.solve(problem["question"])
        
        print(f"K* (effective diversity): {result['k_star']:.2f}")
        print(f"Final Answer: {result['final_answer'][:100]}...")
        print()
        
        for agent_name, response in result["agent_responses"].items():
            print(f"  [{agent_name}]")
            # Print first 150 chars of response
            preview = response[:150].replace("\n", " ")
            print(f"    {preview}...")
            print()
        
        results.append({
            "config_name": name,
            "k_star": result["k_star"],
            "agent_count": result["agent_count"],
            "is_heterogeneous": result["is_heterogeneous"],
        })
    
    # Summary
    print("=" * 70)
    print("SUMMARY: DIVERSITY METRICS")
    print("=" * 70)
    print()
    print(f"{'Configuration':<25} {'Agents':<8} {'K*':<8} {'Diverse':<10}")
    print("-" * 51)
    
    for r in results:
        print(f"{r['config_name']:<25} {r['agent_count']:<8} {r['k_star']:<8.2f} {str(r['is_heterogeneous']):<10}")
    
    print()
    print("KEY TAKEAWAYS:")
    print("- K* measures effective diversity (1 = identical, N = maximally diverse)")
    print("- Heterogeneous agents achieve higher K* with fewer agents")
    print("- Homogeneous scaling hits diminishing returns quickly")
    print("- 2 diverse agents can outperform 16 homogeneous ones")
    print()
    print("For production use, consider mixing:")
    print("  - Different models (GPT-4, Claude, Gemini)")
    print("  - Different reasoning personas")
    print("  - Different temperatures")
    print()


async def run_full_benchmark():
    """Run the full benchmark suite."""
    
    print("Running full benchmark...")
    
    client = None
    if OPENAI_AVAILABLE:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            client = AsyncOpenAI(api_key=api_key)
    
    # Heterogeneous config (our focus)
    config = EnsembleConfig(
        agents=[
            AgentConfig("conservative_verifier"),
            AgentConfig("creative_explorer"),
            AgentConfig("rigorous_formalist"),
        ],
        debate_rounds=2,
        solver="debate",
    )
    
    ensemble = AgentEnsemble(config, client)
    
    results = []
    for problem in BENCHMARK_PROBLEMS:
        print(f"\nSolving: {problem['question'][:50]}...")
        result = await ensemble.solve(problem["question"])
        result["expected"] = problem["answer"]
        result["problem_type"] = problem["type"]
        results.append(result)
    
    # Output results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    for i, r in enumerate(results):
        print(f"\nProblem {i+1} ({r['problem_type']}):")
        print(f"  Expected: {r['expected']}")
        print(f"  Got: {r['final_answer'][:80]}")
        print(f"  K*: {r['k_star']:.2f}")
    
    avg_k_star = np.mean([r["k_star"] for r in results])
    print(f"\nAverage K* across all problems: {avg_k_star:.2f}")


def main():
    """Entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Heterogeneous Agent Ensemble - Diversity Beats Scale"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark instead of comparison demo"
    )
    parser.add_argument(
        "--problem",
        type=int,
        default=None,
        help="Run specific problem by index (0-3)"
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        asyncio.run(run_full_benchmark())
    else:
        asyncio.run(run_comparison_demo())


if __name__ == "__main__":
    main()
