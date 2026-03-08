#!/usr/bin/env python3
"""
Cognitive Dark Matter (CDM) Benchmark Suite
Implementation of evaluation framework from "Cognitive Dark Matter: Measuring What AI Misses" (arXiv:2603.03414)

This POC demonstrates how to measure the "cognitive dark matter" - 
the hidden cognitive processes that traditional benchmarks miss.
"""

import json
import time
import random
import asyncio
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class CDMDomain(Enum):
    """Seven key domains of Cognitive Dark Matter"""
    METACOGNITION = "metacognition"
    COGNITIVE_FLEXIBILITY = "cognitive_flexibility"
    EPISODIC_MEMORY = "episodic_memory"
    LIFELONG_LEARNING = "lifelong_learning"
    ABDUCTIVE_REASONING = "abductive_reasoning"
    SOCIAL_REASONING = "social_reasoning"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"

@dataclass
class CDMTask:
    """A task designed to measure cognitive dark matter"""
    domain: CDMDomain
    task_id: str
    prompt: str
    process_requirements: List[str]  # What cognitive processes should be tracked
    expected_process: Dict[str, Any]  # Expected cognitive process pattern
    surface_answer: str  # What traditional benchmarks would measure
    cdm_indicators: List[str]  # What CDM framework should capture

class CDMBenchmark:
    """Cognitive Dark Matter Benchmark Framework"""
    
    def __init__(self):
        self.tasks = self._initialize_tasks()
        self.results = []
        
    def _initialize_tasks(self) -> List[CDMTask]:
        """Initialize a set of CDM evaluation tasks"""
        return [
            CDMTask(
                domain=CDMDomain.METACOGNITION,
                task_id="meta_confidence",
                prompt="Solve this math problem: What is 17 × 23? Also, rate your confidence in this answer on a scale of 1-10 and explain your reasoning process.",
                process_requirements=["confidence_calibration", "process_awareness", "uncertainty_estimation"],
                expected_process={
                    "shows_working": True,
                    "expresses_confidence": True,
                    "reflects_on_process": True
                },
                surface_answer="391",
                cdm_indicators=["confidence_rating", "process_explanation", "uncertainty_awareness"]
            ),
            CDMTask(
                domain=CDMDomain.COGNITIVE_FLEXIBILITY,
                task_id="perspective_switch",
                prompt="A glass is filled halfway with water. Describe this situation from three different perspectives: a pessimist, an optimist, and a physicist.",
                process_requirements=["perspective_taking", "context_switching", "frame_shifting"],
                expected_process={
                    "multiple_frameworks": True,
                    "smooth_transitions": True,
                    "maintains_coherence": True
                },
                surface_answer="Three descriptions provided",
                cdm_indicators=["perspective_depth", "transition_quality", "framework_consistency"]
            ),
            CDMTask(
                domain=CDMDomain.EPISODIC_MEMORY,
                task_id="temporal_context",
                prompt="I told you earlier that my favorite color is blue. What does this tell you about our conversation history?",
                process_requirements=["temporal_reasoning", "conversation_tracking", "context_integration"],
                expected_process={
                    "acknowledges_limitation": True,
                    "reasons_about_context": True,
                    "maintains_coherence": True
                },
                surface_answer="No prior information available",
                cdm_indicators=["temporal_awareness", "context_reasoning", "limitation_acknowledgment"]
            ),
            CDMTask(
                domain=CDMDomain.ABDUCTIVE_REASONING,
                task_id="best_explanation",
                prompt="You find wet footprints leading from the door to the kitchen, but it's not raining. Generate three possible explanations and identify the most likely one.",
                process_requirements=["hypothesis_generation", "likelihood_assessment", "causal_reasoning"],
                expected_process={
                    "multiple_hypotheses": True,
                    "evaluates_likelihood": True,
                    "considers_evidence": True
                },
                surface_answer="Three explanations with likelihood ranking",
                cdm_indicators=["hypothesis_creativity", "evidence_consideration", "reasoning_quality"]
            ),
            CDMTask(
                domain=CDMDomain.SOCIAL_REASONING,
                task_id="implicit_social",
                prompt="Sarah says 'Nice haircut!' to Mark with a particular tone. Mark looks uncomfortable. What might be happening here?",
                process_requirements=["tone_inference", "social_dynamics", "implicit_communication"],
                expected_process={
                    "infers_tone": True,
                    "considers_context": True,
                    "understands_subtext": True
                },
                surface_answer="Analysis of social interaction",
                cdm_indicators=["tone_sensitivity", "social_awareness", "subtext_understanding"]
            ),
            CDMTask(
                domain=CDMDomain.EMOTIONAL_INTELLIGENCE,
                task_id="emotional_regulation",
                prompt="Describe a strategy for staying calm during a heated argument, including both what to do and what internal processes help maintain emotional control.",
                process_requirements=["emotional_awareness", "regulation_strategies", "process_monitoring"],
                expected_process={
                    "identifies_emotions": True,
                    "provides_strategies": True,
                    "explains_mechanisms": True
                },
                surface_answer="Emotional regulation strategies",
                cdm_indicators=["emotion_awareness", "strategy_depth", "mechanism_understanding"]
            )
        ]
    
    async def evaluate_ai_system(self, ai_function) -> Dict[str, Any]:
        """
        Evaluate an AI system using CDM framework
        ai_function should be an async function that takes a prompt and returns a response
        """
        results = {
            "timestamp": time.time(),
            "domain_scores": {},
            "overall_cdm_score": 0,
            "detailed_results": []
        }
        
        total_cdm_score = 0
        
        for task in self.tasks:
            print(f"Evaluating {task.domain.value}: {task.task_id}")
            
            # Get AI response
            start_time = time.time()
            ai_response = await ai_function(task.prompt)
            response_time = time.time() - start_time
            
            # Evaluate CDM aspects
            cdm_evaluation = self._evaluate_cdm_response(task, ai_response)
            
            task_result = {
                "task": task.task_id,
                "domain": task.domain.value,
                "ai_response": ai_response,
                "response_time": response_time,
                "cdm_score": cdm_evaluation["cdm_score"],
                "process_indicators": cdm_evaluation["process_indicators"],
                "missing_cdm": cdm_evaluation["missing_cdm"]
            }
            
            results["detailed_results"].append(task_result)
            
            if task.domain.value not in results["domain_scores"]:
                results["domain_scores"][task.domain.value] = []
            results["domain_scores"][task.domain.value].append(cdm_evaluation["cdm_score"])
            
            total_cdm_score += cdm_evaluation["cdm_score"]
        
        # Calculate overall scores
        results["overall_cdm_score"] = total_cdm_score / len(self.tasks)
        
        # Calculate domain averages
        for domain in results["domain_scores"]:
            scores = results["domain_scores"][domain]
            results["domain_scores"][domain] = sum(scores) / len(scores)
        
        return results
    
    def _evaluate_cdm_response(self, task: CDMTask, response: str) -> Dict[str, Any]:
        """Evaluate how well a response captures cognitive dark matter"""
        cdm_score = 0
        process_indicators = {}
        missing_cdm = []
        
        # Simple heuristic evaluation (in real implementation, would use more sophisticated analysis)
        response_lower = response.lower()
        
        for indicator in task.cdm_indicators:
            indicator_present = self._check_indicator(indicator, response_lower, task)
            process_indicators[indicator] = indicator_present
            if indicator_present:
                cdm_score += 1
        
        # Normalize score
        max_possible = len(task.cdm_indicators)
        cdm_score = (cdm_score / max_possible) * 100 if max_possible > 0 else 0
        
        # Identify missing CDM elements
        for req in task.process_requirements:
            if not self._check_process_requirement(req, response_lower):
                missing_cdm.append(req)
        
        return {
            "cdm_score": cdm_score,
            "process_indicators": process_indicators,
            "missing_cdm": missing_cdm
        }
    
    def _check_indicator(self, indicator: str, response: str, task: CDMTask) -> bool:
        """Check if a CDM indicator is present in the response"""
        indicator_patterns = {
            "confidence_rating": ["confident", "sure", "certain", "unsure", "scale", "rate"],
            "process_explanation": ["because", "reasoning", "process", "step", "think"],
            "uncertainty_awareness": ["uncertain", "might", "could", "possibly", "unsure"],
            "perspective_depth": ["perspective", "viewpoint", "sees", "views", "from"],
            "transition_quality": ["however", "whereas", "on the other hand", "alternatively"],
            "framework_consistency": ["consistent", "logical", "coherent"],
            "temporal_awareness": ["earlier", "before", "history", "previous", "context"],
            "context_reasoning": ["conversation", "context", "information", "told"],
            "limitation_acknowledgment": ["don't", "cannot", "unable", "no information"],
            "hypothesis_creativity": ["possible", "could", "might", "explanation", "theory"],
            "evidence_consideration": ["evidence", "suggests", "indicates", "supports"],
            "reasoning_quality": ["therefore", "because", "thus", "since", "leads"],
            "tone_sensitivity": ["tone", "sarcastic", "meaning", "implies"],
            "social_awareness": ["social", "relationship", "interaction", "dynamics"],
            "subtext_understanding": ["really", "actually", "implies", "suggests"],
            "emotion_awareness": ["emotion", "feeling", "angry", "calm", "upset"],
            "strategy_depth": ["strategy", "approach", "method", "technique"],
            "mechanism_understanding": ["because", "mechanism", "works", "helps"]
        }
        
        patterns = indicator_patterns.get(indicator, [indicator])
        return any(pattern in response for pattern in patterns)
    
    def _check_process_requirement(self, requirement: str, response: str) -> bool:
        """Check if a process requirement is met"""
        # Simplified check - real implementation would be more sophisticated
        requirement_patterns = {
            "confidence_calibration": ["confident", "certain", "sure", "scale"],
            "process_awareness": ["process", "think", "reasoning", "step"],
            "uncertainty_estimation": ["uncertain", "unsure", "might", "could"],
            "perspective_taking": ["perspective", "viewpoint", "sees"],
            "context_switching": ["different", "alternative", "another"],
            "frame_shifting": ["framework", "view", "approach"],
            "temporal_reasoning": ["time", "earlier", "before", "sequence"],
            "conversation_tracking": ["conversation", "told", "said"],
            "context_integration": ["context", "information", "together"],
            "hypothesis_generation": ["hypothesis", "theory", "explanation", "could"],
            "likelihood_assessment": ["likely", "probable", "chance", "most"],
            "causal_reasoning": ["because", "cause", "reason", "leads"],
            "tone_inference": ["tone", "sarcastic", "meaning"],
            "social_dynamics": ["social", "relationship", "interaction"],
            "implicit_communication": ["implies", "suggests", "really means"],
            "emotional_awareness": ["emotion", "feeling", "emotional"],
            "regulation_strategies": ["strategy", "control", "manage"],
            "process_monitoring": ["monitor", "awareness", "notice"]
        }
        
        patterns = requirement_patterns.get(requirement, [requirement])
        return any(pattern in response for pattern in patterns)

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive CDM evaluation report"""
        report = f"""
COGNITIVE DARK MATTER (CDM) EVALUATION REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}

OVERALL CDM SCORE: {results['overall_cdm_score']:.1f}/100

DOMAIN BREAKDOWN:
"""
        
        for domain, score in results['domain_scores'].items():
            report += f"  {domain.replace('_', ' ').title()}: {score:.1f}/100\n"
        
        report += f"""
DETAILED ANALYSIS:

Traditional benchmarks focus on final answers and surface-level performance.
CDM evaluation reveals the hidden cognitive processes that drive intelligence.

Key Findings:
"""
        
        # Analyze patterns
        high_performers = [d for d, s in results['domain_scores'].items() if s >= 70]
        low_performers = [d for d, s in results['domain_scores'].items() if s < 40]
        
        if high_performers:
            report += f"  STRENGTHS: Strong performance in {', '.join(high_performers)}\n"
        if low_performers:
            report += f"  WEAKNESSES: Significant gaps in {', '.join(low_performers)}\n"
        
        report += f"""
COGNITIVE PROCESS GAPS:
"""
        
        all_missing = []
        for result in results['detailed_results']:
            all_missing.extend(result['missing_cdm'])
        
        missing_counts = {}
        for missing in all_missing:
            missing_counts[missing] = missing_counts.get(missing, 0) + 1
        
        for process, count in sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"  {process.replace('_', ' ').title()}: Missing in {count} tasks\n"
        
        report += f"""
TASK DETAILS:
"""
        
        for result in results['detailed_results']:
            report += f"""
  {result['task'].upper()} ({result['domain']})
    CDM Score: {result['cdm_score']:.1f}/100
    Response Time: {result['response_time']:.2f}s
    Missing CDM: {', '.join(result['missing_cdm']) if result['missing_cdm'] else 'None'}
"""
        
        return report

# Demo AI systems for testing
async def simple_ai_system(prompt: str) -> str:
    """Simulates a basic AI system that gives correct but surface-level answers"""
    await asyncio.sleep(0.1)  # Simulate processing time
    
    responses = {
        "What is 17 × 23?": "391",
        "glass": "A pessimist sees it as half empty, an optimist sees it as half full, and a physicist sees 500ml of H2O in a 1L container.",
        "favorite color": "I don't have information about your favorite color from our conversation history.",
        "wet footprints": "Possible explanations: 1) Someone came in from rain, 2) Wet shoes from puddles, 3) Spilled water. Most likely: wet shoes from outside.",
        "Nice haircut": "This could be genuine praise, sarcasm, or social awkwardness. Mark's discomfort suggests it might be sarcastic or poorly timed.",
        "heated argument": "Take deep breaths, listen actively, speak calmly, and take breaks if needed to maintain emotional control."
    }
    
    for key, response in responses.items():
        if key.lower() in prompt.lower():
            return response
    
    return "I understand the question but need more context to provide a meaningful response."

async def advanced_ai_system(prompt: str) -> str:
    """Simulates a more advanced AI system with better CDM awareness"""
    await asyncio.sleep(0.2)  # Simulate processing time
    
    if "17 × 23" in prompt:
        return """I need to calculate 17 × 23. Let me work through this step by step:
        17 × 23 = 17 × (20 + 3) = (17 × 20) + (17 × 3) = 340 + 51 = 391
        
        I'm quite confident in this answer (8/10) because I used a systematic approach and double-checked the arithmetic. The distributive property makes this calculation reliable."""
    
    elif "glass" in prompt and "water" in prompt:
        return """I'll approach this from three distinct perspectives:
        
        PESSIMIST: 'The glass is half empty - we've already lost half the water, and it's probably going to evaporate or spill soon anyway.'
        
        OPTIMIST: 'The glass is half full! There's plenty of refreshing water to enjoy, and we can always add more.'
        
        PHYSICIST: 'The container holds approximately 500ml of water at room temperature, with the liquid level at 50% capacity. The remaining volume contains air at standard atmospheric pressure.'
        
        Each perspective reveals different cognitive frameworks - emotional valence, future projection, and empirical measurement."""
    
    elif "favorite color" in prompt:
        return """I notice you're referencing something you supposedly told me earlier about your favorite color being blue. However, I don't have access to any previous conversation history - each interaction starts fresh for me. 
        
        This highlights an interesting limitation: while I can reason about hypothetical past conversations, I cannot actually maintain episodic memories across sessions. If you did tell me your favorite color was blue, I would need you to remind me in our current conversation."""
    
    elif "wet footprints" in prompt:
        return """Let me generate multiple explanations and evaluate their likelihood:
        
        HYPOTHESIS 1: Someone came inside with wet shoes (from puddles, sprinklers, or cleaning)
        HYPOTHESIS 2: Someone tracked water from bathroom/sink area
        HYPOTHESIS 3: A pet with wet paws walked through
        
        MOST LIKELY: Wet shoes from outside sources (puddles, lawn sprinklers, car washing). This explains the door-to-kitchen path and doesn't require rain. People often don't realize their shoes are wet from non-rain sources.
        
        Evidence considered: Path direction, absence of rain, typical household traffic patterns."""
    
    elif "Nice haircut" in prompt:
        return """This interaction has several layers of potential meaning:
        
        Sarah's comment could be genuinely complimentary, but Mark's discomfort suggests there might be subtext. Possible scenarios:
        1) SARCASTIC TONE: Sarah might be being passive-aggressive about a bad haircut
        2) POOR TIMING: Even genuine compliments can feel awkward if delivered at wrong moment
        3) SOCIAL DYNAMICS: Their relationship history might make compliments feel loaded
        
        Mark's discomfort is the key signal here - it suggests he's reading something in Sarah's tone or context that feels off, even if the words themselves are positive."""
    
    elif "heated argument" in prompt:
        return """Emotional regulation during conflict involves both behavioral strategies and internal cognitive processes:
        
        BEHAVIORAL STRATEGIES:
        - Slow, deep breathing to activate parasympathetic nervous system
        - Lower your voice (others often mirror this)
        - Take strategic pauses before responding
        
        INTERNAL PROCESSES:
        - Cognitive reappraisal: reframe the situation as problem-solving vs. personal attack
        - Emotional labeling: mentally note 'I'm feeling angry' to create psychological distance
        - Perspective-taking: consider the other person's underlying needs or fears
        
        The key mechanism is that conscious awareness of emotional state creates space between feeling and reaction, allowing prefrontal cortex to override amygdala-driven responses."""
    
    return "I recognize this as a complex question that requires careful consideration of multiple cognitive processes. Let me think through the various dimensions involved..."

async def main():
    """Demo the CDM benchmark system"""
    print("COGNITIVE DARK MATTER (CDM) BENCHMARK DEMO")
    print("==========================================")
    print()
    print("Based on 'Cognitive Dark Matter: Measuring What AI Misses' (arXiv:2603.03414)")
    print("This benchmark evaluates the hidden cognitive processes traditional metrics miss.")
    print()
    
    benchmark = CDMBenchmark()
    
    # Test both AI systems
    print("Testing Simple AI System...")
    simple_results = await benchmark.evaluate_ai_system(simple_ai_system)
    
    print("\nTesting Advanced AI System...")
    advanced_results = await benchmark.evaluate_ai_system(advanced_ai_system)
    
    # Generate reports
    print("\n" + "="*80)
    print("SIMPLE AI SYSTEM RESULTS:")
    print("="*80)
    print(benchmark.generate_report(simple_results))
    
    print("\n" + "="*80)
    print("ADVANCED AI SYSTEM RESULTS:")
    print("="*80)
    print(benchmark.generate_report(advanced_results))
    
    # Save results
    with open('cdm_benchmark_results.json', 'w') as f:
        json.dump({
            'simple_ai': simple_results,
            'advanced_ai': advanced_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to cdm_benchmark_results.json")
    
    # Summary comparison
    print(f"\nCDM PERFORMANCE COMPARISON:")
    print(f"Simple AI:   {simple_results['overall_cdm_score']:.1f}/100")
    print(f"Advanced AI: {advanced_results['overall_cdm_score']:.1f}/100")
    print(f"CDM Gap:     {advanced_results['overall_cdm_score'] - simple_results['overall_cdm_score']:.1f} points")

if __name__ == "__main__":
    asyncio.run(main())
