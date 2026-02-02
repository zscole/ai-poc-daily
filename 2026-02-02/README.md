# Agent Scaling Simulator

**Date**: February 2, 2026  
**Paper**: [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296) (Google Research)

## 🔥 Why This Matters

The AI industry is obsessed with multi-agent systems. Papers like "More Agents Is All You Need" claimed that adding agents consistently improves results. **Google Research just proved this is a myth.**

Their 180-configuration study reveals:
- Centralized multi-agent coordination improves parallelizable tasks by **80.9%**
- But sequential reasoning tasks **degrade by 39-70%** with multi-agent
- Independent agents amplify errors **17.2x** vs **4.4x** for centralized
- Tool-heavy tasks suffer disproportionately from coordination overhead

This POC demonstrates these principles so you can make informed architecture decisions.

## 📊 Key Findings Visualized

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WHEN MULTI-AGENT HELPS                │ WHEN MULTI-AGENT HURTS              │
├───────────────────────────────────────┼─────────────────────────────────────┤
│ ✅ Parallelizable tasks               │ ❌ Sequential reasoning              │
│ ✅ High decomposability (>0.7)        │ ❌ Tool-heavy tasks (>10 tools)     │
│ ✅ Financial analysis, research       │ ❌ Multi-step planning              │
│ ✅ Centralized coordination           │ ❌ Independent (no orchestrator)    │
│    → +80.9% improvement               │    → 39-70% degradation             │
└───────────────────────────────────────┴─────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# No dependencies required (Python 3.8+)
python3 agent_scaling_simulator.py
```

## 🛠️ What It Does

1. **Simulates 5 Agent Architectures**:
   - Single Agent (baseline)
   - Independent (parallel, no communication)
   - Centralized (hub-and-spoke with orchestrator)
   - Decentralized (peer-to-peer mesh)
   - Hybrid (hierarchical + peer coordination)

2. **Tests Against 5 Task Types**:
   - Financial Analysis (parallelizable)
   - Multi-step Planning (sequential)
   - Code Generation (tool-heavy)
   - Web Research (parallelizable)
   - Document Processing (hybrid)

3. **Predicts Optimal Architecture** based on task properties:
   - Decomposability score
   - Tool count
   - Sequential depth
   - Task type

## 📈 Sample Output

```
SIMULATION SUMMARY - Agent Scaling Principles in Action
================================================================================

Task                      Type            Best Arch       Predicted       Match
--------------------------------------------------------------------------------
Financial Analysis        parallelizable  centralized     centralized     ✓
Multi-step Planning       sequential      single_agent    single_agent    ✓
Code Generation           tool_heavy      single_agent    single_agent    ✓
Web Research              parallelizable  single_agent    centralized     ✗
Document Processing       hybrid          hybrid          hybrid          ✓
--------------------------------------------------------------------------------

Prediction Accuracy: 4/5 (80%)
```

## 🧠 The Science

### Error Amplification by Architecture

| Architecture | Error Amplification |
|--------------|---------------------|
| Single Agent | 1.0x (baseline) |
| Centralized | 4.4x |
| Hybrid | 6.0x |
| Decentralized | 8.5x |
| Independent | 17.2x |

**Key insight**: Centralized systems act as a "validation bottleneck" that catches errors before they propagate.

### The Tool-Coordination Trade-off

As tasks require more tools (e.g., a coding agent with 16+ tools), the "tax" of coordinating multiple agents increases disproportionately. Single agent often wins for tool-heavy workflows.

### Capability Saturation

Multi-agent coordination yields diminishing or negative returns once single-agent baselines exceed ~45% accuracy. Better models don't remove the need for multi-agent—they accelerate it, but only when the architecture matches the task.

## 💡 Practical Takeaways

1. **Don't default to multi-agent**. Analyze your task properties first.

2. **Sequential tasks = Single agent**. Period. Multi-step reasoning gets fragmented by coordination overhead.

3. **Parallelizable + Low tools = Centralized**. The orchestrator provides 80%+ gains while containing error amplification.

4. **Tool-heavy = Single agent**. Coordination overhead compounds with tool count.

5. **Measure decomposability**. If your task can't be split into independent subtasks, multi-agent will hurt.

## 📚 References

- [Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296) - Google Research
- [Google Blog Post](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
- [More Agents Is All You Need](https://arxiv.org/abs/2402.05120) - The paper this research challenges

## 🔮 What's Next

The paper's predictive model (R²=0.524) correctly identifies optimal architecture for 87% of unseen configurations. This POC implements a simplified version of that model.

Future directions:
- Integration with real LLM calls to validate simulation
- Automatic task property detection from prompts
- Architecture recommendation API for agent frameworks

---

*Part of the [AI POC Daily](https://github.com/zscole/ai-poc-daily) series*
