# Autonomous ML Research Loop

**Date:** March 8, 2026

**Source:** [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (March 6, 2026, 5700+ stars in 48 hours)

## What This Is

A self-contained implementation of the core pattern from Karpathy's autoresearch: an autonomous loop that proposes architecture and hyperparameter mutations to a small language model, trains each variant on real text data, evaluates validation loss, keeps improvements, and discards failures.

The "researcher" is a programmatic mutation engine that explores 16 different modification types (wider/narrower embeddings, deeper/shallower networks, learning rate adjustments, batch size changes, etc). Each proposed configuration is trained from scratch within a fixed time budget, and only configurations that beat the current best validation loss are accepted.

No GPU required. Runs entirely on CPU using numpy for matrix operations and a character-level transformer trained on Project Gutenberg text (Sherlock Holmes).

## How It Works

1. Download and tokenize a real text corpus at character level
2. Initialize a baseline tiny GPT (2 layers, 48-dim embeddings, 4 heads, ~68K params)
3. For each round:
   - Mutate the current best config (or run baseline on round 1)
   - Build a fresh transformer with the new config
   - Train using random-direction gradient estimation (no autograd needed)
   - Evaluate validation cross-entropy loss
   - Accept if loss improved, reject otherwise
4. Log the full research trajectory with mutation analysis

The training uses a random-direction finite-difference approach: for each parameter tensor, sample a random unit direction, compute the directional derivative via forward-pass perturbation, and update along that direction. This is honest gradient descent without requiring backpropagation infrastructure.

## Running It

```bash
# Default: 8 rounds, 20 seconds per experiment
python3 autoresearch.py

# Custom settings
python3 autoresearch.py --rounds 12 --budget 30

# Different corpus
python3 autoresearch.py --data-url "https://www.gutenberg.org/cache/epub/84/pg84.txt"
```

**Requirements:** Python 3.10+, numpy. No other dependencies.

```bash
pip install numpy
```

## Output

- Console output showing each round's mutation, config, loss, and accept/reject decision
- `research_log.json` with full experiment history
- Improvement trajectory and mutation effectiveness analysis

## Why It Matters

Karpathy's autoresearch represents a shift in how ML research gets done. Instead of a human researcher manually tweaking hyperparameters and architectures, an autonomous agent proposes modifications and evaluates them against a fixed compute budget. The key insight from `program.md` in autoresearch: you stop editing Python and start editing the Markdown instructions that guide the agent.

This POC strips that pattern down to its essentials and makes it runnable on any machine without a GPU or LLM API key. The mutation engine replaces the LLM agent, but the core loop is identical: propose change, train, evaluate, keep or discard.

## Sample Results

From an 8-round run with 15s budget per experiment:

```
Round  1 [baseline        ] val_loss=4.6131 best_so_far=4.6131  <-- accepted
Round  2 [bigger_batch    ] val_loss=4.6103 best_so_far=4.6103  <-- accepted
Round  3 [higher_lr       ] val_loss=4.6048 best_so_far=4.6048  <-- accepted
Round  7 [higher_lr       ] val_loss=4.6043 best_so_far=4.6043  <-- accepted

Mutation Analysis:
  higher_lr : 2 accepted / 0 rejected  (most effective)
  fewer_heads: 0 accepted / 2 rejected (harmful)
```

The research loop discovered that higher learning rates and bigger batches improved this setup, while reducing attention heads hurt performance -- reasonable findings that match known scaling intuitions.
