#!/usr/bin/env python3
"""
Autonomous ML Research Loop

Inspired by Karpathy's autoresearch (github.com/karpathy/autoresearch, March 2026).
Implements the core pattern: an autonomous loop that proposes architecture/hyperparameter
modifications to a small language model, trains each variant on real text data,
evaluates validation loss, keeps improvements, and discards failures.

No GPU required -- trains a character-level transformer on CPU with a 30-second
time budget per experiment. The "researcher" is a programmatic mutation engine
that systematically explores the hyperparameter and architecture space.

Usage:
    python3 autoresearch.py [--rounds 10] [--budget 30] [--data-url URL]
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

DATA_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"  # Sherlock Holmes
DATA_DIR = Path(__file__).parent / "data"
RESULTS_FILE = Path(__file__).parent / "research_log.json"


def download_data(url: str) -> str:
    """Download and cache training text."""
    DATA_DIR.mkdir(exist_ok=True)
    h = hashlib.md5(url.encode()).hexdigest()[:8]
    cached = DATA_DIR / f"corpus_{h}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    print(f"Downloading corpus from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "autoresearch-poc/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    cached.write_text(raw, encoding="utf-8")
    print(f"  cached {len(raw):,} chars -> {cached}")
    return raw


def prepare_data(text: str, block_size: int) -> Tuple[List[int], List[int], dict]:
    """Tokenize at character level and split train/val."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    encoded = [stoi[c] for c in text]
    n = int(0.9 * len(encoded))
    return encoded[:n], encoded[n:], {"stoi": stoi, "itos": itos, "vocab_size": len(chars)}


# ---------------------------------------------------------------------------
# Tiny Transformer (pure Python + minimal math -- no torch/numpy dependency)
# ---------------------------------------------------------------------------
# We use numpy for matrix ops to keep it tractable on CPU.

try:
    import numpy as np
except ImportError:
    print("numpy is required: pip install numpy")
    sys.exit(1)


@dataclass
class ModelConfig:
    """Architecture + hyperparameter configuration."""
    vocab_size: int = 80
    block_size: int = 64
    n_embd: int = 48
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    lr: float = 3e-3
    batch_size: int = 32
    weight_decay: float = 0.01

    def signature(self) -> str:
        """Short hash for dedup."""
        s = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def describe(self) -> str:
        params = self.param_count()
        return (f"embd={self.n_embd} head={self.n_head} layer={self.n_layer} "
                f"bs={self.batch_size} lr={self.lr:.1e} blk={self.block_size} "
                f"drop={self.dropout:.2f} wd={self.weight_decay:.1e} "
                f"params~{params:,}")

    def param_count(self) -> int:
        """Rough parameter estimate."""
        E = self.vocab_size * self.n_embd + self.block_size * self.n_embd  # embeddings
        # per layer: 4 projection matrices (Q,K,V,O) + 2 FFN + 2 layernorms
        head_dim = self.n_embd // self.n_head
        attn = 4 * self.n_embd * self.n_embd  # QKV + out
        ffn = 2 * self.n_embd * (4 * self.n_embd)  # up + down
        ln = 4 * self.n_embd  # 2 layernorms (weight + bias each)
        per_layer = attn + ffn + ln
        final_ln = 2 * self.n_embd
        lm_head = self.n_embd * self.vocab_size
        return E + self.n_layer * per_layer + final_ln + lm_head


# --- numpy-based transformer building blocks ---

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b


class Linear:
    def __init__(self, in_f, out_f, scale=0.02):
        self.w = np.random.randn(in_f, out_f).astype(np.float32) * scale
        self.b = np.zeros(out_f, dtype=np.float32)

    def __call__(self, x):
        return x @ self.w + self.b

    def params(self):
        return [self.w, self.b]


class LayerNorm:
    def __init__(self, dim):
        self.g = np.ones(dim, dtype=np.float32)
        self.b = np.zeros(dim, dtype=np.float32)

    def __call__(self, x):
        return layer_norm(x, self.g, self.b)

    def params(self):
        return [self.g, self.b]


class CausalSelfAttention:
    def __init__(self, cfg: ModelConfig):
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = Linear(cfg.n_embd, cfg.n_embd)

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = np.split(qkv, 3, axis=-1)
        # reshape to (B, nh, T, hd)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        # attention
        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # causal mask
        mask = np.triu(np.ones((T, T), dtype=np.float32) * (-1e9), k=1)
        att = att + mask
        att = softmax(att)
        out = att @ v  # (B, nh, T, hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)

    def params(self):
        return self.qkv.params() + self.proj.params()


class MLP:
    def __init__(self, cfg: ModelConfig):
        self.fc1 = Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc2 = Linear(4 * cfg.n_embd, cfg.n_embd, scale=0.02 / math.sqrt(2 * cfg.n_layer if hasattr(cfg, 'n_layer') else 2))

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

    def params(self):
        return self.fc1.params() + self.fc2.params()


class TransformerBlock:
    def __init__(self, cfg: ModelConfig):
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def params(self):
        return self.ln1.params() + self.attn.params() + self.ln2.params() + self.mlp.params()


class GPT:
    """Minimal GPT for character-level language modeling."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.tok_emb = np.random.randn(cfg.vocab_size, cfg.n_embd).astype(np.float32) * 0.02
        self.pos_emb = np.random.randn(cfg.block_size, cfg.n_embd).astype(np.float32) * 0.02
        self.blocks = [TransformerBlock(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = LayerNorm(cfg.n_embd)
        # Weight-tied lm_head
        self.lm_head_b = np.zeros(cfg.vocab_size, dtype=np.float32)

    def __call__(self, idx):
        """Forward pass. idx: (B, T) integer array. Returns logits (B, T, V)."""
        B, T = idx.shape
        tok = self.tok_emb[idx]  # (B, T, C)
        pos = self.pos_emb[:T]  # (T, C)
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # Weight-tied projection
        logits = x @ self.tok_emb.T + self.lm_head_b
        return logits

    def all_params(self):
        p = [self.tok_emb, self.pos_emb, self.lm_head_b]
        for block in self.blocks:
            p.extend(block.params())
        p.extend(self.ln_f.params())
        return p


# ---------------------------------------------------------------------------
# Training loop (forward-only with numerical gradient for simplicity on CPU)
# Actually, for efficiency we use a finite-difference approach on small batches
# But that's too slow for real training. Instead: manual backprop for cross-entropy
# through the final linear layer, and use random search for architecture params.
#
# Strategy: We do FORWARD-ONLY evaluation. The "research" is architecture/hyperparam
# search. Each config is trained from scratch using a simple SGD on the embedding
# and output layers (the cheapest thing that actually works).
#
# This mirrors autoresearch's pattern: the *agent* proposes changes, the *training*
# validates them. Here the "agent" is our mutation engine.
# ---------------------------------------------------------------------------

def get_batch(data, block_size, batch_size):
    """Sample a random batch of (input, target) pairs."""
    ix = [random.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    x = np.array([data[i:i + block_size] for i in ix], dtype=np.int64)
    y = np.array([data[i + 1:i + block_size + 1] for i in ix], dtype=np.int64)
    return x, y


def cross_entropy_loss(logits, targets):
    """Compute cross-entropy loss. logits: (B,T,V), targets: (B,T)."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)
    # Numerically stable log-softmax
    log_probs = logits_flat - np.log(np.exp(logits_flat).sum(axis=-1, keepdims=True) + 1e-9)
    # Handle numerical issues
    max_logits = logits_flat.max(axis=-1, keepdims=True)
    shifted = logits_flat - max_logits
    log_sum_exp = max_logits.squeeze(-1) + np.log(np.exp(shifted).sum(axis=-1) + 1e-9)
    correct_logits = logits_flat[np.arange(len(targets_flat)), targets_flat]
    loss = -correct_logits + log_sum_exp
    return loss.mean()


def train_model(cfg: ModelConfig, train_data, val_data, time_budget: float) -> dict:
    """
    Train a model with the given config for a fixed time budget.
    Uses a simple gradient estimation approach:
    - Forward pass to get loss
    - Perturb each parameter slightly, measure loss change
    - Update in direction of improvement
    
    For efficiency, we use the "random directions" approach:
    pick random direction, compute directional derivative via finite diff,
    update along that direction. This is slow but honest and works on CPU
    without autograd.
    """
    model = GPT(cfg)
    params = model.all_params()

    best_val_loss = float("inf")
    step = 0
    start = time.time()
    train_losses = []
    val_losses = []
    lr = cfg.lr
    eps = 1e-3  # finite diff epsilon

    # Evaluate initial val loss
    xv, yv = get_batch(val_data, cfg.block_size, min(cfg.batch_size, 16))
    logits = model(xv)
    init_loss = cross_entropy_loss(logits, yv)

    while (time.time() - start) < time_budget:
        # Sample training batch
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size)

        # Forward pass
        logits = model(xb)
        loss = cross_entropy_loss(logits, yb)
        train_losses.append(float(loss))

        # Random-direction gradient estimate for each parameter
        for p in params:
            # Pick a random direction
            direction = np.random.randn(*p.shape).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-9)

            # Perturb forward
            p += eps * direction
            logits_plus = model(xb)
            loss_plus = cross_entropy_loss(logits_plus, yb)

            # Perturb backward
            p -= 2 * eps * direction
            logits_minus = model(xb)
            loss_minus = cross_entropy_loss(logits_minus, yb)

            # Restore
            p += eps * direction

            # Directional derivative
            grad = (loss_plus - loss_minus) / (2 * eps)

            # SGD update along this direction
            p -= lr * grad * direction

            # Weight decay
            if p.ndim > 1:
                p *= (1 - lr * cfg.weight_decay)

        step += 1

        # Periodic validation
        if step % 5 == 0 or (time.time() - start) >= time_budget:
            xv, yv = get_batch(val_data, cfg.block_size, min(cfg.batch_size, 16))
            logits = model(xv)
            val_loss = cross_entropy_loss(logits, yv)
            val_losses.append(float(val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    elapsed = time.time() - start

    # Final validation (average over multiple batches for stability)
    final_vals = []
    for _ in range(5):
        xv, yv = get_batch(val_data, cfg.block_size, min(cfg.batch_size, 16))
        logits = model(xv)
        final_vals.append(float(cross_entropy_loss(logits, yv)))
    final_val_loss = sum(final_vals) / len(final_vals)

    # Generate a sample
    sample = generate(model, val_data[:1], cfg, length=120)

    return {
        "config": asdict(cfg),
        "config_sig": cfg.signature(),
        "init_val_loss": float(init_loss),
        "final_val_loss": final_val_loss,
        "best_val_loss": float(best_val_loss),
        "improvement": float(init_loss) - final_val_loss,
        "steps": step,
        "elapsed_sec": round(elapsed, 1),
        "sample": sample,
        "train_loss_start": float(train_losses[0]) if train_losses else None,
        "train_loss_end": float(train_losses[-1]) if train_losses else None,
    }


def generate(model, seed_tokens, cfg, length=100):
    """Generate text from the model."""
    context = list(seed_tokens[:cfg.block_size])
    generated = []
    for _ in range(length):
        inp = np.array([context[-cfg.block_size:]], dtype=np.int64)
        logits = model(inp)
        # Sample from last position
        probs = softmax(logits[0, -1] / 0.8)  # temperature
        next_tok = np.random.choice(len(probs), p=probs)
        context.append(int(next_tok))
        generated.append(int(next_tok))
    return generated


# ---------------------------------------------------------------------------
# Mutation engine ("the researcher")
# ---------------------------------------------------------------------------

MUTATION_TYPES = [
    "wider",           # increase embedding dim
    "narrower",        # decrease embedding dim
    "deeper",          # add a layer
    "shallower",       # remove a layer
    "more_heads",      # increase attention heads
    "fewer_heads",     # decrease attention heads
    "longer_context",  # increase block size
    "shorter_context", # decrease block size
    "higher_lr",       # increase learning rate
    "lower_lr",        # decrease learning rate
    "bigger_batch",    # increase batch size
    "smaller_batch",   # decrease batch size
    "more_dropout",    # increase dropout
    "less_dropout",    # decrease dropout
    "more_wd",         # increase weight decay
    "less_wd",         # decrease weight decay
]


def mutate_config(cfg: ModelConfig, mutation: Optional[str] = None) -> Tuple[ModelConfig, str]:
    """Apply a random mutation to a config. Returns (new_config, mutation_name)."""
    new = ModelConfig(**{k: v for k, v in asdict(cfg).items()})

    if mutation is None:
        mutation = random.choice(MUTATION_TYPES)

    if mutation == "wider":
        new.n_embd = min(cfg.n_embd + 16, 128)
    elif mutation == "narrower":
        new.n_embd = max(cfg.n_embd - 16, 16)
    elif mutation == "deeper":
        new.n_layer = min(cfg.n_layer + 1, 6)
    elif mutation == "shallower":
        new.n_layer = max(cfg.n_layer - 1, 1)
    elif mutation == "more_heads":
        candidates = [h for h in [1, 2, 4, 8] if h > cfg.n_head and new.n_embd % h == 0]
        if candidates:
            new.n_head = candidates[0]
    elif mutation == "fewer_heads":
        candidates = [h for h in [8, 4, 2, 1] if h < cfg.n_head and new.n_embd % h == 0]
        if candidates:
            new.n_head = candidates[0]
    elif mutation == "longer_context":
        new.block_size = min(cfg.block_size + 16, 128)
    elif mutation == "shorter_context":
        new.block_size = max(cfg.block_size - 16, 16)
    elif mutation == "higher_lr":
        new.lr = min(cfg.lr * 1.5, 0.1)
    elif mutation == "lower_lr":
        new.lr = max(cfg.lr / 1.5, 1e-5)
    elif mutation == "bigger_batch":
        new.batch_size = min(cfg.batch_size * 2, 128)
    elif mutation == "smaller_batch":
        new.batch_size = max(cfg.batch_size // 2, 4)
    elif mutation == "more_dropout":
        new.dropout = min(cfg.dropout + 0.05, 0.5)
    elif mutation == "less_dropout":
        new.dropout = max(cfg.dropout - 0.05, 0.0)
    elif mutation == "more_wd":
        new.weight_decay = min(cfg.weight_decay * 2, 0.5)
    elif mutation == "less_wd":
        new.weight_decay = max(cfg.weight_decay / 2, 1e-5)

    # Ensure n_embd is divisible by n_head
    while new.n_embd % new.n_head != 0:
        new.n_head = max(new.n_head - 1, 1)
        if new.n_head == 1:
            break

    return new, mutation


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------

def run_research(rounds: int, time_budget: float, data_url: str):
    """Run the autonomous research loop."""
    print("=" * 70)
    print("AUTONOMOUS ML RESEARCH LOOP")
    print("Inspired by karpathy/autoresearch (March 2026)")
    print("=" * 70)
    print()

    # Prepare data
    text = download_data(data_url)
    print(f"Corpus: {len(text):,} characters")

    # Use baseline config
    base_cfg = ModelConfig()
    train_data, val_data, vocab = prepare_data(text, base_cfg.block_size)
    base_cfg.vocab_size = vocab["vocab_size"]
    itos = vocab["itos"]

    print(f"Vocabulary: {base_cfg.vocab_size} chars")
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    print(f"Time budget per experiment: {time_budget}s")
    print(f"Research rounds: {rounds}")
    print()

    best_cfg = base_cfg
    best_loss = float("inf")
    history = []
    seen_sigs = set()

    for round_num in range(rounds):
        print(f"--- Round {round_num + 1}/{rounds} ---")

        if round_num == 0:
            # Baseline run
            cfg = base_cfg
            mutation_name = "baseline"
        else:
            # Mutate best known config
            cfg, mutation_name = mutate_config(best_cfg)
            # Skip if we've already tried this exact config
            sig = cfg.signature()
            attempts = 0
            while sig in seen_sigs and attempts < 10:
                cfg, mutation_name = mutate_config(best_cfg)
                sig = cfg.signature()
                attempts += 1

        # Adjust vocab size and block size for data
        cfg.vocab_size = vocab["vocab_size"]
        # Ensure block_size doesn't exceed data
        cfg.block_size = min(cfg.block_size, len(val_data) - 2)

        seen_sigs.add(cfg.signature())

        print(f"  Mutation: {mutation_name}")
        print(f"  Config:   {cfg.describe()}")

        # Re-prepare data if block_size changed
        train_data_r, val_data_r, _ = prepare_data(text, cfg.block_size)

        # Train
        result = train_model(cfg, train_data_r, val_data_r, time_budget)

        # Decode sample
        sample_text = "".join(itos.get(t, "?") for t in result["sample"])
        result["sample_text"] = sample_text

        # Evaluate
        improved = result["final_val_loss"] < best_loss
        result["mutation"] = mutation_name
        result["round"] = round_num + 1
        result["accepted"] = improved

        status = "ACCEPTED (new best)" if improved else "rejected"
        print(f"  Val loss: {result['final_val_loss']:.4f} (init: {result['init_val_loss']:.4f}) "
              f"[{status}]")
        print(f"  Steps:   {result['steps']} in {result['elapsed_sec']}s")
        print(f"  Sample:  {sample_text[:80]}...")
        print()

        if improved:
            best_loss = result["final_val_loss"]
            best_cfg = cfg

        history.append(result)

    # Summary
    print("=" * 70)
    print("RESEARCH COMPLETE")
    print("=" * 70)
    accepted = [h for h in history if h["accepted"]]
    rejected = [h for h in history if not h["accepted"]]
    print(f"Total rounds:  {rounds}")
    print(f"Accepted:      {len(accepted)}")
    print(f"Rejected:      {len(rejected)}")
    print(f"Best val loss: {best_loss:.4f}")
    print(f"Best config:   {best_cfg.describe()}")
    print()

    # Show improvement trajectory
    print("IMPROVEMENT TRAJECTORY:")
    running_best = float("inf")
    for h in history:
        if h["final_val_loss"] < running_best:
            running_best = h["final_val_loss"]
        marker = " <-- best" if h["accepted"] else ""
        print(f"  Round {h['round']:2d} [{h['mutation']:16s}] "
              f"val_loss={h['final_val_loss']:.4f} "
              f"best_so_far={running_best:.4f}{marker}")

    print()
    print("MUTATION ANALYSIS:")
    mutation_stats = {}
    for h in history:
        m = h["mutation"]
        if m not in mutation_stats:
            mutation_stats[m] = {"accepted": 0, "rejected": 0, "avg_loss": []}
        if h["accepted"]:
            mutation_stats[m]["accepted"] += 1
        else:
            mutation_stats[m]["rejected"] += 1
        mutation_stats[m]["avg_loss"].append(h["final_val_loss"])

    for m, stats in sorted(mutation_stats.items()):
        avg = sum(stats["avg_loss"]) / len(stats["avg_loss"])
        print(f"  {m:16s}: {stats['accepted']}A/{stats['rejected']}R  avg_loss={avg:.4f}")

    # Save log
    log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_url": data_url,
        "rounds": rounds,
        "time_budget_per_round": time_budget,
        "best_val_loss": best_loss,
        "best_config": asdict(best_cfg),
        "history": history,
    }
    RESULTS_FILE.write_text(json.dumps(log, indent=2, default=str))
    print(f"\nFull log saved to {RESULTS_FILE}")

    return log


def main():
    parser = argparse.ArgumentParser(description="Autonomous ML Research Loop")
    parser.add_argument("--rounds", type=int, default=8, help="Number of research rounds")
    parser.add_argument("--budget", type=float, default=20, help="Seconds per experiment")
    parser.add_argument("--data-url", default=DATA_URL, help="URL of training corpus")
    args = parser.parse_args()

    run_research(args.rounds, args.budget, args.data_url)


if __name__ == "__main__":
    main()
