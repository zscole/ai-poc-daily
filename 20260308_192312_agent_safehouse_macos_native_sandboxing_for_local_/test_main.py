"""
Tests for Agent Safehouse POC.
Run with: python -m pytest test_main.py -v
"""

import numpy as np
import pytest


# ── Import sanity ─────────────────────────────────────────────────────────────

def test_imports():
    """All public symbols import without error."""
    import main
    assert callable(main.generate_agent_actions)
    assert callable(main.train_policy_classifier)
    assert callable(main.generate_sandbox_profile)
    assert callable(main.run_sandbox_experiments)
    assert callable(main.generate_test_actions)
    assert hasattr(main, "SafehouseAgent")


# ── Data generation ───────────────────────────────────────────────────────────

def test_generate_actions_shape_and_columns():
    """Action dataframe has correct shape and valid label distribution."""
    from main import generate_agent_actions
    df = generate_agent_actions(n_samples=200)
    assert len(df) == 200
    required = {"action_type", "denied", "path_depth", "is_network", "is_exec", "is_write"}
    assert required.issubset(df.columns)
    # Labels must be binary
    assert df["denied"].isin([0, 1]).all()
    # Denial rate must be non-trivial (not all 0 or all 1)
    rate = df["denied"].mean()
    assert 0.05 < rate < 0.95, f"Denial rate {rate:.2f} suspiciously extreme"


def test_action_variance_across_runs():
    """Two calls with seed=None produce different data (real randomness)."""
    from main import generate_agent_actions
    df1 = generate_agent_actions(n_samples=100)
    df2 = generate_agent_actions(n_samples=100)
    # Denied counts are extremely unlikely to be identical across 100 samples
    assert df1["denied"].sum() != df2["denied"].sum() or \
           not df1["path_entropy"].equals(df2["path_entropy"]), \
        "Two unseeded runs produced identical data — likely hardcoded"


# ── ML classifier ─────────────────────────────────────────────────────────────

def test_classifier_performance():
    """Classifier achieves non-trivial F1 and accuracy (>0.65)."""
    from main import generate_agent_actions, train_policy_classifier
    df = generate_agent_actions(n_samples=600)
    pipeline, le, metrics = train_policy_classifier(df)

    assert metrics["test_f1"] > 0.65, f"F1 {metrics['test_f1']:.3f} too low"
    assert metrics["test_accuracy"] > 0.65, f"Accuracy {metrics['test_accuracy']:.3f} too low"
    assert len(metrics["cv_scores"]) == 5
    # CV mean must exceed random (0.5)
    assert metrics["cv_f1_mean"] > 0.5


def test_classifier_returns_pipeline_and_encoder():
    """train_policy_classifier returns a fitted pipeline and LabelEncoder."""
    from main import generate_agent_actions, train_policy_classifier, ACTION_TYPES
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    df = generate_agent_actions(n_samples=400)
    pipeline, le, metrics = train_policy_classifier(df)

    assert isinstance(pipeline, Pipeline)
    assert isinstance(le, LabelEncoder)
    # Encoder should know all action types
    for atype in ACTION_TYPES:
        assert atype in le.classes_


# ── Agent loop ────────────────────────────────────────────────────────────────

def test_agent_loop_output_types():
    """Agent loop returns valid numeric result dict."""
    from main import (
        generate_agent_actions, train_policy_classifier,
        SafehouseAgent, generate_test_actions,
    )
    df = generate_agent_actions(n_samples=400)
    pipeline, le, _ = train_policy_classifier(df)
    agent = SafehouseAgent(pipeline, le)
    actions = generate_test_actions(n=30)
    result = agent.run(actions)

    assert result["total"] == 30
    assert result["allowed"] + result["denied"] == result["total"]
    assert 0.0 <= result["denial_rate"] <= 1.0
    assert isinstance(result["sandbox_verifications"], int)


def test_agent_blocks_sensitive_actions():
    """Agent denies a high proportion of obviously sensitive actions."""
    from main import (
        generate_agent_actions, train_policy_classifier,
        SafehouseAgent,
    )
    df = generate_agent_actions(n_samples=600)
    pipeline, le, _ = train_policy_classifier(df)
    agent = SafehouseAgent(pipeline, le)

    # Craft maximally sensitive actions
    sensitive_actions = [
        {
            "action_type": "file_write",
            "path_depth": 3,
            "has_dotfile": 1,
            "is_system_path": 1,
            "is_network": 0,
            "is_exec": 0,
            "is_write": 1,
            "path_entropy": 3.5,
            "is_sensitive": 1,
        }
        for _ in range(20)
    ]
    result = agent.run(sensitive_actions)
    # Must block at least 70 % of obviously sensitive actions
    assert result["denial_rate"] >= 0.7, (
        f"Agent only denied {result['denial_rate']:.1%} of sensitive actions"
    )


# ── Sandbox profile generation ────────────────────────────────────────────────

def test_sandbox_profile_structure():
    """Generated sandbox profile contains required Seatbelt directives."""
    from main import generate_sandbox_profile
    profile = generate_sandbox_profile(
        allowed_paths=["/tmp/agent_work"],
        allow_network=False,
    )
    assert "(version 1)" in profile
    assert "(deny default)" in profile
    assert "/tmp/agent_work" in profile
    # Network should NOT appear when disabled
    assert "network" not in profile


def test_sandbox_profile_network_flag():
    """Network allow clause appears only when explicitly requested."""
    from main import generate_sandbox_profile
    profile_no_net = generate_sandbox_profile([], allow_network=False)
    profile_net = generate_sandbox_profile([], allow_network=True)
    assert "(allow network*)" not in profile_no_net
    assert "(allow network*)" in profile_net


# ── End-to-end smoke test ─────────────────────────────────────────────────────

def test_end_to_end_pipeline():
    """Full pipeline runs without exceptions and produces sensible output."""
    from main import (
        generate_agent_actions, train_policy_classifier,
        SafehouseAgent, generate_test_actions,
    )
    df = generate_agent_actions(n_samples=300)
    pipeline, le, metrics = train_policy_classifier(df)
    agent = SafehouseAgent(pipeline, le)
    actions = generate_test_actions(n=40)
    result = agent.run(actions)

    # Sanity-check everything is a real number, not NaN
    assert not np.isnan(metrics["test_f1"])
    assert not np.isnan(metrics["test_accuracy"])
    assert not np.isnan(result["denial_rate"])
