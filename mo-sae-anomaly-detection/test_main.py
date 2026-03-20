"""
Tests for MO-SAE implementation.
Checks types, shapes, ranges, and non-trivial behavior — not hardcoded values.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from main import (
    StackedAutoencoder,
    generate_anomaly_dataset,
    train_sae,
    evaluate_objectives,
    pareto_dominates,
    compute_pareto_front,
)


# ── Architecture Tests ───────────────────────────────────────────────────────

def test_sae_output_shape():
    """SAE forward pass must reconstruct to the same shape as input."""
    for hidden_dims in [[16, 8], [64, 32, 8], [128, 64, 32, 8]]:
        model = StackedAutoencoder(input_dim=20, hidden_dims=hidden_dims)
        x = torch.randn(32, 20)
        out = model(x)
        assert out.shape == x.shape, f"Shape mismatch for hidden_dims={hidden_dims}"


def test_sae_parameter_count_positive():
    """SAE must have a non-zero, finite parameter count."""
    model = StackedAutoencoder(input_dim=20, hidden_dims=[64, 32, 8])
    n_params = model.count_parameters()
    assert n_params > 0
    assert np.isfinite(n_params)


def test_sae_reconstruction_error_shape_and_sign():
    """Reconstruction errors must be per-sample, non-negative tensors."""
    model = StackedAutoencoder(input_dim=10, hidden_dims=[8, 4])
    x = torch.randn(50, 10)
    errors = model.reconstruction_error(x)
    assert isinstance(errors, torch.Tensor)
    assert errors.shape == (50,), f"Expected shape (50,), got {errors.shape}"
    assert (errors >= 0).all(), "Reconstruction errors must be non-negative"


def test_deeper_model_has_more_parameters():
    """Deeper/wider SAE must have strictly more parameters than shallower one."""
    small = StackedAutoencoder(input_dim=20, hidden_dims=[16, 8])
    large = StackedAutoencoder(input_dim=20, hidden_dims=[128, 64, 32, 8])
    assert large.count_parameters() > small.count_parameters()


# ── Dataset Tests ────────────────────────────────────────────────────────────

def test_dataset_shape_and_labels():
    """Dataset generation must produce correct shapes and binary labels."""
    X, y = generate_anomaly_dataset(n_normal=100, n_anomaly=20, n_features=10, latent_dim=4)
    assert X.shape == (120, 10)
    assert y.shape == (120,)
    assert set(y.tolist()) == {0, 1}
    assert int(y.sum()) == 20


def test_dataset_anomaly_separation():
    """Anomaly and normal samples should have different distributions."""
    X, y = generate_anomaly_dataset(n_normal=500, n_anomaly=100, n_features=10, latent_dim=4, random_state=7)
    normal_mean_norm = np.linalg.norm(X[y == 0], axis=1).mean()
    anomaly_mean_norm = np.linalg.norm(X[y == 1], axis=1).mean()
    # They should differ (manifold vs off-manifold)
    assert not np.isclose(normal_mean_norm, anomaly_mean_norm, rtol=0.01)


def test_dataset_normal_only():
    """Dataset with n_anomaly=0 should return all-zero labels."""
    X, y = generate_anomaly_dataset(n_normal=50, n_anomaly=0, n_features=5, latent_dim=3)
    assert X.shape == (50, 5)
    assert y.sum() == 0


# ── Training Tests ───────────────────────────────────────────────────────────

def _make_normal_loader(n=200, n_features=10, seed=42, batch_size=32):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    return DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=batch_size, shuffle=True, drop_last=True)


def test_training_returns_correct_number_of_losses():
    """train_sae must return one loss value per epoch."""
    loader = _make_normal_loader()
    model = StackedAutoencoder(input_dim=10, hidden_dims=[8, 4])
    losses = train_sae(model, loader, n_epochs=15, lr=1e-3)
    assert len(losses) == 15


def test_training_reduces_loss():
    """Reconstruction loss must decrease over training epochs (real backprop)."""
    loader = _make_normal_loader(n=300, seed=0)
    model = StackedAutoencoder(input_dim=10, hidden_dims=[8, 4])
    losses = train_sae(model, loader, n_epochs=30, lr=1e-3)
    assert all(l > 0 for l in losses), "All losses must be positive"
    # Average of last 5 epochs < average of first 5 epochs
    early_avg = np.mean(losses[:5])
    late_avg = np.mean(losses[-5:])
    assert late_avg < early_avg, (
        f"Loss should decrease: early={early_avg:.6f}, late={late_avg:.6f}"
    )


def test_trained_model_anomaly_score_separation():
    """After training on normal data, anomalies must yield higher reconstruction error."""
    from sklearn.preprocessing import StandardScaler

    X, y = generate_anomaly_dataset(n_normal=400, n_anomaly=80, n_features=10, latent_dim=4, random_state=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X_normal = torch.FloatTensor(X[y == 0])
    loader = DataLoader(
        TensorDataset(X_normal), batch_size=32, shuffle=True, drop_last=True
    )

    model = StackedAutoencoder(input_dim=10, hidden_dims=[16, 8])
    train_sae(model, loader, n_epochs=40, lr=1e-3)

    errors = model.reconstruction_error(torch.FloatTensor(X)).numpy()
    mean_normal = errors[y == 0].mean()
    mean_anomaly = errors[y == 1].mean()

    assert mean_anomaly > mean_normal, (
        f"Anomaly error ({mean_anomaly:.4f}) must exceed normal error ({mean_normal:.4f})"
    )


# ── Pareto Optimization Tests ────────────────────────────────────────────────

def test_pareto_dominance_basic():
    """A dominates B when better in all objectives."""
    obj_a = {"auc": 0.95, "n_params": 100, "latency_ms": 0.5}
    obj_b = {"auc": 0.80, "n_params": 200, "latency_ms": 2.0}
    assert pareto_dominates(obj_a, obj_b)
    assert not pareto_dominates(obj_b, obj_a)


def test_pareto_dominance_trade_off():
    """When objectives conflict, neither solution dominates."""
    obj_c = {"auc": 0.95, "n_params": 10000, "latency_ms": 5.0}   # best AUC, worst size/speed
    obj_d = {"auc": 0.70, "n_params": 100,   "latency_ms": 0.2}   # worst AUC, best size/speed
    assert not pareto_dominates(obj_c, obj_d)
    assert not pareto_dominates(obj_d, obj_c)


def test_pareto_self_non_dominance():
    """A solution cannot dominate itself."""
    obj = {"auc": 0.85, "n_params": 500, "latency_ms": 1.0}
    assert not pareto_dominates(obj, obj)


def test_compute_pareto_front_identifies_dominated():
    """Non-dominated sorting must correctly exclude dominated solutions."""
    solutions = [
        ({"id": "A"}, {"auc": 0.90, "n_params": 100,  "latency_ms": 1.0}),  # Pareto optimal
        ({"id": "B"}, {"auc": 0.70, "n_params": 500,  "latency_ms": 3.0}),  # dominated by A
        ({"id": "C"}, {"auc": 0.60, "n_params": 50,   "latency_ms": 0.3}),  # Pareto optimal (tiny)
        ({"id": "D"}, {"auc": 0.65, "n_params": 800,  "latency_ms": 4.0}),  # dominated by A and B
    ]
    pareto = compute_pareto_front(solutions)
    pareto_ids = {cfg["id"] for cfg, _ in pareto}

    assert "A" in pareto_ids, "A should be Pareto optimal"
    assert "C" in pareto_ids, "C should be Pareto optimal (smallest/fastest)"
    assert "B" not in pareto_ids, "B is dominated by A"
    assert "D" not in pareto_ids, "D is dominated"


def test_pareto_front_nonempty():
    """Pareto front must always contain at least one solution."""
    solutions = [
        ({"id": str(i)}, {"auc": 0.5 + i * 0.05, "n_params": 1000 - i * 100, "latency_ms": 5.0 - i * 0.5})
        for i in range(5)
    ]
    pareto = compute_pareto_front(solutions)
    assert len(pareto) >= 1


# ── Objective Evaluation Tests ───────────────────────────────────────────────

def test_evaluate_objectives_returns_expected_keys():
    """evaluate_objectives must return auc, n_params, and latency_ms."""
    X, y = generate_anomaly_dataset(n_normal=100, n_anomaly=20, n_features=10, latent_dim=4, random_state=3)
    model = StackedAutoencoder(input_dim=10, hidden_dims=[8, 4])
    obj = evaluate_objectives(model, X.astype(np.float32), y, n_inference_runs=5)
    assert "auc" in obj
    assert "n_params" in obj
    assert "latency_ms" in obj


def test_evaluate_objectives_auc_in_range():
    """AUC must be in [0, 1]."""
    X, y = generate_anomaly_dataset(n_normal=100, n_anomaly=20, n_features=10, latent_dim=4, random_state=5)
    model = StackedAutoencoder(input_dim=10, hidden_dims=[8, 4])
    obj = evaluate_objectives(model, X.astype(np.float32), y, n_inference_runs=5)
    assert 0.0 <= obj["auc"] <= 1.0


def test_evaluate_objectives_latency_positive():
    """Inference latency must be a positive float."""
    X, y = generate_anomaly_dataset(n_normal=50, n_anomaly=10, n_features=10, latent_dim=4, random_state=6)
    model = StackedAutoencoder(input_dim=10, hidden_dims=[8, 4])
    obj = evaluate_objectives(model, X.astype(np.float32), y, n_inference_runs=5)
    assert obj["latency_ms"] > 0.0
    assert np.isfinite(obj["latency_ms"])
