"""
MO-SAE: Multi-Objective Stacked Autoencoders Optimization for Edge Anomaly Detection

CORE TECHNICAL CONTRIBUTION:
MO-SAE applies multi-objective optimization over SAE hyperparameter configurations
to simultaneously optimize detection performance (AUC), model compactness (parameter
count), and inference efficiency (latency), producing a Pareto front of non-dominated
solutions for deployment on heterogeneous, resource-constrained edge devices.

VALID DEMONSTRATION:
Train multiple SAE configurations on normal-only data; evaluate each on three
competing objectives; compute the Pareto front via non-dominated sorting; show
that Pareto-optimal models represent genuine trade-offs between accuracy and
resource efficiency that single-objective optimization cannot capture.

WHY THIS IS HARD:
(1) The three objectives fundamentally conflict — deeper/wider models detect better
    but are slower and larger; (2) SAE reconstruction thresholds must be calibrated
    per model; (3) Non-dominated sorting across a three-dimensional objective space
    is non-trivial; (4) Edge anomaly detection must handle severe class imbalance
    where anomalies are rare relative to normal traffic.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import time
import json

# Device selection: prefer MPS on Apple Silicon, fall back to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class StackedAutoencoder(nn.Module):
    """
    Stacked Autoencoder for unsupervised anomaly detection.
    Trained exclusively on normal samples; anomalies exhibit high reconstruction error.
    Architecture: symmetric encoder-decoder with configurable bottleneck depth.
    """

    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Encoder: input_dim → hidden_dims[-1] (bottleneck)
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: bottleneck → input_dim (mirror of encoder)
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        for d_dim in reversed_dims:
            decoder_layers.append(nn.Linear(prev_dim, d_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = d_dim
        # Remove trailing activation on output layer
        if len(decoder_layers) >= 2:
            decoder_layers = decoder_layers[:-1]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error — higher = more anomalous."""
        self.eval()
        param_device = next(self.parameters()).device
        x = x.to(param_device)
        with torch.no_grad():
            reconstructed = self(x)
            errors = torch.mean((x - reconstructed) ** 2, dim=1)
        return errors.cpu()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def generate_anomaly_dataset(
    n_normal: int = 2000,
    n_anomaly: int = 200,
    n_features: int = 30,
    latent_dim: int = 6,
    random_state=None,
) -> tuple:
    """
    Generate a challenging nonlinear-manifold anomaly detection dataset.

    Normal data: lives on a low-dimensional (latent_dim) nonlinear manifold embedded
    in n_features-dimensional space. Captured by a fixed random tanh decoder.

    Anomaly data: points perturbed off the manifold — close to normal in Euclidean
    distance but structurally inconsistent with the manifold.  A SAE that has fully
    learned the manifold reconstructs normal points well but fails on off-manifold
    anomalies; an underpowered SAE fails both.

    This creates genuine AUC variation across SAE configurations with different
    bottleneck capacities.
    """
    rng = np.random.RandomState(random_state)

    # Fixed random nonlinear "true" generator (the manifold)
    W1 = rng.randn(latent_dim, 16) / np.sqrt(latent_dim)
    W2 = rng.randn(16, n_features) / np.sqrt(16)
    b1 = rng.randn(16) * 0.5

    def manifold_decode(z: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        h = np.tanh(z @ W1 + b1)
        x = h @ W2
        if noise_std > 0:
            x += rng.randn(*x.shape) * noise_std
        return x

    # Normal: latent codes drawn from a standard Gaussian
    z_normal = rng.randn(n_normal, latent_dim)
    normal_data = manifold_decode(z_normal, noise_std=0.08)

    if n_anomaly > 0:
        # Anomaly: start on manifold, then perturb strongly in feature space
        # This makes anomalies "look" locally normal but be off-manifold
        z_anom = rng.randn(n_anomaly, latent_dim)
        on_manifold = manifold_decode(z_anom, noise_std=0.0)
        # Random directions in feature space (not necessarily tangent to manifold)
        perturbation = rng.randn(n_anomaly, n_features)
        # Orthogonalize perturbation w.r.t. manifold tangent (approximate)
        perturbation /= np.linalg.norm(perturbation, axis=1, keepdims=True) + 1e-8
        anomaly_data = on_manifold + perturbation * rng.uniform(0.6, 1.2, (n_anomaly, 1))

        X = np.vstack([normal_data, anomaly_data]).astype(np.float32)
        y = np.concatenate(
            [np.zeros(len(normal_data)), np.ones(len(anomaly_data))]
        ).astype(int)
    else:
        X = normal_data.astype(np.float32)
        y = np.zeros(len(normal_data), dtype=int)

    return X, y


def train_sae(
    model: StackedAutoencoder,
    train_loader: DataLoader,
    n_epochs: int = 40,
    lr: float = 1e-3,
) -> list:
    """
    Train SAE via reconstruction loss on normal-only data.
    Returns per-epoch average loss to confirm genuine learning.
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    epoch_losses = []
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_samples = 0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
        epoch_losses.append(total_loss / total_samples)

    return epoch_losses


def evaluate_objectives(
    model: StackedAutoencoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_inference_runs: int = 100,
) -> dict:
    """
    Evaluate model on three competing objectives:
      1. AUC (detection quality)     — maximize
      2. n_params (model size)       — minimize
      3. latency_ms (inference time) — minimize
    """
    model.to(device)
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)

    # Warm-up pass
    _ = model.reconstruction_error(X_tensor)

    # Latency measurement
    start = time.perf_counter()
    for _ in range(n_inference_runs):
        _ = model.reconstruction_error(X_tensor)
    end = time.perf_counter()
    latency_ms = (end - start) / n_inference_runs * 1000.0

    # Reconstruction errors → anomaly scores (reconstruction_error returns CPU tensor)
    errors = model.reconstruction_error(X_tensor).numpy()
    auc = roc_auc_score(y_test, errors)

    return {
        "auc": float(auc),
        "n_params": model.count_parameters(),
        "latency_ms": float(latency_ms),
    }


def pareto_dominates(obj_a: dict, obj_b: dict) -> bool:
    """
    Return True if solution A Pareto-dominates B.
    Minimization form: [-auc, n_params, latency_ms].
    A dominates B iff A is no worse in every objective and strictly better in at least one.
    """
    a = [-obj_a["auc"], obj_a["n_params"], obj_a["latency_ms"]]
    b = [-obj_b["auc"], obj_b["n_params"], obj_b["latency_ms"]]
    at_least_as_good = all(ai <= bi for ai, bi in zip(a, b))
    strictly_better = any(ai < bi for ai, bi in zip(a, b))
    return at_least_as_good and strictly_better


def compute_pareto_front(solutions: list) -> list:
    """
    Non-dominated sorting: return the subset of solutions not dominated by any other.
    O(n^2) implementation suitable for small configuration spaces.
    """
    pareto = []
    for i, (cfg_i, obj_i) in enumerate(solutions):
        dominated = False
        for j, (cfg_j, obj_j) in enumerate(solutions):
            if i != j and pareto_dominates(obj_j, obj_i):
                dominated = True
                break
        if not dominated:
            pareto.append((cfg_i, obj_i))
    return pareto


def main():
    print("=" * 65)
    print("MO-SAE: Multi-Objective Stacked Autoencoder Optimization")
    print("=" * 65)

    # ── Dataset ──────────────────────────────────────────────────────
    print("\nGenerating synthetic anomaly detection dataset...")
    n_features = 30
    latent_dim = 6   # true manifold dimensionality
    X, y = generate_anomaly_dataset(
        n_normal=800, n_anomaly=100, n_features=n_features,
        latent_dim=latent_dim, random_state=42
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Train: normal samples only (semi-supervised anomaly detection)
    rng = np.random.RandomState(0)
    normal_idx = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]
    rng.shuffle(normal_idx)

    # Reserve 150 normal + all anomalies for test
    test_normal_idx = normal_idx[:150]
    train_normal_idx = normal_idx[300:]

    test_idx = np.concatenate([test_normal_idx, anomaly_idx])
    X_train = torch.FloatTensor(X[train_normal_idx])
    X_test = X[test_idx]
    y_test = y[test_idx]

    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=64, shuffle=True, drop_last=True
    )

    print(f"  Train (normal only): {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples  ({y_test.sum()} anomalies, {(y_test == 0).sum()} normal)")

    # ── SAE Configuration Search Space ───────────────────────────────
    # Key insight: bottleneck dimension (last hidden_dim) controls manifold recovery.
    # The true manifold has latent_dim=6 dims, so:
    #   bottleneck < 6  → underpowered, can't reconstruct normal manifold → poor AUC
    #   bottleneck ≈ 6  → adequate
    #   bottleneck > 6  → good manifold recovery → high AUC but more params/latency
    config_space = [
        {"hidden_dims": [32, 2],          "lr": 1e-3},   # bottleneck=2, far below latent_dim
        {"hidden_dims": [32, 4],          "lr": 1e-3},   # bottleneck=4, below latent_dim
        {"hidden_dims": [64, 6],          "lr": 1e-3},   # bottleneck=6, matches latent_dim
        {"hidden_dims": [64, 32, 6],      "lr": 1e-3},   # deeper, bottleneck=6
        {"hidden_dims": [64, 32, 8],      "lr": 5e-4},   # bottleneck=8, above latent_dim
        {"hidden_dims": [128, 64, 12],    "lr": 5e-4},   # bottleneck=12, more capacity
    ]

    print(f"\nEvaluating {len(config_space)} SAE configurations...")
    print("-" * 65)

    solutions = []
    for i, cfg in enumerate(config_space):
        model = StackedAutoencoder(
            input_dim=n_features, hidden_dims=cfg["hidden_dims"]
        ).to(device)

        losses = train_sae(model, train_loader, n_epochs=20, lr=cfg["lr"])
        obj = evaluate_objectives(model, X_test, y_test, n_inference_runs=10)
        solutions.append((cfg, obj))

        loss_drop = losses[0] - losses[-1]
        print(
            f"  [{i+1:2d}] layers={str(cfg['hidden_dims']):<22} "
            f"AUC={obj['auc']:.4f}  "
            f"params={obj['n_params']:6d}  "
            f"lat={obj['latency_ms']:.3f}ms  "
            f"Δloss={loss_drop:.5f}"
        )

    # ── Pareto Front ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PARETO FRONT (non-dominated solutions):")
    print("=" * 65)
    pareto_front = compute_pareto_front(solutions)
    pareto_front_sorted = sorted(pareto_front, key=lambda x: x[1]["auc"], reverse=True)

    # Compute normalization bounds for efficiency scoring
    all_pareto_aucs = np.array([o["auc"] for _, o in pareto_front_sorted])
    all_pareto_params = np.array([o["n_params"] for _, o in pareto_front_sorted], dtype=float)
    all_pareto_lats = np.array([o["latency_ms"] for _, o in pareto_front_sorted])
    params_range = all_pareto_params.max() - all_pareto_params.min() + 1e-9
    lat_range = all_pareto_lats.max() - all_pareto_lats.min() + 1e-9

    for cfg, obj in pareto_front_sorted:
        # Efficiency score: AUC / (normalized_params + normalized_latency + 1)
        # Measures how much detection quality is achieved per unit of resource cost
        norm_params = (obj["n_params"] - all_pareto_params.min()) / params_range
        norm_lat = (obj["latency_ms"] - all_pareto_lats.min()) / lat_range
        efficiency = obj["auc"] / (norm_params + norm_lat + 1.0)
        print(
            f"  layers={str(cfg['hidden_dims']):<22} "
            f"AUC={obj['auc']:.4f}  "
            f"params={obj['n_params']:6d}  "
            f"lat={obj['latency_ms']:.3f}ms  "
            f"eff={efficiency:.4f}"
        )

    dominated = [s for s in solutions if s not in pareto_front]
    print(f"\n  {len(pareto_front)}/{len(solutions)} configurations on Pareto front")

    # ── Trade-off Analysis ───────────────────────────────────────────
    all_aucs = [o["auc"] for _, o in solutions]
    all_params = [o["n_params"] for _, o in solutions]
    all_lat = [o["latency_ms"] for _, o in solutions]

    print("\nObjective ranges across all solutions:")
    print(f"  AUC:     [{min(all_aucs):.4f}, {max(all_aucs):.4f}]")
    print(f"  Params:  [{min(all_params)}, {max(all_params)}]")
    print(f"  Latency: [{min(all_lat):.3f}ms, {max(all_lat):.3f}ms]")

    if dominated:
        p_auc = np.mean([o["auc"] for _, o in pareto_front])
        d_auc = np.mean([o["auc"] for _, o in dominated])
        print(f"\nPareto front avg AUC:    {p_auc:.4f}")
        print(f"Dominated solutions AUC: {d_auc:.4f}")

    # ── Persist Results ───────────────────────────────────────────────
    results = {
        "n_solutions": len(solutions),
        "n_pareto": len(pareto_front),
        "auc_range": [float(min(all_aucs)), float(max(all_aucs))],
        "params_range": [int(min(all_params)), int(max(all_params))],
        "latency_range_ms": [float(min(all_lat)), float(max(all_lat))],
        "pareto_front": [
            {
                "hidden_dims": cfg["hidden_dims"],
                "auc": obj["auc"],
                "n_params": obj["n_params"],
                "latency_ms": obj["latency_ms"],
            }
            for cfg, obj in pareto_front_sorted
        ],
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to results.json")
    print("MO-SAE Pareto optimization complete.")
    return results


if __name__ == "__main__":
    main()
