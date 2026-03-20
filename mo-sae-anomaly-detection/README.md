# MO-SAE:Multi-Objective Stacked Autoencoders Optimization for Edge Anomaly Detection

Based on: https://arxiv.org/abs/2603.13895v1

## What This Does

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

## Run It

```bash
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies
