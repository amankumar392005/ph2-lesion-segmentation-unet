import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")

BASELINE_RESULTS = RESULTS_DIR / "baseline_test_results.json"
PROPOSED_RESULTS = RESULTS_DIR / "proposed_test_results.json"

BASELINE_HISTORY = RESULTS_DIR / "baseline_history.json"
PROPOSED_HISTORY = RESULTS_DIR / "proposed_history.json"

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Test Results
# ============================================================

with open(BASELINE_RESULTS, "r") as f:
    baseline = json.load(f)

with open(PROPOSED_RESULTS, "r") as f:
    proposed = json.load(f)


# ============================================================
# Metrics
# ============================================================

metrics = ["precision", "recall", "f1", "iou"]

baseline_values = [
    baseline["precision"],
    baseline["recall"],
    baseline["f1"],
    baseline["iou"]
]

proposed_values = [
    proposed["precision"],
    proposed["recall"],
    proposed["f1"],
    proposed["iou"]
]


# ============================================================
# Print Comparison
# ============================================================

print()
print("=" * 70)
print("BASELINE VS PROPOSED MODEL")
print("=" * 70)

print(
    f"{'Metric':<15}"
    f"{'Baseline':>15}"
    f"{'Proposed':>15}"
    f"{'Change':>15}"
)

print("-" * 70)

for metric, b, p in zip(
    metrics,
    baseline_values,
    proposed_values
):

    change = p - b

    print(
        f"{metric.capitalize():<15}"
        f"{b:>15.4f}"
        f"{p:>15.4f}"
        f"{change:>+15.4f}"
    )


# ============================================================
# Save Comparison JSON
# ============================================================

comparison = {
    "baseline": {
        "model": "U-Net + Cross-Entropy",
        "precision": baseline["precision"],
        "recall": baseline["recall"],
        "f1": baseline["f1"],
        "iou": baseline["iou"]
    },

    "proposed": {
        "model": "U-Net + Cross-Entropy + Soft Dice",
        "precision": proposed["precision"],
        "recall": proposed["recall"],
        "f1": proposed["f1"],
        "iou": proposed["iou"]
    },

    "change": {
        "precision": proposed["precision"] - baseline["precision"],
        "recall": proposed["recall"] - baseline["recall"],
        "f1": proposed["f1"] - baseline["f1"],
        "iou": proposed["iou"] - baseline["iou"]
    }
}

comparison_path = RESULTS_DIR / "model_comparison.json"

with open(comparison_path, "w") as f:
    json.dump(comparison, f, indent=4)

print()
print("Comparison saved to:", comparison_path)


# ============================================================
# Metric Comparison Graph
# ============================================================

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline: U-Net + CE"
)

ax.bar(
    x + width / 2,
    proposed_values,
    width,
    label="Proposed: U-Net + CE + Soft Dice"
)

ax.set_ylabel("Score")
ax.set_title("Baseline vs Proposed Model")

ax.set_xticks(x)
ax.set_xticklabels(
    ["Precision", "Recall", "F1 Score", "IoU"]
)

ax.set_ylim(0, 1.05)

ax.legend()

for i, value in enumerate(baseline_values):
    ax.text(
        i - width / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

for i, value in enumerate(proposed_values):
    ax.text(
        i + width / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

metric_graph_path = FIGURES_DIR / "baseline_vs_proposed_metrics.png"

plt.savefig(
    metric_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Metric comparison graph saved to:",
    metric_graph_path
)


# ============================================================
# Load Training Histories
# ============================================================

with open(BASELINE_HISTORY, "r") as f:
    baseline_history = json.load(f)

with open(PROPOSED_HISTORY, "r") as f:
    proposed_history = json.load(f)


# ============================================================
# Training Loss Comparison
# ============================================================

baseline_train = baseline_history["train_loss"]
baseline_val = baseline_history["val_loss"]

proposed_train = proposed_history["train_loss"]
proposed_val = proposed_history["val_loss"]

epochs_baseline = range(
    1,
    len(baseline_train) + 1
)

epochs_proposed = range(
    1,
    len(proposed_train) + 1
)


# ============================================================
# Baseline Loss Curve
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_baseline,
    baseline_train,
    label="Baseline Train Loss"
)

plt.plot(
    epochs_baseline,
    baseline_val,
    label="Baseline Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline Training and Validation Loss")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

baseline_loss_path = (
    FIGURES_DIR /
    "baseline_loss_curve.png"
)

plt.savefig(
    baseline_loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Proposed Loss Curve
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_proposed,
    proposed_train,
    label="Proposed Train Loss"
)

plt.plot(
    epochs_proposed,
    proposed_val,
    label="Proposed Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Proposed Training and Validation Loss")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

proposed_loss_path = (
    FIGURES_DIR /
    "proposed_loss_curve.png"
)

plt.savefig(
    proposed_loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Combined Loss Curve
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    epochs_baseline,
    baseline_val,
    label="Baseline Validation Loss"
)

plt.plot(
    epochs_proposed,
    proposed_val,
    label="Proposed Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss Comparison")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

combined_loss_path = (
    FIGURES_DIR /
    "validation_loss_comparison.png"
)

plt.savefig(
    combined_loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Training graphs created:")
print("-", baseline_loss_path)
print("-", proposed_loss_path)
print("-", combined_loss_path)

print()
print("Model comparison completed.")