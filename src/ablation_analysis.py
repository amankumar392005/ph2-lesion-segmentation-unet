import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

BASELINE_RESULTS = Path("results/baseline_test_results.json")
PROPOSED_RESULTS = Path("results/proposed_test_results.json")

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = Path("results/ablation_analysis.json")
OUTPUT_FIGURE = OUTPUT_DIR / "loss_ablation_comparison.png"


# ============================================================
# Load Results
# ============================================================

with open(BASELINE_RESULTS, "r") as f:
    baseline = json.load(f)

with open(PROPOSED_RESULTS, "r") as f:
    proposed = json.load(f)


# ============================================================
# Extract Metrics
# ============================================================

metrics = [
    "precision",
    "recall",
    "f1",
    "iou"
]

baseline_values = [
    baseline[metric]
    for metric in metrics
]

proposed_values = [
    proposed[metric]
    for metric in metrics
]


# ============================================================
# Calculate Changes
# ============================================================

changes = {}

for metric in metrics:

    baseline_value = baseline[metric]
    proposed_value = proposed[metric]

    changes[metric] = {
        "baseline": baseline_value,
        "proposed": proposed_value,
        "absolute_change": proposed_value - baseline_value,
        "percentage_point_change":
            (proposed_value - baseline_value) * 100
    }


# ============================================================
# Determine Best Model for Each Metric
# ============================================================

best_model = {}

for metric in metrics:

    if proposed[metric] > baseline[metric]:
        best_model[metric] = "Proposed"
    else:
        best_model[metric] = "Baseline"


# ============================================================
# Overall Interpretation
# ============================================================

if (
    proposed["f1"] > baseline["f1"]
    and proposed["iou"] > baseline["iou"]
):
    overall_result = (
        "The proposed compound loss improves "
        "overall segmentation performance."
    )

else:
    overall_result = (
        "The proposed compound loss improves precision "
        "but does not improve overall segmentation "
        "performance on the test split, as F1 and IoU "
        "are lower than the baseline."
    )


# ============================================================
# Save Ablation Results
# ============================================================

ablation_results = {
    "experiment": "Loss Function Ablation",
    "baseline": {
        "architecture": "U-Net",
        "loss": "Cross-Entropy"
    },
    "proposed": {
        "architecture": "U-Net",
        "loss": "Cross-Entropy + Soft Dice"
    },
    "metrics": changes,
    "best_model_by_metric": best_model,
    "overall_interpretation": overall_result
}


with open(OUTPUT_JSON, "w") as f:

    json.dump(
        ablation_results,
        f,
        indent=4
    )


# ============================================================
# Print Results
# ============================================================

print("=" * 70)
print("LOSS FUNCTION ABLATION ANALYSIS")
print("=" * 70)

print()
print(
    f"{'Metric':<15}"
    f"{'Baseline':>12}"
    f"{'Proposed':>12}"
    f"{'Change':>12}"
)

print("-" * 55)

for metric in metrics:

    print(
        f"{metric.capitalize():<15}"
        f"{baseline[metric]:>12.4f}"
        f"{proposed[metric]:>12.4f}"
        f"{proposed[metric] - baseline[metric]:>+12.4f}"
    )

print()
print("Best model by metric:")

for metric in metrics:

    print(
        f"  {metric.capitalize():<10}: "
        f"{best_model[metric]}"
    )

print()
print("Overall interpretation:")
print(overall_result)


# ============================================================
# Create Ablation Figure
# ============================================================

x = np.arange(len(metrics))

width = 0.35

fig, ax = plt.subplots(
    figsize=(10, 6)
)

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
    label="Proposed: U-Net + CE + Dice"
)

ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Score")

ax.set_title(
    "Loss Function Ablation: Baseline vs Proposed"
)

ax.set_xticks(x)

ax.set_xticklabels(
    ["Precision", "Recall", "F1", "IoU"]
)

ax.set_ylim(0, 1.0)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Completion
# ============================================================

print()
print("Ablation analysis saved to:")
print(OUTPUT_JSON)

print()
print("Ablation figure saved to:")
print(OUTPUT_FIGURE)

print()
print("Ablation analysis completed.")
