import json
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RESULTS_DIR = Path("results")

BASELINE_RESULTS = RESULTS_DIR / "baseline_test_results.json"
PROPOSED_RESULTS = RESULTS_DIR / "proposed_test_results.json"
ABLATION_RESULTS = RESULTS_DIR / "ablation_analysis.json"

OUTPUT_FILE = RESULTS_DIR / "final_results_summary.json"


# ============================================================
# Load Experimental Results
# ============================================================

with open(BASELINE_RESULTS, "r") as f:
    baseline = json.load(f)

with open(PROPOSED_RESULTS, "r") as f:
    proposed = json.load(f)

with open(ABLATION_RESULTS, "r") as f:
    ablation = json.load(f)


# ============================================================
# Calculate Changes
# ============================================================

metrics = [
    "precision",
    "recall",
    "f1",
    "iou"
]

comparison = {}

for metric in metrics:

    baseline_value = baseline[metric]
    proposed_value = proposed[metric]

    comparison[metric] = {
        "baseline": baseline_value,
        "proposed": proposed_value,
        "absolute_change": (
            proposed_value - baseline_value
        ),
        "percentage_point_change": (
            proposed_value - baseline_value
        ) * 100
    }


# ============================================================
# Final Results Summary
# ============================================================

final_results = {

    "project": {
        "name": "Lesion Segmentation",
        "dataset": "PH2",
        "image_size": "256x256",
        "train_samples": 140,
        "validation_samples": 30,
        "test_samples": 30,
        "random_seed": 42
    },

    "baseline": {
        "architecture": "U-Net",
        "loss": "Cross-Entropy",
        "precision": baseline["precision"],
        "recall": baseline["recall"],
        "f1": baseline["f1"],
        "iou": baseline["iou"]
    },

    "proposed": {
        "architecture": "U-Net",
        "loss": "Cross-Entropy + Soft Dice",
        "precision": proposed["precision"],
        "recall": proposed["recall"],
        "f1": proposed["f1"],
        "iou": proposed["iou"]
    },

    "comparison": comparison,

    "ablation": {
        "type": "Loss Function Ablation",
        "baseline_loss": "Cross-Entropy",
        "proposed_loss": "Cross-Entropy + Soft Dice",
        "best_model_by_metric":
            ablation["best_model_by_metric"]
    },

    "overall_conclusion": (
        "The proposed compound loss improves precision "
        "but does not improve overall segmentation "
        "performance on the test split. The baseline "
        "achieves higher recall, F1, and IoU."
    )
}


# ============================================================
# Save
# ============================================================

with open(OUTPUT_FILE, "w") as f:

    json.dump(
        final_results,
        f,
        indent=4
    )


# ============================================================
# Print Final Results
# ============================================================

print("=" * 75)
print("FINAL EXPERIMENTAL RESULTS")
print("=" * 75)

print()
print("Dataset:")
print("  PH2")
print("  Train      :", 140)
print("  Validation :", 30)
print("  Test       :", 30)
print("  Image Size :", "256x256")

print()
print("Baseline: U-Net + Cross-Entropy")
print("-" * 45)

print(
    f"Precision : {baseline['precision']:.4f}"
)

print(
    f"Recall    : {baseline['recall']:.4f}"
)

print(
    f"F1 Score  : {baseline['f1']:.4f}"
)

print(
    f"IoU       : {baseline['iou']:.4f}"
)

print()
print("Proposed: U-Net + Cross-Entropy + Soft Dice")
print("-" * 45)

print(
    f"Precision : {proposed['precision']:.4f}"
)

print(
    f"Recall    : {proposed['recall']:.4f}"
)

print(
    f"F1 Score  : {proposed['f1']:.4f}"
)

print(
    f"IoU       : {proposed['iou']:.4f}"
)

print()
print("Changes")
print("-" * 45)

for metric in metrics:

    change = comparison[metric]["absolute_change"]

    print(
        f"{metric.capitalize():<10}: "
        f"{change:+.4f}"
    )

print()
print("Overall conclusion:")
print(final_results["overall_conclusion"])

print()
print("Saved to:")
print(OUTPUT_FILE)