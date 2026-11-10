# ============================================================
# evaluate.py
#
# Purpose:
#   Evaluate the trained baseline U-Net on the unseen
#   PH2 test dataset.
#
# Baseline:
#   U-Net + Cross-Entropy
#
# Metrics:
#   Precision
#   Recall
#   F1 Score
#   IoU
#
# Also saves:
#   - Numerical test results
#   - Sample prediction visualizations
# ============================================================

from pathlib import Path
import json

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import PH2Dataset
from model import UNet


# ============================================================
# 1. Configuration
# ============================================================

DATASET_ROOT = Path("dataset/processed")

MODEL_PATH = Path(
    "models/baseline_unet_ce.pth"
)

RESULTS_DIR = Path("results")

PREDICTION_DIR = (
    RESULTS_DIR / "baseline_predictions"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. Load test dataset
# ============================================================

test_dataset = PH2Dataset(
    DATASET_ROOT,
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# 4. Load trained model
# ============================================================

model = UNet(
    num_classes=2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print(
    "Loaded model:",
    MODEL_PATH
)


# ============================================================
# 5. Metric counters
# ============================================================

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0


# ============================================================
# 6. Evaluate test set
# ============================================================

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Model prediction
        outputs = model(images)

        # Convert logits to predicted class
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # ----------------------------------------------------
        # Calculate confusion values
        # ----------------------------------------------------

        true_positive += (
            (predictions == 1)
            & (masks == 1)
        ).sum().item()

        false_positive += (
            (predictions == 1)
            & (masks == 0)
        ).sum().item()

        false_negative += (
            (predictions == 0)
            & (masks == 1)
        ).sum().item()

        true_negative += (
            (predictions == 0)
            & (masks == 0)
        ).sum().item()


# ============================================================
# 7. Calculate metrics
# ============================================================

# Precision
precision_denominator = (
    true_positive + false_positive
)

if precision_denominator == 0:
    precision = 0.0
else:
    precision = (
        true_positive
        / precision_denominator
    )


# Recall
recall_denominator = (
    true_positive + false_negative
)

if recall_denominator == 0:
    recall = 0.0
else:
    recall = (
        true_positive
        / recall_denominator
    )


# F1 Score
f1_denominator = (
    precision + recall
)

if f1_denominator == 0:
    f1 = 0.0
else:
    f1 = (
        2 * precision * recall
        / f1_denominator
    )


# Intersection over Union
iou_denominator = (
    true_positive
    + false_positive
    + false_negative
)

if iou_denominator == 0:
    iou = 0.0
else:
    iou = (
        true_positive
        / iou_denominator
    )


# ============================================================
# 8. Print results
# ============================================================

print()
print("=" * 50)
print("BASELINE TEST RESULTS")
print("=" * 50)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"IoU       : {iou:.4f}"
)

print()
print("Confusion values:")
print(
    "True Positive :",
    true_positive
)

print(
    "False Positive:",
    false_positive
)

print(
    "False Negative:",
    false_negative
)

print(
    "True Negative :",
    true_negative
)


# ============================================================
# 9. Save numerical results
# ============================================================

results = {
    "model": "U-Net + Cross-Entropy",
    "test_samples": len(test_dataset),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "iou": iou,
    "true_positive": true_positive,
    "false_positive": false_positive,
    "false_negative": false_negative,
    "true_negative": true_negative
}

results_path = (
    RESULTS_DIR / "baseline_test_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print()
print(
    "Results saved to:",
    results_path
)


# ============================================================
# 10. Save sample visualizations
# ============================================================

print()
print("Creating prediction visualizations...")


# Select a few test images
num_visualizations = min(
    5,
    len(test_dataset)
)


for index in range(
    num_visualizations
):

    image, mask = test_dataset[index]

    # Add batch dimension
    image_input = image.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        output = model(
            image_input
        )

        prediction = torch.argmax(
            output,
            dim=1
        )[0].cpu().numpy()

    # Convert image back to H x W x C
    image_np = (
        image.numpy()
        .transpose(1, 2, 0)
    )

    # Convert mask to numpy
    mask_np = mask.numpy()

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 4)
    )

    # Original image
    plt.subplot(
        1, 3, 1
    )

    plt.imshow(image_np)

    plt.title(
        "Original Image"
    )

    plt.axis("off")

    # Ground truth
    plt.subplot(
        1, 3, 2
    )

    plt.imshow(
        mask_np,
        cmap="gray"
    )

    plt.title(
        "Ground Truth"
    )

    plt.axis("off")

    # Prediction
    plt.subplot(
        1, 3, 3
    )

    plt.imshow(
        prediction,
        cmap="gray"
    )

    plt.title(
        "Baseline Prediction"
    )

    plt.axis("off")

    plt.tight_layout()

    save_path = (
        PREDICTION_DIR
        / f"prediction_{index + 1}.png"
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


print(
    "Saved visualizations to:",
    PREDICTION_DIR
)

print()
print("Baseline evaluation completed.")