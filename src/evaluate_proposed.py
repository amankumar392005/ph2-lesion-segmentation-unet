import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PH2Dataset
from model import UNet


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("dataset/processed")
MODEL_PATH = Path("models/proposed_unet_ce_dice.pth")

RESULTS_DIR = Path("results")
PREDICTION_DIR = RESULTS_DIR / "proposed_predictions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# Test Dataset
# ============================================================

test_dataset = PH2Dataset(
    DATASET_ROOT,
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test samples:", len(test_dataset))


# ============================================================
# Load Proposed Model
# ============================================================

model = UNet(num_classes=2).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Loaded model:", MODEL_PATH)


# ============================================================
# Evaluation Variables
# ============================================================

TP = 0
FP = 0
FN = 0
TN = 0


# ============================================================
# Evaluate on Test Set
# ============================================================

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        predictions_np = predictions.cpu().numpy()
        masks_np = masks.cpu().numpy()

        # ----------------------------------------------------
        # Calculate confusion values
        # ----------------------------------------------------

        TP += np.logical_and(
            predictions_np == 1,
            masks_np == 1
        ).sum()

        FP += np.logical_and(
            predictions_np == 1,
            masks_np == 0
        ).sum()

        FN += np.logical_and(
            predictions_np == 0,
            masks_np == 1
        ).sum()

        TN += np.logical_and(
            predictions_np == 0,
            masks_np == 0
        ).sum()


# ============================================================
# Calculate Metrics
# ============================================================

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)

iou = (
    TP / (TP + FP + FN)
    if (TP + FP + FN) > 0
    else 0.0
)


# ============================================================
# Print Results
# ============================================================

print()
print("=" * 50)
print("PROPOSED TEST RESULTS")
print("=" * 50)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"IoU       : {iou:.4f}")

print()
print("Confusion values:")
print("True Positive :", TP)
print("False Positive:", FP)
print("False Negative:", FN)
print("True Negative :", TN)


# ============================================================
# Save Numerical Results
# ============================================================

results = {
    "model": "U-Net + Cross-Entropy + Soft Dice",
    "test_samples": len(test_dataset),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "iou": float(iou),
    "true_positive": int(TP),
    "false_positive": int(FP),
    "false_negative": int(FN),
    "true_negative": int(TN)
}

results_path = RESULTS_DIR / "proposed_test_results.json"

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print()
print("Results saved to:", results_path)


# ============================================================
# Create Qualitative Visualizations
# ============================================================

print()
print("Creating prediction visualizations...")

# Save only first 5 images for now.
# Metrics above were calculated using ALL 30 test images.

num_visualizations = min(5, len(test_dataset))

for i in range(num_visualizations):

    image, mask = test_dataset[i]

    image_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        prediction = torch.argmax(
            output,
            dim=1
        ).squeeze(0).cpu().numpy()

    # --------------------------------------------------------
    # Convert image for visualization
    # --------------------------------------------------------

    image_np = image.permute(1, 2, 0).numpy()

    image_np = (image_np * 255).astype(np.uint8)

    # RGB -> BGR for OpenCV
    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Ground truth mask
    # --------------------------------------------------------

    mask_np = mask.numpy().astype(np.uint8) * 255

    # --------------------------------------------------------
    # Prediction mask
    # --------------------------------------------------------

    prediction_np = prediction.astype(np.uint8) * 255

    # --------------------------------------------------------
    # Resize masks for display
    # --------------------------------------------------------

    mask_bgr = cv2.cvtColor(
        mask_np,
        cv2.COLOR_GRAY2BGR
    )

    prediction_bgr = cv2.cvtColor(
        prediction_np,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Add labels
    # --------------------------------------------------------

    cv2.putText(
        image_bgr,
        "Original Image",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        mask_bgr,
        "Ground Truth",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        prediction_bgr,
        "Proposed Prediction",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Combine images horizontally
    # --------------------------------------------------------

    combined = np.hstack([
        image_bgr,
        mask_bgr,
        prediction_bgr
    ])

    output_path = (
        PREDICTION_DIR /
        f"prediction_{i + 1}.png"
    )

    cv2.imwrite(
        str(output_path),
        combined
    )


print(
    "Saved visualizations to:",
    PREDICTION_DIR
)

print()
print("Proposed evaluation completed.")