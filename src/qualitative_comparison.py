import cv2
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt

from dataset import PH2Dataset
from model import UNet


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("dataset/processed")

BASELINE_MODEL = Path("models/baseline_unet_ce.pth")
PROPOSED_MODEL = Path("models/proposed_unet_ce_dice.pth")

OUTPUT_DIR = Path(
    "results/figures/qualitative_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

NUM_IMAGES = 5


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Load Test Dataset
# ============================================================

test_dataset = PH2Dataset(
    DATASET_ROOT,
    "test"
)

print("Test samples:", len(test_dataset))


# ============================================================
# Load Baseline Model
# ============================================================

baseline_model = UNet(num_classes=2).to(device)

baseline_model.load_state_dict(
    torch.load(
        BASELINE_MODEL,
        map_location=device
    )
)

baseline_model.eval()

print(
    "Loaded baseline:",
    BASELINE_MODEL
)


# ============================================================
# Load Proposed Model
# ============================================================

proposed_model = UNet(num_classes=2).to(device)

proposed_model.load_state_dict(
    torch.load(
        PROPOSED_MODEL,
        map_location=device
    )
)

proposed_model.eval()

print(
    "Loaded proposed:",
    PROPOSED_MODEL
)


# ============================================================
# Generate Qualitative Comparisons
# ============================================================

print()
print("Creating qualitative comparisons...")


for i in range(
    min(NUM_IMAGES, len(test_dataset))
):

    image, mask = test_dataset[i]

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image_tensor = (
        image
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    with torch.no_grad():

        baseline_output = baseline_model(
            image_tensor
        )

        proposed_output = proposed_model(
            image_tensor
        )

        baseline_prediction = torch.argmax(
            baseline_output,
            dim=1
        ).squeeze(0).cpu().numpy()

        proposed_prediction = torch.argmax(
            proposed_output,
            dim=1
        ).squeeze(0).cpu().numpy()

    # --------------------------------------------------------
    # Convert image and masks
    # --------------------------------------------------------

    image_np = image.permute(
        1, 2, 0
    ).numpy()

    mask_np = mask.numpy()

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(
        mask_np,
        cmap="gray"
    )
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Baseline prediction
    axes[2].imshow(
        baseline_prediction,
        cmap="gray"
    )
    axes[2].set_title(
        "Baseline Prediction\nU-Net + CE"
    )
    axes[2].axis("off")

    # Proposed prediction
    axes[3].imshow(
        proposed_prediction,
        cmap="gray"
    )
    axes[3].set_title(
        "Proposed Prediction\nU-Net + CE + Dice"
    )
    axes[3].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"comparison_{i + 1}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


print()
print(
    "Qualitative comparison completed."
)
print(
    "Saved to:",
    OUTPUT_DIR
)