# ============================================================
# train.py
#
# Purpose:
#   Train U-Net for PH2 lesion segmentation.
#
# Baseline:
#   U-Net + Cross-Entropy Loss
#
# Dataset:
#   PH2
#
# Input size:
#   256 x 256
#
# Split:
#   140 train
#   30 validation
#   30 test
# ============================================================

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PH2Dataset
from model import UNet


# ============================================================
# 1. Configuration
# ============================================================

DATASET_ROOT = Path(
    "dataset/processed"
)

MODEL_DIR = Path(
    "models"
)

RESULTS_DIR = Path(
    "results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# Training parameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 30


# ============================================================
# 2. Dataset
# ============================================================

train_dataset = PH2Dataset(
    DATASET_ROOT,
    "train"
)

val_dataset = PH2Dataset(
    DATASET_ROOT,
    "val"
)

test_dataset = PH2Dataset(
    DATASET_ROOT,
    "test"
)


print(
    "Train samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# 3. DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 4. Model
# ============================================================

model = UNet(
    num_classes=2
)

model = model.to(device)


# ============================================================
# 5. Baseline Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 7. Training
# ============================================================

best_val_loss = float("inf")

train_losses = []
val_losses = []


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            masks
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        running_train_loss += (
            loss.item()
            * images.size(0)
        )

    epoch_train_loss = (
        running_train_loss
        / len(train_dataset)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    running_val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            running_val_loss += (
                loss.item()
                * images.size(0)
            )

    epoch_val_loss = (
        running_val_loss
        / len(val_dataset)
    )

    train_losses.append(
        epoch_train_loss
    )

    val_losses.append(
        epoch_val_loss
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {epoch_train_loss:.4f} "
        f"Val Loss: {epoch_val_loss:.4f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if epoch_val_loss < best_val_loss:

        best_val_loss = epoch_val_loss

        torch.save(
            model.state_dict(),
            MODEL_DIR / "baseline_unet_ce.pth"
        )

        print(
            "  -> Best model saved."
        )


# ============================================================
# 8. Save training history
# ============================================================

import json

history = {
    "train_loss": train_losses,
    "val_loss": val_losses
}

with open(
    RESULTS_DIR / "baseline_history.json",
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


print()
print("Training completed.")
print(
    "Best validation loss:",
    best_val_loss
)
print(
    "Model saved at:",
    MODEL_DIR / "baseline_unet_ce.pth"
)