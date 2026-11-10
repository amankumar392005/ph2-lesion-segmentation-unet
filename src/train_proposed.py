import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import PH2Dataset
from model import UNet
from losses import CombinedLoss


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("dataset/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 30


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# Dataset
# ============================================================

train_dataset = PH2Dataset(DATASET_ROOT, "train")
val_dataset = PH2Dataset(DATASET_ROOT, "val")
test_dataset = PH2Dataset(DATASET_ROOT, "test")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# Model
# ============================================================

model = UNet(num_classes=2).to(device)


# ============================================================
# Proposed Compound Loss
# Cross-Entropy + Soft Dice
# ============================================================

criterion = CombinedLoss(dice_weight=1.0)

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training History
# ============================================================

history = {
    "train_loss": [],
    "val_loss": []
}

best_val_loss = float("inf")


# ============================================================
# Training Loop
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    running_train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)


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

            loss = criterion(outputs, masks)

            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)


    # --------------------------------------------------------
    # Print epoch results
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            MODEL_DIR / "proposed_unet_ce_dice.pth"
        )

        print("  -> Best proposed model saved.")


# ============================================================
# Save Training History
# ============================================================

history_path = RESULTS_DIR / "proposed_history.json"

with open(history_path, "w") as f:
    json.dump(history, f, indent=4)


# ============================================================
# Final Output
# ============================================================

print()
print("Proposed model training completed.")
print("Best validation loss:", best_val_loss)
print(
    "Model saved at:",
    MODEL_DIR / "proposed_unet_ce_dice.pth"
)
print(
    "History saved at:",
    history_path
)