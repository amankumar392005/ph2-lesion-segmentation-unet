from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


MANIFEST = Path("dataset/manifest.csv")

df = pd.read_csv(MANIFEST)

print("=" * 70)
print("PH2 IMAGE-MASK VERIFICATION")
print("=" * 70)

print("Total pairs:", len(df))

for _, row in df.head(5).iterrows():

    image_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_array = np.array(image)
    mask_array = np.array(mask)

    print("\nID:", row["id"])
    print("Image size:", image.size)
    print("Mask size:", mask.size)
    print("Image mode:", image.mode)
    print("Mask mode:", mask.mode)

    print(
        "Mask min:",
        mask_array.min(),
        "| Mask max:",
        mask_array.max()
    )

    print(
        "Unique mask values:",
        np.unique(mask_array)
    )

    print(
        "Foreground pixels (>0):",
        np.count_nonzero(mask_array)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(image_array)
    axes[0].set_title("Original Image")

    axes[1].imshow(
        mask_array,
        cmap="gray"
    )
    axes[1].set_title("Ground Truth Lesion Mask")

    axes[2].imshow(image_array)
    axes[2].imshow(
        mask_array,
        cmap="jet",
        alpha=0.4
    )
    axes[2].set_title("Image + Mask Overlay")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(row["id"])
    plt.tight_layout()
    plt.show()