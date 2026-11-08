from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("dataset/processed")

print("=" * 70)
print("VERIFYING PROCESSED DATASET")
print("=" * 70)

for split in ["train", "val", "test"]:

    image_dir = ROOT / split / "images"
    mask_dir = ROOT / split / "masks"

    images = sorted(image_dir.glob("*.png"))
    masks = sorted(mask_dir.glob("*.png"))

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Masks:", len(masks))

    if len(images) != len(masks):
        raise ValueError(
            f"{split}: image/mask count mismatch"
        )

    for image_path, mask_path in zip(images, masks):

        if image_path.stem != mask_path.stem:
            raise ValueError(
                f"Name mismatch: "
                f"{image_path.name} vs {mask_path.name}"
            )

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if image.size != (256, 256):
            raise ValueError(
                f"Wrong image size: "
                f"{image_path}"
            )

        if mask.size != (256, 256):
            raise ValueError(
                f"Wrong mask size: "
                f"{mask_path}"
            )

        mask_array = np.array(mask)

        unique_values = np.unique(mask_array)

        if not np.all(
            np.isin(unique_values, [0, 255])
        ):
            raise ValueError(
                f"Invalid mask values in "
                f"{mask_path}: {unique_values}"
            )

    print("✓ Image/mask counts match")
    print("✓ All images are 256×256")
    print("✓ All masks are 256×256")
    print("✓ All masks contain only 0 and 255")


print("\n" + "=" * 70)
print("DATASET VERIFICATION PASSED")
print("=" * 70)