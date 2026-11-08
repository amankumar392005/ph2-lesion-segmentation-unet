from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split


SEED = 42
IMAGE_SIZE = (256, 256)

MANIFEST = Path("dataset/manifest.csv")
OUTPUT = Path("dataset/processed")


df = pd.read_csv(MANIFEST)

print("=" * 70)
print("PH2 DATASET PREPROCESSING")
print("=" * 70)

print("Total samples:", len(df))

if len(df) != 200:
    raise ValueError(
        f"Expected 200 samples, found {len(df)}"
    )

if df["id"].duplicated().any():
    raise ValueError("Duplicate IDs found.")

for _, row in df.iterrows():

    if not Path(row["image_path"]).is_file():
        raise FileNotFoundError(
            f"Image not found: {row['image_path']}"
        )

    if not Path(row["mask_path"]).is_file():
        raise FileNotFoundError(
            f"Mask not found: {row['mask_path']}"
        )


# --------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=SEED,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=SEED,
    shuffle=True
)


splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}


print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# --------------------------------------------------
# PROCESS IMAGES AND MASKS
# --------------------------------------------------

for split_name, split_df in splits.items():

    image_dir = OUTPUT / split_name / "images"
    mask_dir = OUTPUT / split_name / "masks"

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nProcessing {split_name} "
        f"({len(split_df)} samples)..."
    )

    for _, row in split_df.iterrows():

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        mask = Image.open(
            row["mask_path"]
        ).convert("L")

        # Resize image using bilinear interpolation
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )

        # Resize mask using nearest-neighbor
        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        # Convert mask to NumPy
        mask_array = np.array(mask)

        # Ensure binary mask
        mask_array = (
            mask_array > 127
        ).astype(np.uint8) * 255

        # Verify mask is not empty
        foreground_pixels = np.count_nonzero(
            mask_array
        )

        if foreground_pixels == 0:
            raise ValueError(
                f"Empty lesion mask: {row['id']}"
            )

        # Save processed image
        image.save(
            image_dir / f"{row['id']}.png"
        )

        # Save processed mask
        Image.fromarray(
            mask_array
        ).save(
            mask_dir / f"{row['id']}.png"
        )


# --------------------------------------------------
# SAVE SPLIT INFORMATION
# --------------------------------------------------

split_records = []

for split_name, split_df in splits.items():

    for image_id in split_df["id"]:

        split_records.append({
            "id": image_id,
            "split": split_name
        })


split_info = pd.DataFrame(
    split_records
)

split_info.to_csv(
    OUTPUT / "split.csv",
    index=False
)


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETED")
print("=" * 70)

print("\nSplit distribution:")
print(
    split_info["split"].value_counts()
)

print("\nImage size:", IMAGE_SIZE)

print(
    "Total processed images:",
    len(split_info)
)

print(
    "Output directory:",
    OUTPUT.resolve()
)