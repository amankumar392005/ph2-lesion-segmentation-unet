from pathlib import Path
import pandas as pd

ROOT = Path("dataset/raw/PH2 Dataset images")
OUTPUT = Path("dataset/manifest.csv")

if not ROOT.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{ROOT.resolve()}"
    )

rows = []

case_folders = sorted(
    [p for p in ROOT.iterdir() if p.is_dir()]
)

print("=" * 70)
print("BUILDING PH2 DATASET MANIFEST")
print("=" * 70)

print("Dataset path:", ROOT.resolve())
print("Case folders found:", len(case_folders))

for case_folder in case_folders:

    case_id = case_folder.name

    image_folder = case_folder / f"{case_id}_Dermoscopic_Image"
    mask_folder = case_folder / f"{case_id}_lesion"

    image_files = list(image_folder.glob("*.bmp"))
    mask_files = list(mask_folder.glob("*.bmp"))

    if len(image_files) != 1:
        print(
            f"[ERROR] {case_id}: "
            f"expected 1 dermoscopic image, found {len(image_files)}"
        )
        continue

    if len(mask_files) != 1:
        print(
            f"[ERROR] {case_id}: "
            f"expected 1 lesion mask, found {len(mask_files)}"
        )
        continue

    rows.append({
        "id": case_id,
        "image_path": str(image_files[0]),
        "mask_path": str(mask_files[0])
    })

print("\nValid image-mask pairs:", len(rows))

if len(rows) != 200:
    raise ValueError(
        f"Expected exactly 200 PH2 cases, "
        f"but found {len(rows)} valid pairs."
    )

df = pd.DataFrame(rows)

if df["id"].duplicated().any():
    raise ValueError("Duplicate case IDs found.")

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 70)
print("MANIFEST CREATED SUCCESSFULLY")
print("=" * 70)

print("Saved to:", OUTPUT.resolve())

print("\nFirst 5 entries:")
print(df.head().to_string(index=False))

print("\nTotal pairs:", len(df))