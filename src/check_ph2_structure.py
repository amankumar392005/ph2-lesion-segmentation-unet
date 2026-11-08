from pathlib import Path

ROOT = Path("dataset/raw/PH2 Dataset images")

print("=" * 70)
print("PH2 STRUCTURE CHECK")
print("=" * 70)

cases = sorted([p for p in ROOT.iterdir() if p.is_dir()])

print("Total case folders:", len(cases))

for case in cases[:5]:

    print("\nCASE:", case.name)

    for item in case.rglob("*"):

        if item.is_file():
            print(
                "FILE:",
                item.name,
                "| suffix:",
                repr(item.suffix),
                "| path:",
                item
            )
            