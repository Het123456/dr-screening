"""Exercise 1: look at the MMRDR CFP labels (rough exploration script)."""

import csv
import json
from pathlib import Path

from PIL import Image

# Folder that holds MMRDR-CFP, relative to this file. Change it if your data is elsewhere.
CFP_FOLDER = Path(__file__).resolve().parent.parent.parent / "data" / "MMRDR" / "MMRDR-CFP"
CSV_FILE = CFP_FOLDER / "FP.csv"

LESION_NAMES = [
    "microaneurysm",
    "hard exudate",
    "intraretinal hemorrhage",
    "VB/IRMA",
    "neovascularization",
    "vitreous hemorrhage",
    "retinal detachment",
]

total = 0
grade_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
lesion_counts = [0] * 7
first_row = None

with open(CSV_FILE, newline="") as f:
    for row in csv.DictReader(f):
        total += 1
        if first_row is None:
            first_row = row

        grade_counts[int(row["grade"])] += 1

        # The lesion column is text like "[1, 0, 0, 0, 0, 0, 0]"; json.loads turns it into a list.
        lesions = json.loads(row["lesion"])
        for i in range(7):
            lesion_counts[i] += lesions[i]

print(f"Total images: {total}")

print("\nImages per DR grade:")
for grade, count in grade_counts.items():
    print(f"  grade {grade}: {count} ({100 * count / total:.1f}%)")

print("\nImages with each lesion:")
for name, count in zip(LESION_NAMES, lesion_counts):
    print(f"  {name}: {count}")

# Show the first image with its labels.
lesions = json.loads(first_row["lesion"])
present = [name for name, v in zip(LESION_NAMES, lesions) if v == 1]
print(f"\nFirst image: {first_row['image']}")
print(f"  grade: {first_row['grade']}")
print(f"  lesions present: {present if present else 'none'}")
Image.open(CFP_FOLDER / first_row["image"]).show()
