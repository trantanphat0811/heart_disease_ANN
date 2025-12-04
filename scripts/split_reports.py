"""Split files in `reports/` into `reports/images/` (png) and `reports/tables/` (csv).

Usage:
    python scripts/split_reports.py

This script moves files (not copies). It will create the target
directories if they don't exist.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
IMAGES = REPORTS / "images"
TABLES = REPORTS / "tables"

IMAGES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

def move_files():
    moved = []
    for p in REPORTS.iterdir():
        if p.is_dir():
            continue
        if p.suffix.lower() == ".png":
            dest = IMAGES / p.name
            shutil.move(str(p), str(dest))
            moved.append((p.name, "images"))
        elif p.suffix.lower() == ".csv":
            dest = TABLES / p.name
            shutil.move(str(p), str(dest))
            moved.append((p.name, "tables"))
    return moved

if __name__ == "__main__":
    moved = move_files()
    if not moved:
        print("No files moved — reports/ has no png/csv at root (or already split).")
    else:
        for name, folder in moved:
            print(f"Moved {name} -> reports/{folder}/")
