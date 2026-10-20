"""Jalankan bagian yang dapat direproduksi dari public release."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "01_validate_public_release.py",
    "02_extract_ils_profiles.py",
    "03_build_temporal_split.py",
    "04_recompute_public_outputs.py",
    "05_verify_against_manuscript.py",
]

for script in SCRIPTS:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(HERE / script)], check=True)
