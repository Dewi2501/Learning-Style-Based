"""Periksa struktur dasar public research release."""
from __future__ import annotations

import hashlib
import gzip
import pandas as pd
from paths import PUBLIC_DATA

EXPECTED = {
    "rows": 617666,
    "columns": 55,
    "learners": 565,
    "courses": 149,
    "ils_learners": 461,
}


def sha256_uncompressed(path):
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    df = pd.read_csv(PUBLIC_DATA, low_memory=False)
    got = {
        "rows": len(df),
        "columns": len(df.columns),
        "learners": df["userid"].nunique(),
        "courses": df["courseid"].nunique(),
        "ils_learners": df.loc[df["has_ils"].eq(1), "userid"].nunique(),
    }
    for key, expected in EXPECTED.items():
        if got[key] != expected:
            raise SystemExit(f"FAILED {key}: {got[key]} != {expected}")
        print(f"OK {key}: {got[key]}")
    print("SHA-256 (uncompressed CSV):", sha256_uncompressed(PUBLIC_DATA))


if __name__ == "__main__":
    main()
