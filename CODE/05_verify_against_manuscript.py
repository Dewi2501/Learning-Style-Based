"""Verifikasi bahwa output publik yang dihitung ulang cocok dengan angka audit manuskrip."""
from __future__ import annotations

import json
import math
import pandas as pd
from paths import ROOT, OUT


def close(a, b, tol=0.005):
    return math.isclose(float(a), float(b), abs_tol=tol)


def main():
    ref = ROOT / "OUTPUT" / "manuscript_reference"
    ils = pd.read_csv(OUT / "ils_score_summary.csv")
    split = pd.read_csv(OUT / "primary_split_summary.csv")
    temporal_path = OUT / "release_summary.json"
    if temporal_path.exists():
        temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    else:
        temporal = json.loads((OUT / "temporal_summary.json").read_text(encoding="utf-8"))

    expected_means = {
        "Active-reflective": 2.163,
        "Sensing-intuitive": 1.364,
        "Visual-verbal": -1.030,
        "Sequential-global": -3.725,
    }
    for dim, expected in expected_means.items():
        got = ils.loc[ils["dimension"].eq(dim), "mean_score"].iloc[0]
        if not close(got, expected):
            raise SystemExit(f"ILS mean mismatch {dim}: {got} != {expected}")
        print(f"OK ILS mean {dim}: {got:.3f}")

    expected_split = {"train": 2870, "validation": 728, "test": 652,
                      "Training": 2870, "Validation": 728, "Test": 652}
    for _, row in split.iterrows():
        if int(row.observations) != expected_split[row.group]:
            raise SystemExit(f"Split mismatch {row.group}")
        print(f"OK {row.group}: {int(row.observations)} observations")

    eligible_value = temporal.get("post_ils_eligible_targets", temporal.get("eligible_targets"))
    if int(eligible_value) != 7612:
        raise SystemExit("Post-ILS target count mismatch")
    print("OK post-ILS eligible targets: 7612")


if __name__ == "__main__":
    main()
