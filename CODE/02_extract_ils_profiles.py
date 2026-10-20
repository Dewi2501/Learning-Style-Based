"""Ekstrak satu profil ILS per learner dari release publik."""
from __future__ import annotations

import pandas as pd
from paths import PUBLIC_DATA, DERIVED

SCORE_COLUMNS = [
    "ils_active_reflective_score",
    "ils_sensing_intuitive_score",
    "ils_visual_verbal_score",
    "ils_sequential_global_score",
]


def main():
    use = [
        "userid", "has_ils", "ils_response_id", "ils_reference_courseid",
        "ils_submitted_unix", "ils_submitted_local", *SCORE_COLUMNS,
        "ils_active_reflective_code", "ils_sensing_intuitive_code",
        "ils_visual_verbal_code", "ils_sequential_global_code",
        "ils_ls_class", "ils_semantic_label",
    ]
    df = pd.read_csv(PUBLIC_DATA, usecols=use, low_memory=False)
    profiles = (df.loc[df["has_ils"].eq(1)]
                  .drop_duplicates("userid")
                  .sort_values("userid")
                  .rename(columns={"userid": "learner_id"}))
    path = DERIVED / "learner_ils_scores.csv"
    profiles.to_csv(path, index=False)
    print(f"Wrote {len(profiles)} learner profiles -> {path}")
    for col in SCORE_COLUMNS:
        print(col, "mean=", round(float(profiles[col].mean()), 6))


if __name__ == "__main__":
    main()
