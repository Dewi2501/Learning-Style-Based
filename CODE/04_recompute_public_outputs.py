"""Hitung ringkasan yang dapat direproduksi dari public release."""
from __future__ import annotations

import json
import pandas as pd
from paths import PUBLIC_DATA, DERIVED, OUT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ils = pd.read_csv(DERIVED / "learner_ils_scores.csv")
    split = pd.read_csv(DERIVED / "primary_split_index.csv.gz")
    eligible = pd.read_csv(DERIVED / "post_ils_eligible_targets.csv.gz")

    dimensions = {
        "Active-reflective": "ils_active_reflective_score",
        "Sensing-intuitive": "ils_sensing_intuitive_score",
        "Visual-verbal": "ils_visual_verbal_score",
        "Sequential-global": "ils_sequential_global_score",
    }
    rows = []
    for name, col in dimensions.items():
        s = ils[col].astype(float)
        rows.append({
            "dimension": name,
            "n_learners": int(s.notna().sum()),
            "mean_score": float(s.mean()),
            "mild_abs_1_or_3_count": int(s.abs().isin([1, 3]).sum()),
        })
    pd.DataFrame(rows).to_csv(OUT / "ils_score_summary.csv", index=False)

    split_rows = []
    for group in ["train", "validation", "test"]:
        part = split.loc[split["split"].eq(group)].copy()
        split_rows.append({
            "group": group,
            "observations": len(part),
            "learners": part["learner_id"].nunique(),
            "courses": part["course_id"].nunique(),
            "quizzes": part[["course_id", "quiz_id"]].drop_duplicates().shape[0],
        })
    pd.DataFrame(split_rows).to_csv(OUT / "primary_split_summary.csv", index=False)

    test = split.loc[split["split"].eq("test")]
    summary = {
        "eligible_targets": len(eligible),
        "eligible_learners": int(eligible["learner_id"].nunique()),
        "eligible_courses": int(eligible["course_id"].nunique()),
        "test_ils_age_median_days": float(test["days_since_ils"].median()),
        "test_ils_age_max_days": float(test["days_since_ils"].max()),
    }
    (OUT / "temporal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
