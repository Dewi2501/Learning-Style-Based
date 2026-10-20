"""Recompute ridge models when the anonymized analysis matrix is available.

The public activity release does not contain target quiz scores or the complete
18-variable model matrix. This script intentionally refuses to invent them.
Place the genuine pseudonymous matrix at DATA/analysis/analysis_matrix.csv.gz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from paths import ROOT

MATRIX = ROOT / "DATA" / "analysis" / "analysis_matrix.csv.gz"
LAMBDA_GRID = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
READINESS = [
    "log_count_28d", "active_days_28d", "quiz_activity_28d", "assignment_activity_28d",
    "material_activity_28d", "forum_activity_28d", "attendance_activity_28d",
    "days_since_last_activity", "no_activity_28d", "prior_quiz_count_all",
    "prior_quiz_mean_all", "prior_quiz_missing_all", "prior_quiz_count_course",
    "prior_quiz_mean_course", "prior_quiz_missing_course", "days_since_course_start",
    "days_to_quiz_deadline", "quiz_max_points",
]
ILS = [
    "ils_active_reflective_score", "ils_sensing_intuitive_score",
    "ils_visual_verbal_score", "ils_sequential_global_score",
]


def fit_preprocessor(train_x):
    means = train_x.mean(axis=0, skipna=True)
    x = train_x.fillna(means)
    center = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0).replace(0, 1.0)
    return means, center, scale


def transform(x, means, center, scale):
    return (x.fillna(means) - center) / scale


def ridge_fit(x, y, lam):
    # Intercept tidak dipenalti. Predictor sudah distandardisasi.
    x1 = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(x1.shape[1]) * lam
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(x1.T @ x1 + penalty) @ (x1.T @ y)
    return beta


def predict(x, beta):
    x1 = np.column_stack([np.ones(len(x)), x])
    return np.clip(x1 @ beta, 0.0, 1.0)


def mae(y, p):
    return float(np.mean(np.abs(y - p)) * 100.0)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)) * 100.0)


def run_family(df, features):
    train = df[df.split.eq("train")].copy()
    val = df[df.split.eq("validation")].copy()
    test = df[df.split.eq("test")].copy()
    means, center, scale = fit_preprocessor(train[features])
    xt = transform(train[features], means, center, scale).to_numpy(float)
    xv = transform(val[features], means, center, scale).to_numpy(float)
    xs = transform(test[features], means, center, scale).to_numpy(float)
    yt = train.actual_score.to_numpy(float)
    yv = val.actual_score.to_numpy(float)
    ys = test.actual_score.to_numpy(float)

    rows = []
    for lam in LAMBDA_GRID:
        beta = ridge_fit(xt, yt, lam)
        pv = predict(xv, beta)
        ps = predict(xs, beta)
        rows.append({"lambda": lam, "validation_mae": mae(yv, pv),
                     "test_mae": mae(ys, ps), "test_rmse": rmse(ys, ps)})
    return pd.DataFrame(rows)


def main():
    if not MATRIX.exists():
        raise SystemExit(
            "Missing DATA/analysis/analysis_matrix.csv.gz. "
            "Use the genuine pseudonymous analysis matrix; do not create replacement rows."
        )
    df = pd.read_csv(MATRIX)
    required = set(["split", "actual_score", "learner_id", *READINESS, *ILS])
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit("Missing columns: " + ", ".join(missing))

    out = ROOT / "OUTPUT" / "full_reproduction"
    out.mkdir(parents=True, exist_ok=True)
    run_family(df, READINESS).to_csv(out / "ridge_readiness_grid.csv", index=False)
    run_family(df, READINESS + ILS).to_csv(out / "ridge_readiness_plus_ils_grid.csv", index=False)
    print("Model grids written to", out)


if __name__ == "__main__":
    main()
