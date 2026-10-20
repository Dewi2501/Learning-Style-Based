"""Cluster bootstrap for prediction comparisons on the untouched test set.

Requires DATA/analysis/test_predictions.csv.gz with one row per test observation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from paths import ROOT

PRED = ROOT / "DATA" / "analysis" / "test_predictions.csv.gz"
SEED = 20261019
N_BOOT = 1000


def mae(y, p):
    return np.mean(np.abs(y - p)) * 100.0


def main():
    if not PRED.exists():
        raise SystemExit(
            "Missing DATA/analysis/test_predictions.csv.gz. "
            "Provide genuine test predictions before running bootstrap."
        )
    df = pd.read_csv(PRED)
    required = {"learner_id", "actual_score", "pred_training_mean", "pred_prior_quiz_mean",
                "pred_readiness_lam_0_1", "pred_readiness_ils_lam_0_1",
                "pred_readiness_lam_1", "pred_readiness_ils_lam_1"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit("Missing columns: " + ", ".join(missing))

    rng = np.random.default_rng(SEED)
    learners = df.learner_id.drop_duplicates().to_numpy()
    records = []
    for b in range(N_BOOT):
        sampled = rng.choice(learners, size=len(learners), replace=True)
        pieces = []
        for draw, learner in enumerate(sampled):
            g = df[df.learner_id.eq(learner)].copy()
            g["_draw"] = draw
            pieces.append(g)
        boot = pd.concat(pieces, ignore_index=True)
        y = boot.actual_score.to_numpy(float)
        records.append({
            "replication": b + 1,
            "mae_training_mean": mae(y, boot.pred_training_mean),
            "mae_prior_quiz_mean": mae(y, boot.pred_prior_quiz_mean),
            "delta_prior_minus_training": mae(y, boot.pred_prior_quiz_mean) - mae(y, boot.pred_training_mean),
            "delta_ils_lam_0_1": mae(y, boot.pred_readiness_ils_lam_0_1) - mae(y, boot.pred_readiness_lam_0_1),
            "delta_ils_lam_1": mae(y, boot.pred_readiness_ils_lam_1) - mae(y, boot.pred_readiness_lam_1),
        })
    out = ROOT / "OUTPUT" / "full_reproduction"
    out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result.to_csv(out / "bootstrap_1000.csv.gz", index=False, compression="gzip")
    print(result.quantile([0.025, 0.975], numeric_only=True))


if __name__ == "__main__":
    main()
