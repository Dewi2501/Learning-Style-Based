# Analysis Matrix (for full prognostic reproduction)

The public activity-level release supports reconstruction of ILS metadata, quiz-attempt timing, temporal eligibility, and the primary learner-separated split. It does not contain the target quiz scores and all fields needed for the 18-variable readiness model.

For full model reproduction, place the genuine pseudonymous analysis matrix here as:

`analysis_matrix.csv.gz`

The required column names are listed in `analysis_matrix_schema.csv`.

For bootstrap reproduction, also provide:

`test_predictions.csv.gz`

with these columns:

- `learner_id`
- `actual_score`
- `pred_training_mean`
- `pred_prior_quiz_mean`
- `pred_readiness_lam_0_1`
- `pred_readiness_ils_lam_0_1`
- `pred_readiness_lam_1`
- `pred_readiness_ils_lam_1`

Do not populate these files from manuscript summary statistics. They must come from the actual pseudonymous analytical records/predictions used in the study.
