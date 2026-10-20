# Derived Public Files

These files are generated directly from `DATA/public_release/moodle_learner_activity.csv.gz`.

- `learner_ils_scores.csv`: one row per learner with ILS metadata (461 rows).
- `quiz_attempt_started_index.csv.gz`: quiz attempt-start events (27,682 rows).
- `first_learner_quiz_pairs.csv.gz`: earliest attempt for each learner-course-quiz combination (21,743 rows).
- `post_ils_eligible_targets.csv.gz`: first learner-quiz targets whose start time is strictly after ILS submission (7,612 rows).
- `primary_split_index.csv.gz`: temporal/learner-separated analytical cohort (4,250 rows: 2,870 train, 728 validation, 652 test).

The split files index target observations and ILS metadata. They do not include quiz outcome scores, because those values are not present in the consolidated public activity file supplied for this repository.
