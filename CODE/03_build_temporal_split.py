"""Bangun indeks attempt, target post-ILS, dan split temporal utama."""
from __future__ import annotations

import pandas as pd
from paths import PUBLIC_DATA, DERIVED, OUT

QUIZ_STARTED = r"\mod_quiz\event\attempt_started"


def main():
    use = [
        "event_id", "userid", "courseid", "event_time_unix", "event_datetime_utc",
        "event_datetime_local", "eventname", "module_instance_id", "activity_name",
        "objectid", "has_ils", "ils_reference_courseid", "ils_submitted_unix",
        "ils_active_reflective_score", "ils_sensing_intuitive_score",
        "ils_visual_verbal_score", "ils_sequential_global_score",
    ]
    df = pd.read_csv(PUBLIC_DATA, usecols=use, low_memory=False)
    q = df.loc[df["eventname"].eq(QUIZ_STARTED)].copy()
    q = q.sort_values(["userid", "courseid", "module_instance_id", "event_time_unix", "event_id"])
    q = q.rename(columns={
        "userid": "learner_id", "courseid": "course_id", "module_instance_id": "quiz_id",
        "objectid": "attempt_id", "event_time_unix": "quiz_start_unix",
        "event_datetime_utc": "quiz_start_utc", "event_datetime_local": "quiz_start_local",
        "ils_reference_courseid": "ils_reference_course_id",
    })
    q.to_csv(DERIVED / "quiz_attempt_started_index.csv.gz", index=False, compression="gzip")

    first = q.drop_duplicates(["learner_id", "course_id", "quiz_id"], keep="first").copy()
    first.to_csv(DERIVED / "first_learner_quiz_pairs.csv.gz", index=False, compression="gzip")

    eligible = first.loc[
        first["has_ils"].eq(1)
        & first["ils_submitted_unix"].notna()
        & (first["ils_submitted_unix"] < first["quiz_start_unix"])
    ].copy()
    eligible["days_since_ils"] = (eligible["quiz_start_unix"] - eligible["ils_submitted_unix"]) / 86400.0
    eligible.to_csv(DERIVED / "post_ils_eligible_targets.csv.gz", index=False, compression="gzip")

    cut1 = pd.Timestamp("2021-08-01T00:00:00Z").timestamp()
    cut2 = pd.Timestamp("2022-01-01T00:00:00Z").timestamp()
    train = eligible.loc[eligible["quiz_start_unix"] < cut1].copy()
    train_users = set(train["learner_id"])
    validation = eligible.loc[
        (eligible["quiz_start_unix"] >= cut1)
        & (eligible["quiz_start_unix"] < cut2)
        & (~eligible["learner_id"].isin(train_users))
    ].copy()
    prior_users = set(eligible.loc[eligible["quiz_start_unix"] < cut2, "learner_id"])
    test = eligible.loc[
        (eligible["quiz_start_unix"] >= cut2)
        & (~eligible["learner_id"].isin(prior_users))
    ].copy()

    parts = []
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        part = part.copy()
        part["split"] = name
        parts.append(part)
    split = pd.concat(parts, ignore_index=True)
    split.to_csv(DERIVED / "primary_split_index.csv.gz", index=False, compression="gzip")

    print("quiz attempts started:", len(q))
    print("first learner-quiz pairs:", len(first))
    print("post-ILS eligible targets:", len(eligible), "learners:", eligible.learner_id.nunique(), "courses:", eligible.course_id.nunique())
    print("primary split:", len(train), len(validation), len(test))


if __name__ == "__main__":
    main()
