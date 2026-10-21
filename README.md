# Behavioral Prediction of Learning Preferences using Supervised Machine Learning

This repository contains the data, Python code, and CSV outputs used for the reproducibility package of the manuscript **“Behavioral Prediction of Learning Preferences using Supervised Machine Learning.”**

## Repository structure

```text
DATA/
  compiled_pendikom_moodle_logs.csv
  data_manifest.csv

CODE/
  01_ml_baselines_and_splits.py
  02_course_heterogeneity.py
  03_multitask_erm_and_domain_generalization.py
  04_domain_invariance_diagnostic.py
  run_all_domain_generalization.py

OUTPUT/
  *.csv
```

## DATA

`compiled_pendikom_moodle_logs.csv` is the primary compiled Moodle dataset used by the experiment scripts. The dataset contains anonymized learner-level behavioral records and ILS-related fields. The study dataset contains 565 unique learners from 50 courses, with complete ILS labels available for 461 learners.

The file is retained in the same row-wrapped CSV representation used in the executed experiment. The Python scripts include the corresponding loader and do not require manual reformatting.

## CODE

The scripts implement the main experimental workflow:

1. `01_ml_baselines_and_splits.py` — conventional supervised baselines, learner-level random folds, course-disjoint folds, predictions, metrics, feature lists, and fold assignments.
2. `02_course_heterogeneity.py` — course-level heterogeneity analysis, eta-squared diagnostics, opportunity variation, and course-identity prediction.
3. `03_multitask_erm_and_domain_generalization.py` — multi-task supervised ERM and robustness-oriented risk/domain variants.
4. `04_domain_invariance_diagnostic.py` — diagnostic analysis of recoverable course information from original and learned representations.
5. `run_all_domain_generalization.py` — runs the four core scripts sequentially.

All paths are repository-relative. The scripts read from `DATA/` and write core generated CSV files to `OUTPUT/`.

## OUTPUT

`OUTPUT/` contains the complete CSV result bundle from the executed experiment, including:

- fold-level supervised metrics and predictions;
- course heterogeneity diagnostics;
- neural multi-task and robustness-variant results;
- representation diagnostics;
- statistical comparisons and error analyses;
- manuscript-oriented summary tables;
- an output manifest (`CSV_OUTPUT_MANIFEST.csv`).

## Running the core experiment

From the repository root:

```bash
python CODE/run_all_domain_generalization.py
```

Required Python packages are listed in `requirements.txt`.

## Notes on reproducibility

- Random seeds used by the scripts are fixed where specified in the executed experiment.
- Course identifiers are used for grouped validation and diagnostics, not as predictive features in the supervised preference models.
- The existing `OUTPUT/` files preserve the complete executed result bundle. Re-running the core scripts will overwrite the corresponding core CSV outputs with newly generated results.

## Data privacy

The repository dataset is anonymized. Learner identifiers are non-identifying analysis IDs, and no directly identifying personal information is included in this reproducibility package.
