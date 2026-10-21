# Behavioral Prediction of Learning Preferences using Supervised Machine Learning

This repository contains the dataset, Python source code, and experiment outputs associated with the reproducibility package for the manuscript:

**“Behavioral Prediction of Learning Preferences using Supervised Machine Learning”**

The repository is intended to support transparency, reproducibility, and further research on the prediction of Felder–Silverman learning preferences from Moodle behavioral data using supervised machine learning.

## Manuscript Status

The manuscript associated with this repository is currently **under peer review**.

Full bibliographic information will be added to this repository once the article has been formally published. Researchers who use the dataset, code, experimental workflow, or results from this repository are encouraged to cite the corresponding journal article when the final citation becomes available.

The official citation will be provided in the **Citation** section of this README after publication.

## Repository Structure

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

requirements.txt
README.md
```

## DATA

### `compiled_pendikom_moodle_logs.csv`

This file is the primary compiled Moodle dataset used throughout the experiments.

The dataset contains anonymized learner-level behavioral records derived from Moodle activity data together with the ILS-related fields required for the learning-preference prediction experiments.

The compiled study dataset contains:

- **565 unique learners** with Moodle behavioral records;
- **461 learners with complete ILS labels** used in the supervised learning-preference experiments.

The available ILS labels represent the four dimensions of the Felder–Silverman Learning Style Model (FSLSM):

- Active–Reflective;
- Sensing–Intuitive;
- Visual–Verbal;
- Sequential–Global.

The dataset is retained in the same row-wrapped CSV representation used during the executed experiments. The accompanying Python scripts contain the appropriate loading procedure, so manual restructuring of the CSV file is not required before running the experiments.

### `data_manifest.csv`

This file provides supporting information about the data files included in the reproducibility package.

## CODE

The `CODE/` directory contains the Python scripts used to implement the main experimental workflow.

### `01_ml_baselines_and_splits.py`

Implements the conventional supervised machine-learning experiments, including:

- learner-level random validation folds;
- course-disjoint validation folds;
- supervised baseline models;
- fold assignments;
- model predictions;
- performance metrics; and
- feature information used in the experiments.

### `02_course_heterogeneity.py`

Examines behavioral heterogeneity across courses, including:

- course-level behavioral variation;
- eta-squared diagnostics;
- learning-opportunity variation; and
- prediction of course identity from behavioral features.

These analyses are used to examine the extent to which Moodle behavioral patterns may differ across instructional contexts.

### `03_multitask_erm_and_domain_generalization.py`

Implements the multi-task supervised learning and domain-generalization experiments, including empirical risk minimization and robustness-oriented learning variants.

The experiments are designed to evaluate whether learning-preference prediction remains effective when behavioral distributions vary across course contexts.

### `04_domain_invariance_diagnostic.py`

Performs diagnostic analyses of course-related information contained in:

- the original behavioral feature space; and
- learned representations produced by the predictive models.

The analysis provides additional evidence regarding the degree to which learned representations retain or reduce course-specific information.

### `run_all_domain_generalization.py`

Runs the four core experiment scripts sequentially using repository-relative paths.

All core scripts read their input data from `DATA/` and write generated CSV results to `OUTPUT/`.

## OUTPUT

The `OUTPUT/` directory contains the CSV result bundle generated from the executed experiments.

The outputs include:

- supervised learning metrics at the fold level;
- learner-level model predictions;
- random-fold and course-disjoint evaluation results;
- course heterogeneity diagnostics;
- multi-task learning results;
- robustness and domain-generalization results;
- representation-invariance diagnostics;
- statistical comparison results;
- error analyses;
- manuscript-oriented summary tables; and
- `CSV_OUTPUT_MANIFEST.csv`, which documents the generated output files.

The provided outputs preserve the results from the executed experimental workflow used to support the manuscript.

## Running the Core Experiments

From the repository root, run:

```bash
python CODE/run_all_domain_generalization.py
```

The required Python dependencies are listed in:

```text
requirements.txt
```

Install the dependencies before running the experiments, for example:

```bash
pip install -r requirements.txt
```

## Reproducibility Notes

- Random seeds are fixed where specified in the experimental scripts.
- Data splitting procedures are implemented directly in the provided code.
- Course identifiers are used for grouped validation, heterogeneity analysis, and diagnostic purposes.
- Course identifiers are **not used as predictive features** in the supervised learning-preference models.
- The experiments distinguish conventional learner-level random validation from course-disjoint evaluation to assess predictive robustness across course contexts.
- The existing files in `OUTPUT/` preserve the complete result bundle from the executed experiments.
- Re-running the scripts may overwrite the corresponding generated CSV files in `OUTPUT/`.

## Data Privacy

The dataset included in this repository has been anonymized for research and reproducibility purposes.

Learner identifiers are represented using non-identifying analysis IDs. No directly identifying personal information is included in the publicly shared reproducibility package.

The shared data are intended to support verification of the reported experiments while protecting learner privacy.

## Research Use

This repository may be used for:

- reproduction of the reported experiments;
- comparison with alternative supervised learning algorithms;
- learning-preference prediction research;
- Moodle learning-analytics research;
- cross-course robustness studies;
- domain-generalization experiments; and
- further methodological studies using anonymized educational behavioral data.

Researchers extending this work are encouraged to clearly document any changes to preprocessing, data partitioning, feature construction, learning algorithms, or evaluation protocols so that subsequent results remain comparable.

## Citation

The manuscript associated with this repository is currently under peer review.

The final journal citation and DOI will be added here after publication.

Until the article is formally published, please refer to this repository and manuscript using the manuscript title:

> **Behavioral Prediction of Learning Preferences using Supervised Machine Learning**

### Citation after publication

```text
Citation information will be provided here after the associated
journal article has been formally published.
```

The BibTeX entry will also be added to this section once the final publication metadata and DOI become available.

## Availability

This repository serves as the reproducibility archive accompanying the manuscript **“Behavioral Prediction of Learning Preferences using Supervised Machine Learning.”**

Updates related to the final journal publication, citation information, and DOI will be provided in this repository following publication.
