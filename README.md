# Learning-Style-Based Adaptive Model with Temporal Validation in Moodle

This repository provides research data and reproducibility materials associated with the manuscript:

**“Learning-Style-Based Adaptive Model with Temporal Validation in Moodle”**

Authors:

- **Dewi Rosita** — Computer Education Study Program, Faculty of Teacher Training and Education, Universitas Mulawarman, Indonesia
- **Mei Vita Romadon Ningrum** — Geography Education Study Program, Faculty of Teacher Training and Education, Universitas Mulawarman, Indonesia

Corresponding author: **Dewi Rosita**  
Email: **dewi.rosita@fkip.unmul.ac.id**

---

## Manuscript Status

The manuscript associated with this repository has been **submitted to Jurnal FASILKOM, Universitas Muhammadiyah Riau (UMRI), Indonesia**.

The manuscript has not yet been formally published. Repository contents may be updated during the editorial process to maintain consistency with the submitted manuscript and any subsequent revisions.

---

## Repository Purpose

This repository supports reproducibility of a retrospective Moodle learning-analytics study examining the temporal relationship between learner activity, previous academic performance, and Index of Learning Styles (ILS) preference scores.

The study evaluates whether ILS preference information provides additional predictive value for subsequent quiz performance when temporal ordering and learner separation are explicitly considered.

The repository is intended to provide sufficient research material for:

- examination of Moodle learner activity records;
- reconstruction of the analytical dataset;
- temporal learning-analytics experiments;
- analysis of ILS preference scores;
- reproducibility and verification of the reported experimental workflow.

---

## Dataset Summary

The Moodle archive used in the study contains:

| Component | Number |
|---|---:|
| Courses | 149 |
| Learners with Moodle log activity | 565 |
| Standard Moodle log events | 631,728 |
| Complete ILS respondents | 461 |
| Completed quiz attempts | 27,682 |
| First learner-quiz pairs retained during the audit | 21,743 |

The Moodle activity records cover the period from **September 2019 to June 2022**.

The ILS data contain responses covering the four Felder-Silverman preference dimensions:

1. Active / Reflective
2. Sensing / Intuitive
3. Visual / Verbal
4. Sequential / Global

---

## Main Data File

### `moodle_learner_activity_master.csv`

This file is a consolidated research table constructed from the relational Moodle data.

The master file preserves learner-level and course-level relationships required for analysis while organizing Moodle events in a single event-level table.

A learner may therefore occur in multiple rows because each row represents a recorded Moodle activity or event.

The dataset includes identifiers and research variables required to associate Moodle activity with:

- learner;
- course;
- Moodle component;
- activity or event type;
- event time;
- learning resource or activity;
- quiz and assessment information where applicable;
- linked ILS information where available.

Learners without complete ILS information are retained in the Moodle activity dataset rather than being deleted from the archive.

The ILS-linked analytical cohort can be selected from the master dataset when required.

---

## Data Privacy

The repository is intended for research and reproducibility purposes.

Direct personal information that is not required for the reported analyses is excluded from the research release. Personally identifying fields such as learner names, personal email addresses, and IP addresses are not intended to be used as analytical predictors.

Identifiers retained in the research dataset are used to preserve relationships among Moodle records and analytical observations.

Users of this repository should not attempt to re-identify individuals represented in the dataset.

---

## Complete Raw Moodle Export

The public repository provides the consolidated research dataset and selected reproducibility materials.

The **complete original Moodle export**, including the full relational table structure used to construct the research dataset, is not distributed directly through this public repository.

Researchers who require access to the complete raw Moodle research archive for legitimate academic or reproducibility purposes may contact:

**First / Corresponding Author**  
Dewi Rosita  
Computer Education Study Program  
Faculty of Teacher Training and Education  
Universitas Mulawarman  
Email: **dewi.rosita@fkip.unmul.ac.id**

or:

**Research Assets Contact**  
Faculty of Teacher Training and Education  
Universitas Mulawarman  
Email: **skripsi@fkip.unmul.ac.id**

Requests for the complete raw data may be subject to verification of the intended research use and any applicable institutional, ethical, privacy, or data-use requirements.

---

## Important Note on the Raw Data

The CSV file available in this repository is a research-oriented consolidated representation of the Moodle archive and should not be interpreted as a direct database dump of Moodle.

The original archive contains multiple relational tables. These tables were linked using their internal Moodle identifiers during data preparation.

Researchers requiring inspection of the complete relational Moodle structure should request access to the original research archive through the contacts listed above.

---

## Research Scope

The dataset was used for retrospective and temporal analysis.

The study does **not** assume that learning-style preferences represent fixed learner abilities, nor does it claim that matching instructional materials to a learning style automatically improves learning outcomes.

ILS scores are treated as preference-related information whose incremental predictive contribution is evaluated empirically.

The associated study also distinguishes between:

- performance- and behavior-based academic support; and
- optional and revisable learner presentation preferences.

---

## Reproducibility

The analytical workflow includes:

1. Moodle archive auditing;
2. learner and course relationship checking;
3. reconstruction of ILS dimensional scores;
4. temporal ordering of learner activities and assessments;
5. construction of pre-prediction behavioral features;
6. chronological and learner-separated data splitting;
7. baseline and ridge-regression modeling;
8. comparison of models with and without ILS information;
9. bootstrap-based uncertainty analysis.

Additional scripts and derived research files may be added to this repository as part of the reproducibility materials.

---

## Responsible Use

This repository is provided for academic research, verification, and reproducibility.

Users should:

- preserve learner confidentiality;
- avoid attempts at re-identification;
- avoid redistributing restricted raw records without authorization;
- clearly distinguish the consolidated research dataset from the complete Moodle database export;
- cite the associated manuscript or final publication when using this dataset.

---

## Citation

The associated manuscript has been submitted for publication.

Until the article is formally published, please refer to the work as:

> Rosita, D., & Ningrum, M. V. R.  
> “Learning-Style-Based Adaptive Model with Temporal Validation in Moodle.”  
> Manuscript submitted to Jurnal FASILKOM, Universitas Muhammadiyah Riau (UMRI), Indonesia.

The citation information in this section will be updated after publication.

---

## Contact

For questions regarding the dataset, reproducibility materials, or research methodology:

**Dewi Rosita**  
Computer Education Study Program  
Faculty of Teacher Training and Education  
Universitas Mulawarman, Indonesia  
Email: **dewi.rosita@fkip.unmul.ac.id**

For requests related to complete research assets:
**skripsi@fkip.unmul.ac.id**
