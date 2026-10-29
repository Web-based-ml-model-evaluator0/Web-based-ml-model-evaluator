# Sample Datasets for Web-Based ML Model Evaluator

This directory contains verified sample datasets for testing the upload, validation, preview, preprocessing, training, and evaluation pipelines of the Web-Based ML Model Evaluator.

## 1. Breast Cancer Wisconsin Diagnostic Dataset

- **File:** `sample_data/breast_cancer.csv`
- **Task:** Binary Classification
- **Source:** UCI Machine Learning Repository / Scikit-Learn dataset repository
- **Instances:** 569 rows
- **Total Columns:** 31 (30 numeric input features + 1 target variable)
- **Target Variable:** `diagnosis`
  - Class distribution:
    - `benign`: 357 (62.74%)
    - `malignant`: 212 (37.26%)
- **Feature Attributes:**
  - 10 real-valued characteristics computed for each cell nucleus:
    - `radius` (mean of distances from center to points on the perimeter)
    - `texture` (standard deviation of gray-scale values)
    - `perimeter`
    - `area`
    - `smoothness` (local variation in radius lengths)
    - `compactness` (perimeter^2 / area - 1.0)
    - `concavity` (severity of concave portions of the contour)
    - `concave points` (number of concave portions of the contour)
    - `symmetry`
    - `fractal dimension` ("coastline approximation" - 1)
  - Features are computed for mean, standard error, and "worst" (largest) values (3 × 10 = 30 features).
- **Missing Values:** None.
- **Suitability:** Baseline classification verification for Sprint 1 (Sprint 1 scope requirement F).
