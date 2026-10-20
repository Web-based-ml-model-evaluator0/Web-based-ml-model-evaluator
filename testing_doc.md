# Web Based ML Model Evaluator — Sprint 1 Testing Documentation

**Team:** Team 16  
**Project:** Web Based ML Model Evaluator  
**Sprint:** Sprint 1  
**Testing type:** Manual functional testing

## 1. Objective

The objective of testing is to verify that the application can upload and validate CSV datasets, handle invalid input, train machine-learning models, display evaluation metrics, and show actual versus predicted values.

## 2. Test Environment

- Operating System: Windows
- Programming Language: Python
- User Interface: Streamlit
- Data Processing: Pandas
- Machine Learning: Scikit-learn
- Test Datasets: Breast Cancer dataset, Diabetes dataset, and custom CSV test files

## 3. Test Cases and Results

| Test ID | Test Scenario | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| TC-01 | Upload a valid CSV dataset | Dataset loads successfully | Dataset loaded successfully during testing | PASS |
| TC-02 | Upload a malformed CSV file | Application reports an input error | Invalid CSV was rejected during testing | PASS |
| TC-03 | Validate dataset information | Display rows, columns, missing values, and duplicate rows | Dataset information was displayed | PASS |
| TC-04 | Train Random Forest Regression on diabetes.csv | Model trains and displays regression metrics | App ran successfully; MAE, RMSE, R², and prediction table appeared | PASS |
| TC-05 | Use a regression target containing "unknown" | Application displays an error and stops training | One non-numeric salary value was detected and training was stopped | PASS |
| TC-06 | Test a dataset containing missing values | App handles missing feature values and completes evaluation when data is sufficient | Evaluation completed; classification metrics displayed as 0.0000. Further investigation is required | NEEDS REVIEW |
| TC-07 | Test Logistic Regression classification | Model trains and displays classification metrics | Not yet confirmed | PENDING |
| TC-08 | Test Decision Tree classification | Model trains and displays classification metrics | Not yet confirmed | PENDING |
| TC-09 | Test Linear Regression | Model trains and displays MAE, RMSE, and R² | Not yet confirmed | PENDING |
| TC-10 | Test Decision Tree Regression | Model trains and displays MAE, RMSE, and R² | Not yet confirmed | PENDING |
| TC-11 | Download prediction results | Predictions download as a CSV file | Not yet confirmed | PENDING |

## 4. Defects and Observations

### Observation 1: Invalid regression target

A custom dataset contained the text value "unknown" in the numerical salary target column. The application detected the invalid value and stopped training with a clear error message. This behavior was verified after updating the validation code.

### Observation 2: Classification metrics on a small dataset

A small dataset containing missing feature values produced Accuracy, Precision, Recall, and F1 Score values of 0.0000. Although the application completed evaluation, the result requires investigation. Small test sets and incorrect predictions may cause this outcome.

### Observation 3: Remaining model tests

The remaining model configurations and CSV prediction download must be tested before final testing sign-off.

## 5. Conclusion

Initial manual testing verified CSV upload, invalid CSV rejection, dataset inspection, Random Forest regression evaluation, and rejection of a non-numeric regression target.

Additional testing is required for the remaining model configurations, prediction download, and the classification result that produced zero-valued metrics. Sprint 1 testing should be considered complete only after the required acceptance criteria have been verified.

## 6. Evidence

Attach screenshots of:

1. Successful dataset upload and preview.
2. Regression evaluation metrics.
3. Invalid regression target error message.
4. Classification evaluation results.
5. Successful prediction CSV download.

Record the actual result and date for each additional test when it is performed.
