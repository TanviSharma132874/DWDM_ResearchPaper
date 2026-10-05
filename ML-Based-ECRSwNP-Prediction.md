# Summary: ML model to predict ECRSwNP (Liu et al., 2026)

*American Journal of Rhinology & Allergy, 40(4), 316-326. DOI: 10.1177/19458924261425834*

Only the abstract, references and metadata were available to me. Anything marked "my inference" is not stated in the paper.

## 1. Problem and aim
Eosinophilic chronic rhinosinusitis with nasal polyps (ECRSwNP) is normally confirmed by biopsy after surgery. The authors built a model that predicts it **before surgery** from routine clinical data, so treatment can be individualized earlier.

## 2. Data
- 331 patients with chronic rhinosinusitis with nasal polyps, collected retrospectively.
- Training set 223, testing set 98.
- Single source: the Second Hospital & Clinical Medical School, Lanzhou University. The data are not public.

## 3. Pipeline
Patients (331) → split into train (223) and test (98) → LASSO + multivariate logistic regression → train 4 ML models with cross-validation → pick XGBoost → evaluate (AUC, decision curve, NRI/IDI), explain (SHAP), and check prognosis (Kaplan-Meier).

## 4. Algorithms used
- **LASSO and logistic regression:** LASSO shrinks weak variables to zero. Logistic regression then keeps the independent predictors.
- **Four ML classifiers:** the abstract names only XGBoost. It builds trees one after another, with each tree correcting the previous trees' errors, and uses regularization to limit overfitting.
- **SHAP:** scores each variable's push on a prediction, both globally (importance ranking) and per patient.
- **Decision curve analysis:** compares the model's net benefit with treating everyone or no one.
- **NRI and IDI:** measure how much the model improves classification and separation over a baseline.
- **Kaplan-Meier:** checks whether predicted groups differ in time to recurrence.

## 5. Key predictors
1. Peripheral blood eosinophil percentage
2. Visual analog scale (symptom severity)
3. Ethmoid/maxillary sinus ratio (from CT)
4. Nasal polyps score (from endoscopy)

## 6. Results
- XGBoost was the best model.
- AUC was **0.981** in training and **0.928** in testing.
- It showed the best net benefit across risk thresholds.
- It also predicted postoperative recurrence.

## 7. Limitations
The abstract states none, so these are my inference:
- Retrospective, single-center data, with no external validation mentioned.
- Small sample, which raises overfitting risk.
- The train-to-test AUC drop (0.981 to 0.928) hints at overfitting.
- 223 + 98 = 321, not 331, so 10 patients are unaccounted for.
- Blood eosinophils are already a known strong marker, so the model may partly rediscover it.

## 8. Context
Earlier work includes the JESREC score, blood eosinophil studies, and ML or nomogram models (Thorwarth 2021, Zhou 2023, Xiong 2024, Yuan 2025), plus CT-based deep learning. This paper's added value is the combination of XGBoost, SHAP, decision-curve analysis and a recurrence check.

## 9. Takeaways for you
- It is a clean template for an applied-ML workflow: feature selection, model comparison, calibration and decision-curve checks, and explainability.
- You could rebuild it as a data analytics portfolio project in Python with `LassoCV`, `xgboost` and `shap`, using simulated data since the real data aren't public.

## 10. Questions to ask when reading the full text
- Was the test set truly held out, including during feature selection and tuning?
- How many patients were ECRSwNP-positive?
- What threshold defined "eosinophilic"?
- What do the calibration plot and the SHAP summary plot show?





