# Summary: Machine Learning Diagnosis of Chronic Rhinosinusitis (CRS) from Pre-Treatment Patient-Generated Health Data

*Raghavan AM, Aboueisha MA, Prohnitchi I, Cvancara DJ, Humphreys IM, Jafari A, Abuzeid WM. Am J Rhinol Allergy, 2025; 39(3):229-236.*

> **Note on sources:** Based on the abstract and later papers that cite this study (full text was not accessible). Items marked **(?)** are not verified and should be checked against the paper's Methods and Results.

## Goal
Test whether a machine learning model can flag **probable CRS before any rhinologist-directed treatment**, using only data the patient generates (questionnaires and history). This is a binary classification problem with two endpoints:
- **Primary:** sinonasal inflammation on CT, defined as a Lund-Mackay score (LMS) of 5 or more.
- **Secondary:** LMS of 5 or more **and** at least 2 cardinal symptoms of CRS.

## Data
- **Cohort:** 543 patients evaluated at a tertiary care rhinology clinic who then had CT imaging with LMS.
- **Collection:** patient-reported outcome measures and other patient data were collected through an electronic platform before the in-person visit.
- **Features:** 57 predictors extracted from the patient-generated health data.
- **Split:** 90:10 train/test (naive test set), plus cross-validation.
- **Not verified (?):** the full list of 57 predictors, class prevalence, and missing-data handling.

## Workflow (flowchart)

```
543 patients, tertiary rhinology clinic
                │
                ▼
Pre-visit data via electronic platform (PROMs + other patient data)
                │
                ▼
57 predictors extracted from patient-generated health data
                │
                ▼
CT imaging → Lund-Mackay score (LMS) → define labels
   ├── Primary:   LMS ≥ 5
   └── Secondary: LMS ≥ 5 AND ≥ 2 cardinal symptoms
                │
                ▼
Split: 90% train │ 10% naive test set
                │
                ▼
Train with cross-validation (feature selection / tuning (?))
   ├── Random Forest
   ├── Deep Neural Network
   ├── XGBoost
   └── Logistic / linear regression (baseline)
                │
                ▼
Evaluate on 10% test set → compare models
                │
                ▼
Best model = XGBoost
```

## Algorithms and methods
- **Models compared:** XGBoost, Random Forest (RF), Deep Neural Network (DNN), and logistic/linear regression as the baseline.
- The ML models were reported to outperform the linear regression model.
- **Not verified (?):** hyperparameters, DNN architecture, class-imbalance handling, and any feature-importance method.

## Results

| Endpoint (XGBoost, best model) | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Primary: LMS ≥ 5 | 71.3% | 74.5% | 38.9% | 91.9% |
| Secondary: LMS ≥ 5 + ≥ 2 cardinal symptoms | 79.8% | 85.5% | 36.4% | 97.7% |

- Accuracy across algorithms ranged from 74.5% to 85.5%.
- The pattern is **high specificity but low sensitivity** (about 36-39%): the model rules CRS in well but misses roughly 60% of true cases.
- The authors conclude the model accurately predicted probable CRS (the secondary endpoint).
- A later review rated the study at serious risk of bias because of low sensitivity and potential overfitting.

## Limitations

**Issues I noticed (useful for critical reading):**
1. **Tiny test set.** 10% of 543 is about 54 patients, so metrics are unstable. Repeated splits or nested CV would be more convincing.
2. **Low sensitivity.** About 36-39% sensitivity is weak for a triage tool, whose purpose is to avoid missing patients.
3. **Accuracy is a misleading headline** when classes are imbalanced. The secondary endpoint's 85.5% accuracy is inflated by its very high specificity.
4. **Spectrum or selection bias.** All patients were referred to a tertiary clinic and then sent for CT, so results may not transfer to primary care, where triage would actually be used.
5. **LMS ≥ 5 is a convenient threshold, not a gold standard.** CT inflammation correlates imperfectly with symptoms, and the cut-off drives prevalence and performance.
6. **Single centre, no external validation.**
7. **Weak baseline.** Beating "linear regression" is a low bar; a regularised logistic regression is the fair comparator.
8. **Clinical utility unproven.** No evidence yet that predictions reduce unnecessary CT scans or speed up referral.
9. **No calibration or decision-curve analysis (?)**, to be confirmed in the paper.

## Takeaways for your own ML/analytics projects
- Report **sensitivity and specificity**, not just accuracy, especially for imbalanced screening problems.
- Use a **strong simple baseline** (regularised logistic regression) before claiming gains from XGBoost or DNNs.
- With small samples, prefer **repeated or nested cross-validation** over a single 90:10 split.
- Validate on an **external cohort** that matches the intended deployment setting.
- Add **calibration plots, confidence intervals and decision-curve analysis**.
