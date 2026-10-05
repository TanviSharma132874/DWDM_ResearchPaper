# Summary: Clinical Key Features Uncovered by Blood Eosinophilia-Based Machine Learning Classification of Chronic Rhinosinusitis (CRS)

*Ishikawa M, Jiang Z, Nguyen CH, Hatsukawa H, Mamitsuka H. Int Forum Allergy Rhinol, vol. 16 (2026; online late 2025).*

> **Note on sources:** Based on the abstract, figure legends and later papers that cite this study (full text was not accessible). Items marked **(?)** are not verified and should be checked against the paper's Methods and Results.

## Goal
Use exploratory data analysis and machine learning to find **which clinical features distinguish CRS patients by the presence and severity of blood eosinophilia**, to gain insight into CRS pathophysiology. This is **explanatory feature discovery**, not a clinical prediction tool.

## Data
- **Cohort:** 399 patients with CRS (univariate analyses).
- **Labels** come from the absolute eosinophil count (AEC):
  - **Setting-1 (existence):** AEC cutoff of 500/µL, giving 2 classes.
  - **Setting-2 (severity):** cutoffs of 500 and 1500/µL, giving 3 classes (below 500, 500-1500, 1500 or above).
- **Features:** figure legends mention CRP, *Dermatophagoides farinae* (DFA, house dust mite), inhaled and oral corticosteroid use, neutrophil-to-lymphocyte ratio (NLR), nasal polyps (NP) and VAS score. The abstract adds blood basophil count, comorbid chronic eosinophilic pneumonia (CEP), and sinus CT scores.
- **Not verified (?):** the full feature list, centre and country, and the train/test or cross-validation design.

## Workflow (flowchart)

```
399 CRS patients
        │
        ▼
Define blood-eosinophilia labels from AEC
   ├── Setting-1: AEC <500 vs ≥500 /µL        (2 classes)
   └── Setting-2: <500 │ 500–1500 │ ≥1500 /µL (3 classes)
        │
        ▼
Exploratory univariate analyses (clinical, blood, CT, NP, etc.)
   → 17 significant features (both settings)
   → 8 additional features (Setting-2 only)
        │
        ▼
Train and compare ML classifiers (LR, RF, SVM, XGBoost) (?)
   → evaluate with ROC / AUC (mean ± SD)
        │
        ▼
Best model per setting
   ├── Setting-1: Random Forest
   └── Setting-2: XGBoost
        │
        ▼
SHAP feature importance on best model (?)
        │
        ▼
Key features → basophil involvement in eosinophilia
```

## Algorithms and methods
- **Best models:** Random Forest in Setting-1 and XGBoost (eXtreme Gradient Boosting) in Setting-2.
- **Likely comparison set (?):** Logistic Regression, Random Forest, SVM and XGBoost, inferred from figure-legend abbreviations. ROC/AUC is reported as mean ± SD.
- **Interpretability (?):** SHAP appears in the figure legends, suggesting SHAP-based feature importance.
- **Not verified (?):** hyperparameters, imbalance handling, and the validation scheme.

## Results
- Univariate analyses showed significant differences for **17 features in both settings** and **8 additional features in Setting-2**.
- With increasing severity of eosinophilia, eosinophilic CRS **without** nasal polyps became more common.
- For **AEC ≥ 1500/µL**, the five most important classification features were:
  1. High CRP
  2. High blood basophil count
  3. Comorbid chronic eosinophilic pneumonia (CEP)
  4. Low maxillary sinus CT scores
  5. Low NP scores
- The authors conclude that ML classification revealed **basophil involvement** in peripheral blood eosinophilia.
- **Not verified (?):** the Setting-1 top features and AUC values per model.

## Limitations

**Issues I noticed (useful for critical reading):**
1. **Importance is not causation.** The basophil finding is hypothesis-generating. Basophils and eosinophils come from the same blood differential and both reflect type 2 inflammation, so the link may partly reflect shared biology or measurement.
2. **Labels come from a continuous blood value.** The 500 and 1500/µL thresholds are conventional but arbitrary, and counts near a cutoff are noisy and vary day to day.
3. **Counter-intuitive findings need checking.** Severe eosinophilia going with *low* maxillary CT scores and *low* NP scores is unexpected, since polyps and sinus opacification usually track eosinophilic disease. It may reflect the CEP subgroup or confounding.
4. **Small subgroups.** With about 399 patients and a 3-class setting, the AEC ≥ 1500/µL and CEP groups are probably small, so feature importances may be unstable (inference).
5. **Correlated features** (CT scores, blood counts, CRP) make importance rankings unstable. Look for SHAP or permutation importance with confidence intervals.
6. **Validation unclear (?).** Without an external or held-out cohort, the key features should be treated as exploratory.
7. **Not directly comparable to prediction studies.** This paper explains and phenotypes, so its metrics should not be compared with triage or diagnostic models.

## Takeaways for your own ML/analytics projects
- Use ML for **feature discovery** only with caution: report stability of importances across resamples.
- Avoid **dichotomising continuous outcomes** where possible, or show sensitivity analyses on the cutoffs.
- Combine **univariate tests, multivariable models and SHAP** rather than relying on one view of importance.
- Treat associations as **hypotheses to test in an external cohort**.
