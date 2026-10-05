# Summary: ML Mortality Prediction for ICU Patients with Acute Rhinosinusitis (ARS)

*Sun, Wang & Yang, Medicine (2026) 105:17*

## Goal
Predict **in-hospital mortality** (died vs. survived) in ICU patients with acute rhinosinusitis, and make the model explainable with SHAP. This is a binary classification problem.

## Data
- **3 public ICU databases**, 559 patients total after exclusions:
  - **MIMIC-IV v3.1** (n=392): used for **training**
  - **MIMIC-III CareVue** (n=89): **external validation 1**
  - **eICU v2.0** (n=78): **external validation 2**
- **Cohort filters:** adults over 18, ICD-9 461 / ICD-10 J01 diagnosis, first ICU admission only, ICU stay of at least 24 hours.
- **Outcome:** in-hospital mortality, about 10% overall (56 deaths), so the classes are imbalanced.
- **Features:** demographics, vitals, labs (BUN, creatinine, Hb, platelets, WBC, HCO₃⁻), severity scores (SOFA, SAPS II, OASIS, GCS, Charlson), ventilation, vasopressors, and comorbidities.

## Workflow (flowchart)

```
MIMIC-IV (538)      MIMIC-III (102)      eICU (99)
      │                   │                  │
      ▼                   ▼                  ▼
Exclude: non-first admission, age ≤18, ICU stay <24 h
      │                   │                  │
      ▼                   ▼                  ▼
 n = 392              n = 89             n = 78
      │                   │                  │
      └───────── Standardize variables ──────┘
                          │
                          ▼
     Drop variables with >30% missing
     Impute the rest (MICE: 5 datasets, 20 iterations)
                          │
                          ▼
     Boruta feature selection (run on EACH database)
     → take the UNION of confirmed features
                          │
                          ▼
     Train 9 ML models on MIMIC-IV
     (SMOTE if death:survival ratio <0.3,
      5-fold CV × 3 repeats, grid search, L2 regularization)
                          │
                          ▼
     External validation on MIMIC-III and eICU
     (AUC, sensitivity, specificity, accuracy)
                          │
                          ▼
     Pick best model → Random Forest
                          │
                          ▼
     SHAP interpretation (global + individual)
```

## Algorithms
- **Feature selection: Boruta.** It builds a random forest, adds randomly shuffled "shadow" copies of the features, and keeps only real features that beat the shadows. Features are labeled confirmed, tentative, or rejected.
- **Nine models compared:** Random Forest, XGBoost, Logistic Regression, SVM, GBM, KNN, CART, LASSO, Neural Network.
- **Tuning:**
  - RF: node size, max nodes, number of trees.
  - SVM: C and sigma (RBF kernel).
  - KNN: k.
- **Imbalance handling:** SMOTE.
- **Interpretability: SHAP** (kernelshap in R), with bar plots, beeswarm, dependence, force, and waterfall plots.
- **Tools:** R 4.2.0 (`caret`, `Boruta`, `pROC`, `kernelshap`, `shapviz`, `mice`).

## Results

**Boruta top features by database:**
- MIMIC-IV: platelets, BUN, SOFA
- MIMIC-III: SOFA, platelets, weight
- eICU: temperature, GCS, SAPS II, Hb

**Best model performance (Random Forest, external validation):**

| Dataset | AUC (95% CI) | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| MIMIC-III | 0.711 (0.564-0.858) | 0.692 | 0.697 | 0.730 |
| eICU | 0.799 (0.664-0.933) | 1.000 | 0.611 | 0.769 |

Other models in eICU: KNN had the highest AUC (0.824) but only 0.564 accuracy, and LASSO had AUC 0.803 with 0.917 specificity. RF was chosen on overall balance, not top AUC.

**SHAP findings (top predictors):**
1. **OASIS score** (most important)
2. **SAPS II**
3. **Weight** (higher weight lowers risk, stabilizing above about 100 kg)
4. **BUN in the first 24 h**
5. **SOFA**
6. Also temperature, HCO₃⁻, hemoglobin, and creatinine

Risk rises steeply with OASIS and SAPS II, then plateaus.

## Limitations

**Stated by the authors:**
- Retrospective design, with selection bias and residual confounding
- Small external validation sets (n=89 and n=78)
- No ARS-specific details (imaging such as Lund-Mackay, microbiology, disease subtype)
- Needs prospective validation

**Issues I noticed (useful for critical reading):**
1. **Weak performance.** An AUC of 0.711 on MIMIC-III is only moderate, and the confidence intervals are very wide (0.56-0.86). Calling the model "robust" and a "reliable tool" is an overstatement.
2. **Very few events.** Only about 56 deaths overall, and roughly 6 and 13 deaths in the two validation sets. Sensitivity of 1.000 on 6 deaths is not meaningful, yet the authors attribute it to L2 regularization and grid search.
3. **Misplaced L2 claim.** Random forest has no L2 penalty in the usual sense, so the explanation is questionable.
4. **Possible leakage and redundancy.** OASIS, SAPS II, and SOFA already summarize the same physiology (and include BUN, GCS, and temperature), so the model largely re-uses existing severity scores. It adds little over them, and no baseline comparison against SOFA, SAPS II, or OASIS alone is shown.
5. **Boruta "union" approach.** Taking the union of features across databases lets the external sets influence feature selection, which weakens the independence of the validation. Boruta was also run on each external dataset, so the validation is not fully out-of-sample.
6. **Mixed model-selection logic.** RF was picked even though KNN and LASSO had higher AUC in eICU, and the paper says "RF performed best across all datasets", which its own Table 2 doesn't support.
7. **Questionable SHAP interpretations.** The paper suggests fever and high weight are "protective" and that very high BUN lowers risk. These are likely artifacts of small data and correlated features, not causal findings. Units are also inconsistent (BUN in "ng/mL", Hb in "mmol/L").
8. **Population definition.** ICU patients coded with ARS may really be admitted for other conditions (sepsis, ventilation), so ARS may be incidental. Over half were mechanically ventilated, which is unusual for sinusitis.
9. **No calibration or decision-curve analysis** reported, only AUC, sensitivity, and specificity.
10. **Wording inconsistency.** The text switches between "sinusitis" and "ARS", and reports "primary care" applicability for an ICU-only model.

## Takeaways for your own ML/analytics projects
- A solid template: **external validation across databases, multiple imputation, a model comparison, and SHAP** for interpretability.
- With few positive events, **report CIs, use bootstrapping, and avoid over-interpreting** perfect sensitivity.
- Always benchmark against a **simple baseline** (e.g., logistic regression on SOFA).
- Keep **feature selection inside the training data only** to avoid leakage.
- Add **calibration plots** and clinical-utility analysis, not just AUC.

## Quick comparison with Paper 1
| | Paper 1 (CRSwNP) | Paper 2 (ICU ARS) |
|---|---|---|
| Task | Polyp vs. no polyp | In-hospital death |
| Data | 206, single center | 559, 3 public databases |
| Core method | Logistic regression + LASSO + nomogram | Boruta + 9 ML models + RF |
| Validation | Internal split | External, other databases |
| Interpretability | Nomogram | SHAP |
| Best AUC | 0.96 (likely inflated) | 0.80 (more modest, more honest) |

