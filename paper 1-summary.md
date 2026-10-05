# Summary: Risk Prediction Model for Chronic Sinusitis with Nasal Polyps (CRSwNP)

*Li et al., European Archives of Oto-Rhino-Laryngology, 2025*

## Goal
Build a simple, fast model that predicts whether a chronic sinusitis (CRS) patient has nasal polyps (CRSwNP) or not (CRSsNP). It is a binary classification problem.

## Data
- **206 patients** from one hospital (Dalian Third People's Hospital, 2018-2024): 105 with polyps, 101 without.
- **Variables (~28):** demographics (height, weight, age, BMI), smoking and drinking, comorbidities (asthma, diabetes, hypertension, allergic rhinitis, deviated septum), blood markers (WBC, neutrophils, eosinophils, cholesterol, TG), bacterial culture, and two clinical scores: **Lund-Mackay (LM)** from CT and **Lund-Kennedy (LK)** from nasal endoscopy.
- **Split:** 70/30 random split, 146 training (73 per class) and 60 validation (28 / 32).

## Workflow (flowchart)

```
206 CRS patients (retrospective data)
            │
            ▼
Inclusion/exclusion filtering (age 18-70, no fungal sinusitis, no tumor, no prior surgery, no autoimmune disease)
            │
            ▼
Random split 70:30
   ├── Training set (n=146)         └── Validation set (n=60)
            │                                    │
            ▼                                    │
Univariate logistic regression                   │
(keep variables with P < 0.1)                    │
            │                                    │
            ▼                                    │
Multivariate logistic regression                 │
(full, forward, backward, stepwise)              │
→ best model by lowest AIC (116.56)              │
            │                                    │
            ▼                                    │
LASSO regression (cross-validated lambda)        │
→ 7 variables at "1-SE" lambda                   │
            │                                    │
            ▼                                    │
Combine with clinical judgment → 4 final variables
(Alcohol, Bacterial culture, LM score, LK score)
            │                                    │
            ▼                                    │
Build nomogram (points 0-100 → probability)      │
            │                                    ▼
Evaluate on training set ──────────► Evaluate on validation set
(ROC, calibration, DCA)               (ROC, calibration, DCA)
```

## Algorithm and model
- **Core method:** logistic regression, with feature selection via univariate screening, then AIC stepwise selection, then LASSO (L1 regularization).
- **LASSO detail:** the "lambda.min" line gives the lowest error, and the "1-SE" line gives the simplest model within one standard error of the minimum. They used **1-SE** for a sparser model.
- **Final equation:**

  `logit(P) = 4.003 + 0.863·Drink − 3.041·Bacteria + 0.335·LM + 0.424·LK`

- **Nomogram:** each variable maps to points (0-100), the total points map to a probability, so clinicians can use it without software.
- **Tools:** R 4.1.2 (`MASS` for stepwise, `stats` for models, `rms` for validation and nomogram).

## Results

| Metric | Training | Validation |
|---|---|---|
| AUC | 0.917 (CI 0.874-0.960) | 0.960 (CI 0.915-1.000) |
| Calibration | Close to ideal line, mean absolute error 0.02 | Close to ideal line |
| DCA | Net benefit above the "treat all/none" lines across a wide threshold range | Same |

Single-factor AUCs were only about 0.86 each, so the combined model improved discrimination.

**Significant factors:** alcohol consumption, bacterial culture, LM score, LK score. Deviated septum, allergic rhinitis, and eosinophil % also appeared in univariate or LASSO results. LM and LK were by far the strongest (p < 0.001).

## Limitations

**Stated by the authors:**
- Single center
- Small validation set (n=60)
- No public datasets for prognostic data
- Needs larger prospective multicenter validation

**Issues I noticed (useful for your critical reading):**
1. **Circularity and leakage:** the LK score includes polyp grading, so using it to predict polyps is partly predicting the outcome from itself. LASSO also kept "pathology" as a predictor, which essentially defines the label.
2. **Validation AUC (0.96) is higher than training AUC (0.917).** This is unusual and likely reflects the tiny validation set (wide CI). The paper also calls it "external" validation in the discussion, but it is only internal (random split).
3. **No cross-validation or bootstrap** on the final model. A single split on 206 samples is unstable.
4. **Lenient p < 0.1 threshold** for variable screening, and univariate screening itself is a weak selection method.
5. **Data quality problems:** heights as low as 17 cm, inconsistent tables (Table 3 z-values and signs don't match estimates, "Emphysema" listed twice in Table 2), and calibration plot axes labelled "nonadherence", which looks copied from another study.
6. **Only 12 bacterial-positive vs. 6** cases, so the strong bacteria coefficient (−3.04) is based on very few observations and is likely unstable.
7. **Retrospective, surgical-patient data** may carry selection bias (all patients had surgery).
8. **No comparison** against other ML models (random forest, XGBoost) despite calling it a "machine learning" study. LASSO is used only for feature selection.

## Takeaways for your own ML/analytics projects
- A clear pipeline (screen, select, build, validate) with a nomogram is good for interpretability.
- Always check for **target leakage** and **overfitting**, and use cross-validation or bootstrapping on small data.
- Report calibration and decision-curve analysis, not just AUC.
- Compare against baseline and alternative models.

