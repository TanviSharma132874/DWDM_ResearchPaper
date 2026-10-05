# Summary: Explainable Machine Learning Model for Postoperative Recurrence in Pediatric Chronic Rhinosinusitis (CRS)

*Jiang S, Qi B, Xie S, Xie Z, Zhang H, Jiang W. Otolaryngol Head Neck Surg, 2025; 172:1044-1052 (online December 2024).*

> **Note on sources:** Based on the abstract; full text was not accessible. Items marked **(?)** are not verified and should be checked against the paper's Methods and Results.

## Goal
Develop an **interpretable ML model** to predict **postoperative recurrence in children with CRS** after functional endoscopic sinus surgery (FESS). This is a binary classification problem, and the final model is also published as a web calculator.

## Data
- **Source:** retrospective study at Xiangya Hospital of Central South University (Changsha, China).
- **Cohort:** 148 pediatric CRS patients treated with FESS between January 2015 and January 2022.
- **Features:** demographic characteristics and peripheral blood inflammatory indices, plus calculated inflammation indices such as the systemic immune-inflammation index (SII) and pan-immune-inflammation value (PIV).
- **Not verified (?):** recurrence definition, follow-up length, recurrence rate, train/test split ratio, and the full feature list.

## Workflow (flowchart)

```
148 pediatric CRS patients, FESS, 2015-2022 (Xiangya Hospital)
                │
                ▼
Collect demographics + peripheral blood counts
Calculate inflammation indices (SII, PIV, ...)
                │
                ▼
Train 3 ML algorithms → compare by AUC
   Random Forest was best (AUC = 0.728)
                │
                ▼
Reduce features by importance + tune parameters
                │
                ▼
Final RF model with 4 features
   SII │ PIV │ E% (eosinophils) │ L% (lymphocytes)
   Internal validation AUC = 0.779
                │
                ▼
Interpretation
   ├── SHAP (global importance)
   └── Ceteris Paribus profiles (individual / nonlinear effects)
                │
                ▼
Deploy as interactive web app (Shiny)
```

## Algorithms and methods
- **Three ML algorithms** were trained and compared by AUC; Random Forest (RF) won. **(?)** The other two are not named in the abstract.
- **Feature reduction:** features were ranked by importance and reduced, then parameters were tuned, giving a final 4-feature RF.
- **Interpretability:** SHAP (Shapley Additive Explanations) for global importance, and Ceteris Paribus profiles for how predictions change as one feature varies.
- **Deployment:** the final model was turned into an interactive web tool (R Shiny).

## Results
- **Initial model comparison:** RF had the best discriminative ability, with AUC = 0.728.
- **Final model:** a 4-feature RF (SII, PIV, E%, L%) reached AUC = 0.779 in internal validation.
- **Global importance:** L% (lymphocyte percentage) and E% (eosinophil percentage) contributed most to the model.
- **Nonlinear effects:** SII, PIV, L% and E% each showed complex, nonlinear relationships with recurrence.
- **Not verified (?):** sensitivity, specificity, calibration, and confidence intervals.

## Limitations

**Issues I noticed (useful for critical reading):**
1. **Small sample.** 148 patients is small for ML, so splits will be noisy and the recurrence group is likely a few dozen children. Confidence intervals are essential.
2. **Moderate discrimination.** AUCs of 0.73-0.78 are "fair". The abstract's "promising accuracy" is generous.
3. **Validation is internal only.** "Validation" appears to mean internal validation at one centre, with no external cohort, so generalisability is unknown.
4. **The AUC rose after feature reduction (0.728 to 0.779).** This can happen from tuning and selection on the same small data, which risks optimism. Check whether selection was nested inside cross-validation.
5. **Blood indices are routine but noisy.** Eosinophil and lymphocyte percentages vary with infection, allergy, medication and time of day, and composite indices like SII and PIV are derived from the same counts, so they are highly correlated. This weakens SHAP and Ceteris Paribus interpretations.
6. **Retrospective design (2015-2022).** Surgical technique and postoperative care may have changed over the period (?).
7. **No comparison with a simple logistic regression** is mentioned in the abstract. Check whether RF actually beat it.
8. **A web tool is not clinical validation.** Publishing a calculator does not show that it improves care.

## Takeaways for your own ML/analytics projects
- With small clinical datasets, use **nested cross-validation** so feature selection and tuning do not leak into the test estimate.
- Always include a **regularised logistic regression baseline**.
- Pair **SHAP (global)** with **Ceteris Paribus or partial-dependence profiles (individual / nonlinear)**, as this paper did.
- Report **calibration and decision-curve analysis**, not only AUC.
- Validate on an **external cohort** before releasing a prediction tool.
