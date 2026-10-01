# Summary: Machine Learning for Personalised Prediction of Revision Endoscopic Sinus Surgery (ESS)

*Nuutinen et al., PLOS ONE, 2022*

## Goal
Predict which chronic rhinosinusitis (CRS) patients will need **revision ESS** (a repeat sinus surgery) after their first (baseline) ESS, and explain which factors drive that risk for each individual. This is a binary classification problem.

## Data
- **Source:** electronic health records from the Helsinki University Hospital ENT department (HUS, Finland).
- **Sampling:** patients with ICD-10 rhinitis/sinusitis codes (J01, J30-J33) seen in 2005, 2007, 2009, 2011 and 2013 (n = 5,080). Follow-up ran until 2019.
- **Final cohort:** **767 surgical CRS patients** aged 16 or older (27 under 16 excluded), of whom **111 (14.5%) had revision ESS**. Revisions happened on average 30.3 months after baseline ESS.
- **Features (~28):**
  - Demographics (age, sex).
  - Comorbidities: asthma, allergy, CRSwNP, NERD (NSAID-exacerbated respiratory disease), immunodeficiency or suspicion of it, diabetes, obesity, and others.
  - Healthcare-use features: number of visits before and after baseline ESS, visit frequency, and days from first visit to ESS.
- **Extraction:** structured codes plus keyword mining of free clinical text.
- **Split:** 70% training and 30% test, stratified by outcome.

## Workflow (flowchart)

```
5,080 patients with rhinitis/sinusitis codes (sampled 2005-2013)
                │
                ▼
Keep surgical CRS patients with baseline ESS, age ≥16 → n = 767
                │
                ▼
Extract features (structured EHR + free-text keyword mining)
                │
                ▼
Stratified split: 70% training │ 30% test
                │
   ┌────────────┴────────────────────────────────┐
   ▼                                             ▼
Pipeline (a): model comparison &            Pipeline (b): interpretability
data-collection-time analysis               & univariate models
   │                                             │
Step 1: Sequential Forward Selection         Predefined variables
        (SFS, AUROC, CV, top 15 vars)             │
   │                                         Step 2: Grid-search hyperparameters
Step 2: Grid-search hyperparameters               │
   │                                         Step 3: Train on train + validation
Step 3: Train final model                         │
   │                                         Step 4: Test on held-out 30%
Step 4: Evaluate on 30% test fold                 │
   │                                         SHAP + partial dependence plots
Repeat 10× (10 train/test reformulations)    (gradient boosting, 6-month data)
and average
```

## Algorithms and methods
- **Three classifiers compared:** Logistic Regression (LR), Random Forest (RF), Gradient Boosting (GB, XGBoost).
- **Imbalance handling:** class-weighted loss functions (no SMOTE).
- **Feature selection:** SFS maximising AUROC. Variables are ranked by how early SFS picks them, with 15 points for the first and 14 for the second, summed over 10 runs (Eq. 1, max 150).
- **Tuning:** `GridSearchCV`.
- **Four analyses:**
  1. Univariate logistic models.
  2. Classifier comparison.
  3. Effect of data-collection window: 0, 3, 6 or 12 months after baseline ESS (LR).
  4. Interpretability with **SHAP** (`TreeExplainer`) and **partial dependence plots** (`pdpbox`).
- **Metrics:** AUROC, AUPRC, sensitivity, specificity, F1. Baselines were checked by training on randomised labels (AUROC about 0.5, AUPRC about 0.15).
- **Tools:** Python (`sklearn`, `xgboost`, `mlxtend`, `shap`, `pdpbox`, `numpy`, `pandas`).

## Results

**Classifier comparison (6-month window, best number of variables):**

| Model | Best AUROC | # Variables |
|---|---|---|
| Logistic Regression | 0.744 | 6 |
| Gradient Boosting | 0.741 | 8 |
| Random Forest | 0.737 | 11 |

All three performed about the same, so a simple LR was as good as the black-box models. AUPRC was about 0.35 against a 0.145 baseline.

**Effect of follow-up window (LR, all variables):**

| Data collected until | AUROC | Sensitivity | Specificity |
|---|---|---|---|
| Baseline ESS (0 mo) | 0.682 | 0.55 | 0.71 |
| 3 months | 0.715 | 0.61 | 0.73 |
| 6 months | 0.744 | 0.62 | 0.75 |
| 12 months | 0.784 | 0.61 | 0.79 |

**Top predictors (SFS rank and SHAP):**
1. Number of visits in the 6 months after baseline ESS (strongest, univariate AUROC 0.77 for 12 months).
2. CRSwNP.
3. Asthma.
4. Immunodeficiency or suspicion of it.
5. NERD.
6. Allergy.
7. Visit frequency before ESS.
8. Age.

**Non-linear effects found by PDP (the main novelty):**
- **Age:** lowest risk at 16-30 years, higher at 30-65, and slightly lower again above 70.
- **Pre-ESS visits:** a U-shape. Patients with 10-25 visits had lower risk than those with fewer than 10 or more than 25.
- **Time from first visit to ESS:** risk drops sharply after about 100 days, so a short wait means higher risk.

## Limitations

**Stated by the authors:**
- Data are from a relatively small number of patients.
- Care practices changed over the 2005-2019 period.
- Patients may have had revision surgery elsewhere (they argue this is minimal, since over 90% of ESS is public).
- No time-series modelling (e.g., LSTM).
- The baseline ESS may not be the patient's first, and EHR data only start in 2005.
- Missing variables: postoperative treatment, symptom scores, polyp score, Lund-Mackay CT score, medication, smoking, eosinophils, and extent of surgery.
- Replication in other populations is needed.

**Issues I noticed (useful for critical reading):**
1. **Likely target leakage with the timing of "future" data.** Post-op visits in the first 6-12 months are the top predictor, but many revisions happen within that same window (Fig. 1a). Patients already heading to revision surgery naturally have more visits. The 12-month model's higher AUROC (0.784) partly reflects this, and it is not a true early prediction.
2. **Visits measure health-service behaviour, not just disease.** Visit counts depend on referral patterns, physician habits and patient preference, so the conclusion that visit frequency "signals poor control" is an interpretation, not a finding.
3. **Unequal follow-up and no survival analysis.** Revision patients were followed for 30 months on average and non-revision patients for 82. Treating this as plain binary classification ignores censoring. A Cox or time-to-event model would suit the question better.
4. **Moderate performance.** AUROC of about 0.74 and sensitivity of about 0.6 means roughly 40% of revisions are missed. The authors' own scale calls 0.7-0.8 only "fair", yet the abstract and conclusions sound more positive.
5. **Small test set.** 30% of 767 is about 230 patients, with only about 33 revision cases. Results are averaged over 10 random re-splits, which is repeated hold-out rather than a true external dataset, although the paper calls the test fold "external".
6. **Unstable feature selection.** SFS picked different top-10 variables for each classifier (e.g., memory disorders and obesity for GB, diabetes and cancer for RF). Some selected variables, such as memory disorders, have no clear clinical rationale and look like noise.
7. **Odd univariate statistics.** An odds ratio of 112 for 12-month visits with a CI of 105-120 is implausibly narrow. Averaging CIs over 10 reformulations of the same data understates uncertainty, and the scaling of the variables isn't explained.
8. **Inconsistencies.** The abstract gives RF AUROC as 0.730 but the results say 0.737. The text also lists the AUPRC values in an order that doesn't match Table 4.
9. **Subjective features.** "Suspicion of immunodeficiency" and text-mined NERD depend on keyword extraction and physician judgement, so they may not transfer to other hospitals.
10. **Correlated features weaken SHAP and PDP.** Partial dependence assumes independent features, but days-to-ESS and visit counts are correlated (r = 0.51).
11. **No calibration or decision-curve analysis.** The paper promotes "personalised risk" but never shows that predicted probabilities are well calibrated.
12. **Their own result undercuts the ML angle.** LR matched GB and RF, which fits the literature finding that ML often does not beat LR on clinical tabular data. The non-linear effects are interesting, but they didn't improve accuracy.

## Takeaways for your own ML/analytics projects
- Test whether a complex model beats a **simple logistic regression** baseline. Here it did not.
- **Time-window design matters.** Decide exactly what data would be available at prediction time, or you create leakage.
- When follow-up time differs between patients, consider **survival analysis** instead of binary labels.
- Use **SHAP and PDP** to find non-linear effects such as the U-shape, which linear models miss.
- Report **AUPRC against the prevalence baseline** for imbalanced data, and use a randomised-label sanity check as they did.
- Add calibration plots, confidence intervals, and a truly external test set.

## Quick comparison across the three papers

| | Paper 1 (CRSwNP) | Paper 2 (ICU ARS) | Paper 3 (Revision ESS) |
|---|---|---|---|
| Task | Polyp vs. no polyp | In-hospital death | Revision surgery |
| Data | 206, single centre | 559, 3 public databases | 767, single region |
| Feature selection | Univariate, AIC, LASSO | Boruta | Sequential forward selection |
| Models | LR + nomogram | 9 models, RF best | LR, GB, RF (all similar) |
| Validation | Internal split | External databases | Repeated 70/30 hold-out |
| Interpretability | Nomogram | SHAP | SHAP + PDP |
| Best AUROC | 0.96 (likely inflated) | 0.80 | 0.74-0.78 |
| Main concern | Label leakage (LK score) | Few events, wide CIs | Temporal leakage, censoring |

