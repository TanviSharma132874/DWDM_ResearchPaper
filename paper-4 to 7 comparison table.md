# Comparison of Machine Learning Papers on Chronic Rhinosinusitis (CRS)

> Papers 1 and 2 use figures from the original summary file. Fields marked **n/v** were not available from the abstracts accessed. Items marked **(?)** are inferred and should be checked against the full papers.

## Table 1: Task and data

| Paper | Topic | Task / outcome | Sample | Data source |
|---|---|---|---|---|
| 1 | CRSwNP | Polyp vs. no polyp | 206 | Single centre |
| 2 | ICU acute rhinosinusitis | In-hospital death | 559 (66 deaths, 11.8%) | 3 public databases |
| 3 | Revision sinus surgery (Nuutinen 2022) | Revision ESS (yes/no) | 767 (111 revisions, 14.5%) | Helsinki hospital records, 2005-2019 |
| 4 | Diagnosis (Raghavan 2025) | LMS ≥5, and LMS ≥5 plus ≥2 symptoms | 543 | Tertiary clinic, patient-reported data, 57 predictors |
| 5 | Blood eosinophilia (Ishikawa 2026) | AEC class (2 classes at 500/µL; 3 classes at 500 and 1500/µL) | 399 | n/v (likely single centre) |
| 6 | Comorbid CEP in CRS (Ishikawa 2025) | CEP / ECRS features | n/v (19 ECRS within the CEP group) | n/v (likely single centre) |
| 7 | Pediatric recurrence (Jiang 2025) | Postoperative recurrence after FESS | 148 children | Xiangya Hospital, 2015-2022 |

## Table 2: Methods and results

| Paper | Feature selection | Models | Validation | Interpretability | Best result | Main concern |
|---|---|---|---|---|---|---|
| 1 | Univariate, AIC, LASSO | LR + nomogram | Internal split | Nomogram | AUROC 0.96 | Label leakage (LK score), likely inflated |
| 2 | Boruta | 9 models, RF best | External databases | SHAP | AUROC 0.80 | Few events, wide CIs |
| 3 | Sequential forward selection | LR, GB, RF (all similar) | Repeated 70/30 hold-out | SHAP + PDP | AUROC 0.74-0.78 | Temporal leakage, censoring |
| 4 | n/v | XGBoost, RF, DNN, LR | 90:10 split + CV | n/v | XGBoost AUC 0.71 (primary), 0.80 (secondary); sensitivity ~0.37 | Tiny test set, low sensitivity |
| 5 | Univariate screen | LR, RF, SVM, XGBoost (?); RF and XGBoost best | n/v | SHAP | AUC n/v | Association not causation, arbitrary cutoffs |
| 6 | n/v | LR, SVM, RF, XGBoost; RF and XGBoost best | n/v | SHAP + decision tree | AUC n/v; top features: eosinophils, CRP, WBC | Rare outcome, circular predictors |
| 7 | Importance-based reduction to 4 features | 3 models, RF best | Internal validation | SHAP + Ceteris Paribus | AUC 0.728, then 0.779 after reduction | Small n, internal validation only |

## Patterns across the papers

- **Tree ensembles dominate.** RF or XGBoost was the best or joint-best model in Papers 2 and 4-7. In Paper 3 all three models performed about the same.
- **Discrimination is moderate.** Most honest AUCs fall between 0.70 and 0.80 (Papers 2, 3, 4 and 7). Only Paper 1's 0.96 looks inflated.
- **Sensitivity is a recurring weakness.** Paper 4 reports about 37%, and Paper 3 about 60%. High specificity or accuracy can hide this.
- **SHAP is near-universal** (Papers 2, 3, 5, 6, 7), often paired with PDP, Ceteris Paribus or a decision tree to check nonlinear effects.
- **Validation is weak.** Only Paper 2 used external data. The others rely on internal splits, repeated hold-out or a single centre.
- **Small samples dominate.** Sizes run from 148 to 767, and the key subgroups (CEP cases, recurrences) are much smaller.
- **Two study types.** Papers 1, 2, 3, 4 and 7 are prediction studies. Papers 5 and 6 are feature-discovery studies, so their metrics are not comparable to the others.
- **Recurring flaws:** leakage (Papers 1 and 3), small test sets (Paper 4), arbitrary cut-offs (Papers 4 and 5), and no calibration analysis in any paper as far as could be verified.
