

Summary: Predicting Who Benefits from Sinus Surgery (CRS), Generative AI vs. Supervised ML
1. Problem and goal
Condition: Chronic Rhinosinusitis (CRS), a long-term sinus inflammation. Patients respond very differently to endoscopic sinus surgery (ESS).
Question: Using only pre-operative data, can we identify patients who will not improve, so surgery could have been avoided?
Success definition: SNOT-22 (a quality-of-life score, 0–110) drops by ≥ 8.9 points at 6 months (the MCID).
Class 1 = improved (surgery worked).
Class 0 = did not improve (the minority class the model must find).
Comparison: Supervised ML (LR, SVM, Naïve Bayes, Random Forest, MLP) vs. generative AI (ChatGPT-5 Thinking, MedGPT-5, Gemini 2.5 Pro, Perplexity Sonar, Claude Sonnet 4.5, DeepSeek-V3.2).
2. Dataset
NIH-funded multicenter study (NCT01332136), two cohorts: 791 patients and 355 patients.
After merging and keeping only patients who had ESS, 524 surgical cases remained.
Features: demographics, socioeconomic factors, phenotype (with or without polyps), comorbidities, CT Lund–Mackay score, endoscopy score, baseline SNOT-22.
Test set: n = 105 (85 Class 1, 20 Class 0).
3. Pipeline / flowchart
        EHR pre-operative data (no post-op fields → no leakage)
                           │
                 Data cleaning + encoding
                           │
          ┌────────────────┴─────────────────┐
          ▼                                  ▼
   PATH 1: Supervised ML              PATH 2: Generative AI
   Train LR / SVM / NB / RF / MLP     Serialize patient row into a prompt
   (80/20 stratified split,           (expert-otolaryngologist role,
    class weights / focal loss)        MCID = 8.9, chain-of-thought)
          │                                  │
          │                          Run k=5 times per patient,
          │                          majority vote
          ▼                                  ▼
   Surgery recommendation (0/1)      Surgery recommendation (0/1) + confidence
          └────────────────┬─────────────────┘
                           ▼
        Compare: AUROC, AUPRC, F1, sensitivity/specificity,
        calibration, decision-curve net benefit, DeLong/McNemar tests
4. Key methods
Leakage control: post-op variables removed; encoders and scalers fit inside CV folds; no patient overlap across splits.
Imbalance handling: class weights for LR and trees, focal or weighted loss for the MLP.
Best model: MLP with 1 hidden layer of 400 neurons.
GenAI protocol:
Zero-shot prompt with a strict output schema: PREDICTION: 0/1 and CONFIDENCE: one of 5 levels.
Temperature 0.1–0.5, top-p 0.7–0.95, 5 replicates, with logging of model version, date and prompt hash.
5. Main results
Model	AUROC	F1	AP
MLP (ours)	0.66	0.83	0.86
Claude Sonnet 4.5	0.63	0.81	0.85
DeepSeek-V3.2	0.62	0.77	0.85
ChatGPT-5 Thinking	0.58	0.77	0.84
MedGPT-5	0.49	0.72	0.81
Gemini 2.5 Pro	0.40	0.62	0.78
Perplexity Sonar	0.36	0.61	0.78
MLP confusion matrix: [9, 11; 5, 80]. It has the best Class 0 recall (0.45) among the top models.
Treat-all bias: most LLMs recommend surgery for nearly everyone. MedGPT caught 0 of 20 non-responders and Perplexity caught 0, giving high accuracy but no clinical value.
Accuracy is misleading under imbalance (about 81% of test patients are responders), so AUROC, AP and Class 0 recall matter more.
6. How ChatGPT "reasoned" (the heuristic algorithm)
Expected improvement = 45% × baseline SNOT-22.
Multiply by bracket factors for baseline SNOT-22 (×0.5 up to ×1.2), endoscopy score, CT Lund–Mackay score, and a ×1.05 polyp bonus.
Apply penalty multipliers for comorbidities: depression and fibromyalgia ×0.7, smoker ×0.85, COPD ×0.8, asthma ×0.9, OSA and diabetes ×0.9, GERD ×0.95, prior surgery ×0.85, age ≥ 65 ×0.9.
Predicted score = baseline − adjusted improvement. If Δ > 9 → recommend surgery (1), else 0.
Confidence is set by the distance of Δ from the threshold.
Clinicians found this logic sensible, but the multiplicative stacking over-penalizes comorbidities, the confidence bins are coarse and poorly calibrated, and the cutpoints were never tuned on data.

7. Feature importance (MLP, permutation importance)
Top drivers, in order: baseline SNOT-22 (by far the largest), age, CT Lund–Mackay score, allergy testing, prior surgery, polyps. Smaller effects came from COPD, septal deviation, income, insurance, sex and race, which the authors flag for fairness monitoring. These match what the LLM said it was using, so the LLM's explanations are plausible even though its predictions are weaker.

8. RAG experiment
Adding retrieved guideline text (EPOS/AAO-HNS excerpts, BM25, top-5) did not help (AUROC 0.57 vs. 0.58, with 21 false positives). The guidelines repeat general heuristics the LLM already knows and add no patient-specific signal.

9. Conclusion
The authors propose an ML-first, GenAI-augmented workflow:

A calibrated tabular ML model handles primary triage and risk estimates.
GenAI handles plain-language explanations, patient-facing summaries and auditable rationales.
10. Limitations (stated and my own observations)
Stated by the authors:

Retrospective, single clinical domain.
The 8.9 MCID cutoff is a pragmatic threshold, so labels are noisy near it.
LLMs were restricted to binary and confidence outputs.
The RAG corpus was small.
Weaknesses worth noting if you cite or reproduce this:

Small test set: only 20 Class 0 patients, so the metrics are unstable and the confidence intervals would be wide.
Modest discrimination: AUROC 0.66 is only a little better than the best LLM, and Claude (0.63) is close. The abstract's "superior" claim is stronger than the numbers support.
Missing evidence: the abstract mentions calibration, decision-curve net benefit and subgroup analyses, but the paper shows no calibration plots, Brier scores or net-benefit curves.
Inconsistencies: the abstract lists four LLMs but the results cover six. Table I lists Logistic Regression at 0.85 accuracy, the same as the MLP, and Naïve Bayes at 0.30. The RAG test counts (24 Class 0, 81 Class 1) differ from the main test split (20 and 85).
Missing data was simply deleted, which can bias the sample.
Zero-shot only: no few-shot or fine-tuned LLM comparison, and results depend on model versions from a single date (2025-10-21).
Hyperparameter tuning of the baselines is not described in detail.
11. Takeaways for you (data analytics angle)
Accuracy alone is misleading on imbalanced data. Use AUROC, AUPRC, F1 and per-class recall.
Tabular models like MLPs and tree ensembles still beat LLMs on structured prediction tasks.
Avoiding data leakage and calibrating probabilities matter as much as picking a model.
Permutation importance is a simple, model-agnostic way to explain predictions.
