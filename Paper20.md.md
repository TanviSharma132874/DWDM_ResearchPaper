PAPER 20 | THEORETICAL, MODELING OR SIMULATION STUDY
Personalized health monitoring using explainable AI: bridging trust in predictive healthcare
M. Sree Vani, Rayapati Venkata Sudhakar, A. Mahendar, Sukanya Ledalla, Marepalli Radha, M. Sunitha
Scientific Reports, 2025; 15: 31892
Links: DOI | Consensus summary
## 1. Background and objective
Many accurate deep learning models are black boxes, so clinicians do not trust them.
Existing explanation methods are often generic or post-hoc, rarely patient-specific, and tested only on narrow datasets.
Objective: build a model that is both accurate and explainable, giving a prediction plus the reasons behind it.
## 2. Dataset used
Dataset: MIMIC-III, a public, de-identified critical-care database.

## 3. Methodology flowchart

Figure 1. Data-to-model pipeline
Step by step
Take an ICU subset of MIMIC-III with complete vitals, labs and demographics.
Clean and scale the data, reduce dimensions with PCA, and balance the classes.
Train PersonalCareNet: three dense layers (128, 64, 32) with ReLU, dropout 0.3, optional attention, sigmoid output; Adam optimiser, learning rate 0.001, batch 64, early stopping.
Predict risk for unseen patients.
Run SHAP to show which features pushed each prediction up or down (local) and which matter overall (global).

## 4. Results

Figure 2. Prediction plus explanation for the clinician
## 5. What it means
The model beats nine baselines on accuracy while still giving readable explanations, so interpretability does not have to cost much performance.
SHAP force plots explain one patient’s risk; summary and feature-importance plots show what drives risk across all patients.
## 6. Limitations to mention
Evaluated on one dataset (MIMIC-III) with no external validation; the authors list cross-dataset validation as future work.
The abstract mentions a CNN with attention (“CHARMS”), while the Methods describe dense layers with an optional attention step. Quote the Methods description.
Text says 100 epochs, but the accuracy and loss figures cover 30 epochs.
Random undersampling produced the balanced set. Check that it was applied without leaking test data into training.
## 7. Suggested presentation outline (6 slides)
## 8. Likely questions and short answers
Q: What is SHAP?
A: A method that gives each input feature a score for how much it pushed one prediction up or down, based on Shapley values from game theory.
Q: Local vs global explanation?
A: Local explains one patient’s prediction; global shows which features matter across all patients.
Q: Why undersample?
A: The outcome classes were imbalanced, and balancing stops the model from favouring the majority class.
## 9. How it fits the team project
Your team’s explainable-AI section: it shows how the AI methods in the CRS papers can be made trustworthy.

| KEY TAKEAWAY (say this first in your presentation) PersonalCareNet predicts patient risk from ICU data with 97.86% accuracy and explains every prediction with SHAP, both for the whole population (global) and for each individual patient (local). |
| --- |

| Item | Details |
| --- | --- |
| Dataset | MIMIC-III (Medical Information Mart for Intensive Care III) |
| Full database | Over 40,000 ICU patients with demographics, vital signs, lab tests, records, discharge summaries |
| Features used | Age, gender, heart rate, respiratory rate, systolic and diastolic blood pressure, oxygen saturation, glucose, selected lab indicators |
| Cleaning | Removed records with more than 30% missingness or inconsistent values |
| Final dataset | 10,432 samples, class-balanced by random undersampling |
| Preprocessing | Mean/median imputation, mode for categories, z-score scaling, PCA keeping 95% of variance |
| Split | 70% training, 15% validation, 15% test (stratified) |

| Metric | PersonalCareNet |
| --- | --- |
| Accuracy | 97.86% (best baselines: AutoGluon 93.0%, TabNet 92.5%) |
| AUC | 98.3% |
| Precision | 96.2% |
| Recall | 95.4% |
| F1-score | 95.8% |
| Training / validation accuracy | About 99% / 97.5% |

| Slide | Content |
| --- | --- |
| 1. Problem | Accurate AI is not enough; clinicians need reasons. |
| 2. Dataset | MIMIC-III subset: 10,432 samples, features listed. |
| 3. Pipeline | Show Figure 1. |
| 4. Model + XAI | Show Figure 2: prediction plus SHAP, local and global. |
| 5. Results | 97.86% accuracy, AUC 98.3%, vs AutoGluon 93.0%. |
| 6. Limits + take-home | Single dataset; trust comes from explanation plus validation. |

| CHECK AGAINST THE FULL TEXT BEFORE PRESENTING This write-up is based on the abstract and the openly readable parts of the paper. Confirm these details in the PDF: Name and definition of the binary outcome label being predicted. Whether undersampling happened before or after the train/test split. The full comparison table (Table 2) if you quote more baselines. |
| --- |
