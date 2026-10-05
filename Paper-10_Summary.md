# Paper 10 — Artificial Intelligence in Rhinology: Diagnostic Accuracy and Barriers

## Reference
**B. H. Shrikrishna, G. Deepa**, *The Application and Diagnostic Accuracy of Artificial Intelligence in Rhinology: A Review*, Cureus, 2025.

## Type of Paper
Systematic/literature review.

## Main Problem
The paper investigates how accurately Artificial Intelligence can diagnose nose and sinus disorders and what barriers prevent AI systems from being used routinely in clinical practice.

## Methodology
The review searched **PubMed** using:

`(Rhinology) AND (Artificial Intelligence) AND (Accuracy)`

- Records identified: **27**
- Studies included: **12**
- Inclusion criteria included AI application, human participants, diagnostic focus, sample size of at least 10, available full text, and reported diagnostic performance.

## Algorithms / AI Methods
One included study with **543 patients** compared:
- **XGBoost**
- **Random Forest**
- **Deep Neural Network (DNN)**
- **Logistic Regression**

## Dataset
The review used a literature dataset:
- Database: PubMed
- Records found: 27
- Studies included: 12
- One reported study contained 543 patients.

## Reported Performance

| Metric | Reported Range |
|---|---:|
| Accuracy | 74.5–85.5% |
| Sensitivity | 36–39% |
| Specificity | 92–98% |

## Major Barriers
- Small or single-hospital datasets
- Limited testing on other populations
- Difficulty integrating AI into clinical workflows
- Lack of clinician trust when predictions cannot be explained
- Need for validation in realistic clinical settings

## Relevance to Our Project
For a CRS/sinus-risk prediction system, this paper shows that accuracy alone is not enough. **Sensitivity is especially important because missing patients with the disease can be clinically significant.**

## Key Takeaway
> A useful medical AI system needs good predictive performance **and** clinical reliability, validation, and interpretability.
