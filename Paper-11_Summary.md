# Paper 11 — Explainable AI in Disease Prediction

## Reference
**R. Alkhanbouli, H. M. A. Almadhani, F. Alhosani, M. C. E. Simsekler**, *The role of explainable artificial intelligence in disease prediction: a systematic literature review and future research directions*, BMC Medical Informatics and Decision Making, 2025.

## Type of Paper
Systematic literature review.

## Main Problem
Many AI models can make accurate medical predictions but behave like **black boxes**. Clinicians may find it difficult to understand why a model produced a particular prediction.

The paper studies the use of **Explainable AI (XAI)** in disease prediction.

## Methodology
The review searched:
- Scopus
- PubMed
- Web of Science

A PRISMA-based selection process was used.

- Articles identified: **76**
- Articles excluded: **46**
- Studies synthesized: **30**

## Main XAI Methods

| XAI Method | Share |
|---|---:|
| SHAP | 38% |
| LIME | 26% |
| Grad-CAM | 5% |
| Fuzzy Logic | 5% |
| PDP | 5% |

## Important Techniques

### SHAP
Assigns importance values to input features and can provide global and local explanations.

### LIME
Explains an individual prediction using a simpler local model.

### Grad-CAM
Highlights image regions that influenced an image-based prediction.

### PDP
Shows how a feature affects model predictions.

## Major Findings
- SHAP was the most frequently used XAI technique.
- LIME was the second most common.
- Many studies still depend on limited datasets.
- Model complexity can reduce interpretability.
- Many studies rely on only one data type.
- Clinicians may find explanations difficult to interpret.

## Relevance to Our Project
For a CRS prediction model, **SHAP can explain which variables contribute to the predicted risk**, such as PM2.5, symptoms, clinical measurements, or other patient features.

## Key Takeaway
> A medical prediction model should not only say **what** the prediction is; XAI should help explain **why** the model made that prediction.
