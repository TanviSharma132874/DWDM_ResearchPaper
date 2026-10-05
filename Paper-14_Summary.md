# Paper 14 — Active Feature Acquisition via Explainability-Driven Ranking

## Reference
Paper based on **Güney et al., ICML 2025**, PMLR 267:20748–20765.

## Type of Paper
Research paper on **Explainability-Driven Active Feature Acquisition**.

## Main Problem
Static feature selection uses the same features for every patient. However, the most useful features can vary from patient to patient.

The paper uses **local explanations** to create patient-specific feature rankings and then learns a policy for selecting the next feature.

## Static vs Dynamic Selection

**Static selection:** Same features are selected for every patient.

**Dynamic AFA:** Different patients can receive different feature-acquisition sequences.

## Main Method
1. Train a predictor.
2. Apply a **local explanation method**.
3. Obtain **per-instance feature importance**.
4. Rank features for each instance.
5. Reframe AFA as a feature-prediction problem.
6. Train a **decision-transformer policy network** using the rankings.
7. At test time, acquire features sequentially according to their importance.

## Algorithms / Techniques
- Local explanation methods
- Feature importance ranking
- **Decision Transformer**
- Sequential Active Feature Acquisition

## Dataset
The paper reports experiments on **multiple datasets**.

The supplied summary does not provide the individual dataset names.

## Findings
The paper reports that the approach:
- Outperformed state-of-the-art AFA methods in predictive accuracy.
- Improved acquisition efficiency.
- Demonstrated that XAI can be used to learn decision policies, not only explain final predictions.

## Why This Approach Helps

### Patient-specific
Feature rankings are generated for each instance.

### Efficient Acquisition
Important features can be collected earlier.

### Avoids Hard Reinforcement Learning
The approach learns from explanation rankings rather than directly solving a difficult sparse-reward RL problem.

## Example for a CRS Project
Suppose a patient has:
- PM2.5 exposure
- Nasal symptom score
- IgE
- CT-related information
- Clinical measurements

An explainability model could rank these features differently for different patients. The AFA policy could then acquire the most useful missing feature first.

## Limitation
The learned policy inherits weaknesses or biases from the explanation method used to generate the feature rankings.

## Relevance to Our Project
If a CRS model uses SHAP to rank inputs such as PM2.5, nasal symptom scores, and IgE, those rankings could potentially help decide which missing feature to collect first.

## Key Takeaway
> Explainability can be used not only to explain predictions but also to decide **which feature should be collected next**.
