# Paper 15 — Evaluating Active Feature Acquisition Methods Safely from Past Data

## Reference
Based on **von Kleist et al., JMLR 26(60):1–84, 2025**.

## Type of Paper
Research paper on **safe/offline evaluation of Active Feature Acquisition policies**.

## Main Problem
Deploying an AFA agent changes which features are collected.

This can create a **distribution shift** between historical data and the data that would be produced by the new AFA policy.

Therefore, past data cannot simply be used to evaluate a new AFA agent as if the acquisition process had remained unchanged.

## Proposed Evaluation Framework
The paper defines **Active Feature Acquisition Performance Evaluation (AFAPE)**.

The goal is to estimate:
- Counterfactual acquisition costs
- Misclassification costs
- Overall expected performance

from retrospective data.

## Key Assumptions

### 1. NDE — No Direct Effect
Acquiring a feature does not change the underlying feature values.

### 2. NUC — No Unobserved Confounding
Past acquisition decisions depend only on observed features.

## Methods / Estimators
The paper discusses:
- **Direct Method (DM)**
- **Inverse Probability Weighting (IPW)**
- **Double Reinforcement Learning (DRL)**

It also describes a semi-offline reinforcement-learning framework when assumptions hold.

## Important Concepts

| Term | Meaning |
|---|---|
| Counterfactual | What would have happened if a different action had been taken |
| Positivity | Every possible action has some chance of occurring in historical data |
| Direct Method | Predict outcomes with a model |
| IPW | Re-weight observations according to historical action probability |
| DRL | Combines estimation approaches to improve reliability |
| Semi-offline RL | Combines historical data with partly simulated acquisition |

## Heart-Attack Example
The paper uses a diagnostic example:
1. Patient presents with chest pain.
2. Acquire diagnostic feature A1.
3. Acquire laboratory test A2.
4. Produce a diagnosis.
5. Pay acquisition costs.
6. If the diagnosis is wrong, pay a misclassification cost.

Therefore:

**Total Cost = Feature Acquisition Costs + Misclassification Cost**

AFAPE estimates expected total cost for a new AFA agent using historical data.

## Dataset / Evaluation Setting
The paper focuses on retrospective healthcare data and explains the evaluation problem through a heart-attack diagnosis example. The key contribution is the **evaluation framework**, rather than a new CRS-specific prediction dataset.

## Relevance to Our Project
Before deploying an AFA tool for CRS, evaluation should consider:
- Which tests the model would request.
- How much those tests would cost.
- How prediction errors affect patients.
- Whether historical data is sufficient to evaluate the new policy safely.

## Limitation
The evaluation methods depend on assumptions such as no direct effect, no unobserved confounding, and appropriate positivity. If these assumptions are violated, estimates can become unreliable.

## Key Takeaway
> AFA systems should not be evaluated naively on historical data because changing the acquisition policy can change the data distribution itself.
