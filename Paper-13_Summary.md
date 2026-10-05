# Paper 13 — Learning-To-Measure: In-Context Active Feature Acquisition

## Reference
Paper based on **Kobayashi et al., arXiv, 2025**.

## Type of Paper
Research paper on **in-context Active Feature Acquisition (AFA)** and meta-learning.

## Main Problem
Real medical data often has:
- Missing features
- Retrospective missingness
- Few labels

Training a separate model for every new task can be slow and data-hungry.

## Proposed Approach
The paper presents **Learning-to-Measure (L2M)**, a meta-learning framework with:
1. Reliable uncertainty estimation on unseen tasks.
2. An agent that chooses which feature to acquire next.

The approach works **in-context**, without retraining the model for every new task.

## Algorithm / Method
1. Pre-train a sequence model autoregressively on many tasks with arbitrary missingness.
2. Introduce a new task using observed data as context.
3. Estimate uncertainty over the label.
4. Select the feature with the highest **conditional mutual information**.
5. Acquire the selected feature.
6. Add it to the context.
7. Re-estimate uncertainty.
8. Continue until uncertainty is sufficiently low.
9. Produce the final prediction.

## Feature Selection Criterion
The greedy agent selects the next feature using **Conditional Mutual Information (CMI)**.

CMI measures how much additional information a feature provides about the label, given the information already observed.

## Dataset
Experiments were reported on:
- **Synthetic tabular benchmarks**
- **Real tabular benchmarks**

The supplied summary does not specify individual dataset names.

## Findings
- Tested on synthetic and real tabular benchmarks.
- Matched or surpassed task-specific baselines.
- Improvements were strongest when labels were scarce and missingness was high.

## Traditional AFA vs L2M

| Traditional AFA | L2M |
|---|---|
| Often retrains for a task | Handles a new task in-context |
| Requires task-specific learning | Uses a pre-trained sequence model |
| Can be expensive | Avoids per-task retraining |
| Sequential feature acquisition | Sequential feature acquisition |

## Limitation
Greedy mutual-information selection can be **myopic**. A feature may be weak individually but highly useful when combined with another feature.

## Relevance to Our Project
An L2M-style approach could potentially decide **which missing clinical/environmental feature should be collected next** without training a completely new model for every task.

## Key Takeaway
> One pre-trained model can serve many feature-acquisition tasks by using the available data as context.
