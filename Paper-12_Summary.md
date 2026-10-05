# Paper 12 — Cost-Aware Active Feature Acquisition Under Realistic Clinical Constraints

## Reference
**J. C. Bingham et al.**, *Cost-Aware Active Feature Acquisition for Differential Diagnosis under Realistic Clinical Availability Constraints*, medRxiv, 2026.

> **Note:** The supplied paper summary identifies this as a medRxiv preprint and states that it was not peer reviewed.

## Type of Paper
Research/preprint on **Active Feature Acquisition (AFA)**.

## Main Problem
Active Feature Acquisition decides **which diagnostic test or feature should be obtained next**.

Traditional AFA evaluations can be overly optimistic because they may assume every test is available, ignore repeated patients, simplify costs, or ignore realistic time constraints.

## Main Method
The paper proposes/evaluates **EIG-Cost**.

**EIG = Expected Information Gain**

The idea is to select a test when the expected information it provides is worth its cost.

## General Workflow
1. Patient arrives with available information.
2. Candidate test panels are scored.
3. Expected information gain is estimated.
4. Test cost is considered.
5. The best-value test within the budget is selected.
6. The diagnosis estimate is updated.
7. The process continues while budget remains and uncertainty is sufficiently high.
8. A final prediction is produced.

## Dataset
The study uses **MIMIC-IV acute admissions**:

- Acute admissions: **64,766**
- Patients: **39,884**
- Conditions: **21**
- Features: **55**
- Test panels: **30**
- Patient-level budgets: **$30–$60**
- Five patient-level resamples were used.

Test-panel pricing used the **2026 Medicare fee schedule**.

## Methods Compared
EIG-Cost was compared against eight published AFA methods.

Under realistic availability:
- **3 of 8 methods collapsed to a vitals-only baseline**
- Remaining methods were adapted to the setting
- EIG-Cost remained comparatively robust

## Important Concepts

| Term | Meaning |
|---|---|
| Expected Information Gain | Expected reduction in diagnostic uncertainty from a test |
| Monte-Carlo Estimate | Average over simulated possible test outcomes |
| MIMIC-IV | Public database of hospital patient records |
| Macro-F1 | Average F1 across conditions |
| Patient-level split | All admissions from one patient stay in either train or test |

## Relevance to Our Project
CRS diagnosis can involve CT, endoscopy, blood tests, and clinical measurements. These tests can be expensive or unavailable to every patient.

The cost-aware AFA idea could help decide **which diagnostic information should be collected first**.

## Limitations
- The work is a preprint and not yet peer reviewed.
- Evaluation uses MIMIC-IV acute-care data.
- Pricing is based on one Medicare fee schedule.
- Results may differ in other hospitals and healthcare systems.

## Key Takeaway
> The best diagnostic feature is not necessarily the most informative one; it should provide enough information **relative to its cost and availability**.
