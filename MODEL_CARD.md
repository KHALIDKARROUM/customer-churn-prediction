# Model Card: Telco Churn Classifier

## Intended use

This model demonstrates churn-risk ranking and retention analysis for the bundled Telco dataset. It may support exploratory prioritization and portfolio demonstrations. It is not approved for autonomous customer decisions, pricing, eligibility, or production outreach.

## Inputs and target

The model uses tenure, contract, total charges, internet service, monthly charges, and payment method. The target is the dataset's `Churn` field. Blank `TotalCharges` values are removed, leaving 7,032 records.

The dataset does not specify an as-of date or future outcome window. Consequently, a score should be described as association with the recorded churn label, not a guaranteed probability of churn in a defined future period.

## Training and evaluation

Candidate selection uses five-fold cross-validation on the training partition only. The selected classifier is sigmoid-calibrated, its decision threshold is derived from training-only out-of-fold predictions, and final metrics are calculated once on a stratified 20% holdout. `artifacts/model_metadata.json` contains the current generated results.

Population charts use out-of-fold probabilities so that records are not displayed with scores from a model that trained on those same records.

## Important limitations

- The sample is small, cross-sectional, and may not represent a current customer population.
- Feature importance describes model behavior, not causal effects.
- Contract and payment behavior can proxy for economic or demographic circumstances.
- Calibration and thresholds can deteriorate when products, prices, or customer behavior change.
- Retention recommendations are rules based on observed associations and have not been validated by experiments.

## Required checks before operational use

- Confirm dataset provenance, redistribution rights, consent, and retention policies.
- Define an as-of date and prediction horizon, then rebuild the training table without future leakage.
- Measure performance and calibration by relevant customer subgroups.
- Monitor feature drift, label drift, and intervention outcomes.
- Set the threshold using explicit false-positive and false-negative costs.
- Provide human review and a way to audit every scored record and model version.
