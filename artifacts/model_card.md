# Model Card

## Intended use

This model ranks simulated financial transactions by fraud risk using only information
available before transaction completion. It is a research and portfolio artifact, not a
production authorization system.

## Data and validation

- Dataset rows: 6,362,620
- Fraud prevalence: 0.1291%
- Temporal split: steps 1-520 train, 521-631 validation, later steps test
- Account identifiers are excluded to reduce memorization and privacy risk.
- `isFlaggedFraud` is evaluated as a baseline but excluded from model features.
- Post-transaction sender and recipient balances are excluded from model features.

## Test performance

| model                       |   threshold |   roc_auc |   average_precision |   precision |   recall |     f1 |     f2 |   balanced_accuracy |   accuracy |   true_negatives |   false_positives |   false_negatives |   true_positives |
|:----------------------------|------------:|----------:|--------------------:|------------:|---------:|-------:|-------:|--------------------:|-----------:|-----------------:|------------------:|------------------:|-----------------:|
| Existing rule               |      0.5000 |    0.5028 |              0.0195 |      1.0000 |   0.0056 | 0.0111 | 0.0070 |              0.5028 |     0.9861 |            88214 |                 0 |              1245 |                7 |
| Logistic regression         |      0.0607 |    0.9700 |              0.6585 |      0.5329 |   0.6334 | 0.5788 | 0.6104 |              0.8128 |     0.9871 |            87519 |               695 |               459 |              793 |
| Histogram gradient boosting |      0.3801 |    0.9987 |              0.9331 |      0.8231 |   0.8770 | 0.8492 | 0.8657 |              0.9372 |     0.9956 |            87978 |               236 |               154 |             1098 |

The operating threshold (0.380144) was selected on the validation set
by maximizing F2, which weights recall more heavily than precision.

## Most influential features

| feature             |   importance_mean |   importance_std |
|:--------------------|------------------:|-----------------:|
| log_oldbalance_org  |           0.79858 |          0.00905 |
| log_amount          |           0.71423 |          0.01076 |
| log_oldbalance_dest |           0.21939 |          0.00513 |
| type_transfer       |           0.21187 |          0.00635 |
| type_cash_out       |           0.09974 |          0.00325 |
| type_cash_in        |           0.06549 |          0.00134 |
| type_payment        |           0.06531 |          0.00530 |
| hour                |           0.04377 |          0.00470 |
| step                |           0.00000 |          0.00000 |
| day                 |           0.00000 |          0.00000 |

Permutation importance describes predictive contribution, not causality.

## Limitations

- PaySim is simulated data and does not represent every real fraud pattern.
- Production integration must verify that every input is available at authorization time.
- Financial loss and investigation cost are not provided. The F2 threshold is therefore a
  transparent proxy, not a fully optimized business decision.
- Performance should be monitored over time for drift and subgroup failures before
  deployment.
