# VC Investment Decision Classifier

Predicts whether a VC would **invest (1)** or **pass (0)** on a startup, from 7 inputs:
six criteria scored 1–5 — Team, Technology, Market, Value Proposition, Competitive Advantage,
Socially Impactful — plus the Funding amount.

It learns from ~765 hand-evaluated startups in `datafile.csv`. The two human evaluators
(William, Wout) agreed 99% of the time, so their decisions are combined into one label
(conflicting rows are dropped).

## Why gradient boosting (not an LLM)

The data is small and tabular (7 numeric inputs, ~765 rows). A gradient-boosting classifier
(`sklearn.HistGradientBoostingClassifier`) is faster, more accurate, and fully interpretable
for this shape of problem. It handles the class imbalance (~26% invest) via balanced class
weights and tolerates missing scores natively.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Train

```bash
.venv/bin/python train.py
```

Outputs 5-fold cross-validated metrics, a confusion matrix, feature importances, and writes
`model.joblib`.

## Predict

```bash
# TEAM TECH MARKET VALUEPROP COMPADV SOCIAL FUNDING
.venv/bin/python predict.py 3 2 3 2 4 1 220000
# -> Decision: 1 (INVEST), Invest probability: 91.9%
```

## Current performance (5-fold CV)

| Metric | Value |
|---|---|
| Accuracy | 80.9% (baseline "always pass" = 74.2%) |
| ROC-AUC | 0.871 |
| F1 (invest) | 0.657 |
| Precision / Recall (invest) | 0.611 / 0.711 |

**Most predictive inputs:** Funding amount and Value Proposition, then Competitive Advantage.

## Possible improvements

- Swap in XGBoost / LightGBM (`pip install xgboost`) for a likely small accuracy bump.
- Add a logistic-regression baseline for plain-English per-criterion coefficients.
- Tune the decision threshold (currently 0.5) to trade precision vs. recall.
