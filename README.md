# VC Investment Decision Classifier

Predicts whether a VC would **invest (1)** or **pass (0)** on a startup, from 7 inputs:
six criteria scored 1–5 — Team, Technology, Market, Value Proposition, Competitive Advantage,
Socially Impactful — plus the Funding amount.

It learns from ~765 hand-evaluated startups in [data/datafile.csv](file:///Users/williamragnarsson/Programming/vc-ai-analyst-llm-training/data/datafile.csv). The two human evaluators
(William, Wout) agreed 99% of the time, so their decisions are combined into one label
(conflicting rows are dropped).

## Why gradient boosting (not an LLM)

The data is small and tabular (7 numeric inputs, ~765 rows). A gradient-boosting classifier
(`sklearn.HistGradientBoostingClassifier`) is faster, more accurate, and fully interpretable
for this shape of problem. It handles the class imbalance (~26% invest) via balanced class
weights and tolerates missing scores natively.

---

## Directory Structure

```text
vc-ai-analyst-llm-training/
├── data/
│   └── datafile.csv               # Raw startup evaluations
├── models/
│   ├── model.joblib               # Trained Python classifier bundle
│   └── model.onnx                 # Exported ONNX model for Node.js
├── python/
│   ├── requirements.txt           # Python packages
│   ├── train.py                   # Train step script
│   ├── test_python.py             # CLI prediction tool (Python)
│   └── export_onnx.py             # ONNX format exporter script
├── node/
│   ├── package.json               # Node.js configuration
│   ├── package-lock.json
│   └── test_onnx.js               # CLI prediction tool (Node.js)
└── README.md                      # System documentation
```

---

## Setup

### Python Environment Setup
```bash
python -m venv .venv
.venv/bin/pip install -r python/requirements.txt
```

### Node.js Environment Setup
```bash
cd node
npm install
cd ..
```

---

## Train

```bash
.venv/bin/python python/train.py
```

Outputs 5-fold cross-validated metrics, a confusion matrix, feature importances, and writes [models/model.joblib](file:///Users/williamragnarsson/Programming/vc-ai-analyst-llm-training/models/model.joblib).

---

## Predict / Inference

### 1. Python Inference
Use the [python/test_python.py](file:///Users/williamragnarsson/Programming/vc-ai-analyst-llm-training/python/test_python.py) CLI helper to query predictions with the scikit-learn model:
```bash
# TEAM TECH MARKET VALUEPROP COMPADV SOCIAL FUNDING
.venv/bin/python python/test_python.py 3 2 3 2 4 1 220000
# -> Decision: 1 (INVEST), Invest probability: 91.9%
```

### 2. Export to ONNX (for Node.js / Vercel / Next.js)
`model.joblib` only loads in Python. To run the model in a JavaScript runtime with no Python, export it to the language-neutral ONNX format:

```bash
.venv/bin/python python/export_onnx.py   # writes models/model.onnx
```

### 3. JavaScript / Node.js Inference
Load the exported ONNX model inside Node (e.g. in a Next.js route handler) with `onnxruntime-node`. 

Run the test script to verify:
```bash
node node/test_onnx.js
```

#### Tensor Contract:
- **Input** `float_input` — float32, shape `[1, 7]`, in this exact column order:
  `Team, Technology, Market, Value Proposition, Competitive Advantage, Socially Impactful, funding`
- **Outputs** `label` (int64 `[1]`, 0 = pass / 1 = invest) and
  `probabilities` (float32 `[1, 2]`, index `[1]` = invest probability)

Verified to produce identical predictions to the sklearn model (Stellerian → 1 @ 0.919, DroneSpotter → 0 @ 0.000).

---

## Current performance (5-fold CV)

| Metric | Value |
|---|---|
| Accuracy | 80.9% (baseline "always pass" = 74.2%) |
| ROC-AUC | 0.871 |
| F1 (invest) | 0.657 |
| Precision / Recall (invest) | 0.611 / 0.711 |

**Most predictive inputs:** Funding amount and Value Proposition, then Competitive Advantage.

---

## Possible improvements

- Swap in XGBoost / LightGBM (`pip install xgboost`) for a likely small accuracy bump.
- Add a logistic-regression baseline for plain-English per-criterion coefficients.
- Tune the decision threshold (currently 0.5) to trade precision vs. recall.
