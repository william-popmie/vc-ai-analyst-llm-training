"""Train a VC investment-decision classifier.

Inputs (per startup): six 1-5 criteria scores + funding amount.
Output: 1 (invest) / 0 (don't invest), predicting how the VC evaluators would decide.

Run:  .venv/bin/python train.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
import joblib

DATA_FILE = "datafile.csv"
MODEL_FILE = "model.joblib"

CRITERIA = [
    "Team",
    "Technology",
    "Market",
    "Value Proposition",
    "Competitive Advantage",
    "Socially Impactful",
]
FEATURES = CRITERIA + ["funding"]


def parse_funding(value):
    """'$220.000' -> 220000.0 (European thousands separator). Blank -> NaN."""
    if not isinstance(value, str) or not value.strip():
        return np.nan
    cleaned = value.replace("$", "").replace(".", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def to_label(value):
    v = str(value).strip().lower()
    if v == "yes":
        return 1
    if v == "no":
        return 0
    return np.nan


def load_data():
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

    will = df["Invest William"].map(to_label)
    wout = df["Invest Wout"].map(to_label)

    # Combined label: agree -> that value; only one present -> use it;
    # conflict or both missing -> drop.
    label = pd.Series(np.nan, index=df.index)
    both = will.notna() & wout.notna()
    label[both & (will == wout)] = will[both & (will == wout)]
    label[will.notna() & wout.isna()] = will[will.notna() & wout.isna()]
    label[wout.notna() & will.isna()] = wout[wout.notna() & will.isna()]

    conflicts = int((both & (will != wout)).sum())

    X = pd.DataFrame(index=df.index)
    for c in CRITERIA:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["funding"] = df["Funding amount"].map(parse_funding)

    mask = label.notna()
    X, y = X[mask].reset_index(drop=True), label[mask].astype(int).reset_index(drop=True)
    return X, y, conflicts


def main():
    X, y, conflicts = load_data()
    pos_rate = y.mean()
    print(f"Usable labeled rows: {len(y)}  (dropped {conflicts} rater conflicts)")
    print(f"Class balance: {int(y.sum())} invest / {int((1 - y).sum())} pass "
          f"({pos_rate:.1%} positive)")
    print(f"Baseline (always 'pass') accuracy: {1 - pos_rate:.1%}\n")

    model = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=42, max_iter=300, learning_rate=0.05
    )

    # Stratified 5-fold cross-validated metrics (stable estimate on small data).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("=== 5-fold cross-validated performance ===")
    print(classification_report(y, pred, target_names=["pass (0)", "invest (1)"], digits=3))
    print(f"ROC-AUC: {roc_auc_score(y, proba):.3f}")
    print(f"F1 (invest): {f1_score(y, pred):.3f}")
    print("Confusion matrix (rows=true, cols=pred) [pass, invest]:")
    print(confusion_matrix(y, pred), "\n")

    # Feature importance via permutation on a held-out split.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    fitted = model.fit(X_tr, y_tr)
    perm = permutation_importance(
        fitted, X_te, y_te, scoring="f1", n_repeats=20, random_state=42
    )
    print("=== Feature importance (permutation, drop in F1) ===")
    order = np.argsort(perm.importances_mean)[::-1]
    for i in order:
        print(f"  {FEATURES[i]:<22} {perm.importances_mean[i]:+.3f} "
              f"(± {perm.importances_std[i]:.3f})")

    # Fit final model on ALL data and save.
    final = model.fit(X, y)
    joblib.dump({"model": final, "features": FEATURES}, MODEL_FILE)
    print(f"\nSaved trained model -> {MODEL_FILE}")


if __name__ == "__main__":
    main()
