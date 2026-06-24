"""Predict an invest (1) / pass (0) decision from the 7 inputs.

Usage:
    .venv/bin/python predict.py TEAM TECH MARKET VALUEPROP COMPADV SOCIAL FUNDING

Example (Stellerian, funding 220000):
    .venv/bin/python predict.py 3 2 3 2 4 1 220000
"""

import sys
import pandas as pd
import joblib

MODEL_FILE = "model.joblib"


def predict(team, technology, market, value_prop, comp_adv, social, funding):
    bundle = joblib.load(MODEL_FILE)
    model, features = bundle["model"], bundle["features"]
    row = pd.DataFrame(
        [[team, technology, market, value_prop, comp_adv, social, funding]],
        columns=features,
    )
    proba = float(model.predict_proba(row)[0, 1])
    return int(proba >= 0.5), proba


def main():
    args = sys.argv[1:]
    if len(args) != 7:
        print(__doc__)
        sys.exit(1)
    vals = [float(a) for a in args]
    decision, proba = predict(*vals)
    label = "INVEST" if decision == 1 else "PASS"
    print(f"Decision: {decision} ({label})")
    print(f"Invest probability: {proba:.1%}")


if __name__ == "__main__":
    main()
