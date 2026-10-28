
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/openf1/within_stint_pace_loss.csv")
OUTPUT = Path("data/openf1/race_holdout_validation.csv")

df = pd.read_csv(INPUT)

required = [
    "Country",
    "SessionKey",
    "DriverNumber",
    "StintNumber",
    "Compound",
    "TyreLife",
    "MedianPaceLossSeconds",
]

df = df.dropna(subset=required).copy()
df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")
df["MedianPaceLossSeconds"] = pd.to_numeric(
    df["MedianPaceLossSeconds"], errors="coerce"
)
df = df.dropna(subset=["TyreLife", "MedianPaceLossSeconds"])

stint_keys = [
    "Country", "SessionKey", "DriverNumber", "StintNumber", "Compound"
]

# Keep each stint-age observation unique.
df = df.drop_duplicates(subset=stint_keys + ["TyreLife"])

races = sorted(df["Country"].unique())
if len(races) < 3:
    raise RuntimeError("Need at least three races for holdout validation.")

results = []

for held_out in races:
    train = df[df["Country"] != held_out]
    test = df[df["Country"] == held_out]

    for compound in sorted(test["Compound"].unique()):
        train_c = train[train["Compound"] == compound]
        test_c = test[test["Compound"] == compound]

        if train_c.empty or test_c.empty:
            continue

        # Estimate the median observed pace loss at each tyre age
        # using training races only.
        curve = train_c.groupby("TyreLife")[
            "MedianPaceLossSeconds"
        ].median()

        # Baseline: predict zero pace change at every tyre age.
        # This is the reference model, not a physical tyre model.
        predictions = []
        actuals = []

        for _, row in test_c.iterrows():
            age = row["TyreLife"]
            actual = row["MedianPaceLossSeconds"]

            # Only evaluate ages represented in the training races.
            if age not in curve.index:
                continue

            predictions.append(float(curve.loc[age]))
            actuals.append(float(actual))

        if not actuals:
            continue

        actuals = np.asarray(actuals)
        predictions = np.asarray(predictions)

        zero_predictions = np.zeros_like(actuals)

        results.append({
            "HeldOutRace": held_out,
            "Compound": compound,
            "EvaluatedObservations": len(actuals),
            "Curve_MAE": float(np.mean(np.abs(actuals - predictions))),
            "ZeroBaseline_MAE": float(
                np.mean(np.abs(actuals - zero_predictions))
            ),
            "Curve_Bias": float(np.mean(predictions - actuals)),
        })

out = pd.DataFrame(results)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)

print("Races:", races)
print("Saved:", OUTPUT)

if out.empty:
    print("No comparable compound/age groups across train and test races.")
else:
    print("\nPer-race holdout results:")
    print(out.round(3).to_string(index=False))

    print("\nAverage error by compound:")
    print(
        out.groupby("Compound")[
            ["Curve_MAE", "ZeroBaseline_MAE", "Curve_Bias"]
        ]
        .mean()
        .round(3)
        .to_string()
    )

    print(
        "\nInterpretation: a lower Curve_MAE than ZeroBaseline_MAE "
        "means the historical curve predicts observed pace loss better "
        "than predicting zero. It does not establish causal tyre degradation."
    )
