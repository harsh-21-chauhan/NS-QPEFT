import pandas as pd

df = pd.read_csv("results/predictions.csv")

profiles = df.groupby("profile")

for name, group in profiles:

    mae = abs(
        group["output_fidelity"] -
        group["Predicted"]
    ).mean()

    print(name)
    print("MAE:", round(mae,4))
    print()