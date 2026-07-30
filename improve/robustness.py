import math
import os
import torch
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from ns_qpeft import NSQPEFT

os.makedirs("results", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/dataset.csv")

features = [
    "depth",
    "gate_count",
    "num_qubits",
    "cx_count",
    "remote_gate_count",
    "teleportation_count",
    "bell_fidelity",
    "latency",
    "waiting_time",
    "link_loss",
    "gate_error",
    "readout_error"
]

model = NSQPEFT()

model.load_state_dict(
    torch.load("models/ns_qpeft_model.pth")
)

model.eval()

results = []

# -----------------------------
# Evaluate Each Profile
# -----------------------------

for profile in df["profile"].unique():

    subset = df[df["profile"] == profile]

    X = subset[features].values
    y = subset["output_fidelity"].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X = torch.FloatTensor(X)

    with torch.no_grad():
        prediction = model(X)

    prediction = prediction.numpy().flatten()

    mae = mean_absolute_error(y, prediction)

    rmse = math.sqrt(
        mean_squared_error(y, prediction)
    )

    r2 = r2_score(y, prediction)

    mape = mean_absolute_percentage_error(
        y,
        prediction
    ) * 100

    results.append({
        "Profile": profile,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    })

result_df = pd.DataFrame(results)

print(result_df)

result_df.to_csv(
    "results/robustness_results.csv",
    index=False
)

print("\nRobustness analysis completed.")