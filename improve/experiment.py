import os
import math
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

X = df[features].values
y = df["output_fidelity"].values

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X = torch.FloatTensor(X)

# -----------------------------
# Load Model
# -----------------------------

model = NSQPEFT()

model.load_state_dict(
    torch.load("models/ns_qpeft_model.pth")
)

model.eval()

# -----------------------------
# Prediction
# -----------------------------

with torch.no_grad():

    prediction = model(X)

prediction = prediction.numpy().flatten()

# -----------------------------
# Metrics
# -----------------------------

mae = mean_absolute_error(y, prediction)

rmse = math.sqrt(
    mean_squared_error(y, prediction)
)

r2 = r2_score(y, prediction)

mape = mean_absolute_percentage_error(
    y,
    prediction
) * 100

print("\n========== RESULTS ==========")

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")
print(f"MAPE : {mape:.2f}%")

# -----------------------------
# Save Predictions
# -----------------------------

result = df.copy()

result["Predicted"] = prediction

result["Absolute_Error"] = abs(
    result["output_fidelity"] -
    result["Predicted"]
)

result.to_csv(
    "results/predictions.csv",
    index=False
)

print("\nPredictions saved.")

# -----------------------------
# Save Metrics
# -----------------------------

with open(
    "results/experiment_metrics.txt",
    "w"
) as f:

    f.write(f"MAE  : {mae:.6f}\n")
    f.write(f"RMSE : {rmse:.6f}\n")
    f.write(f"R2   : {r2:.6f}\n")
    f.write(f"MAPE : {mape:.2f}%\n")

print("Experiment completed successfully.")