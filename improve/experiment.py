import time
import torch
import pandas as pd
import numpy as np

import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from ns_qpeft import NSQPEFT

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

start = time.time()

with torch.no_grad():
    prediction = model(X)

end = time.time()

prediction = prediction.numpy().flatten()

# -----------------------------
# Metrics
# -----------------------------

mae = mean_absolute_error(y, prediction)

rmse = np.sqrt(
    mean_squared_error(y, prediction)
)

r2 = r2_score(y, prediction)

print("\n========== RESULTS ==========")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")
print(f"Inference Time : {end-start:.6f} sec")

# -----------------------------
# Save Predictions
# -----------------------------

result = df.copy()

result["Predicted"] = prediction

result["Error"] = abs(
    result["output_fidelity"] -
    result["Predicted"]
)


os.makedirs("results", exist_ok=True)

result.to_csv(
    "results/predictions.csv",
    index=False
)

print("\npredictions.csv generated successfully.")