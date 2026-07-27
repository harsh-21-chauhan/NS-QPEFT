import pandas as pd
import torch
import numpy as np

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

with torch.no_grad():

    prediction = model(X)

prediction = prediction.numpy().flatten()

# -----------------------------
# Metrics
# -----------------------------

mae = mean_absolute_error(y, prediction)

rmse = np.sqrt(
    mean_squared_error(y, prediction)
)

r2 = r2_score(y, prediction)

print()

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# -----------------------------
# Save Predictions
# -----------------------------

results = df.copy()

results["Predicted"] = prediction

results.to_csv(
    "predictions.csv",
    index=False
)

print("\nPredictions saved.")