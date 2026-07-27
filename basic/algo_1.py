import pandas as pd

import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from ns_qpeft import NSQPEFT
import torch
import torch.nn as nn

df = pd.read_csv("dataset.csv")

# print(df.head())

features = [
    "depth",
    "gate_count",
    "cx_count",
    "remote_gate_count",
    "teleportation_count",
    "bell_fidelity",
    "latency",
    "waiting_time",
    "link_loss"
]

X = df[features]
y = df["output_fidelity"]


scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=features
)

# print(X_scaled.head())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)




# Fake Model we will replace this with

model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=1000,
    random_state=42
)




# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

joblib.dump(model, "algorithm1_model.pkl")

print("Model saved successfully.")

sample = X_scaled.iloc[[0]]

prediction = model.predict(sample)

print("Predicted Output Fidelity:", prediction[0])
print("Actual Output Fidelity:", y.iloc[0])

model = NSQPEFT()
dummy = torch.randn(4, 9)
output = model(dummy)
print(output)
