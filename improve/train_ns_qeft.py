import copy
import pandas as pd
import torch
import torch.nn as nn
import os
import math
import time
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from ns_qpeft import NSQPEFT

# -------------------------------------------------
# Configuration
# -------------------------------------------------

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
PATIENCE = 20

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

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
y = df["output_fidelity"].values.reshape(-1,1)

# -------------------------------------------------
# Normalize
# -------------------------------------------------

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# -------------------------------------------------
# Split Dataset
# -------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

# -------------------------------------------------
# Tensor Conversion
# -------------------------------------------------

X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)

y_train = torch.FloatTensor(y_train)
y_val = torch.FloatTensor(y_val)
y_test = torch.FloatTensor(y_test)

# -------------------------------------------------
# DataLoader
# -------------------------------------------------

train_loader = DataLoader(
    TensorDataset(X_train,y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(X_val,y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------------------------------
# Model
# -------------------------------------------------

model = NSQPEFT()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

# -------------------------------------------------
# Training
# -------------------------------------------------

best_loss = float("inf")
best_model = None
counter = 0

train_history = []
val_history = []
history = []

start_time = time.time()

for epoch in range(EPOCHS):

    # ------------------ Training ------------------

    model.train()

    train_loss = 0

    for x_batch,y_batch in train_loader:

        optimizer.zero_grad()

        prediction = model(x_batch)

        loss = criterion(
            prediction,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------- Validation ------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for x_batch,y_batch in val_loader:

            prediction = model(x_batch)

            loss = criterion(
                prediction,
                y_batch
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    train_history.append(train_loss)
    val_history.append(val_loss)

    history.append({
    "epoch": epoch + 1,
    "train_loss": train_loss,
    "validation_loss": val_loss,
    "learning_rate": optimizer.param_groups[0]["lr"]
    })

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train {train_loss:.6f} | "
        f"Val {val_loss:.6f}"
    )

    # --------------- Early Stopping ----------------

    if val_loss < best_loss:

        best_loss = val_loss

        best_model = copy.deepcopy(
            model.state_dict()
        )

        counter = 0

    else:

        counter += 1

        if counter >= PATIENCE:

            print("\nEarly Stopping")

            break

# -------------------------------------------------
# Load Best Model
# -------------------------------------------------

model.load_state_dict(best_model)



os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

training_time = time.time() - start_time

torch.save(
    model.state_dict(),
    "models/ns_qpeft_model.pth"
)

print("\nBest Model Saved")

# -------------------------------------------------
# Testing
# -------------------------------------------------

model.eval()

with torch.no_grad():

    prediction = model(X_test)

prediction = prediction.numpy()
truth = y_test.numpy()

# -------------------------------------------------
# Metrics
# -------------------------------------------------

mae = mean_absolute_error(
    truth,
    prediction
)

rmse = math.sqrt(
    mean_squared_error(
        truth,
        prediction
    )
)

r2 = r2_score(
    truth,
    prediction
)

mape = mean_absolute_percentage_error(
    truth,
    prediction
) * 100

print("\n----------------------------")
print(f"Test MAE   : {mae:.6f}")
print(f"Test RMSE  : {rmse:.6f}")
print(f"Test R²    : {r2:.6f}")
print(f"Test MAPE  : {mape:.2f}%")
print(f"Train Time : {training_time:.2f} sec")
print("----------------------------")

# -------------------------------------------------
# Save Training Log
# -------------------------------------------------

history_df = pd.DataFrame(history)

history_df.to_csv(
    "results/training_log.csv",
    index=False
)

# -------------------------------------------------
# Save Loss Curve
# -------------------------------------------------

loss_df = pd.DataFrame({
    "epoch": range(1, len(train_history) + 1),
    "train_loss": train_history,
    "validation_loss": val_history
})

loss_df.to_csv(
    "results/loss_curve.csv",
    index=False
)

# -------------------------------------------------
# Save Metrics (TXT)
# -------------------------------------------------

with open(
    "results/training_metrics.txt",
    "w"
) as f:

    f.write(f"Best Validation Loss : {best_loss:.6f}\n")
    f.write(f"Test MAE             : {mae:.6f}\n")
    f.write(f"Test RMSE            : {rmse:.6f}\n")
    f.write(f"Test R2              : {r2:.6f}\n")
    f.write(f"Test MAPE            : {mape:.2f}%\n")
    f.write(f"Training Time (sec)  : {training_time:.2f}\n")

# -------------------------------------------------
# Save Metrics (CSV)
# -------------------------------------------------

metrics_df = pd.DataFrame({

    "Metric": [

        "Best Validation Loss",
        "MAE",
        "RMSE",
        "R2",
        "MAPE",
        "Training Time (sec)"

    ],

    "Value": [

        best_loss,
        mae,
        rmse,
        r2,
        mape,
        training_time

    ]

})

metrics_df.to_csv(
    "results/final_metrics.csv",
    index=False
)

print("\nTraining completed successfully.")
