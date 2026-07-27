import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ns_qpeft import NSQPEFT

# Load dataset
df = pd.read_csv("dataset.csv")


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

X = df[features].values
y = df["output_fidelity"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Model
model = NSQPEFT()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 200

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

# Evaluation
model.eval()

with torch.no_grad():

    predictions = model(X_test)

    mse = criterion(predictions, y_test)

    mae = torch.mean(torch.abs(predictions - y_test))

print("\nTest MSE:", mse.item())
print("Test MAE:", mae.item())

# Save model
torch.save(model.state_dict(), "ns_qpeft_model.pth")

print("Model saved successfully.")