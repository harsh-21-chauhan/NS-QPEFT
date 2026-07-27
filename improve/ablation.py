import copy
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

from torch.utils.data import DataLoader,TensorDataset

from baseline_model import BaselineModel
from ns_qpeft import NSQPEFT

# -------------------------

BATCH_SIZE = 64
EPOCHS = 100

# -------------------------

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

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)

y_train=torch.FloatTensor(y_train)
y_test=torch.FloatTensor(y_test)

train_loader=DataLoader(
    TensorDataset(X_train,y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)



def train_model(model):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    for epoch in range(EPOCHS):

        model.train()

        for x_batch,y_batch in train_loader:

            optimizer.zero_grad()

            prediction=model(x_batch)

            loss=criterion(
                prediction,
                y_batch
            )

            loss.backward()

            optimizer.step()

    model.eval()

    with torch.no_grad():

        prediction=model(X_test)

    prediction=prediction.numpy()

    mae=mean_absolute_error(
        y_test,
        prediction
    )

    r2=r2_score(
        y_test,
        prediction
    )

    return mae,r2


baseline = BaselineModel()

baseline_mae,baseline_r2 = train_model(
    baseline
)

nsqpeft = NSQPEFT()

ns_mae,ns_r2 = train_model(
    nsqpeft
)

print("\n-----------")

print("Baseline")

print("MAE :",baseline_mae)
print("R2  :",baseline_r2)

print("\nNS-QPEFT")

print("MAE :",ns_mae)
print("R2  :",ns_r2)