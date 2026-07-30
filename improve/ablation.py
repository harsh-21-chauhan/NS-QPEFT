import math
import os
import time
import torch
import torch.nn as nn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from baseline_model import BaselineModel
from ns_qpeft import NSQPEFT

os.makedirs("results", exist_ok=True)

# -----------------------------
# Dataset
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

def evaluate(model):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    start=time.time()

    for epoch in range(100):

        model.train()

        optimizer.zero_grad()

        prediction=model(X_train)

        loss=criterion(
            prediction,
            y_train
        )

        loss.backward()

        optimizer.step()

    training_time=time.time()-start

    model.eval()

    with torch.no_grad():

        prediction=model(X_test)

    prediction=prediction.numpy()

    truth=y_test.numpy()

    mae=mean_absolute_error(
        truth,
        prediction
    )

    rmse=math.sqrt(
        mean_squared_error(
            truth,
            prediction
        )
    )

    r2=r2_score(
        truth,
        prediction
    )

    mape=mean_absolute_percentage_error(
        truth,
        prediction
    )*100

    return [
        mae,
        rmse,
        r2,
        mape,
        training_time
    ]

baseline=BaselineModel()

baseline_result=evaluate(
    baseline
)

nsqpeft=NSQPEFT()

ns_result=evaluate(
    nsqpeft
)

result=pd.DataFrame({

    "Model":[
        "Baseline",
        "NS-QPEFT"
    ],

    "MAE":[
        baseline_result[0],
        ns_result[0]
    ],

    "RMSE":[
        baseline_result[1],
        ns_result[1]
    ],

    "R2":[
        baseline_result[2],
        ns_result[2]
    ],

    "MAPE":[
        baseline_result[3],
        ns_result[3]
    ],

    "Training_Time":[
        baseline_result[4],
        ns_result[4]
    ]

})

print(result)

result.to_csv(
    "results/ablation_results.csv",
    index=False
)