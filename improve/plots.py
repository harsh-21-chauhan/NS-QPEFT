import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ------------------------------------------------
# Training Loss
# ------------------------------------------------

loss = pd.read_csv("results/loss_curve.csv")

plt.figure(figsize=(6.5,4))

plt.plot(
    loss["epoch"],
    loss["train_loss"],
    linewidth=2.2,
    label="Training Loss"
)

plt.plot(
    loss["epoch"],
    loss["validation_loss"],
    linewidth=2.2,
    label="Validation Loss"
)

plt.xlabel("Epoch",fontsize=13)
plt.ylabel("Loss",fontsize=13)

plt.tick_params(labelsize=11)

plt.grid(alpha=0.3)

plt.legend(fontsize=11)

plt.tight_layout()

plt.savefig(
    "results/training_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------
# Prediction Scatter
# ------------------------------------------------

prediction = pd.read_csv(
    "results/predictions.csv"
)

plt.figure(figsize=(5.8,5.2))

plt.scatter(
    prediction["output_fidelity"],
    prediction["Predicted"],
    s=18,
    alpha=0.65
)

plt.plot(
    [0.84,1.01],
    [0.84,1.01],
    "--",
    linewidth=2
)

plt.xlim(0.84,1.01)
plt.ylim(0.84,1.01)

plt.xlabel("Actual Output Fidelity",fontsize=13)
plt.ylabel("Predicted Output Fidelity",fontsize=13)

plt.tick_params(labelsize=11)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/prediction_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------
# Residual Plot
# ------------------------------------------------

plt.figure(figsize=(6,4))

plt.scatter(

    prediction["Predicted"],

    prediction["Absolute_Error"],

    alpha=0.5

)

plt.xlabel("Predicted Fidelity")
plt.ylabel("Absolute Error")
plt.title("Residual Plot")

plt.grid(True)

plt.savefig(

    "results/residual_plot.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

# ------------------------------------------------
# Ablation
# ------------------------------------------------

ablation = pd.read_csv(
    "results/ablation_results.csv"
)

plt.figure(figsize=(4.8,4))

bars = plt.bar(
    ablation["Model"],
    ablation["MAE"],
    width=0.55
)

for bar in bars:

    y = bar.get_height()

    plt.text(
        bar.get_x()+bar.get_width()/2,
        y+0.0006,
        f"{y:.4f}",
        ha="center",
        fontsize=11
    )

plt.ylabel("MAE",fontsize=13)

plt.tick_params(labelsize=11)

plt.grid(axis="y",alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/ablation_mae.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------
# Robustness
# ------------------------------------------------

robust = pd.read_csv(
    "results/robustness_results.csv"
)

plt.figure(figsize=(6,4))

plt.bar(

    robust["Profile"],

    robust["MAE"]

)

plt.ylabel("MAE")

plt.title("Robustness Analysis")

plt.savefig(

    "results/robustness_mae.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("\nAll plots generated successfully.")