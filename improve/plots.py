import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/predictions.csv")

# Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(df["output_fidelity"], df["Predicted"], alpha=0.6)
plt.xlabel("Actual Fidelity")
plt.ylabel("Predicted Fidelity")
plt.title("Predicted vs Actual")
plt.grid(True)
plt.savefig("results/prediction_scatter.png")
plt.close()

# Residual Plot
residual = df["output_fidelity"] - df["Predicted"]

plt.figure(figsize=(6,4))
plt.scatter(df["Predicted"], residual, alpha=0.6)
plt.axhline(0,color='r')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True)
plt.savefig("results/residual_plot.png")
plt.close()

print("Plots saved.")