import os
import math
import random
import numpy as np
import pandas as pd
import time

from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

from config import *
from feature_extractor import FeatureExtractor
from noise_model import NetworkNoiseModel

start = time.time()

# ---------------------------------------
# Reproducibility
# ---------------------------------------

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------
# Create folders
# ---------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------------------
# Components
# ---------------------------------------

feature_extractor = FeatureExtractor()

network_generator = NetworkNoiseModel()

ideal_simulator = AerSimulator()

dataset = []

# ---------------------------------------
# Fidelity Function
# ---------------------------------------

def output_fidelity(ideal_counts, noisy_counts):

    states = set(ideal_counts.keys()) | set(noisy_counts.keys())

    total_ideal = sum(ideal_counts.values())

    total_noisy = sum(noisy_counts.values())

    fidelity = 0.0

    for state in states:

        p = ideal_counts.get(state,0) / total_ideal

        q = noisy_counts.get(state,0) / total_noisy

        fidelity += math.sqrt(p*q)

    return round(fidelity,6)

# ---------------------------------------
# Dataset Generation
# ---------------------------------------

for sample in range(NUM_SAMPLES):

    num_qubits = random.randint(
        MIN_QUBITS,
        MAX_QUBITS
    )

    depth = random.randint(
        MIN_DEPTH,
        MAX_DEPTH
    )

    qc = random_circuit(
        num_qubits=num_qubits,
        depth=depth,
        max_operands=2,
        measure=True
    )

    features = feature_extractor.extract_features(qc)
    
    if features["cx_count"] == 0:
     continue

    network = network_generator.sample_network()

    noise_model = network_generator.create_noise_model(network)

    transpiled = transpile(
        qc,
        ideal_simulator
    )

    noisy_simulator = AerSimulator(
        noise_model=noise_model
    )

        # -----------------------------
    # Ideal Circuit
    # -----------------------------

    ideal_job = ideal_simulator.run(

        transpiled,

        shots=SHOTS

    )

    ideal_counts = ideal_job.result().get_counts()

    # -----------------------------
    # Noisy Circuit
    # -----------------------------

    noisy_job = noisy_simulator.run(

        transpiled,

        shots=SHOTS

    )

    noisy_counts = noisy_job.result().get_counts()

    fidelity = output_fidelity(

        ideal_counts,

        noisy_counts

    )

    dataset.append({

        **features,

        "profile":network["profile"],

        "bell_fidelity":network["bell_fidelity"],

        "latency":network["latency"],

        "waiting_time":network["waiting_time"],

        "link_loss":network["link_loss"],

        "gate_error":network["gate_error"],

        "readout_error":network["readout_error"],

        "output_fidelity":fidelity

    })

    if (sample+1)%250==0:

        print(f"{sample+1}/{NUM_SAMPLES} completed")

            # -----------------------------
    # Ideal Circuit
    # -----------------------------

    ideal_job = ideal_simulator.run(

        transpiled,

        shots=SHOTS

    )

    ideal_counts = ideal_job.result().get_counts()

    # -----------------------------
    # Noisy Circuit
    # -----------------------------

    noisy_job = noisy_simulator.run(

        transpiled,

        shots=SHOTS

    )

    noisy_counts = noisy_job.result().get_counts()

    fidelity = output_fidelity(

        ideal_counts,

        noisy_counts

    )

    dataset.append({

        **features,

        "profile":network["profile"],

        "bell_fidelity":network["bell_fidelity"],

        "latency":network["latency"],

        "waiting_time":network["waiting_time"],

        "link_loss":network["link_loss"],

        "gate_error":network["gate_error"],

        "readout_error":network["readout_error"],

        "output_fidelity":fidelity

    })

    if (sample+1)%250==0:

        print(f"{sample+1}/{NUM_SAMPLES} completed")

# ---------------------------------------
# Save Dataset
# ---------------------------------------

df = pd.DataFrame(dataset)

df.to_csv(

    "data/dataset.csv",

    index=False

)

print()

print("Dataset Created Successfully")

print(df.head())

print()

print("Saved to data/dataset.csv")

# ---------------------------------------
# Dataset Summary
# ---------------------------------------

with open(

    "results/dataset_summary.txt",

    "w"

) as f:

    f.write("========== DATASET SUMMARY ==========\n\n")

    f.write(f"Samples : {len(df)}\n")

    f.write(f"Average Depth : {df['depth'].mean():.2f}\n")

    f.write(f"Average Gate Count : {df['gate_count'].mean():.2f}\n")

    f.write(f"Average CX Count : {df['cx_count'].mean():.2f}\n")

    f.write(f"Average Remote Gates : {df['remote_gate_count'].mean():.2f}\n")

    f.write(f"Average Bell Fidelity : {df['bell_fidelity'].mean():.4f}\n")

    f.write(f"Average Latency : {df['latency'].mean():.2f}\n")

    f.write(f"Average Waiting Time : {df['waiting_time'].mean():.2f}\n")

    f.write(f"Average Link Loss : {df['link_loss'].mean():.4f}\n")

    f.write(f"Average Output Fidelity : {df['output_fidelity'].mean():.4f}\n")

print()

print("Dataset Summary Saved")

end = time.time()

print(f"\nGeneration Time: {(end-start)/60:.2f} minutes")