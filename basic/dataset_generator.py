import random
import pandas as pd

from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import transpile

simulator = AerSimulator()

dataset = []

noise_model = NoiseModel()

cx_error = depolarizing_error(0.02, 2)
noise_model.add_all_qubit_quantum_error(cx_error, ["cx"])


def output_fidelity(ideal_counts, noisy_counts):
    total = sum(ideal_counts.values())
    fidelity = 0

    for state in set(ideal_counts) | set(noisy_counts):
        p = ideal_counts.get(state, 0) / total
        q = noisy_counts.get(state, 0) / total
        fidelity += min(p, q)

    return round(fidelity, 4)

for i in range(100):

    qc = random_circuit(
        num_qubits=4,
        depth=random.randint(4, 10),
        measure=True
    )

    depth = qc.depth()
    gate_count = qc.size()
    cx_count = qc.count_ops().get("cx", 0)

    # QPU Mapping
    qubit_location = {
        0: "A",
        1: "A",
        2: "B",
        3: "B"
    }

    remote_gate_count = 0

    for instruction in qc.data:
        if instruction.operation.name == "cx":
            control = instruction.qubits[0]._index
            target = instruction.qubits[1]._index

            if qubit_location[control] != qubit_location[target]:
                remote_gate_count += 1

    bell_fidelity = round(random.uniform(0.85, 1.00), 3)
    latency = round(random.uniform(5, 100), 2)
    waiting_time = round(random.uniform(5, 50), 2)
    link_loss = round(random.uniform(0.00, 0.20), 3)
    teleportation_count = remote_gate_count

    qc_transpiled = transpile(qc, simulator)
    ideal_job = simulator.run(qc_transpiled, shots=1024)
    ideal_counts = ideal_job.result().get_counts()

    noisy_job = simulator.run(
    qc_transpiled,
    shots=1024,
    noise_model=noise_model
)
    noisy_counts = noisy_job.result().get_counts()

    fidelity = output_fidelity(ideal_counts, noisy_counts)

    dataset.append({
        "depth": depth,
        "gate_count": gate_count,
        "cx_count": cx_count,
        "remote_gate_count": remote_gate_count,
        "teleportation_count": teleportation_count,
        "bell_fidelity": bell_fidelity,
        "latency": latency,
        "waiting_time": waiting_time,
        "link_loss": link_loss,
        "ideal_counts": str(ideal_counts),
        "noisy_counts": str(noisy_counts),
        "output_fidelity": fidelity
    })

df = pd.DataFrame(dataset)

df.to_csv("dataset.csv", index=False)

print(df.head())
print("\nDataset saved as dataset.csv")