from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator

# Create simulator
simulator = AerSimulator()

# Generate random circuit
qc = random_circuit(
    num_qubits=4,
    depth=6,
    measure=True
)

# Extract features
depth = qc.depth()
size = qc.size()
num_qubits = qc.num_qubits
ops = qc.count_ops()

# Run circuit
job = simulator.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()

# Print
print(qc)
print("\nDepth:", depth)
print("Size:", size)
print("Qubits:", num_qubits)
print("Operations:", ops)
print("Counts:", counts)