import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import depolarizing_error
from qiskit_aer import AerSimulator

#basic Quantum circuit 

qc = QuantumCircuit(4,4)

qc.h(0)
qc.cx(0,1)
qc.cx(0,2)
qc.cx(2,3)
qc.measure_all()

# print(qc)

# depth means it contains 5 layers in the whole operation
print(qc.depth())

#gate count - size gives the no of operations will be done
print(qc.size())

# no of qubits in this - 4
print(qc.num_qubits)

# count every gate
print(qc.count_ops())

qubit_location = {
    0: "A",
    1: "A",
    2: "B",
    3: "B"
}



# detect remote cx

remote_gate_count = 0

for instruction in qc.data:

    if instruction.operation.name == "cx":

        control = instruction.qubits[0]._index
        target = instruction.qubits[1]._index

        if qubit_location[control] != qubit_location[target]:
            remote_gate_count += 1

print(remote_gate_count)


features = {

"depth": qc.depth(),

"gate_count": qc.size(),

"remote_gate_count": remote_gate_count,

"cx_count": qc.count_ops().get("cx",0),

"qubits": qc.num_qubits
}

print(features)





#simulator details
simulator = AerSimulator()


# Noise Modelling in this 

noise = NoiseModel()

error = depolarizing_error(0.02,2)

noise.add_all_qubit_quantum_error(
    error,
    ["cx"]
)

job1 = simulator.run(qc, shots=1000)

job = simulator.run(
    qc,
    noise_model=noise,
    shots=1000
)


res = job.result()
res1 = job1.result()


print(res.get_counts())
print(res1.get_counts())