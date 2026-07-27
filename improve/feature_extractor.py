from qiskit import QuantumCircuit


class FeatureExtractor:

    def extract_features(self, qc: QuantumCircuit):

        depth = qc.depth()

        gate_count = qc.size()

        num_qubits = qc.num_qubits

        operations = qc.count_ops()

        cx_count = operations.get("cx", 0)

        # -------------------------
        # Dynamic QPU Mapping
        # -------------------------

        midpoint = num_qubits // 2

        qubit_location = {}

        for i in range(num_qubits):

            if i < midpoint:
                qubit_location[i] = "A"
            else:
                qubit_location[i] = "B"

        # -------------------------
        # Count Remote Gates
        # -------------------------

        remote_gate_count = 0

        for instruction in qc.data:

            if instruction.operation.name != "cx":
                continue

            control = instruction.qubits[0]._index
            target = instruction.qubits[1]._index

            if qubit_location[control] != qubit_location[target]:

                remote_gate_count += 1

        teleportation_count = remote_gate_count

        return {

            "depth": depth,

            "gate_count": gate_count,

            "num_qubits": num_qubits,

            "cx_count": cx_count,

            "remote_gate_count": remote_gate_count,

            "teleportation_count": teleportation_count

        }