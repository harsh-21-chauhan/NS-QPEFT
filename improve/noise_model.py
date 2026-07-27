import random

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
    thermal_relaxation_error
)


class NetworkNoiseModel:

    def __init__(self):

        self.profiles = {

            "Excellent": {

                "latency": (5,10),
                "link_loss": (0.00,0.01),
                "readout_error": (0.001,0.003)

            },

            "Good": {

                "latency": (10,20),
                "link_loss": (0.01,0.03),
                "readout_error": (0.003,0.007)

            },

            "Average": {

                "latency": (20,50),
                "link_loss": (0.03,0.07),
                "readout_error": (0.007,0.015)

            },

            "Poor": {

                "latency": (50,100),
                "link_loss": (0.07,0.15),
                "readout_error": (0.015,0.03)

            }

        }

    # ----------------------------------

    def sample_network(self):

        profile = random.choice(
            list(self.profiles.keys())
        )

        p = self.profiles[profile]

        latency = random.uniform(
            *p["latency"]
        )

        link_loss = random.uniform(
            *p["link_loss"]
        )

        # Link Loss -> Bell Fidelity

        bell_fidelity = max(
            0.80,
            1 - link_loss + random.uniform(-0.005,0.005)
        )

        # Bell Fidelity -> Gate Error

        gate_error = max(
            0.001,
            (1-bell_fidelity)*0.25
        )

        # Latency -> Waiting Time

        waiting_time = latency*0.5

        readout_error = random.uniform(
            *p["readout_error"]
        )

        return {

            "profile":profile,

            "bell_fidelity":round(
                bell_fidelity,4
            ),

            "latency":round(
                latency,2
            ),

            "waiting_time":round(
                waiting_time,2
            ),

            "link_loss":round(
                link_loss,4
            ),

            "gate_error":gate_error,

            "readout_error":readout_error

        }

    # ----------------------------------

    def create_noise_model(self,params):

        noise_model = NoiseModel()

        # Depolarizing Error

        dep_error = depolarizing_error(
            params["gate_error"],
            2
        )

        noise_model.add_all_qubit_quantum_error(
            dep_error,
            ["cx"]
        )

        # Thermal Relaxation

        T1 = 100

        T2 = 80

        thermal_error = thermal_relaxation_error(

            T1,

            T2,

            params["waiting_time"]

        )

        noise_model.add_all_qubit_quantum_error(

            thermal_error,

            ["id","x","sx"]

        )

        # Readout Error

        p = params["readout_error"]

        readout = ReadoutError([

            [1-p,p],

            [p,1-p]

        ])

        noise_model.add_all_qubit_readout_error(
            readout
        )

        return noise_model