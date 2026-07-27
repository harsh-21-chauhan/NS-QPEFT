from noise_model import NetworkNoiseModel

generator = NetworkNoiseModel()

params = generator.sample_network()

noise = generator.create_noise_model(params)

print(params)

print(noise)