# -------------------------
# Dataset
# -------------------------

NUM_SAMPLES = 8000

SHOTS = 1024

# -------------------------
# Circuit
# -------------------------

MIN_QUBITS = 4
MAX_QUBITS = 8

MIN_DEPTH = 5
MAX_DEPTH = 20

# -------------------------
# Training
# -------------------------

BATCH_SIZE = 64

EPOCHS = 200

LEARNING_RATE = 1e-3

PATIENCE = 20

# -------------------------
# Dataset Split
# -------------------------

TRAIN_RATIO = 0.80

VALID_RATIO = 0.10

TEST_RATIO = 0.10

# -------------------------
# Random Seed
# -------------------------

SEED = 42