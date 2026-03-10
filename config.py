import torch

IMG_SIZE = 224
BATCH_SIZE = 64

# train from scratch
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3

# transfer learning
# NUM_EPOCHS = 10
# LEARNING_RATE = 1e-4

NUM_CLASSES = 2  # cat & dog
SEED = 42
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
