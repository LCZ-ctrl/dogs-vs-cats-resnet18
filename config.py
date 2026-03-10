import torch

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 80
LEARNING_RATE = 0.001
NUM_CLASSES = 2  # cat & dog
SEED = 42
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
