import torch
import random
import numpy as np
import os
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def save_model(model, path: Path):
    torch.save(model.state_dict(), path)


def calculate_accuracy(outputs, labels):
    score, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def calculate_correct(outputs, labels):
    return (outputs.argmax(1) == labels).sum().item()
