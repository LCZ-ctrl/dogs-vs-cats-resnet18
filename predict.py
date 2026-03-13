import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from PIL import Image
from config import *
from model import ResNet18
from dataset import get_test_loader


def predict_random_one():
    model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint_path = "models/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = get_test_loader()
    test_dataset = test_loader.dataset

    idx = random.randint(0, len(test_dataset) - 1)
    img_tensor, img_id = test_dataset[idx]

    filename = test_dataset.imgs[idx]
    img_path = test_dataset.test_path / filename
    original_pil = Image.open(img_path).convert('RGB')

    with torch.no_grad():
        inputs = img_tensor.unsqueeze(0).to(DEVICE)
        outputs = model(inputs)

        probs = F.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

        text = 'cat' if pred.item() == 0 else 'dog'
        conf_percent = conf.item() * 100

    print(f"Image ID: {img_id}")
    print(f"Prediction: {text}")
    print(f"Confidence: {conf_percent:.2f}%")
    print(f"Probability: [Cat: {probs[0] * 100:.2f}% | Dog: {probs[1] * 100:.2f}%]")

    plt.imshow(original_pil)
    plt.title(f"ID: {img_id} | Prediction: {text} ({conf_percent:.1f}%)")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    predict_random_one()
