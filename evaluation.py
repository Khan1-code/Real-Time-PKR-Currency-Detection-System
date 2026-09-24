print("EVALUATION STARTED")

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===== SETTINGS =====
DATA_DIR = "data"
CKPT_PATH = os.path.join("models", "best_model.pth")
BATCH_SIZE = 32
# ====================

def build_model(num_classes: int):
    # Must match your app.py / train.py architecture (mobilenet_v2 + replaced classifier)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # IMPORTANT: Use SAME normalization as train.py/app.py
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Dataset classes (from folder names):", test_dataset.classes)

    # Load checkpoint dict saved by train.py
    ckpt = torch.load(CKPT_PATH, map_location=device)
    classes = ckpt["classes"]              # class order used during training
    state_dict = ckpt["model_state"]

    print("Checkpoint classes:", classes)

    # Build model and load weights
    model = build_model(num_classes=len(classes))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Evaluate
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Report
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    print("\n✅ Classification Report (Precision / Recall / F1):")
    print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

    print("✅ Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
