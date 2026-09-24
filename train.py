import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data"
MODEL_OUT = "models/best_model.pth"
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(25),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=test_tfms)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_tfms)

    class_names = train_ds.classes
    print("Classes:", class_names)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs("models", exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total else 0
        print(f"Epoch {epoch+1}/{EPOCHS} | loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": class_names}, MODEL_OUT)
            print("✅ Saved best model:", MODEL_OUT)

    # Evaluate on test set (Accuracy + Precision/Recall/F1)
    ckpt = torch.load(MODEL_OUT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    print("\n✅ Classification Report (Precision / Recall / F1):")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("✅ Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
