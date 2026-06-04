import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SMRNet
from dataset import MRIDataset
from utils import calculate_metrics


device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = MRIDataset("splits/adni_train.csv")
val_dataset = MRIDataset("splits/adni_val.csv")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = SMRNet(num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

best_acc = 0.0

for epoch in range(100):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs, _ = model(images)
            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc, sen, spe, f1 = calculate_metrics(targets, preds)

    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, ACC={acc:.4f}, SEN={sen:.4f}, SPE={spe:.4f}, F1={f1:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_smrnet.pth")
