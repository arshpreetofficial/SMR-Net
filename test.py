import torch
from torch.utils.data import DataLoader
from model import SMRNet
from dataset import MRIDataset
from utils import calculate_metrics


device = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset = MRIDataset("splits/oasis_test.csv")
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = SMRNet(num_classes=3).to(device)
model.load_state_dict(torch.load("best_smrnet.pth", map_location=device))
model.eval()

preds, targets = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs, attention = model(images)
        pred = torch.argmax(outputs, dim=1)

        preds.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())

acc, sen, spe, f1 = calculate_metrics(targets, preds)

print("External Validation Results")
print(f"ACC: {acc:.4f}")
print(f"SEN: {sen:.4f}")
print(f"SPE: {spe:.4f}")
print(f"F1 : {f1:.4f}")
