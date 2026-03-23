from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import numpy as np
import torch

def train(model, train_loader, cost_function, optimizer, device):
    model.train()
    total_loss = 0

    for x in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        xx = model(x)
        loss = cost_function(xx, x)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

def evaluate(model, test_dataset, device, sigma=4):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for img, mask, label, defect in test_dataset:
            img = img.unsqueeze(0).to(device)
            xx = model(img)

            error_map = (img - xx) ** 2
            error_map = error_map.mean(dim=1).squeeze().cpu().numpy()
            error_smooth = gaussian_filter(error_map, sigma=sigma)
            all_scores.append(error_smooth.max())
            all_labels.append(label)

    auroc = roc_auc_score(all_labels, all_scores)
    print(f'AUROC: {auroc:.4f}')
    return auroc