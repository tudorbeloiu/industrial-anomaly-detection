import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AnomalyDataset, AnomalyDatasetTest, find_mean_std
from autoencoder import IndustryEncDec
from loss import CombinedLoss
from train import train, evaluate

BASE_DIR = './data'
SAVE_DIR = './models'
os.makedirs(SAVE_DIR, exist_ok=True)

categories = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

results ={}

device = "cuda" if torch.cuda.is_available() else "cpu"

for cat in categories:
    print(f'Training: {cat}')
    train_dir = os.path.join(BASE_DIR, cat, 'train', 'good')
    test_dir = os.path.join(BASE_DIR, cat, 'test')
    mask_dir = os.path.join(BASE_DIR, cat, 'ground_truth')

    
    mean, std = find_mean_std(train_dir)

    train_dataset = AnomalyDataset(train_dir, mean, std)
    test_dataset = AnomalyDatasetTest(test_dir, mask_dir, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


    model = IndustryEncDec().to(device)
    cost_function = CombinedLoss(alpha=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(SAVE_DIR, f'model_{cat}.pth')


    for epoch in range(60):
        avg_loss = train(model, train_loader, cost_function, optimizer, device)
        scheduler.step(avg_loss)
        print(f'epoch {epoch+1} | loss: {avg_loss:.6f}')

        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f'early stopping at epoch {epoch+1}')
                break


    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.to(device)
    print(f'results for {cat}:')
    evaluate(model, test_dataset, device, sigma=4)