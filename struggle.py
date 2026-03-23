import torch
import torch.nn as nn
from torchvision import models

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

import os

from dataset import find_mean_std, AnomalyDataset, AnomalyDatasetTest
from torch.utils.data import DataLoader

# resnet = models.resnet18(weights='IMAGENET1K_V1')
# for i, child in enumerate(resnet.children()):
#     print(f'{i}: {child.__class__.__name__}')

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights='IMAGENET1K_V1')
        children = list(resnet.children())

        self.layer1 = nn.Sequential(*children[:5]) #early features
        self.layer2 = children[5] #mid features
        self.layer3 = children[6] #deeper features

    def forward(self, x):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        
        return f2, f3 # 128x32x32 and 256x16x16
    

def normal_distrib(extractor, train_loader, device):
    extractor.eval()

    all_f2 = []
    all_f3 = []

    with torch.no_grad():
        for img in train_loader:
            img = img.to(device)

            f2, f3 = extractor(img)
            all_f2.append(f2.cpu()) # we don t want to run out of GPU so we move the feature maps of each batch back to RAM
            all_f3.append(f3.cpu())

   
    all_f2 = torch.cat(all_f2, dim=0)
    all_f3 = torch.cat(all_f3, dim = 0)

    # f3 is 16x16 => upsample to 32x32
    f3_upsample = nn.functional.interpolate(
        input = all_f3, size=all_f2.shape[2:], mode='bilinear', align_corners=False
    )
    combined = torch.cat([all_f2, f3_upsample], dim = 1)
    # right now we have (N, 384, 32, 32) => N images, 384 features at 32x32 positions
    
    N,C, H, W = combined.shape
    combined = combined.permute(0, 2, 3, 1).reshape(N * H * W, C).numpy() # each patch is sa sample

    mean = np.mean(combined, axis = 0)
    cov = np.cov(combined, rowvar=False)

    return mean, cov, H, W

def score_test_images(extractor, img, mean, cov_inv, H, W, device):
    extractor.eval()
    
    with torch.no_grad():
        # add batch dimension (3, 256, 256) → (1, 3, 256, 256)
        img = img.unsqueeze(0).to(device)
        
        # get features from ResNet
        f2, f3 = extractor(img)
        
        # make f3 same spatial size as f2
        f3_up = nn.functional.interpolate(
            f3, size=f2.shape[2:], mode='bilinear', align_corners=False
        )
        
        # sttack features: 128 + 256 = 384 features per patch
        features = torch.cat([f2, f3_up], dim=1)
    
    # convert to numpy table: one row per patch, one column per feature
    features = features.squeeze(0)          # remove batch dim
    features = features.permute(1, 2, 0)    # channels last: (H, W, 384)
    features = features.cpu().numpy()
    patches = features.reshape(-1, features.shape[-1])  # (H*W, 384)
    
    # how far is each patch from "normal"
    diff = patches - mean                          # difference from normal
    scores = np.sum(diff @ cov_inv * diff, axis=1) # Mahalanobis distance
    
    score_map = scores.reshape(H, W)
    return score_map

def evaluate_feature(extractor, test_dataset, mean, cov, H, W, device, sigma=4):
    cov_inv = np.linalg.inv(cov)

    all_scores = []
    all_labels = []

    for img, mask, label, defect in test_dataset:
        score_map = score_test_images(extractor, img, mean, cov_inv, H, W, device)
        score_smooth = gaussian_filter(score_map, sigma=sigma)
        all_scores.append(score_smooth.max())
        all_labels.append(label)

    auroc = roc_auc_score(all_labels, all_scores)
    print(f"AUROC : {auroc:.4f}")
    return auroc

if __name__ == "__main__":
    BASE_DIR = './data'
    categories = ['cable', 'metal_nut', 'transistor', 'capsule', 'screw', 'carpet', 'tile']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = FeatureExtractor().to(device)

    for cat in categories:
        print(f"{cat}")

        train_dir = os.path.join(BASE_DIR, cat, "train", "good")
        test_dir = os.path.join(BASE_DIR, cat, "test")
        mask_dir = os.path.join(BASE_DIR, cat, "ground_truth")

        mean, std = find_mean_std(train_dir)

        train_dataset = AnomalyDataset(train_dir, mean, std)
        test_dataset =AnomalyDatasetTest(test_dir, mask_dir, mean, std)
        
        train_loader = DataLoader(train_dataset, batch_size = 16, shuffle=True)

        feat_mean, cov, H, W = normal_distrib(extractor, train_loader, device)
        evaluate_feature(extractor, test_dataset, feat_mean, cov, H, W, device, sigma=4)
