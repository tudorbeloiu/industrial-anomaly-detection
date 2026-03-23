import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

from PIL import Image

class AnomalyDataset(Dataset):
    def __init__(self, root_dir, mean, std):
        super().__init__()
        image_paths = []
        for f in os.listdir(root_dir):
            if f.endswith('.png'):
                image_paths.append(os.path.join(root_dir, f))

        self.img_paths = sorted(image_paths)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')
        return self.transform(img)
    
class AnomalyDatasetTest(Dataset):
    def __init__(self, test_dir, mask_dir, mean, std):
        super().__init__()

        self.samples = []
        good_dir = os.path.join(test_dir, 'good')
        
        for f in os.listdir(good_dir):
             if f.endswith('.png') or f.endswith('.jpg'):
                self.samples.append({
                    "img_path": os.path.join(good_dir, f),
                    "mask_path": None,
                    "label": 0,
                    "defect": 'good'
                })

        for defect in sorted(os.listdir(test_dir)):
            if defect == "good":
                continue

            defect_dir = os.path.join(test_dir, defect)
            defect_mask_dir = os.path.join(mask_dir, defect)

            if not os.path.isdir(defect_dir):
                continue

            for f in sorted(os.listdir(defect_dir)):
                mask_name = f.replace('.png', '_mask.png')
                self.samples.append({
                    "img_path": os.path.join(defect_dir, f),
                    "mask_path": os.path.join(defect_mask_dir, mask_name),
                    "label": 1,
                    "defect": defect
                })

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        just_img = self.samples[index]

        img = Image.open(just_img['img_path']).convert('RGB')
        img = self.transform(img)

        if just_img['mask_path']:
            mask = Image.open(just_img['mask_path']).convert('L')
            mask = self.mask_transform(mask)
        else:
            mask = torch.zeros(1, 256, 256)

        return img, mask, just_img['label'], just_img['defect']

def find_mean_std(root_dir):
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = base_transform(img)
        for c in range(3):
            mean[c] += img[c].mean()
            std[c] += img[c].std()
    
    mean /= len(image_paths)
    std /= len(image_paths)
    return mean.tolist(), std.tolist()