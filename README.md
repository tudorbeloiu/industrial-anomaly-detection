# MVTec Anomaly Detection

Unsupervised anomaly detection on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset using two approaches: a convolutional autoencoder trained from scratch, and a feature-based method using a pretrained ResNet18 with Mahalanobis distance scoring.

## Results

Image-level AUROC on the MVTec AD test set:

| Category    | Autoencoder | Feature Extractor | Best   |
|-------------|-------------|-------------------|--------|
| grid        | **0.98**    | –                 | 0.98   |
| wood        | **0.98**    | –                 | 0.98   |
| hazelnut    | **0.97**    | –                 | 0.97   |
| toothbrush  | **0.95**    | –                 | 0.95   |
| leather     | **0.94**    | –                 | 0.94   |
| tile        | 0.70        | **0.94**          | 0.94   |
| metal_nut   | 0.46        | **0.94**          | 0.94   |
| cable       | 0.56        | **0.88**          | 0.88   |
| zipper      | **0.87**    | –                 | 0.87   |
| capsule     | 0.69        | **0.82**          | 0.82   |
| bottle      | **0.80**    | –                 | 0.80   |
| transistor  | 0.54        | **0.79**          | 0.79   |
| pill        | **0.76**    | –                 | 0.76   |
| carpet      | **0.64**    | 0.63              | 0.64   |
| screw       | **0.65**    | 0.52              | 0.65   |

The autoencoder works well on texture categories (grid, wood, leather, hazelnut) where defects break regular patterns. It struggles with complex objects (cable, metal_nut, transistor) where normal images already have high visual variation. The feature extractor handles those cases better by comparing semantic features rather than raw pixels.

Carpet and screw remain the weakest categories — carpet has very complex textures and screw has extremely subtle defects.

## Approach

### Autoencoder (`main.py`)

A convolutional encoder-decoder with BatchNorm, trained to reconstruct normal images. At test time, the reconstruction error highlights anomalous regions.

- **Architecture**: 3 encoder blocks + latent → 3 decoder blocks, bottleneck at 256 × 16 × 16
- **Loss**: 70% MSE + 30% SSIM — MSE handles pixel accuracy, SSIM preserves structural patterns
- **Training**: Adam optimizer with ReduceLROnPlateau scheduler, early stopping
- **Scoring**: Gaussian-smoothed reconstruction error map, max value as anomaly score
- **Normalization**: Per-category mean/std computed from training images

### Feature Extractor (`struggle.py`)

Uses a frozen ResNet18 (pretrained on ImageNet) to extract mid-level features from layers 2 and 3. A multivariate Gaussian is fit to the patch-level features of normal training images. Test images are scored by Mahalanobis distance from this distribution.

- **No training required** — feature extraction and Gaussian fitting takes minutes
- **Multi-scale features**: layer2 (128 channels, local textures) + layer3 (256 channels, structural patterns) concatenated into 384-dimensional patch descriptors
- **Scoring**: Mahalanobis distance at each spatial position, Gaussian-smoothed, max as anomaly score

## Project Structure

```
├── autoencoder.py         # Encoder-decoder architecture
├── dataset.py             # Dataset classes and per-category normalization
├── loss.py                # MSE + SSIM combined loss
├── train.py               # Training loop and evaluation
├── main.py                # Autoencoder pipeline (all 15 categories)
├── struggle.py            # ResNet18 feature-based pipeline
└── README.md
```

## Usage

### Setup

Download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it so each category is a subdirectory under `./data/`:

```
data/
├── bottle/
│   ├── train/good/
│   ├── test/
│   └── ground_truth/
├── cable/
│   ├── ...
```

Install dependencies:

```bash
pip install torch torchvision torchmetrics scikit-learn scipy pillow
```

### Run the autoencoder on all categories

```bash
python main.py
```

### Run the feature extractor on categories where the autoencoder struggles

```bash
python struggle.py
```

Edit `BASE_DIR` in either script to point to your dataset location.

## Dataset

This project uses the **MVTec Anomaly Detection Dataset (MVTec AD)**, which is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0).

**The dataset is not included in this repository.** Download it from [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad).

If you use this dataset, please cite:

> Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, Carsten Steger: *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*; International Journal of Computer Vision 129(4):1038-1059, 2021.

> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger: *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*; IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 9584-9592, 2019.
