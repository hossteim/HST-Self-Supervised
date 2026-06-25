# HST Morphological Mapping Pipeline — Inference

This repository contains the inference-only pipeline for the self-supervised
content-based image retrieval system described in the paper:

> *A Self-Supervised Framework for Scalable Content-Based Search in Astronomical Data Archives*
> Teimoorinia et al.

## Overview

The pipeline takes HST drizzled FITS images as input and produces:
- **MAP1**: Per-source morphological cluster assignments (400 classes on a 20×20 grid)
- **MAP2**: Per-image field-level cluster assignments (144 classes on a 12×12 grid)

## Folder Structure

```
pipeline/
├── README.md
├── run_inference_pipeline.ipynb  # Main inference notebook
├── pipeline_helpers.py          # Model architectures & utility functions
├── data/
│   ├── fits/                    # ← Place your input *_drz.fits files here
│   └── cutouts/                 # ← Extracted cutouts are saved here
└── models/
    ├── vicreg_encoder_rotinv_dim128_128px_Epoch_100it_42k.pth  (VICReg-128, cutout encoder)
    ├── vicreg_encoder_rotinv_dim256_256px_Epoch_200.pth        (VICReg-256, field encoder)
    ├── train_latents_dec_model.pt                               (DEC MAP1, 400 clusters)
    ├── dec_map2_model.pth                                       (DEC MAP2, 144 clusters)
    ├── vae_compressor_best.pth                                  (VAE, 528→128 dim)
    ├── scaler_528.pkl                                           (Pre-fit StandardScaler)
    └── map2_cluster_to_cell.json                                (Grid layout mapping)
```

## Requirements

### Python packages

```
torch
torchvision
numpy
scipy
scikit-learn
scikit-image
astropy
matplotlib
joblib
tqdm
Pillow
```

Install with:
```bash
pip install torch torchvision numpy scipy scikit-learn scikit-image astropy matplotlib joblib tqdm Pillow
```

### SExtractor (optional — only needed for source detection)

SExtractor is required only if you want to run the source detection step (Step 3
in the notebook). If you already have pre-extracted cutouts, you can skip it.

Install on Linux:
```bash
sudo apt-get install sextractor
```

Install on macOS (via Homebrew):
```bash
brew install sextractor
```

After installation, ensure `sex` is on your PATH, or update `SEXTRACTOR_BIN` in
the notebook configuration cell to point to your installation.

## Usage

1. Place your HST `*_drz.fits` files in `data/fits/`
2. Open `run_inference_pipeline.ipynb`
3. Run all cells in order

The notebook will:
- Detect sources and extract cutouts (or load pre-extracted ones)
- Assign each source to a MAP1 morphological cluster
- Compute the image-level fingerprint
- Assign each image to a MAP2 field-level cluster on the reorganized 12×12 grid

## Notes

- All model checkpoints are inference-only — no training data is required.
- The pipeline runs on CPU or GPU (auto-detected via PyTorch).
- The grid layout in `map2_cluster_to_cell.json` is fixed and matches the
  published MAP2 figures. It maps raw DEC cluster IDs to topology-aware
  positions computed via UMAP + Hungarian assignment during training.
