"""
pipeline_helpers.py — Utility functions and model definitions for the HST DRZ
morphological mapping pipeline.

This module contains:
- Model architectures (VICRegEncoder, DECHead, DECHeadSimple, DeepVAE)
- Image preprocessing (normalization, cutout extraction)
- SExtractor wrapper
- Dataset class for DataLoader
"""

import math
import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from astropy.io import fits as astrofits
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from torch.utils.data import Dataset


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class VICRegEncoder(nn.Module):
    """
    VICReg self-supervised encoder for grayscale astronomical images.
    5-layer CNN backbone → adaptive pooling → 2-layer MLP projector.
    Used at both 128px (cutout-level) and 256px (full-image-level).
    """
    def __init__(self, projection_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x):
        return self.projector(self.backbone(x))


class DECHead(nn.Module):
    """
    Deep Embedded Clustering head (MAP1, K=400).
    Computes Student-t soft assignments from embeddings to cluster centroids.
    Optionally includes an MLP transform before distance computation.
    Returns (q, z) where q is soft assignment and z is the (optionally transformed) embedding.
    """
    def __init__(self, input_dim, n_clusters, hidden=256, use_mlp=False, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.mlp = (nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, input_dim),
        ) if use_mlp else nn.Identity())
        self.mu = nn.Parameter(torch.randn(n_clusters, input_dim))
        nn.init.xavier_uniform_(self.mu)

    def forward(self, x):
        z = self.mlp(x)
        z2 = (z ** 2).sum(1, keepdim=True)
        mu2 = (self.mu ** 2).sum(1, keepdim=True).T
        dist2 = z2 + mu2 - 2 * (z @ self.mu.T)
        q = (1.0 + dist2 / self.alpha).pow(-(self.alpha + 1.0) * 0.5)
        q = q / (q.sum(1, keepdim=True) + 1e-12)
        return q, z


class DECHeadSimple(nn.Module):
    """
    Simple DEC clustering head (MAP2, K=144).
    No MLP transform — just distance-based soft assignment.
    """
    def __init__(self, input_dim, n_clusters, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.mu = nn.Parameter(torch.randn(n_clusters, input_dim))

    def forward(self, x):
        dist2 = torch.cdist(x, self.mu, p=2) ** 2
        q = (1.0 + dist2 / self.alpha).pow(-(self.alpha + 1.0) * 0.5)
        return q / (q.sum(1, keepdim=True) + 1e-12)


class DeepVAE(nn.Module):
    """
    Variational Autoencoder for compressing the 528-dim combined feature vector
    into a 128-dim latent space. At inference, only the encoder (mu) is used.
    """
    def __init__(self, input_dim=528, latent_dim=128):
        super().__init__()
        self.encoder_block = nn.Sequential(
            nn.Linear(input_dim, 384), nn.LayerNorm(384), nn.ReLU(),
            nn.Linear(384, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 192), nn.LayerNorm(192), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(192, latent_dim)
        self.fc_logvar = nn.Linear(192, latent_dim)

    def forward(self, x):
        h = self.encoder_block(x)
        return self.fc_mu(h)


# =============================================================================
# DATASET
# =============================================================================

class CutoutDataset(Dataset):
    """
    PyTorch Dataset for normalized cutout images.
    Converts float32 arrays [0,1] to PIL → tensor with mean=0.5, std=0.5 normalization.
    """
    def __init__(self, cutouts):
        self.cutouts = cutouts.astype(np.float32)

    def __len__(self):
        return len(self.cutouts)

    def __getitem__(self, idx):
        img = self.cutouts[idx]
        img = Image.fromarray((img * 255).astype(np.uint8))
        t = TF.to_tensor(img)
        t = TF.normalize(t, (0.5,), (0.5,))
        return t


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def normalize_cutouts(imgs):
    """
    Percentile clip [0, 99] + min-max normalize each image in-place.
    Input: (N, H, W) float32 array. Output: same array, values in [0, 1].
    """
    for i in range(len(imgs)):
        lo, hi = np.percentile(imgs[i], [0, 99])
        im = np.clip(imgs[i], lo, hi)
        imgs[i] = (im - im.min()) / (im.max() - im.min() + 1e-6)
    return imgs


def normalize_single_image(img):
    """Percentile clip + min-max normalize a single (H, W) image."""
    lo, hi = np.percentile(img, [0, 99])
    img = np.clip(img, lo, hi)
    return (img - img.min()) / (img.max() - img.min() + 1e-6)


def resize_to_256(sci):
    """Resize a full-size science image to 256x256 with bicubic interpolation."""
    return resize(sci, (256, 256), order=3, anti_aliasing=True,
                  preserve_range=True).astype(np.float32)


def load_fits_science_image(fits_path):
    """Load and clean the SCI extension from a FITS file. Returns float32 array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with astrofits.open(str(fits_path), memmap=False, ignore_missing_end=True) as hdul:
            sci = np.nan_to_num(hdul[1].data.astype(np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)
    return sci


# =============================================================================
# SEXTRACTOR
# =============================================================================

def run_sextractor(fits_path, work_dir, sextractor_bin, nnw_file):
    """
    Run SExtractor on a FITS file to detect sources.
    
    Args:
        fits_path: Path to the FITS file
        work_dir: Directory to write the output catalog
        sextractor_bin: Path to the 'sex' binary
        nnw_file: Path to default.nnw neural network weights file
    
    Returns:
        Path to the output source catalog text file
    """
    base = Path(fits_path).stem
    out_txt = os.path.join(work_dir, base + ".txt")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with astrofits.open(fits_path, memmap=False, ignore_missing_end=True) as hdul:
            sci = np.nan_to_num(hdul[1].data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            wht = np.nan_to_num(hdul[2].data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            sci_hdr = hdul[1].header
            wht_hdr = hdul[2].header

    with tempfile.TemporaryDirectory() as tmpdir:
        sci_tmp = os.path.join(tmpdir, "sci.fits")
        wht_tmp = os.path.join(tmpdir, "wht.fits")
        sex_cfg = os.path.join(tmpdir, "default.sex")
        paramfile = os.path.join(tmpdir, "default.param")
        convfile = os.path.join(tmpdir, "default.conv")
        nnw_tmp = os.path.join(tmpdir, "default.nnw")

        shutil.copy(nnw_file, nnw_tmp)
        astrofits.writeto(sci_tmp, sci, header=sci_hdr, overwrite=True)
        astrofits.writeto(wht_tmp, wht, header=wht_hdr, overwrite=True)

        with open(sex_cfg, "w") as f:
            f.write(
                f"CATALOG_NAME     {out_txt}\n"
                f"CATALOG_TYPE     ASCII_HEAD\n"
                f"PARAMETERS_NAME  {paramfile}\n"
                f"FILTER           Y\n"
                f"FILTER_NAME      {convfile}\n"
                f"WEIGHT_TYPE      MAP_WEIGHT\n"
                f"WEIGHT_IMAGE     {wht_tmp}\n"
                f"DETECT_MINAREA   100\n"
                f"DETECT_THRESH    3\n"
                f"DEBLEND_NTHRESH  32\n"
                f"DEBLEND_MINCONT  1\n"
                f"CLEAN            Y\n"
                f"CLEAN_PARAM      1.0\n"
                f"BACK_TYPE        AUTO\n"
                f"BACK_SIZE        64\n"
                f"BACK_FILTERSIZE  3\n"
                f"PHOT_APERTURES   5\n"
                f"PHOT_AUTOPARAMS  2.5, 3.5\n"
                f"MAG_ZEROPOINT    26.0\n"
                f"SEEING_FWHM      0.9\n"
                f"GAIN             1.0\n"
                f"VERBOSE_TYPE     QUIET\n"
                f"STARNNW_NAME     {nnw_tmp}\n"
            )
        with open(paramfile, "w") as f:
            f.write(
                "X_IMAGE\nY_IMAGE\nALPHA_J2000\nDELTA_J2000\n"
                "A_IMAGE\nB_IMAGE\nTHETA_IMAGE\nKRON_RADIUS\n"
                "ISOAREA_IMAGE\nISOAREAF_IMAGE\nBACKGROUND\nTHRESHOLD\n"
            )
        with open(convfile, "w") as f:
            f.write("CONV NORM\n1 2 1\n2 4 2\n1 2 1\n")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"

        subprocess.run(
            [sextractor_bin, sci_tmp, "-c", sex_cfg],
            check=True, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    return out_txt


# =============================================================================
# CUTOUT EXTRACTION
# =============================================================================

# Constants
_FALLBACK_MULT = 2.0
_MIN_L, _MAX_L = 32, 512
CUTOUT_SIZE = 128
MAX_CUTS_PER_FILE = 200


def _read_sextractor_catalog(txt_path):
    """Parse a SExtractor ASCII_HEAD catalog into an (N, 11) array."""
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                rows.append([float(x) for x in parts[:11]])
            except ValueError:
                continue
    if not rows:
        return np.empty((0, 11), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _compute_cutout_side(a_img, b_img):
    """Compute cutout side length from SExtractor A_IMAGE and B_IMAGE."""
    if np.isfinite(a_img) and np.isfinite(b_img):
        L = int(math.ceil(2.0 * _FALLBACK_MULT * max(float(a_img), float(b_img))))
    else:
        L = _MIN_L
    return max(_MIN_L, min(_MAX_L, (L // 2) * 2))


def _crop_square(img, xc, yc, L):
    """Extract an LxL square cutout centered at (xc, yc). Returns None if out of bounds."""
    half = L // 2
    x_min, x_max = xc - half, xc + half
    y_min, y_max = yc - half, yc + half
    H, W = img.shape
    if x_min < 0 or y_min < 0 or x_max > W or y_max > H:
        return None
    cut = img[y_min:y_max, x_min:x_max]
    if cut.shape != (L, L) or np.isnan(cut).any():
        return None
    return cut


def extract_cutouts(fits_path, catalog_txt, cutout_size=128, max_cutouts=200):
    """
    Extract source cutouts from a FITS image using a SExtractor catalog.
    
    Returns:
        cutouts: (N, cutout_size, cutout_size) float32 array
        se_params: (N, 11) float32 array of SExtractor parameters per source
    """
    sci = load_fits_science_image(fits_path)

    catalog = _read_sextractor_catalog(catalog_txt)
    if catalog.size == 0:
        return (np.empty((0, cutout_size, cutout_size), dtype=np.float32),
                np.empty((0, 11), dtype=np.float32))

    images, params = [], []
    np.random.seed(42)
    order = np.random.permutation(len(catalog))

    for idx in order:
        row = catalog[idx]
        xc = int(round(float(row[0])))
        yc = int(round(float(row[1])))
        L_native = _compute_cutout_side(row[4], row[5])

        if L_native <= cutout_size:
            cut = _crop_square(sci, xc, yc, cutout_size)
        else:
            big = _crop_square(sci, xc, yc, L_native)
            cut = (resize(big, (cutout_size, cutout_size), order=3,
                          anti_aliasing=True, preserve_range=True).astype(np.float32)
                   if big is not None else None)

        if cut is None:
            continue
        images.append(cut.astype(np.float32, copy=False))
        params.append(row)
        if len(images) >= max_cutouts:
            break

    if not images:
        return (np.empty((0, cutout_size, cutout_size), dtype=np.float32),
                np.empty((0, 11), dtype=np.float32))
    return np.stack(images), np.stack(params)


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_vicreg_model(path, projection_dim, device):
    """Load a VICReg encoder from a checkpoint file."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        dim = ckpt.get("latent_dim", projection_dim)
        model = VICRegEncoder(projection_dim=dim).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = VICRegEncoder(projection_dim=projection_dim).to(device)
        model.load_state_dict(ckpt)
    model.eval()
    return model


def load_dec_map1(path, device):
    """Load the DEC MAP1 clustering model (K=400). Returns (model, K)."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = DECHead(
        input_dim=ckpt["latent_dim"],
        n_clusters=int(ckpt["k"]),
        use_mlp=ckpt.get("use_mlp", False),
        alpha=ckpt.get("alpha", 1.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, int(ckpt["k"])


def load_dec_map2(path, device, vae_latent_dim=128, dec_k=144):
    """Load the DEC MAP2 clustering model (K=144)."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = DECHeadSimple(vae_latent_dim, dec_k).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


def load_vae(path, device, input_dim=528, latent_dim=128):
    """Load the VAE compressor model (encoder only used at inference)."""
    model = DeepVAE(input_dim, latent_dim).to(device)
    model.load_state_dict(
        torch.load(str(path), map_location=device, weights_only=False), strict=False
    )
    model.eval()
    return model
