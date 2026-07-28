"""
Entry point: sets up seeds/device, builds the SHT basis and the sphere-warped
digit dataset, builds the model, trains it, and reports final validation
accuracy.

Run with:
    python main.py
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from sht import SHT
from data import SphereTexture, build_dataset
from model import SphericalCNN
from training import train

# ---------------------------------------------------------------------------
# Reproducibility: fix seeds for both the torch and numpy RNGs so runs are
# repeatable (dataset split, weight init, batch shuffling, augmentation, etc).
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# CUDA is assumed to be available. If you ever need to run on CPU instead,
# change this line to torch.device("cpu").
# ---------------------------------------------------------------------------
device = torch.device("cuda")

# ---------------------------------------------------------------------------
# Spherical-harmonic resolution: degrees 0..l_max are used, on a 40x80
# equiangular grid. Higher l_max + denser grid = less information lost
# when warping digits onto the sphere, at the cost of more compute.
# ---------------------------------------------------------------------------
L_MAX = 16
N_THETA, N_PHI = 40, 80


def main():
    print("Building SHT basis and spherically-wrapped digit dataset...")
    sht = SHT(n_theta=N_THETA, n_phi=N_PHI, l_max=L_MAX, device=device)
    texture = SphereTexture(lon_max_deg=60, lat_max_deg=60)
    X, y = build_dataset(sht, texture, device=device)

    # Stratified 80/20 train/val split (keeps class balance in both sets).
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=0.2, random_state=0, stratify=y.numpy()
    )
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"train: {tuple(X_train.shape)}, val: {tuple(X_val.shape)}  "
          f"(n_harmonics = {sht.n_harm} for l_max={L_MAX})")

    model = SphericalCNN(sht, dropout=0.3).to(device)
    train(
        model, sht, X_train, y_train, X_val, y_val,
        epochs=40, batch_size=32, lr=2e-3, weight_decay=1e-4, augment=True,
        device=device,
    )

    with torch.no_grad():
        model.eval()
        final_acc = (
            model(X_val.to(device), sht.lm_degree).argmax(1).cpu() == y_val
        ).float().mean().item()
    print(f"Final (best-checkpoint) validation accuracy on sphere-warped digits: {final_acc:.3f}")


if __name__ == "__main__":
    main()
