# Spherical CNN — Rotation-Equivariant Digit Classification

A convolutional neural network that operates on **spherical harmonics** instead of a flat pixel grid, built to classify handwritten digits ([sklearn's `load_digits`](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset)) after warping each image onto a small patch of a sphere.

Ordinary CNNs exploit *translation* equivariance — shift the image, and the features shift with it. This project builds the analogous idea for *rotation* equivariance on a sphere: every layer is designed so that rotating the input sphere and then running the network gives the same result as running the network and then rotating the output.

## Why a sphere?

Any well-behaved signal on a sphere's surface can be decomposed into a weighted sum of **spherical harmonics** $Y_l^m(\theta, \phi)$ — the sphere's version of a Fourier basis, indexed by:

- **degree `l`** — how fine-grained the detail is (like frequency)
- **order `m`** — how the pattern is oriented around the polar axis

The key fact this project is built around: rotating the sphere about its polar axis by angle `α` turns into an almost trivial operation in harmonic space — every coefficient just gets multiplied by a complex phase:

$$\hat f_{l,m} \;\rightarrow\; \hat f_{l,m} \cdot e^{i m \alpha}$$

No resampling, no interpolation error. That one property is exploited three times in this codebase:

1. **Free, exact data augmentation** — "rotate the input" becomes one complex multiply.
2. **A rotation-equivariant convolution** — a layer that only mixes channels *within* a fixed `(l, m)` slot automatically commutes with rotation, since rotation never moves a coefficient to a different slot.
3. **A rotation-invariant classifier feature** — since rotation only changes phase, never magnitude, taking `|·|` of the final layer's output discards "how it was rotated" while keeping "what digit it is."

## Pipeline

```
flat 8x8 digit
  → upsample to 28x28
  → project onto a patch of the sphere (SphereTexture)
  → sample at grid points (theta, phi)
  → Spherical Harmonic Transform → complex coefficients
  → SphericalCNN (SpectralConv + SpatialReLU, x3)
  → |coefficients|  (rotation-invariant real features)
  → fully connected classifier → digit 0–9
```

## Project structure

```
.
├── main.py       # seeds, device, constants, and the orchestration script
├── sht.py        # SHT: spherical harmonic transform (forward/inverse)
├── data.py       # SphereTexture (image → sphere patch) + dataset builder
├── model.py      # SpectralConv, SpatialReLU, and the SphericalCNN architecture
└── training.py   # rotation augmentation + the training loop
```

| File | Responsibility |
|---|---|
| `main.py` | Fixes RNG seeds, sets device to CUDA, defines harmonic resolution (`l_max`, grid size), and runs the full pipeline: build SHT → build dataset → stratified train/val split → train → report accuracy. |
| `sht.py` | Builds the spherical-harmonic basis on an equiangular grid; converts signals between spatial (grid-point values) and harmonic (coefficient) representations. |
| `data.py` | Warps flat digit images onto a bounded lat/lon patch of the sphere via bilinear interpolation, then transforms the whole dataset into harmonic coefficients. |
| `model.py` | `SpectralConv` (learnable per-degree complex channel mixing — the rotation-equivariant "convolution"), `SpatialReLU` (nonlinearity applied via a round-trip through spatial space), and `SphericalCNN` (stacks 3 conv+activation blocks into a full classifier). |
| `training.py` | `augment_rotate` (the free rotation-augmentation trick) and `train` (Adam, weight decay, cosine LR schedule, best-validation-checkpoint tracking/restoration). |

## Requirements

- Python 3.9+
- PyTorch (with CUDA support — this project assumes a GPU is available)
- NumPy
- SciPy (recent enough to provide `scipy.special.sph_harm_y`)
- scikit-learn

```bash
pip install torch numpy scipy scikit-learn
```

## Usage

```bash
python main.py
```

This will:
1. Build the spherical-harmonic basis (`l_max=16` on a 40×80 grid).
2. Load and warp all 1,797 digit images onto the sphere, converting each to harmonic coefficients.
3. Split into an 80/20 stratified train/validation set.
4. Train `SphericalCNN` for 40 epochs with rotation augmentation, dropout, weight decay, and a cosine LR schedule.
5. Restore the best-validation-accuracy checkpoint and report final accuracy.

## Model architecture

```
input (1 channel, 289 harmonic coefficients, complex)
 → SpectralConv(1 → 16)  → SpatialReLU
 → SpectralConv(16 → 32) → SpatialReLU
 → SpectralConv(32 → 64) → SpatialReLU
 → |·|   (complex magnitude → real, rotation-invariant)
 → flatten → Dropout
 → Linear(→128) → ReLU → Linear(→10)
```

## Notes / limitations

- The sphere patch only covers ±60° of longitude/latitude — most of the sphere is unused background. This keeps the problem tractable while still exercising genuine spherical structure.
- `l_max=16` and a 40×80 grid are a resolution trade-off; increasing either gives less information loss when warping digits onto the sphere, at the cost of more compute.
- This is a research/educational project, not a benchmark-optimized model — the point is demonstrating rotation equivariance on the sphere, not maximizing digit-classification accuracy (which flat CNNs will trivially beat on this dataset).

## License

Add your preferred license here (e.g. MIT).
