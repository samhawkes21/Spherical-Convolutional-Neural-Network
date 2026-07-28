# Spherical CNN — Rotation-Equivariant Digit Classification

A convolutional neural network that operates on spherical harmonics instead of a flat pixel grid, built to classify handwritten digits ([sklearn's `load_digits`](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset)) after warping each image onto a small patch of a sphere.

Ordinary CNNs exploit *translation* equivariance — shift the image, and the features shift with it. This project builds the analogous idea for *rotation* equivariance on a sphere: every layer is designed so that rotating the input sphere and then running the network gives the same result as running the network and then rotating the output.

## Why a sphere?

Any signal on a sphere's surface can be decomposed into a weighted sum of **spherical harmonics** $Y_l^m(\theta, \phi)$ — the sphere's version of a Fourier basis, indexed by:

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
  → project onto a patch of the sphere 
  → sample at grid points 
  → Spherical Harmonic Transform → complex coefficients
  → SphericalCNN 
  → absolute values of coefficients
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


