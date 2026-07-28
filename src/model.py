"""
The SphericalCNN model, including its two custom layers.

SpectralConv:
    A learnable, complex-valued channel mixing applied *within each harmonic
    degree l independently*. Because it never mixes coefficients across
    different l or across different m, and because a rotation only changes
    the phase of each coefficient (see training.py) without moving it
    to a different (l, degree) group, this operation commutes with rotation
    -- i.e. it is rotation-equivariant.

SpatialReLU:
    A pointwise ReLU cannot be applied usefully directly to complex harmonic
    coefficients. Instead, this layer temporarily leaves harmonic space:
    inverse transform -> ReLU in the real spatial domain -> forward transform
    back to harmonic space. Unlike a "ModReLU" (which only rescales each
    coefficient's magnitude and therefore can never move energy between
    frequency bands), a pointwise nonlinearity in the spatial domain spreads
    energy across *all* frequencies once transformed back -- this is where
    much of a spherical CNN's expressive power comes from.

SphericalCNN:
    Three [SpectralConv -> SpatialReLU] blocks (channels 1 -> 16 -> 32 -> 64),
    followed by a magnitude operation that converts the final complex harmonic
    coefficients into rotation-invariant real features, then a small fully
    connected classifier head.

    Why `.abs()` before the classifier: a rotation about the polar axis only
    multiplies each coefficient by a complex phase exp(i*m*alpha) -- it never
    changes that coefficient's magnitude. So taking the magnitude of the final
    layer's output discards "which rotation was applied" while keeping
    everything else, which is exactly the invariance a digit classifier wants
    (a rotated 7 is still a 7).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv(nn.Module):
    """Per-degree learnable complex linear map between channels.

    For each degree l in [0, l_max], learns a separate (in_ch, out_ch)
    complex weight matrix, and applies it to every coefficient of that
    degree (i.e. every m for that l) identically.
    """

    def __init__(self, in_ch, out_ch, l_max):
        super().__init__()
        self.l_max = l_max
        # One (in_ch, out_ch) complex weight matrix per degree l.
        # Scaled by 1/sqrt(in_ch) for roughly unit-variance activations at init.
        w = torch.randn(l_max + 1, in_ch, out_ch, dtype=torch.cfloat) / np.sqrt(in_ch)
        self.weight = nn.Parameter(w)

    def forward(self, x, lm_degree):
        """
        Args:
            x: (N, in_ch, n_harm) complex tensor of harmonic coefficients.
            lm_degree: (n_harm,) tensor giving the degree l of each coefficient.

        Returns:
            (N, out_ch, n_harm) complex tensor.
        """
        out = torch.zeros(
            x.shape[0], self.weight.shape[-1], x.shape[-1],
            dtype=torch.cfloat, device=x.device,
        )
        # For each degree l, select just the coefficients with that degree
        # and apply that degree's weight matrix to mix input -> output channels.
        for l in range(self.l_max + 1):
            mask = lm_degree == l
            out[:, :, mask] = torch.einsum("nci,co->noi", x[:, :, mask], self.weight[l])
        return out


class SpatialReLU(nn.Module):
    """ReLU nonlinearity applied in the spatial domain.

    Pipeline: harmonic coefficients -> inverse SHT -> real part -> ReLU
    -> forward SHT -> harmonic coefficients.
    """

    def __init__(self, sht):
        super().__init__()
        self.sht = sht  # shared SHT instance, so no basis is recomputed per-layer

    def forward(self, x):
        """
        Args:
            x: (N, C, n_harm) complex tensor of harmonic coefficients.

        Returns:
            (N, C, n_harm) complex tensor, after the spatial-domain ReLU.
        """
        spatial = self.sht.inverse(x).real   # back to (N, C, n_grid), real-valued
        spatial = F.relu(spatial)            # the actual nonlinearity
        return self.sht.forward(spatial)     # forward again -> (N, C, n_harm) complex


class SphericalCNN(nn.Module):
    def __init__(self, sht, n_classes=10, dropout=0.3):
        super().__init__()
        l_max = sht.l_max

        # Three spectral-conv + spatial-ReLU blocks, growing channel width.
        self.conv1 = SpectralConv(1, 16, l_max)
        self.act1 = SpatialReLU(sht)
        self.conv2 = SpectralConv(16, 32, l_max)
        self.act2 = SpatialReLU(sht)
        self.conv3 = SpectralConv(32, 64, l_max)
        self.act3 = SpatialReLU(sht)

        # Classifier head, operating on flattened rotation-invariant features.
        n_harm = sht.n_harm
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * n_harm, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x, lm_degree):
        """
        Args:
            x: (N, 1, n_harm) complex tensor -- harmonic coefficients of the
                input signal (one input channel).
            lm_degree: (n_harm,) tensor giving the degree l of each
                coefficient, needed by every SpectralConv layer.

        Returns:
            (N, n_classes) real tensor of class logits.
        """
        x = self.act1(self.conv1(x, lm_degree))
        x = self.act2(self.conv2(x, lm_degree))
        x = self.act3(self.conv3(x, lm_degree))

        x = x.abs()                      # complex -> real, rotation-invariant
        x = x.reshape(x.shape[0], -1)    # flatten channels + harmonics
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
