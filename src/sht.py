"""
Spherical Harmonic Transform (SHT).

Spherical harmonics Y_l^m are the sphere's analog of a Fourier basis: any
signal defined on the sphere's surface can be decomposed into a weighted
sum of these basis functions, indexed by a *degree* l (>= 0, roughly "how
fine-grained") and an *order* m (-l <= m <= l, roughly "how it varies
around the polar axis").

`SHT` builds this basis on a fixed sampling grid and provides:
  - forward():  spatial signal  -> harmonic coefficients
  - inverse():  harmonic coefficients -> spatial signal

It also records each coefficient's (l, m) indices, because:
  - `l` (degree) is what SpectralConv mixes channels within (see model.py)
  - `m` (order) is what the free rotation augmentation acts on (see
    training.py) -- rotating about the polar axis is a pure phase
    shift per order m.
"""

import numpy as np
import torch
from scipy.special import sph_harm_y


def make_equiangular_grid(n_theta, n_phi):
    """Build a regular equiangular (theta, phi) sampling grid on the sphere.

    theta ranges over (0, pi) (avoiding the poles exactly, where the
    coordinate system is singular) and phi ranges over a full [0, 2*pi).

    Returns:
        Two flat 1D arrays (theta, phi), one entry per grid point, in
        row-major (theta-major) order.
    """
    theta = np.linspace(1e-3, np.pi - 1e-3, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
    return Theta.ravel(), Phi.ravel()


class SHT:
    """Precomputes a spherical-harmonic basis on an equiangular grid and
    exposes forward/inverse transforms between spatial and harmonic space.
    """

    def __init__(self, n_theta, n_phi, l_max, device="cuda"):
        self.l_max = l_max
        theta, phi = make_equiangular_grid(n_theta, n_phi)
        self.theta, self.phi = theta, phi

        # Build one basis column Y_l^m(theta, phi) for every (l, m) pair with
        # l in [0, l_max] and m in [-l, l], evaluated at every grid point.
        # Also remember each column's (l, m) so later layers can index by them.
        cols, lm_l, lm_m = [], [], []
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                cols.append(sph_harm_y(l, m, theta, phi))
                lm_l.append(l)
                lm_m.append(m)
        Y = np.stack(cols, axis=1)  # shape (n_grid, n_harm), complex

        # Y: the basis matrix itself (used for the inverse transform).
        self.Y = torch.tensor(Y, dtype=torch.complex64, device=device)
        # Y_pinv: pseudo-inverse of Y (used for the forward transform), since
        # the sampling grid is not exactly orthonormal w.r.t. the harmonics.
        self.Y_pinv = torch.linalg.pinv(self.Y)

        # Per-coefficient bookkeeping: degree l (int, for SpectralConv) and
        # order m (float, for the rotation-phase multiply in augmentation).
        self.lm_degree = torch.tensor(lm_l, dtype=torch.long, device=device)
        self.lm_order = torch.tensor(lm_m, dtype=torch.float32, device=device)

        self.n_harm = self.Y.shape[1]   # total number of harmonic coefficients
        self.n_grid = self.Y.shape[0]   # total number of spatial grid points

    def forward(self, f):
        """Spatial signal(s) -> harmonic coefficients.

        Args:
            f: (..., n_grid) real or complex tensor of spatial samples.

        Returns:
            (..., n_harm) complex tensor of harmonic coefficients.
        """
        return f.to(torch.complex64) @ self.Y_pinv.T

    def inverse(self, coeffs):
        """Harmonic coefficients -> spatial signal.

        Args:
            coeffs: (..., n_harm) complex tensor of harmonic coefficients.

        Returns:
            (..., n_grid) complex tensor of spatial samples (take `.real`
            if you know the underlying signal should be real-valued).
        """
        return coeffs @ self.Y.T
