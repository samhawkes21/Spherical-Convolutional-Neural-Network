"""
Data setup: wrapping flat digit images onto the sphere, and building the
full harmonic-coefficient dataset the model trains on.

Combines two concerns that only ever get used together:
  - SphereTexture / make grid-sampling: how a flat image maps onto a patch
    of the sphere's surface.
  - build_dataset: loading sklearn's digit images, projecting each one onto
    the sphere via SphereTexture, and transforming the result into harmonic
    coefficients ready to feed into SphericalCNN.
"""

import numpy as np
import torch
from scipy.ndimage import zoom
from sklearn.datasets import load_digits


class SphereTexture:
    """Maps a flat image onto a bounded longitude/latitude patch of the sphere.

    Points on the sphere outside the patch (i.e. outside +/-lon_max_deg and
    +/-lat_max_deg) are treated as background and sampled as 0.
    """

    def __init__(self, lon_max_deg=60, lat_max_deg=60):
        # Half-width of the patch in radians (image spans +/- this range).
        self.lon_max = np.deg2rad(lon_max_deg)
        self.lat_max = np.deg2rad(lat_max_deg)

    def sample(self, img, theta, phi):
        """Bilinearly sample `img` at each spherical coordinate (theta, phi).

        Args:
            img: (h, w) array, the flat source image.
            theta: array of polar angles (0 at north pole, pi at south pole).
            phi: array of azimuthal angles.

        Returns:
            Array the same shape as theta/phi: the interpolated pixel value
            at each point, or 0 if that point falls outside the image patch.
        """
        h, w = img.shape

        # Convert spherical (theta, phi) to a local "latitude/longitude"
        # centered on the equator/prime-meridian, where the image patch lives.
        lat = np.pi / 2 - theta
        lon = (phi + np.pi) % (2 * np.pi) - np.pi  # wrap into (-pi, pi]

        # Which sphere points actually fall inside the image's patch?
        inside = (np.abs(lon) <= self.lon_max) & (np.abs(lat) <= self.lat_max)

        # Rescale (lon, lat) in [-max, max] to normalized image coords [0, 1].
        u = np.clip((lon / self.lon_max + 1) / 2, 0, 1)
        v = np.clip((lat / self.lat_max + 1) / 2, 0, 1)

        # Convert normalized coords to fractional pixel indices, then do
        # standard bilinear interpolation between the 4 neighboring pixels.
        src_x, src_y = u * (w - 1), v * (h - 1)
        x0 = np.floor(src_x).astype(int)
        y0 = np.floor(src_y).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        wx, wy = src_x - x0, src_y - y0

        Ia, Ib, Ic, Id = img[y0, x0], img[y0, x1], img[y1, x0], img[y1, x1]
        out = (
            Ia * (1 - wx) * (1 - wy)
            + Ib * wx * (1 - wy)
            + Ic * (1 - wx) * wy
            + Id * wx * wy
        )

        # Anything outside the patch is background (0), not interpolated junk.
        return np.where(inside, out, 0.0)


def build_dataset(sht, texture, out_size=28, device="cuda"):
    """Load sklearn digits, warp each onto the sphere, and transform to
    harmonic coefficients.

    Args:
        sht: an `SHT` instance (provides the grid to sample onto and the
            forward transform into harmonic space).
        texture: a `SphereTexture` instance (defines how the flat image maps
            onto the sphere's surface).
        out_size: side length (in pixels) each 8x8 digit is upsampled to
            before being projected onto the sphere.
        device: device to build the coefficient tensor on.

    Returns:
        X: (n_samples, 1, n_harm) complex tensor -- harmonic coefficients,
           with a leading channel dimension of 1 (single input channel).
        y: (n_samples,) long tensor of digit labels (0-9).
    """
    data = load_digits()
    images = data.images / 16.0   # normalize pixel range (raw data is 0-16)
    labels = data.target

    # Sample each upsampled image onto the sphere's grid points.
    signals = np.zeros((len(images), sht.n_grid), dtype=np.float32)
    for i, img in enumerate(images):
        big = zoom(img, out_size / img.shape[0], order=1)  # upsample 8x8 -> out_size x out_size
        signals[i] = texture.sample(big, sht.theta, sht.phi)

    # Convert spatial samples to harmonic coefficients (no gradient needed here).
    signals_t = torch.tensor(signals, device=device)
    with torch.no_grad():
        coeffs = sht.forward(signals_t)

    X = coeffs.unsqueeze(1)  # add channel dim: (n_samples, 1, n_harm)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y
