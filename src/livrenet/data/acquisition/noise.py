"""Bounded image-coherent and independent 3-D noise."""

import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def rms(v):
    return float(np.sqrt(np.mean(np.sum(np.asarray(v, dtype=float) ** 2, axis=1))))


def unit_field(points, uv, camera, n, seed, mode):
    rng = np.random.default_rng(seed)
    iid = rng.normal(size=points.shape)
    iid /= rms(iid)
    # Smooth random depth bias sampled in image coordinates; do not subtract
    # its mean: a real coherent error may include a translation-like bias.
    spacing = 4.0
    shape = (
        math.ceil(camera.height / spacing) + 4,
        math.ceil(camera.width / spacing) + 4,
    )
    grid = gaussian_filter(
        rng.normal(size=shape),
        sigma=n["coherent_correlation_pixels"] / spacing,
        mode="reflect",
    )
    grid /= max(float(np.std(grid)), 1e-12)
    grid += rng.normal(0, 0.3)
    bias = map_coordinates(
        grid, [uv[:, 1] / spacing + 2, uv[:, 0] / spacing + 2], order=1, mode="nearest"
    )
    rays = points - np.asarray(camera.position)
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    coherent = rays * bias[:, None]
    coherent /= rms(coherent)
    fraction = n["mixed_coherent_power_fraction"]
    if mode == "iid":
        combined = iid
    elif mode == "coherent":
        combined = coherent
    elif mode == "mixed":
        combined = math.sqrt(1 - fraction) * iid + math.sqrt(fraction) * coherent
    else:
        raise ValueError("Unknown noise mode")
    combined = combined / rms(combined)
    # Truncated, RMS-normalized perturbations. This is not an unbounded
    # Gaussian distribution. No points/support are deleted to hide outliers.
    lengths = np.linalg.norm(combined, axis=1)
    order = np.argsort(lengths, kind="stable")
    probabilities = np.linspace(0.0, 1.0, len(lengths))
    magnitudes = np.interp(
        probabilities,
        n["radial_quantile_probabilities"],
        n["radial_quantiles_unit_rms"],
    )
    assigned = np.empty_like(magnitudes)
    assigned[order] = magnitudes
    combined *= (assigned / np.maximum(lengths, 1e-12))[:, None]
    combined /= rms(combined)
    limit = n["maximum_point_error_rms_multiplier"]
    for _ in range(100):
        lengths = np.linalg.norm(combined, axis=1)
        if lengths.max() <= limit + 1e-10:
            break
        combined *= np.minimum(1.0, limit / np.maximum(lengths, 1e-12))[:, None]
        combined /= rms(combined)
    else:
        raise ValueError("Failed to bound noise tails")
    return combined
