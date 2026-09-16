"""Numerical helpers that do not depend on xarray (so they can be unit tested anywhere)."""
from __future__ import annotations

import numpy as np


def area_stats(values) -> dict:
    """Statistics over all valid (finite) pixels in a 2-D array."""
    a = np.asarray(values, dtype="float64").ravel()
    n_total = int(a.size)
    good = a[np.isfinite(a)]
    n_valid = int(good.size)
    if n_valid == 0:
        return dict(val_mean=None, val_median=None, val_min=None, val_max=None,
                    val_std=None, n_valid=0, n_total=n_total)
    return dict(
        val_mean=float(good.mean()),
        val_median=float(np.median(good)),
        val_min=float(good.min()),
        val_max=float(good.max()),
        val_std=float(good.std(ddof=0)) if n_valid > 1 else 0.0,
        n_valid=n_valid,
        n_total=n_total,
    )


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(d))


def nearest_valid_cell(lats, lons, valid_mask, lat, lon, max_km=15.0):
    """Index (i, j) of the nearest grid cell that holds data (sea, not land).

    lats, lons: 1-D coordinate arrays; valid_mask: 2-D bool array [lat, lon].
    Returns None if nothing valid lies within max_km.
    """
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    glat, glon = np.meshgrid(lats, lons, indexing="ij")
    dist = haversine_km(glat, glon, lat, lon)
    dist = np.where(np.asarray(valid_mask), dist, np.inf)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    if not np.isfinite(dist[idx]) or dist[idx] > max_km:
        return None
    return int(idx[0]), int(idx[1]), float(dist[idx])


def circular_mean_deg(values) -> float | None:
    a = np.asarray(values, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    r = np.radians(a)
    ang = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    return float(ang % 360.0)


def mask_valid(values, valid_range=None, attrs=None):
    """Set physically impossible values to NaN.

    Uses the configured valid_range and, when present, the dataset's own valid_min/valid_max attributes.
    """
    a = np.array(values, dtype="float64", copy=True)
    lo, hi = -np.inf, np.inf
    if valid_range:
        lo, hi = float(valid_range[0]), float(valid_range[1])
    if attrs:
        if attrs.get("valid_min") is not None:
            lo = max(lo, float(np.asarray(attrs["valid_min"]).ravel()[0]))
        if attrs.get("valid_max") is not None:
            hi = min(hi, float(np.asarray(attrs["valid_max"]).ravel()[0]))
    with np.errstate(invalid="ignore"):
        a[(a < lo) | (a > hi)] = np.nan
    return a
