"""Copernicus Marine Service (CMEMS) sources via the official `copernicusmarine` toolbox.

Needs a free account: https://data.marine.copernicus.eu/register
Set COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..db import Observation
from ..stats import area_stats, nearest_valid_cell
from .base import Source, SourceError


def _coord(ds, *names):
    for n in names:
        if n in ds.coords or n in ds.dims:
            return n
    raise SourceError(f"None of {names} found in dataset coordinates {list(ds.coords)}")


class _CopernicusBase(Source):
    required_env = ("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")

    def _open(self, bbox, start: date, end: date, end_hour=23):
        import copernicusmarine  # imported lazily so tests run without it

        lon_min, lat_min, lon_max, lat_max = bbox
        # Open lazily with only the spatial subset (metadata only, no download), then clip the
        # requested dates to what the dataset actually holds. Reprocessed products end months
        # before today and NRT products only keep recent weeks, so requests often overhang.
        ds = copernicusmarine.open_dataset(
            dataset_id=self.cfg["dataset_id"],
            variables=list(self.cfg["variables"].keys()),
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            coordinates_selection_method="inside",
        )
        tname = _coord(ds, "time")
        times = pd.to_datetime(ds[tname].values)
        t0 = pd.Timestamp(datetime(start.year, start.month, start.day))
        t1 = pd.Timestamp(datetime(end.year, end.month, end.day, end_hour, 59))
        if len(times) == 0 or t1 < times.min() or t0 > times.max():
            ds.close()
            return None
        return ds.sel({tname: slice(max(t0, times.min()), min(t1, times.max()))})

    def _values(self, ds, var):
        da = ds[var]
        # drop a depth dimension of length one if present
        for d in list(da.dims):
            if d not in (_coord(ds, "time"), _coord(ds, "latitude", "lat"), _coord(ds, "longitude", "lon")):
                da = da.isel({d: 0})
        tname = _coord(ds, "time")
        la, lo = _coord(ds, "latitude", "lat"), _coord(ds, "longitude", "lon")
        da = da.transpose(tname, la, lo)
        vals = np.asarray(da.values, dtype="float64")
        if var in (self.cfg.get("kelvin_to_celsius") or []):
            # some products already deliver Celsius; only convert values that look like Kelvin
            if np.isfinite(vals).any() and np.nanmedian(vals) > 200:
                vals = vals - 273.15
        times = pd.to_datetime(ds[tname].values).to_pydatetime()
        return times, np.asarray(ds[la].values), np.asarray(ds[lo].values), vals


class CopernicusGrid(_CopernicusBase):
    """Area statistics over each configured area for gridded daily products."""

    def fetch(self, start: date, end: date):
        ds = self._open(self.union_bbox(), start, end)
        if ds is None:
            return []
        out = []
        try:
            for var, code in self.cfg["variables"].items():
                times, lats, lons, vals = self._values(ds, var)
                for acode, area in self.areas().items():
                    lon_min, lat_min, lon_max, lat_max = area["bbox"]
                    iy = np.where((lats >= lat_min) & (lats <= lat_max))[0]
                    ix = np.where((lons >= lon_min) & (lons <= lon_max))[0]
                    if iy.size == 0 or ix.size == 0:
                        # area smaller than a grid cell (e.g. 0.05 deg SST): use the cell at its centre
                        iy = np.array([int(np.argmin(np.abs(lats - (lat_min + lat_max) / 2)))])
                        ix = np.array([int(np.argmin(np.abs(lons - (lon_min + lon_max) / 2)))])
                    sub = vals[:, iy.min():iy.max() + 1, ix.min():ix.max() + 1]
                    for k, t in enumerate(times):
                        s = area_stats(sub[k])
                        out.append(Observation(self.code, code, acode, t, **s))
        finally:
            ds.close()
        return out


class CopernicusPoint(_CopernicusBase):
    """Hourly model series at the nearest valid sea cell to each configured point."""

    def fetch(self, start: date, end: date):
        extra = int(self.cfg.get("include_forecast_days") or 0)
        today = datetime.utcnow().date()
        if end >= today:
            end = today + timedelta(days=extra) if extra else today
        out = []
        for pcode in self.cfg["points"]:
            p = self.point(pcode)
            bbox = (p["lon"] - 0.12, p["lat"] - 0.12, p["lon"] + 0.12, p["lat"] + 0.12)
            ds = self._open(bbox, start, end)
            if ds is None:
                continue
            try:
                cell = None
                for var, code in self.cfg["variables"].items():
                    times, lats, lons, vals = self._values(ds, var)
                    if vals.size == 0:
                        continue
                    if cell is None:
                        valid = np.isfinite(vals).any(axis=0)
                        cell = nearest_valid_cell(lats, lons, valid, p["lat"], p["lon"], max_km=15)
                        if cell is None:
                            raise SourceError(f"No sea cell within 15 km of {pcode} in {self.cfg['dataset_id']}")
                    i, j, _ = cell
                    series = vals[:, i, j]
                    for t, v in zip(times, series):
                        out.append(Observation(self.code, code, pcode, t, float(v), n_valid=int(np.isfinite(v)), n_total=1))
            finally:
                ds.close()
        return out
