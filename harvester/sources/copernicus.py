"""Copernicus Marine Service (CMEMS) sources via the official `copernicusmarine` toolbox.

Needs a free account: https://data.marine.copernicus.eu/register
Set COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD.

In config.yaml a source's `variables` maps dataset variable names to our codes, either
    {CHL: chl}
or, when the exact name in the dataset is uncertain or the units need converting,
    {DIATO: {code: diatoms, match: diatom}}
    {dissic: {code: dic, scale: 1000}}        # mol m-3 -> mmol m-3
For 3-D model variables a spec can also ask for values at fixed depths and an isoline depth:
    {thetao: {code: temp, depths: [10, 30, 50], surface_code: temp_surface}}   # -> temp_10m, temp_30m ...
    {so: {code: salinity, depths: [100], interface: {code: interface_depth, value: 37.5}}}
`match` is looked for (case-insensitively) in the variable name, standard_name and long_name
if the given name is not in the dataset.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..db import Observation
from ..stats import area_stats, mask_valid, nearest_valid_cell
from .base import Source, SourceError


def _coord(ds, *names):
    for n in names:
        if n in ds.coords or n in ds.dims:
            return n
    raise SourceError(f"None of {names} found in dataset coordinates {list(ds.coords)}")


def variable_specs(cfg: dict) -> list[dict]:
    """Normalise a source's `variables` block into [{name, code, match}]."""
    out = []
    for name, spec in (cfg.get("variables") or {}).items():
        if isinstance(spec, dict):
            out.append({"name": name, "code": spec["code"], "match": spec.get("match"),
                        "scale": spec.get("scale"), "depths": spec.get("depths"),
                        "surface_code": spec.get("surface_code"), "interface": spec.get("interface")})
        else:
            out.append({"name": name, "code": spec, "match": None, "scale": None,
                        "depths": None, "surface_code": None, "interface": None})
    return out


def spec_codes(spec: dict) -> list[str]:
    """Variable codes a spec produces."""
    if spec.get("depths") or spec.get("interface"):
        codes = [f"{spec['code']}_{int(d)}m" for d in (spec.get("depths") or [])]
        if spec.get("surface_code"):
            codes.insert(0, spec["surface_code"])
        if spec.get("interface"):
            codes.append(spec["interface"]["code"])
        return codes
    return [spec["code"]]


def nearest_level(depths, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(depths, dtype="float64") - float(target))))


def isoline_depth(profile, depths, value: float):
    """Depth where a profile first reaches `value` going down, linearly interpolated.

    profile: array [..., z] (NaN below the seabed); depths: 1-D ascending, positive down.
    Returns NaN where the value is never reached, 0 where the surface already reaches it.
    """
    prof = np.asarray(profile, dtype="float64")
    z = np.asarray(depths, dtype="float64")
    flat = prof.reshape(-1, prof.shape[-1])
    out = np.full(flat.shape[0], np.nan)
    for i, col in enumerate(flat):
        ok = np.isfinite(col)
        if not ok.any():
            continue
        c, zz = col[ok], z[ok]
        hit = np.where(c >= value)[0]
        if hit.size == 0:
            continue
        k = hit[0]
        if k == 0:
            out[i] = 0.0
            continue
        c0, c1 = c[k - 1], c[k]
        frac = (value - c0) / (c1 - c0) if c1 != c0 else 0.0
        out[i] = zz[k - 1] + frac * (zz[k] - zz[k - 1])
    return out.reshape(prof.shape[:-1])


def resolve_variable(ds_vars: dict, spec: dict) -> str | None:
    """Find a spec's variable in a dataset: exact name, case-insensitive name, then `match` in attributes."""
    if spec["name"] in ds_vars:
        return spec["name"]
    lower = {k.lower(): k for k in ds_vars}
    if spec["name"].lower() in lower:
        return lower[spec["name"].lower()]
    if spec.get("match"):
        m = spec["match"].lower()
        for k, attrs in ds_vars.items():
            text = " ".join([k, str(attrs.get("standard_name", "")), str(attrs.get("long_name", ""))]).lower()
            if m in text:
                return k
    return None


def current_stats(u, v):
    """Speed statistics over pixels, plus the direction the area-mean current flows towards (deg from north)."""
    u = np.asarray(u, dtype="float64")
    v = np.asarray(v, dtype="float64")
    speed = np.hypot(u, v)
    stats = area_stats(speed)
    ok = np.isfinite(u) & np.isfinite(v)
    direction = None
    if ok.any():
        direction = float(np.degrees(np.arctan2(u[ok].mean(), v[ok].mean())) % 360.0)
    return stats, direction


class _CopernicusBase(Source):
    required_env = ("COPERNICUSMARINE_SERVICE_USERNAME", "COPERNICUSMARINE_SERVICE_PASSWORD")
    _units_logged: set = set()

    def _open(self, bbox, start: date, end: date, end_hour=23):
        import copernicusmarine  # imported lazily so tests run without it

        lon_min, lat_min, lon_max, lat_max = bbox
        kwargs = dict(
            dataset_id=self.cfg["dataset_id"],
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            coordinates_selection_method="inside",
        )
        if self.cfg.get("max_depth") is not None:
            # model products have many depth levels; only fetch the surface layer
            kwargs.update(minimum_depth=0, maximum_depth=float(self.cfg["max_depth"]))
        names = [s["name"] for s in variable_specs(self.cfg)]
        # Open lazily (metadata only, no download). If a variable name is wrong, open everything
        # and resolve names from the dataset itself rather than failing.
        try:
            ds = copernicusmarine.open_dataset(variables=names, **kwargs)
        except Exception as e:
            if "variable" not in str(e).lower():
                raise
            ds = copernicusmarine.open_dataset(**kwargs)
        tname = _coord(ds, "time")
        times = pd.to_datetime(ds[tname].values)
        t0 = pd.Timestamp(datetime(start.year, start.month, start.day))
        t1 = pd.Timestamp(datetime(end.year, end.month, end.day, end_hour, 59))
        if len(times) == 0 or t1 < times.min() or t0 > times.max():
            ds.close()
            return None
        return ds.sel({tname: slice(max(t0, times.min()), min(t1, times.max()))})

    def _resolve(self, ds):
        ds_vars = {k: dict(ds[k].attrs) for k in ds.data_vars}
        resolved = []
        for spec in variable_specs(self.cfg):
            actual = resolve_variable(ds_vars, spec)
            if actual is None:
                raise SourceError(f"{self.cfg['dataset_id']}: no variable '{spec['name']}'. "
                                  f"Available: {sorted(ds_vars)}")
            resolved.append({**spec, "actual": actual})
            key = (self.code, actual)
            if key not in _CopernicusBase._units_logged:
                _CopernicusBase._units_logged.add(key)
                print(f"[{self.code}]   {actual} -> {spec['code']}  units: {ds_vars[actual].get('units', '?')}")
        return resolved

    def _profile(self, ds, var, scale=None):
        """Times, lats, lons, depths (positive down, ascending) and values [time, depth, lat, lon]."""
        da = ds[var]
        tname = _coord(ds, "time")
        la, lo = _coord(ds, "latitude", "lat"), _coord(ds, "longitude", "lon")
        zdims = [d for d in da.dims if d not in (tname, la, lo)]
        if len(zdims) != 1:
            raise SourceError(f"{var}: expected one vertical dimension, found {zdims}")
        zd = zdims[0]
        da = da.transpose(tname, zd, la, lo)
        vals = np.asarray(da.values, dtype="float64")
        depths = np.abs(np.asarray(ds[zd].values, dtype="float64"))      # 'elevation' is negative
        order = np.argsort(depths)
        vals, depths = vals[:, order], depths[order]
        if scale:
            vals = vals * float(scale)
        times = pd.to_datetime(ds[tname].values).to_pydatetime()
        return times, np.asarray(ds[la].values), np.asarray(ds[lo].values), depths, vals

    def _depth_fields(self, ds, spec):
        """{code: values [time, lat, lon]} for a spec with depths / interface."""
        times, lats, lons, depths, prof = self._profile(ds, spec["actual"], spec.get("scale"))
        vmeta = self.config.get("variables", {})
        fields = {}
        if spec.get("surface_code"):
            c = spec["surface_code"]
            fields[c] = mask_valid(prof[:, 0], vmeta.get(c, {}).get("valid_range"))
        for d in spec.get("depths") or []:
            c = f"{spec['code']}_{int(d)}m"
            k = nearest_level(depths, d)
            if abs(depths[k] - d) > max(3.0, 0.3 * d):
                raise SourceError(f"{spec['actual']}: no model level near {d} m (nearest {depths[k]:.1f} m); "
                                  f"is max_depth deep enough?")
            fields[c] = mask_valid(prof[:, k], vmeta.get(c, {}).get("valid_range"))
        if spec.get("interface"):
            ic = spec["interface"]
            iso = isoline_depth(np.moveaxis(prof, 1, -1), depths, float(ic["value"]))   # [time, lat, lon]
            fields[ic["code"]] = mask_valid(iso, vmeta.get(ic["code"], {}).get("valid_range"))
        return times, lats, lons, fields

    def _values(self, ds, var, code, scale=None):
        da = ds[var]
        tname = _coord(ds, "time")
        la, lo = _coord(ds, "latitude", "lat"), _coord(ds, "longitude", "lon")
        # keep only the level nearest the surface of any extra dimension (e.g. depth or elevation)
        for d in list(da.dims):
            if d not in (tname, la, lo):
                idx = 0
                if d in ds.coords:
                    idx = int(np.argmin(np.abs(np.asarray(ds[d].values, dtype="float64"))))
                da = da.isel({d: idx})
        da = da.transpose(tname, la, lo)
        vals = np.asarray(da.values, dtype="float64")
        if var in (self.cfg.get("kelvin_to_celsius") or []):
            if np.isfinite(vals).any() and np.nanmedian(vals) > 200:
                vals = vals - 273.15                                    # some products already deliver Celsius
        if scale:
            vals = vals * float(scale)                                  # unit conversion set in config.yaml
        # drop physically impossible pixels (e.g. unflagged coastal artefacts) before any statistics
        vals = mask_valid(vals, self.config.get("variables", {}).get(code, {}).get("valid_range"))
        times = pd.to_datetime(ds[tname].values).to_pydatetime()
        return times, np.asarray(ds[la].values), np.asarray(ds[lo].values), vals


def _area_slices(lats, lons, bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    iy = np.where((lats >= lat_min) & (lats <= lat_max))[0]
    ix = np.where((lons >= lon_min) & (lons <= lon_max))[0]
    if iy.size == 0 or ix.size == 0:
        # area smaller than a grid cell: use the cell at its centre
        iy = np.array([int(np.argmin(np.abs(lats - (lat_min + lat_max) / 2)))])
        ix = np.array([int(np.argmin(np.abs(lons - (lon_min + lon_max) / 2)))])
    return slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1)


class CopernicusGrid(_CopernicusBase):
    """Area statistics over each configured area for gridded daily products.

    With `derive: currents` and variables for eastward (uo) and northward (vo) velocity,
    it stores current speed (area statistics) and the direction the mean current flows towards.
    """

    def fetch(self, start: date, end: date):
        ds = self._open(self.union_bbox(), start, end)
        if ds is None:
            return []
        out = []
        try:
            specs = self._resolve(ds)
            if self.cfg.get("derive") == "currents":
                return self._currents(ds, specs)
            for spec in specs:
                if spec.get("depths") or spec.get("interface"):
                    times, lats, lons, fields = self._depth_fields(ds, spec)
                else:
                    times, lats, lons, vals = self._values(ds, spec["actual"], spec["code"], spec.get("scale"))
                    fields = {spec["code"]: vals}
                for code, vals in fields.items():
                    for acode, area in self.areas().items():
                        sy, sx = _area_slices(lats, lons, area["bbox"])
                        sub = vals[:, sy, sx]
                        for k, t in enumerate(times):
                            out.append(Observation(self.code, code, acode, t, **area_stats(sub[k])))
        finally:
            ds.close()
        return out

    def _currents(self, ds, specs):
        by_code = {s["code"]: s for s in specs}
        try:
            u_spec, v_spec = by_code["current_u"], by_code["current_v"]
        except KeyError:
            raise SourceError("derive: currents needs variables mapped to current_u and current_v")
        times, lats, lons, u = self._values(ds, u_spec["actual"], "current_u")
        _, _, _, v = self._values(ds, v_spec["actual"], "current_v")
        out = []
        for acode, area in self.areas().items():
            sy, sx = _area_slices(lats, lons, area["bbox"])
            for k, t in enumerate(times):
                stats, direction = current_stats(u[k, sy, sx], v[k, sy, sx])
                out.append(Observation(self.code, "current_speed", acode, t, **stats))
                out.append(Observation(self.code, "current_dir", acode, t, direction,
                                       n_valid=stats["n_valid"], n_total=stats["n_total"]))
        return out


class CopernicusPoint(_CopernicusBase):
    """Hourly model series at the nearest valid sea cell to each configured point."""

    def fetch(self, start: date, end: date):
        extra = int(self.cfg.get("forecast_days") or self.cfg.get("include_forecast_days") or 0)
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
                for spec in self._resolve(ds):
                    times, lats, lons, vals = self._values(ds, spec["actual"], spec["code"], spec.get("scale"))
                    if vals.size == 0:
                        continue
                    if cell is None:
                        valid = np.isfinite(vals).any(axis=0)
                        cell = nearest_valid_cell(lats, lons, valid, p["lat"], p["lon"], max_km=15)
                        if cell is None:
                            raise SourceError(f"No sea cell within 15 km of {pcode} in {self.cfg['dataset_id']}")
                    i, j, _ = cell
                    for t, val in zip(times, vals[:, i, j]):
                        out.append(Observation(self.code, spec["code"], pcode, t, float(val),
                                               n_valid=int(np.isfinite(val)), n_total=1))
            finally:
                ds.close()
        return out
