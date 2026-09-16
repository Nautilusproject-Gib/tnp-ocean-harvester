"""NASA Ocean Biology DAAC Level-3 mapped products (e.g. PAR) via `earthaccess`.

Needs a free NASA Earthdata login: https://urs.earthdata.nasa.gov/users/new
Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np

from ..db import Observation
from ..stats import area_stats, mask_valid
from .base import Source

_logged_in = False


def _force_ipv4():
    """GitHub-hosted runners have no IPv6 route; NASA hosts publish IPv6 addresses, which can
    surface as '[Errno 101] Network is unreachable'. Make requests/urllib3 use IPv4 only."""
    import socket
    import urllib3.util.connection as urllib3_conn
    urllib3_conn.allowed_gai_family = lambda: socket.AF_INET


def _login():
    global _logged_in
    _force_ipv4()
    import earthaccess
    if not _logged_in:
        auth = earthaccess.login(strategy="environment")
        if not auth or not getattr(auth, "authenticated", True):
            raise RuntimeError("Earthdata login failed - check EARTHDATA_USERNAME / EARTHDATA_PASSWORD")
        _logged_in = True
    return earthaccess


def granule_date(granule) -> date:
    """Start date of a CMR granule (dict-like earthaccess DataGranule)."""
    umm = granule["umm"]
    begin = umm["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
    return datetime.fromisoformat(begin.replace("Z", "+00:00")).date()


class NasaL3m(Source):
    required_env = ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD")

    def fetch(self, start: date, end: date):
        import xarray as xr

        earthaccess = _login()
        lon_min, lat_min, lon_max, lat_max = self.union_bbox()
        results = earthaccess.search_data(
            short_name=self.cfg["short_name"],
            temporal=(start.isoformat(), f"{end.isoformat()}T23:59:59"),
            bounding_box=(lon_min, lat_min, lon_max, lat_max),
            granule_name=self.cfg.get("granule_pattern", "*.DAY.*"),
        )
        if not results:
            return []
        var = self.cfg.get("variable", "par")
        code = self.cfg.get("code", var)
        out = []
        files = earthaccess.open(results)
        for g, f in zip(results, files):
            d = granule_date(g)
            with xr.open_dataset(f, engine="h5netcdf") as ds:
                lat = ds["lat"].values
                lon = ds["lon"].values
                iy = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                ix = np.where((lon >= lon_min) & (lon <= lon_max))[0]
                block = ds[var].isel(lat=slice(iy.min(), iy.max() + 1),
                                     lon=slice(ix.min(), ix.max() + 1)).load()
                blat, blon, vals = block["lat"].values, block["lon"].values, block.values.astype("float64")
                vals = mask_valid(vals, self.config.get("variables", {}).get(code, {}).get("valid_range"))
            for acode, area in self.areas().items():
                a_lon_min, a_lat_min, a_lon_max, a_lat_max = area["bbox"]
                jy = np.where((blat >= a_lat_min) & (blat <= a_lat_max))[0]
                jx = np.where((blon >= a_lon_min) & (blon <= a_lon_max))[0]
                if jy.size == 0 or jx.size == 0:
                    # area smaller than one pixel: use the pixel containing the centre
                    cy = (a_lat_min + a_lat_max) / 2
                    cx = (a_lon_min + a_lon_max) / 2
                    jy = np.array([int(np.argmin(np.abs(blat - cy)))])
                    jx = np.array([int(np.argmin(np.abs(blon - cx)))])
                sub = vals[jy.min():jy.max() + 1, jx.min():jx.max() + 1]
                out.append(Observation(self.code, code, acode, datetime(d.year, d.month, d.day), **area_stats(sub)))
        return out
