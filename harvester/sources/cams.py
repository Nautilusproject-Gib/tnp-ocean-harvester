"""CAMS global reanalysis (EAC4) from the Copernicus Atmosphere Data Store, via `cdsapi`.

Needs a free ADS account: https://ads.atmosphere.copernicus.eu
Accept the EAC4 licence on the dataset page, then set ADS_API_KEY
(the personal access token shown on your ADS profile page).
Requests are queued on the ADS side and can take minutes each.
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..db import Observation
from ..stats import nearest_valid_cell
from .base import Source

ADS_URL = "https://ads.atmosphere.copernicus.eu/api"


class CamsEac4(Source):
    required_env = ("ADS_API_KEY",)

    def fetch(self, start: date, end: date):
        import cdsapi
        import xarray as xr

        p = self.point(self.cfg["point"])
        area = [p["lat"] + 0.8, p["lon"] - 0.8, p["lat"] - 0.8, p["lon"] + 0.8]  # N, W, S, E
        client = cdsapi.Client(url=os.environ.get("ADS_API_URL", ADS_URL), key=os.environ["ADS_API_KEY"],
                               quiet=True, progress=False)
        name_map = {
            "duaod550": "dust_aerosol_optical_depth_550nm",
            "aod550": "total_aerosol_optical_depth_550nm",
        }
        request = {
            "date": [f"{start.isoformat()}/{end.isoformat()}"],
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "variable": [name_map[v] for v in self.cfg["variables"]],
            "area": area,
            "data_format": "netcdf",
        }
        out = []
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "eac4.download"
            try:
                client.retrieve("cams-global-reanalysis-eac4", request).download(str(target))
            except Exception as e:
                if "valid combination" in str(e):
                    # EAC4 runs months behind real time; dates past its end are simply not there yet
                    print(f"[{self.code}]   no EAC4 data for {start} .. {end} (outside the reanalysis period)")
                    return []
                raise
            paths = [target]
            if zipfile.is_zipfile(target):
                with zipfile.ZipFile(target) as z:
                    z.extractall(tmp)
                paths = [Path(tmp) / n for n in z.namelist() if n.endswith(".nc")]
            for path in paths:
                with xr.open_dataset(path) as ds:
                    tname = "valid_time" if "valid_time" in ds.coords else "time"
                    la = "latitude" if "latitude" in ds.coords else "lat"
                    lo = "longitude" if "longitude" in ds.coords else "lon"
                    for var, code in self.cfg["variables"].items():
                        if var not in ds:
                            continue
                        da = ds[var].transpose(tname, la, lo)
                        vals = np.asarray(da.values, dtype="float64")
                        cell = nearest_valid_cell(ds[la].values, ds[lo].values, np.isfinite(vals).any(axis=0),
                                                  p["lat"], p["lon"], max_km=60)
                        if cell is None:
                            continue
                        i, j, _ = cell
                        times = pd.to_datetime(ds[tname].values).to_pydatetime()
                        for t, v in zip(times, vals[:, i, j]):
                            out.append(Observation(self.code, code, self.cfg["point"], t, float(v),
                                                   n_valid=int(np.isfinite(v)), n_total=1))
        return out
