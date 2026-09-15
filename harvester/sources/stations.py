"""Station and API sources that need no heavy scientific libraries.

- NOAA NCEI Integrated Surface Database (ISD / "global-hourly"), Gibraltar LXGB
- Open-Meteo historical weather (ERA5) and air quality (CAMS)
- NASA AERONET sun photometer daily averages
"""
from __future__ import annotations

import io
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ..db import Observation
from .base import Source

# ----------------------------------------------------------------------------------------------
# NOAA ISD
# ----------------------------------------------------------------------------------------------
ISD_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"
_BAD_QC = set("2367")


def _qc_ok(q: str) -> bool:
    return q not in _BAD_QC


def parse_isd_wnd(s):
    """'320,1,N,0051,1' -> (direction deg or None, speed m/s or None)."""
    if not isinstance(s, str) or s.count(",") < 4:
        return None, None
    d, dq, typ, sp, sq = s.split(",")[:5]
    direction = None if d == "999" or not _qc_ok(dq) else float(d)
    if sp == "9999" or not _qc_ok(sq):
        speed = None
    else:
        speed = float(sp) / 10.0
    if typ == "C":  # calm
        speed, direction = 0.0, None
    elif typ == "V":  # variable direction
        direction = None
    return direction, speed


def parse_isd_signed(s, missing="+9999", scale=10.0):
    """'+0215,1' -> 21.5"""
    if not isinstance(s, str) or "," not in s:
        return None
    v, q = s.split(",")[:2]
    if v in (missing, "9999", "+99999", "99999") or not _qc_ok(q):
        return None
    return float(v) / scale


def parse_isd_gust(s):
    """OC1 '0123,1' -> 12.3 m/s"""
    if not isinstance(s, str) or "," not in s:
        return None
    v, q = s.split(",")[:2]
    if v == "9999" or not _qc_ok(q):
        return None
    return float(v) / 10.0


def parse_isd_precip(s):
    """AA1 '12,0004,9,1' -> (12 hours, 0.4 mm)"""
    if not isinstance(s, str) or s.count(",") < 3:
        return None, None
    hh, depth, _cond, q = s.split(",")[:4]
    if hh == "99" or depth == "9999" or not _qc_ok(q):
        return None, None
    return int(hh), float(depth) / 10.0


def parse_isd_csv(text: str, source: str, location: str, start: date | None = None, end: date | None = None):
    df = pd.read_csv(io.StringIO(text), dtype=str, low_memory=False)
    if df.empty:
        return []
    df["t"] = pd.to_datetime(df["DATE"], errors="coerce").dt.round("h")
    df = df.dropna(subset=["t"])
    if start is not None:
        df = df[df["t"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["t"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
    records: dict[tuple, float] = {}

    def put(t, var, v):
        if v is not None and np.isfinite(v):
            records[(t, var)] = v  # later reports in the same hour win

    for row in df.itertuples(index=False):
        t = row.t.to_pydatetime()
        d = getattr(row, "WND", None)
        direction, speed = parse_isd_wnd(d)
        put(t, "wind_speed", speed)
        put(t, "wind_dir", direction)
        put(t, "air_temp", parse_isd_signed(getattr(row, "TMP", None)))
        if "OC1" in df.columns:
            put(t, "wind_gust", parse_isd_gust(getattr(row, "OC1", None)))
        for col in ("AA1", "AA2"):
            if col in df.columns:
                hours, mm = parse_isd_precip(getattr(row, col, None))
                if hours in (1, 3, 6, 12, 24):
                    put(t, f"precip_{hours}h", mm)
        if "SLP" in df.columns:
            put(t, "slp", parse_isd_signed(getattr(row, "SLP", None), missing="99999"))
    return [Observation(source, var, location, t, v, n_valid=1, n_total=1)
            for (t, var), v in sorted(records.items())]


class NceiIsd(Source):
    def fetch(self, start: date, end: date):
        out = []
        for year in range(start.year, end.year + 1):
            r = self.http_get(ISD_URL.format(year=year, station=self.cfg["station_id"]), timeout=300)
            if r.status_code == 404:
                continue
            out.extend(parse_isd_csv(r.text, self.code, self.cfg["point"], start, end))
        return out


# ----------------------------------------------------------------------------------------------
# Open-Meteo
# ----------------------------------------------------------------------------------------------
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"


def parse_openmeteo_hourly(js: dict, variables: dict, source: str, location: str):
    hourly = js.get("hourly") or {}
    times = hourly.get("time") or []
    out = []
    for api_var, code in variables.items():
        vals = hourly.get(api_var)
        if vals is None:
            continue
        for t, v in zip(times, vals):
            if v is None:
                continue
            out.append(Observation(source, code, location, datetime.fromisoformat(t), float(v), n_valid=1, n_total=1))
    return out


class _OpenMeteo(Source):
    endpoint = OM_ARCHIVE

    def extra_params(self):
        return {}

    def fetch(self, start: date, end: date):
        today = datetime.utcnow().date()
        end = min(end, today)
        out = []
        for pcode in self.cfg["points"]:
            p = self.point(pcode)
            params = {
                "latitude": p["lat"], "longitude": p["lon"],
                "start_date": start.isoformat(), "end_date": end.isoformat(),
                "hourly": ",".join(self.cfg["variables"].keys()),
                "timezone": "GMT",
                **self.extra_params(),
            }
            import os
            if os.environ.get("OPEN_METEO_API_KEY"):
                params["apikey"] = os.environ["OPEN_METEO_API_KEY"]
            r = self.http_get(self.endpoint, params=params, timeout=180)
            if r.status_code >= 400:
                continue
            out.extend(parse_openmeteo_hourly(r.json(), self.cfg["variables"], self.code, pcode))
        return out


class OpenMeteoArchive(_OpenMeteo):
    endpoint = OM_ARCHIVE

    def extra_params(self):
        p = {"wind_speed_unit": "ms", "precipitation_unit": "mm", "temperature_unit": "celsius"}
        if self.cfg.get("model"):
            p["models"] = self.cfg["model"]
        return p


class OpenMeteoAirQuality(_OpenMeteo):
    endpoint = OM_AIR


# ----------------------------------------------------------------------------------------------
# AERONET
# ----------------------------------------------------------------------------------------------
AERONET_URL = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"


def parse_aeronet_daily(text: str, source: str):
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("AERONET_Site")), None)
    if header_idx is None:
        return []
    df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])), low_memory=False)
    df = df.replace(-999.0, np.nan)
    date_col = next(c for c in df.columns if c.startswith("Date("))
    out = []
    for _, row in df.iterrows():
        site = str(row["AERONET_Site"]).strip()
        loc = f"aeronet_{site.lower()}"
        t = datetime.strptime(str(row[date_col]).strip(), "%d:%m:%Y")
        for col, code in (("AOD_500nm", "aeronet_aod_500"), ("440-870_Angstrom_Exponent", "aeronet_ae_440_870")):
            if col in df.columns and pd.notna(row[col]):
                out.append(Observation(source, code, loc, t, float(row[col]), n_valid=1, n_total=1))
    return out


class Aeronet(Source):
    def fetch(self, start: date, end: date):
        out = []
        for site in self.cfg["sites"]:
            params = {
                "site": site,
                "year": start.year, "month": start.month, "day": start.day,
                "year2": end.year, "month2": end.month, "day2": end.day,
                "AOD15": 1, "AVG": 20, "if_no_html": 1,
            }
            r = self.http_get(AERONET_URL, params=params, timeout=180)
            out.extend(parse_aeronet_daily(r.text, self.code))
        return out
