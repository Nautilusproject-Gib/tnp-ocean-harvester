"""Station and API sources that need no heavy scientific libraries.

- NOAA NCEI Integrated Surface Database (ISD / "global-hourly"), Gibraltar LXGB
- Open-Meteo historical weather (ERA5) and air quality (CAMS)
- NASA AERONET sun photometer daily averages
- Open-Meteo weather forecast (recent days and the week ahead)
- IOC Sea Level Station Monitoring Facility tide gauges (Algeciras)
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
# NCEI retired the www.ncei.noaa.gov/data/global-hourly download path on 31 July 2026.
# The same CSV files now live in the NOAA Open Data Dissemination bucket on AWS.
ISD_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{station}.csv"
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
            url = ISD_URL.format(year=year, station=self.cfg["station_id"])
            r = self.http_get(url, timeout=300)
            if r.status_code in (403, 404):
                print(f"[{self.code}]   no file for {year} yet ({r.status_code}) at {url}")
                continue
            out.extend(parse_isd_csv(r.text, self.code, self.cfg["point"], start, end))
        return out


# ----------------------------------------------------------------------------------------------
# Open-Meteo
# ----------------------------------------------------------------------------------------------
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_AIR = "https://air-quality-api.open-meteo.com/v1/air-quality"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"


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


class OpenMeteoForecast(Source):
    """The last few days and the week ahead from Open-Meteo's forecast models (no account).

    Recent days fill the few days ERA5 lags behind; days after today are published as forecasts.
    The requested date range is ignored: every run fetches past_days back and forecast_days ahead.
    """

    def fetch(self, start: date, end: date):
        out = []
        for pcode in self.cfg["points"]:
            p = self.point(pcode)
            params = {
                "latitude": p["lat"], "longitude": p["lon"],
                "hourly": ",".join(self.cfg["variables"].keys()),
                "past_days": int(self.cfg.get("past_days", 7)),
                "forecast_days": int(self.cfg.get("forecast_days", 7)) + 1,     # +1: today counts as a day
                "wind_speed_unit": "ms", "precipitation_unit": "mm", "temperature_unit": "celsius",
                "timezone": "GMT",
            }
            if self.cfg.get("model"):
                params["models"] = self.cfg["model"]
            r = self.http_get(OM_FORECAST, params=params, timeout=120)
            if r.status_code >= 400:
                continue
            out.extend(parse_openmeteo_hourly(r.json(), self.cfg["variables"], self.code, pcode))
        return out


# ----------------------------------------------------------------------------------------------
# IOC Sea Level Station Monitoring Facility
# ----------------------------------------------------------------------------------------------
IOC_URL = "https://www.ioc-sealevelmonitoring.org/service.php"


def parse_ioc_sealevel(records: list, sensors=("rad", "prs", "flt", "enc", "pr1", "bub"),
                       spike_m: float = 0.3, min_per_hour: int = 20) -> pd.Series:
    """IOC minute records [{slevel, stime, sensor}] -> hourly mean sea level (m), spikes removed.

    Uses the first sensor in `sensors` that has data. A reading more than `spike_m` from the
    15-minute running median is dropped. An hour needs `min_per_hour` good readings.
    """
    if not isinstance(records, list) or not records:
        return pd.Series(dtype="float64")                  # the service answers errors with a JSON object
    df = pd.DataFrame(records)
    if not {"slevel", "stime"} <= set(df.columns):
        return pd.Series(dtype="float64")
    if "sensor" in df.columns:
        present = list(df["sensor"].dropna().unique())
        pick = next((x for x in sensors if x in present), present[0] if present else None)
        if pick is not None:
            df = df[df["sensor"] == pick]
    df["t"] = pd.to_datetime(df["stime"], errors="coerce")
    df["v"] = pd.to_numeric(df["slevel"], errors="coerce")
    s = df.dropna(subset=["t", "v"]).drop_duplicates("t").set_index("t")["v"].sort_index()
    s = s[(s > -20) & (s < 20)]
    if s.empty:
        return pd.Series(dtype="float64")
    med = s.rolling("15min", center=True, min_periods=3).median()
    s = s[(s - med).abs() <= spike_m]
    g = s.resample("1h")
    hourly = g.mean().where(g.count() >= min_per_hour)
    return hourly.dropna()


class IocSeaLevel(Source):
    def fetch(self, start: date, end: date):
        import time as _time
        today = datetime.utcnow().date()
        end = min(end, today)
        out = []
        station, point = self.cfg["station"], self.cfg["point"]
        step = int(self.cfg.get("request_days", 7))
        cur = start
        while cur <= end:
            stop = min(end, cur + timedelta(days=step - 1))
            params = {"query": "data", "format": "json", "code": station,
                      "timestart": cur.isoformat(), "timestop": (stop + timedelta(days=1)).isoformat()}
            r = self.http_get(IOC_URL, params=params, timeout=180)
            try:
                records = r.json() if r.status_code < 400 else []
            except ValueError:
                records = []
            hourly = parse_ioc_sealevel(records, tuple(self.cfg.get("sensors") or
                                                        ("rad", "prs", "flt", "enc", "pr1", "bub")))
            hourly = hourly[(hourly.index >= pd.Timestamp(cur)) & (hourly.index < pd.Timestamp(stop + timedelta(days=1)))]
            for t, v in hourly.items():
                out.append(Observation(self.code, "sea_level_hourly", point, t.to_pydatetime(), float(v),
                                       n_valid=1, n_total=1))
            cur = stop + timedelta(days=1)
            _time.sleep(float(self.cfg.get("pause_seconds", 1.0)))       # be gentle with a free service
        return out


# ----------------------------------------------------------------------------------------------
# AERONET
# ----------------------------------------------------------------------------------------------
AERONET_URL = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"


def parse_aeronet_daily(text: str, source: str):
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("AERONET_Site")), None)
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
            rows = parse_aeronet_daily(r.text, self.code)
            if not rows:
                snippet = " | ".join(r.text.strip().splitlines()[:6])[:400]
                print(f"[{self.code}]   no rows for site {site}; server said: {snippet}")
            out.extend(rows)
        return out
