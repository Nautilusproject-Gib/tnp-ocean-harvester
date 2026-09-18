"""NEMO sightings: reading, cleaning, and joining to the sea conditions.

The source is either a CSV committed to the repo or the NEMO API; everything after that is identical,
so swapping one for the other is a config change.

Privacy rules applied here, not later:
  - observer names, user IDs and free-text notes never reach anything published;
  - notes are kept only in the TNP-only file, where they are needed to verify strandings;
  - published positions are rounded to a grid cell, and sensitive species are rounded harder.

Times: NEMO records local Gibraltar time, and midnight means the time of day was not known.
Daily conditions are matched on the local date (so a 00:30 sighting belongs to the night before);
anything needing the hour is matched on the exact UTC instant, and skipped when the time is unknown.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import derived as dv

LOCAL_TZ = "Europe/Gibraltar"
DROP_COLUMNS = ("user", "observer", "notes", "email", "device", "phone")


def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


# ------------------------------------------------------------------------------------------------
# Reading and cleaning
# ------------------------------------------------------------------------------------------------
COLUMN_ALIASES = {
    "id": "record_id", "record_id": "record_id",
    "reported": "local_time", "sighted": "local_time", "datetime": "local_time", "date": "local_time",
    "parent": "group", "group": "nemo_group", "category": "group",
    "species": "species", "user": "user", "notes": "notes", "comment": "notes",
    "lat": "lat", "latitude": "lat", "lon": "lon", "lng": "lon", "longitude": "lon",
    "verified": "verified", "visibility": "visibility", "condition": "condition",
}


def _col(df: pd.DataFrame, name: str, default="") -> pd.Series:
    """A column as a Series, or a filled Series when the export does not carry it."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def read_records(text_or_rows, time_format: str | None = "%d/%m/%Y %H:%M") -> pd.DataFrame:
    """CSV text (or a list of dicts from the API) -> a frame with our column names."""
    if isinstance(text_or_rows, str):
        import csv
        import io
        # The NEMO export carries more fields than header names (the last column has no title), which
        # pandas handles by shifting columns or dropping data. Read the header, count the real fields,
        # and name the extras before parsing.
        rows = list(csv.reader(io.StringIO(text_or_rows)))
        if not rows:
            return pd.DataFrame(columns=["local_time"])
        header = [h.strip().lstrip("\ufeff") for h in rows[0]]
        width = max(len(r) for r in rows[1:]) if len(rows) > 1 else len(header)
        extras = ["visibility", "condition", "extra3"][:max(0, width - len(header))]
        df = pd.DataFrame(
            [r + [""] * (len(header + extras) - len(r)) for r in rows[1:]],
            columns=header + extras).replace("", np.nan)
    else:
        df = pd.DataFrame(list(text_or_rows))
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    # the export's header is one name short, so the last column arrives unnamed
    df = df.rename(columns={c: COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower())
                            for c in df.columns})
    if "visibility" not in df.columns:
        extra = [c for c in df.columns if c.startswith("unnamed")]
        if extra:
            df = df.rename(columns={extra[0]: "visibility"})
    if time_format:
        df["local_time"] = pd.to_datetime(df["local_time"], format=time_format, errors="coerce")
    else:
        df["local_time"] = pd.to_datetime(df["local_time"], errors="coerce")
    return df.dropna(subset=["local_time"])


def to_utc(local: pd.Series) -> pd.Series:
    """Local Gibraltar time -> UTC. The hour that repeats each October is read as the first
    occurrence (summer time); the hour that does not exist each March is nudged forward."""
    aware = local.dt.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent="shift_forward")
    return aware.dt.tz_convert("UTC").dt.tz_localize(None)


def assign_area(lat, lon, areas: dict, fallback_km: float = 40.0):
    """(area, exact) for a position: the smallest area containing it, else the nearest area centre.

    A record just outside every box still happened in water the dashboard measures, so it is matched
    to the nearest area and marked as not exact rather than thrown away.
    """
    from .stats import haversine_km
    best, best_size = None, np.inf
    for code, a in areas.items():
        lon_min, lat_min, lon_max, lat_max = a["bbox"]
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            size = (lon_max - lon_min) * (lat_max - lat_min)
            if size < best_size:
                best, best_size = code, size
    if best:
        return best, True
    near, dist = None, np.inf
    for code, a in areas.items():
        lon_min, lat_min, lon_max, lat_max = a["bbox"]
        d = haversine_km(lat, lon, (lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
        if d < dist:
            near, dist = code, d
    return (near, False) if dist <= fallback_km else (None, False)


def grid_cell(lat, lon, km: float = 1.0) -> str:
    """Position rounded to a grid cell, for publishing."""
    step = km / 111.0
    return f"{round(lat / step) * step:.3f},{round(lon / step) * step:.3f}"


def clean(df: pd.DataFrame, config: dict, bbox=None) -> pd.DataFrame:
    """Normalise records: times, area, flags, effort counts, and personal fields removed."""
    areas = config.get("areas", {})
    out = pd.DataFrame(index=df.index)
    out["record_id"] = _col(df, "record_id", None)
    out["local_time"] = df["local_time"]
    out["time_known"] = ~((df["local_time"].dt.hour == 0) & (df["local_time"].dt.minute == 0))
    out["utc_time"] = to_utc(df["local_time"])
    out["local_date"] = df["local_time"].dt.date.astype(str)
    out["species"] = (_col(df, "species", "Unknown").astype(str).str.strip()
                      .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown", "null": "Unknown"}))
    out["group"] = (_col(df, "group", "Unrecorded").astype(str).str.strip()
                    .replace({"": "Unrecorded", "nan": "Unrecorded", "None": "Unrecorded",
                                "-": "Unrecorded", "null": "Unrecorded"}))   # legacy records carry "null"
    out["lat"] = pd.to_numeric(_col(df, "lat", np.nan), errors="coerce")
    out["lon"] = pd.to_numeric(_col(df, "lon", np.nan), errors="coerce")
    out["verified"] = _col(df, "verified").astype(str).str.strip().str.lower().eq("yes")
    out["public"] = ~_col(df, "visibility").astype(str).str.strip().str.lower().eq("hidden")
    out["condition"] = _col(df, "condition").astype(str).str.strip().str.lower().replace("nan", "")
    out["notes"] = _col(df, "notes").fillna("").astype(str)

    # effort: how many records and how many different people that day, counted before identities go
    if "user" in df.columns:
        by_day = df.assign(d=out["local_date"]).groupby("d")["user"]
        out["day_records"] = out["local_date"].map(by_day.size())
        out["day_contributors"] = out["local_date"].map(by_day.nunique())
    else:
        counts = out.groupby("local_date").size()
        out["day_records"] = out["local_date"].map(counts)
        out["day_contributors"] = np.nan

    out = out.dropna(subset=["lat", "lon"])
    if bbox:
        lon_min, lat_min, lon_max, lat_max = bbox
        inside = out["lat"].between(lat_min, lat_max) & out["lon"].between(lon_min, lon_max)
        out = out[inside]
    placed = [assign_area(la, lo, areas) for la, lo in zip(out["lat"], out["lon"])]
    out["area"] = [p[0] for p in placed]
    out["area_exact"] = [p[1] for p in placed]
    out["cell_1km"] = [grid_cell(la, lo) for la, lo in zip(out["lat"], out["lon"])]
    return out.sort_values("utc_time").reset_index(drop=True)


def tag_species(df: pd.DataFrame, species_cfg: dict) -> pd.DataFrame:
    """Add the flags from species.yaml: gelatinous type, invasive status, sting risk, sensitivity."""
    gel = species_cfg.get("gelatinous", {}) or {}
    drifters = set(gel.get("drifter") or [])
    water = set(gel.get("water_column") or [])
    invasive = species_cfg.get("invasive", {}) or {}
    stinging = set(species_cfg.get("stinging") or [])
    sensitive = set(species_cfg.get("sensitive") or [])
    df = df.copy()
    df["gelatinous"] = np.where(df["species"].isin(drifters), "drifter",
                                np.where(df["species"].isin(water), "water_column", ""))
    df["invasive"] = df["species"].map(invasive).fillna("")
    df["stinging"] = df["species"].isin(stinging)
    df["sensitive"] = df["species"].isin(sensitive)
    return df


# ------------------------------------------------------------------------------------------------
# Conditions at the time of each sighting
# ------------------------------------------------------------------------------------------------
def tide_state(times, fit: dict | None, step_minutes: int = 10) -> pd.DataFrame:
    """Height above mean level, whether the tide is rising, and hours from the nearest high water.

    The tide is predicted once across the whole span rather than per record, which keeps this quick
    even for thousands of sightings.
    """
    cols = ["tide_height", "tide_rising", "hours_from_high"]
    # nanoseconds throughout: a timezone conversion can leave an index in microseconds, and mixing
    # the two silently puts every sighting in the wrong decade
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(times))).values.astype("datetime64[ns]"))
    if fit is None or len(idx) == 0:
        return pd.DataFrame(index=range(len(idx)), columns=cols)
    grid = pd.date_range(idx.min().floor("h") - pd.Timedelta(days=1),
                         idx.max().ceil("h") + pd.Timedelta(days=1), freq=f"{step_minutes}min")
    pred = dv.predict_tide(fit, grid, include_mean=False)
    highs = np.array([t.value for t, kind, _ in dv.tide_extremes(pred) if kind == "high"], dtype="int64")
    ns = idx.values.astype("datetime64[ns]").astype("int64")
    slot = np.searchsorted(grid.values.astype("datetime64[ns]").astype("int64"), ns)
    slot = np.clip(slot, 1, len(grid) - 2)
    height = pred.values[slot]
    rising = pred.values[slot + 1] > pred.values[slot]
    if highs.size:
        pos = np.clip(np.searchsorted(highs, ns), 1, highs.size - 1)
        before, after = highs[pos - 1], highs[pos]
        nearest = np.where(np.abs(ns - before) <= np.abs(after - ns), before, after)
        hours = (ns - nearest) / 3.6e12
    else:
        hours = np.full(len(idx), np.nan)
    return pd.DataFrame({"tide_height": np.round(height, 2), "tide_rising": rising,
                         "hours_from_high": np.round(hours, 1)}, index=range(len(idx)))


def attach_conditions(records: pd.DataFrame, merged_by_key: dict, variables: list,
                      tide_fit: dict | None = None) -> pd.DataFrame:
    """Add a column per configured variable, matched on the local date and the record's area.

    variables: [{variable, location}] where location "area" means the record's own sea area.
    """
    out = records.copy()
    dates = pd.to_datetime(out["local_date"])
    for spec in variables:
        var, loc = spec["variable"], spec.get("location", "area")
        col = var if loc == "area" else f"{var}"
        values, anomalies = [], []
        for when, area in zip(dates, out["area"]):
            key = (var, area if loc == "area" else loc)
            m = merged_by_key.get(key)
            v = a = np.nan
            if m is not None and when in m.index:
                v = float(m.loc[when, "value"])
                same_doy = m[(m.index.dayofyear >= when.dayofyear - 7) & (m.index.dayofyear <= when.dayofyear + 7)]
                if len(same_doy) >= 5:
                    a = v - float(same_doy["value"].mean())
            values.append(_r(v))
            anomalies.append(_r(a, 2))
        out[col] = values
        if spec.get("anomaly", True):
            out[f"{col}_anomaly"] = anomalies
    if tide_fit is not None:
        known = out["time_known"].values
        tides = tide_state(pd.DatetimeIndex(out["utc_time"]), tide_fit)
        for c in tides.columns:
            out[c] = np.where(known, tides[c].values, None)
    moon = [dv.moon_illumination(t) for t in out["utc_time"]]
    out["moon_illumination_pct"] = [round(m * 100) for m in moon]
    return out


# ------------------------------------------------------------------------------------------------
# Indices and watches
# ------------------------------------------------------------------------------------------------
def daily_index(records: pd.DataFrame, mask, min_share_records: int = 3):
    """Daily count of the selected records, and their share of everything reported that day.

    The share is the effort-robust index: it holds up when reporting is busy or quiet.
    """
    if records.empty:
        return pd.DataFrame(columns=["count", "total", "share"])
    day = pd.to_datetime(records["local_date"])
    total = day.value_counts().sort_index()
    sel = day[mask.values if hasattr(mask, "values") else mask].value_counts().sort_index()
    df = pd.DataFrame({"count": sel, "total": total}).fillna({"count": 0})
    df["share"] = np.where(df["total"] >= min_share_records, df["count"] / df["total"], np.nan)
    return df.sort_index()


def bloom_events(index: pd.DataFrame, min_count: int = 3, min_share: float = 0.4,
                 min_days: int = 2, max_gap_days: int = 1):
    """Days that are busy for this group both in number and in share, run together into events."""
    if index.empty:
        return [], None
    full = pd.date_range(index.index.min(), index.index.max(), freq="D")
    counts = index["count"].reindex(full).fillna(0)
    share = index["share"].reindex(full)
    flags = (counts >= min_count) & ((share >= min_share) | share.isna())
    events = []
    for a, b in dv.runs(flags.values, min_days, max_gap_days):
        window = counts.iloc[a:b + 1]
        k = int(np.argmax(window.values))
        events.append({"start": full[a].strftime("%Y-%m-%d"), "end": full[b].strftime("%Y-%m-%d"),
                       "days": int(b - a + 1), "records": int(window.sum()),
                       "peak_date": full[a + k].strftime("%Y-%m-%d"), "peak_records": int(window.max()),
                       "peak_share": _r(share.iloc[a + k], 2)})
    last = len(full) - 1
    status = {"date": full[last].strftime("%Y-%m-%d"), "records": int(counts.iloc[last]),
              "share": _r(share.iloc[last], 2), "state": "none"}
    if events and events[-1]["end"] == status["date"]:
        status.update(state="bloom", since=events[-1]["start"], days=events[-1]["days"])
    elif flags.iloc[last]:
        status.update(state="elevated", days=1)
    return events, status


def invasive_watch(records: pd.DataFrame, species_cfg: dict) -> list:
    """One entry per watchlist species: status, first report, first verified record, records per year.

    Only a verified record sets first_verified, because that is what goes public.
    """
    listed = species_cfg.get("invasive", {}) or {}
    out = []
    for species, status in listed.items():
        sub = records[records["species"] == species]
        ver = sub[sub["verified"]]
        entry = {"species": species, "status": status, "records": int(len(sub)),
                 "first_reported": sub["local_date"].min() if len(sub) else None,
                 "last_reported": sub["local_date"].max() if len(sub) else None,
                 "first_verified": ver["local_date"].min() if len(ver) else None,
                 "verified_records": int(len(ver)),
                 "by_year": {str(y): int(n) for y, n in
                             pd.to_datetime(sub["local_date"]).dt.year.value_counts().sort_index().items()},
                 "areas": sorted({a for a in sub["area"].dropna().unique()}),
                 "cells": int(sub["cell_1km"].nunique())}
        out.append(entry)
    order = {"established": 0, "emerging": 1, "watch": 2}
    return sorted(out, key=lambda e: (order.get(e["status"], 3), -e["records"]))


STRANDING_WORDS = r"dead|stranded|strand|washed up|washed-up|beached|carcass|decompos"


def stranding_candidates(records: pd.DataFrame) -> pd.DataFrame:
    """Records that look like strandings: the condition field if NEMO has one, otherwise the notes."""
    cond = records["condition"].astype(str).str.lower()
    by_condition = cond.str.contains("dead|strand|carcass", regex=True, na=False)
    notes = records["notes"].astype(str).str.lower()
    by_notes = notes.str.contains(STRANDING_WORDS, regex=True, na=False)
    out = records[by_condition | by_notes].copy()
    out["evidence"] = np.where(by_condition[by_condition | by_notes], "condition field", "notes")
    # which word matched, so the list is usable without carrying the note itself
    import re
    found = notes[by_condition | by_notes].str.findall(STRANDING_WORDS)
    out["matched_word"] = [", ".join(sorted(set(f))) if isinstance(f, list) else "" for f in found]
    return out


def match_strandings(candidates: pd.DataFrame, log: pd.DataFrame, days: int = 2, km: float = 3.0) -> dict:
    """Reconcile app records against the TNP strandings log: matched, log only, app only."""
    if log is None or log.empty:
        return {"matched": [], "log_only": [], "app_only": candidates["record_id"].tolist()}
    log = log.copy()
    log["date"] = pd.to_datetime(log["date"], errors="coerce")
    matched, used = [], set()
    for _, rec in candidates.iterrows():
        when = pd.Timestamp(rec["local_date"])
        near = log[(log["date"] >= when - pd.Timedelta(days=days)) & (log["date"] <= when + pd.Timedelta(days=days))]
        if "species" in near.columns:
            near = near[near["species"].astype(str).str.lower().str[:6] == str(rec["species"]).lower()[:6]]
        if {"lat", "lon"} <= set(near.columns) and len(near):
            from .stats import haversine_km
            near = near[[haversine_km(rec["lat"], rec["lon"], la, lo) <= km
                         for la, lo in zip(near["lat"], near["lon"])]]
        if len(near):
            i = near.index[0]
            used.add(i)
            matched.append({"record_id": rec["record_id"], "log_index": int(i), "date": rec["local_date"],
                            "species": rec["species"]})
    app_only = [r["record_id"] for _, r in candidates.iterrows()
                if r["record_id"] not in {m["record_id"] for m in matched}]
    log_only = [int(i) for i in log.index if i not in used]
    return {"matched": matched, "log_only": log_only, "app_only": app_only}
