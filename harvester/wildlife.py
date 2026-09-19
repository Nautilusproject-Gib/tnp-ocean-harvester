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
    "reported": "local_time", "sighted": "local_time", "datetime": "local_time",
    "local_time": "local_time", "date": "local_time",
    "parent": "group", "group": "nemo_group", "category": "group",
    "species": "species", "user": "user", "notes": "notes", "comment": "notes",
    "lat": "lat", "latitude": "lat", "lon": "lon", "lng": "lon", "longitude": "lon",
    "verified": "verified", "visibility": "visibility", "public": "visibility",
    "condition": "condition",
}

# An export can carry several columns that all look like the sighting time: a full timestamp, a bare
# date, sometimes both. Keep the richest one and drop the rest, in this order.
TIME_PREFERENCE = ("reported", "sighted", "datetime", "local_time", "date")


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
    # Drop the spare time columns before renaming, or two of them become "local_time" and every
    # later lookup gets a two-column frame instead of a series.
    lowered = {c: str(c).strip().lower() for c in df.columns}
    timeish = [c for c, low in lowered.items() if COLUMN_ALIASES.get(low) == "local_time"]
    if len(timeish) > 1:
        rank = {name: i for i, name in enumerate(TIME_PREFERENCE)}
        keep = min(timeish, key=lambda c: rank.get(lowered[c], len(rank)))
        df = df.drop(columns=[c for c in timeish if c != keep])
    # the export's header is one name short, so the last column arrives unnamed
    df = df.rename(columns={c: COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower())
                            for c in df.columns})
    # NEMO's own export splits the top group ("parent") from the sub-group ("group"); a tidied
    # export usually has one column called "group" holding the top group. Use whichever arrived.
    if "group" not in df.columns and "nemo_group" in df.columns:
        df = df.rename(columns={"nemo_group": "group"})
    if "visibility" not in df.columns:
        extra = [c for c in df.columns if c.startswith("unnamed")]
        if extra:
            df = df.rename(columns={extra[0]: "visibility"})
    raw_time = df["local_time"]
    if time_format:
        df["local_time"] = pd.to_datetime(raw_time, format=time_format, errors="coerce")
        # A configured format that does not fit the file would silently throw every record away, so
        # fall back to working the format out rather than publishing an empty wildlife section.
        if len(df) and df["local_time"].isna().mean() > 0.5:
            df["local_time"] = pd.to_datetime(raw_time, errors="coerce")
    else:
        df["local_time"] = pd.to_datetime(raw_time, errors="coerce")
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


def _keys(df: pd.DataFrame) -> pd.Series:
    """"Group|Species" for each record, the key a group-qualified list entry is written against."""
    return _col(df, "group", "").astype(str).str.strip() + "|" + df["species"].astype(str).str.strip()


def matches(df: pd.DataFrame, entries) -> pd.Series:
    """Which records match a species.yaml list.

    NEMO names a species relative to its group, so a common dolphin is recorded as "Common" and a
    fin whale as "Fin". Those bare names collide across groups: "Blue" is a blue whale under Whales
    and a blue shark under Sharks. An entry written as "Whales|Blue" matches only in that group; a
    bare entry still matches any group, so short lists of unambiguous names stay readable.
    """
    entries = set(entries or [])
    qualified = {e for e in entries if "|" in e}
    bare = entries - qualified
    hit = df["species"].astype(str).str.strip().isin(bare)
    if qualified:
        hit = hit | _keys(df).isin(qualified)
    return hit


def lookup(df: pd.DataFrame, mapping: dict, default="") -> pd.Series:
    """Values from a species.yaml mapping, group-qualified keys taking precedence over bare ones."""
    mapping = mapping or {}
    qualified = {k: v for k, v in mapping.items() if "|" in k}
    bare = {k: v for k, v in mapping.items() if "|" not in k}
    out = df["species"].astype(str).str.strip().map(bare)
    if qualified:
        out = _keys(df).map(qualified).fillna(out)
    return out.fillna(default)


def tag_species(df: pd.DataFrame, species_cfg: dict) -> pd.DataFrame:
    """Add the flags from species.yaml: gelatinous type, invasive status, sting risk, sensitivity."""
    gel = species_cfg.get("gelatinous", {}) or {}
    df = df.copy()
    df["gelatinous"] = np.where(matches(df, gel.get("drifter")), "drifter",
                                np.where(matches(df, gel.get("water_column")), "water_column", ""))
    df["invasive"] = lookup(df, species_cfg.get("invasive", {}))
    df["stinging"] = matches(df, species_cfg.get("stinging"))
    df["sensitive"] = matches(df, species_cfg.get("sensitive"))
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


# ------------------------------------------------------------------------------------------------
# Watch species: one species followed closely, with the sea conditions behind each appearance
# ------------------------------------------------------------------------------------------------
def _series(merged_by_key: dict, var: str, location: str):
    """A daily series for one variable at one location, or None."""
    m = merged_by_key.get((var, location))
    if m is None or getattr(m, "empty", True):
        return None
    return m["value"].astype(float)


def window_mean(series, end, days: int, offset: int = 0):
    """Mean of a series over the `days` before (and including) `end`, skipping the last `offset` days.

    offset=1 with days=3 means the three days leading up to the day before: what the sea had been
    doing before anybody looked, rather than the conditions on the night itself.
    """
    if series is None:
        return np.nan
    last = pd.Timestamp(end) - pd.Timedelta(days=offset)
    win = series.loc[(series.index <= last) & (series.index > last - pd.Timedelta(days=days))]
    return float(win.mean()) if len(win) else np.nan


def seasonal_norm(series, when, half_window: int = 10, min_years: int = 3):
    """The usual value for this time of year, from every other year in the record.

    The same calendar window in OTHER years, so a bloom year cannot flatter its own baseline.
    """
    if series is None:
        return np.nan
    when = pd.Timestamp(when)
    doy, others = when.dayofyear, series[series.index.year != when.year]
    if not len(others):
        return np.nan
    near = others[(others.index.dayofyear - doy).map(lambda d: min(abs(d), 365 - abs(d))) <= half_window]
    if near.index.year.nunique() < min_years:
        return np.nan
    return float(near.mean())


def cluster_records(sub: pd.DataFrame, max_gap_days: int = 3) -> list:
    """Records of one species run together into appearances: dates within `max_gap_days` of each other.

    A rare species does not need the share-of-effort test that bloom_events applies; two reports on
    consecutive nights is already the thing we want to look at.
    """
    if sub.empty:
        return []
    day = pd.to_datetime(sub["local_date"]).sort_values()
    events, start, prev, n = [], day.iloc[0], day.iloc[0], 0
    for d in day:
        if (d - prev).days > max_gap_days:
            events.append({"start": start.strftime("%Y-%m-%d"), "end": prev.strftime("%Y-%m-%d"),
                           "days": int((prev - start).days) + 1, "records": n})
            start, n = d, 0
        prev, n = d, n + 1
    events.append({"start": start.strftime("%Y-%m-%d"), "end": prev.strftime("%Y-%m-%d"),
                   "days": int((prev - start).days) + 1, "records": n})
    return events


DRIVER_RULES = (
    # (key, label, how it is decided) - all thresholds are in the config so they can be argued with
    ("warm_sea", "Warm sea"),
    ("heatwave", "Marine heatwave"),
    ("upwelling", "Upwelling before"),
    ("onshore", "Onshore wind"),
    ("calm", "Calm days before"),
    ("rich_water", "Chlorophyll up"),
    ("dark_moon", "Dark moon"),
)


def event_conditions(ev: dict, merged_by_key: dict, area: str, cfg: dict) -> dict:
    """The sea conditions around one appearance, and which known drivers were in play.

    Everything here is measured, not inferred: each driver is a named threshold on a series the
    harvester already collects, and the thresholds live in config.yaml.
    """
    wind_loc = cfg.get("wind_location", "gibraltar_airport")
    start = pd.Timestamp(ev["start"])
    sst = _series(merged_by_key, "sst", area)
    chl = _series(merged_by_key, "chl", area)
    wind = _series(merged_by_key, "wind_speed", wind_loc)
    up = _series(merged_by_key, "upwelling_index", wind_loc)
    lead = int(cfg.get("lead_days", 3))

    sst_now = window_mean(sst, ev["end"], ev["days"])
    sst_usual = seasonal_norm(sst, start)
    chl_now = window_mean(chl, ev["end"], ev["days"])
    chl_usual = seasonal_norm(chl, start)
    wind_before = window_mean(wind, start, lead, offset=1)
    wind_usual = seasonal_norm(wind, start)
    up_before = window_mean(up, start, int(cfg.get("upwelling_lead_days", 5)), offset=1)
    moon = float(np.mean([dv.moon_illumination(pd.Timestamp(ev["start"]) + pd.Timedelta(days=i)) * 100
                          for i in range(ev["days"])]))

    warm = sst_now - sst_usual if np.isfinite(sst_now) and np.isfinite(sst_usual) else np.nan
    chl_up = chl_now - chl_usual if np.isfinite(chl_now) and np.isfinite(chl_usual) else np.nan
    calm = (wind_before < wind_usual * float(cfg.get("calm_ratio", 0.8))
            if np.isfinite(wind_before) and np.isfinite(wind_usual) else False)

    drivers = []
    if np.isfinite(warm) and warm >= float(cfg.get("warm_sea_c", 1.0)):
        drivers.append("warm_sea")
    if np.isfinite(up_before) and up_before >= float(cfg.get("upwelling_threshold", 500)):
        drivers.append("upwelling")
    if np.isfinite(up_before) and up_before <= -float(cfg.get("upwelling_threshold", 500)):
        drivers.append("onshore")
    if calm:
        drivers.append("calm")
    # "above average" fires on half of all days by construction, so ask for a real margin
    if (np.isfinite(chl_now) and np.isfinite(chl_usual)
            and chl_now >= chl_usual * float(cfg.get("rich_water_ratio", 1.25))):
        drivers.append("rich_water")
    if moon <= float(cfg.get("dark_moon_pct", 35)):
        drivers.append("dark_moon")
    return {"sst": _r(sst_now, 1), "sst_vs_usual": _r(warm, 1),
            "chl": _r(chl_now, 2), "chl_vs_usual": _r(chl_up, 2),
            "wind_before": _r(wind_before, 1), "wind_usual": _r(wind_usual, 1),
            "upwelling_before": _r(up_before, 0), "moon_pct": int(round(moon)),
            "drivers": drivers}


def condition_profile(sub: pd.DataFrame, merged_by_key: dict, area: str, cfg: dict) -> list:
    """Conditions on the days this species was seen, against the same dates in other years.

    The control is the same calendar window in other years, so season is held still and what is
    left is how those particular days differed.
    """
    wind_loc = cfg.get("wind_location", "gibraltar_airport")
    wanted = [("sst", "Sea temperature", "°C", 1, area),
              ("chl", "Chlorophyll", "mg/m³", 2, area),
              ("wind_speed", "Wind", "km/h", 1, wind_loc),
              ("upwelling_index", "Upwelling index", "m³/s per km", 0, wind_loc)]
    days = sorted({d for d in sub["local_date"]})
    out = []
    for var, label, unit, nd, loc in wanted:
        s = _series(merged_by_key, var, loc)
        if s is None:
            continue
        on = [float(s.loc[pd.Timestamp(d)]) for d in days if pd.Timestamp(d) in s.index]
        usual = [v for v in (seasonal_norm(s, d) for d in days) if np.isfinite(v)]
        if not on or not usual:
            continue
        out.append({"variable": var, "label": label, "unit": unit,
                    "on_days": _r(float(np.mean(on)), nd), "usual": _r(float(np.mean(usual)), nd),
                    "n": len(on)})
    return out


def watch_report(records: pd.DataFrame, merged_by_key: dict, cfg: dict, triggers: dict | None = None) -> dict:
    """Everything the dashboard shows about one watched species."""
    key = cfg.get("species")
    sub = records[matches(records, [key])] if key else records.iloc[0:0]
    name = cfg.get("name") or (key.split("|", 1)[1] if key and "|" in key else key)
    report = {"key": key, "name": name, "scientific": cfg.get("scientific", ""),
              "note": cfg.get("note", ""), "records": int(len(sub)), "events": [], "profile": [],
              "by_year": {}, "by_month": [0] * 12, "first": None, "last": None, "areas": {}}
    if sub.empty:
        return report
    day = pd.to_datetime(sub["local_date"])
    area = cfg.get("area") or (sub["area"].value_counts().idxmax() if sub["area"].notna().any() else None)
    report.update(first=sub["local_date"].min(), last=sub["local_date"].max(), area=area,
                  by_year={str(y): int(n) for y, n in day.dt.year.value_counts().sort_index().items()},
                  by_month=[int((day.dt.month == m).sum()) for m in range(1, 13)],
                  areas={a: int(n) for a, n in sub["area"].value_counts().items()},
                  verified=int(sub["verified"].sum()))
    events = cluster_records(sub, int(cfg.get("max_gap_days", 3)))
    for ev in events:
        ev.update(event_conditions(ev, merged_by_key, area, cfg))
    if triggers:
        dv.attach_triggers(events, triggers, int(cfg.get("trigger_lookback_days", 10)))
        for ev in events:
            kinds = {t["type"] for t in ev.get("triggers", [])}
            if "heatwave" in kinds and "heatwave" not in ev["drivers"]:
                ev["drivers"].insert(0, "heatwave")
                if "warm_sea" in ev["drivers"]:
                    ev["drivers"].remove("warm_sea")   # the heatwave is the better way to say it
            # An overlapping upwelling event only counts when the index itself agrees: the event list
            # is for the Spanish coast, and the wind here can be doing the opposite on the day.
            offshore = ev.get("upwelling_before")
            if ("upwelling" in kinds and "upwelling" not in ev["drivers"]
                    and (offshore is None or offshore >= 0)):
                ev["drivers"].append("upwelling")
    report["events"] = sorted(events, key=lambda e: e["start"], reverse=True)
    report["profile"] = condition_profile(sub, merged_by_key, area, cfg)
    report["driver_labels"] = {k: v for k, v in DRIVER_RULES}
    return report


def invasive_watch(records: pd.DataFrame, species_cfg: dict) -> list:
    """One entry per watchlist species: status, first report, first verified record, records per year.

    Only a verified record sets first_verified, because that is what goes public.
    """
    listed = species_cfg.get("invasive", {}) or {}
    out = []
    for key, status in listed.items():
        sub = records[matches(records, [key])]
        species = key.split("|", 1)[1] if "|" in key else key      # the group is a filter, not a name
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
