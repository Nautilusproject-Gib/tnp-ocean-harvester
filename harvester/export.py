"""Export compact JSON files for the website dashboard.

Produces, under export.output_dir:
  meta.json                          variables, locations, sources, generation time
  latest.json                        latest daily value per variable/location, with anomaly
  daily/<variable>__<location>.json  merged daily series + day-of-year climatology
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .db import Database
from .stats import circular_mean_deg

SUM_VARS = {"precip", "precip_1h", "precip_3h", "precip_6h", "precip_12h", "precip_24h"}
MAX_VARS = {"wind_gust"}


def daily_aggregate(series: list[tuple], variable: str) -> pd.Series:
    """[(datetime, value), ...] -> pd.Series indexed by date."""
    if not series:
        return pd.Series(dtype="float64")
    df = pd.DataFrame(series, columns=["t", "v"])
    df["t"] = pd.to_datetime(df["t"])
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    g = df.groupby(df["t"].dt.normalize())["v"]
    if variable.endswith("_dir"):
        out = g.apply(lambda s: circular_mean_deg(s.values))
    elif variable in SUM_VARS:
        # only report a daily total when (nearly) every hour is present
        counts = g.count()
        out = g.sum().where(counts >= 20)
    elif variable in MAX_VARS:
        out = g.max()
    else:
        out = g.mean()
    return out.astype("float64").dropna()


def merge_by_priority(per_source: dict[str, pd.Series], priority: list[str]) -> pd.DataFrame:
    """Pick, for each day, the value from the highest-priority source that has one."""
    order = [s for s in priority if s in per_source] + [s for s in per_source if s not in priority]
    merged = pd.DataFrame(columns=["value", "source"])
    for src in order:
        s = per_source[src]
        new = s[~s.index.isin(merged.index)]
        if not new.empty:
            merged = pd.concat([merged, pd.DataFrame({"value": new.values, "source": src}, index=new.index)])
    return merged.sort_index()


def climatology(merged: pd.DataFrame, min_years=3):
    if merged.empty:
        return None
    years = merged.index.year.nunique()
    if years < min_years:
        return None
    df = merged.copy()
    df["doy"] = np.minimum(df.index.dayofyear, 365)
    vals = df["value"].astype(float)
    # 15-day centred window, wrapped around the year, to smooth sparse satellite days
    clim = []
    for d in range(1, 366):
        win = [((d - 1 + k) % 365) + 1 for k in range(-7, 8)]
        sel = vals[df["doy"].isin(win)]
        if sel.size < 5:
            clim.append([d, None, None, None])
        else:
            clim.append([d, round(float(sel.mean()), 4), round(float(sel.quantile(0.1)), 4),
                         round(float(sel.quantile(0.9)), 4)])
    return {"years": int(years), "first_year": int(merged.index.year.min()),
            "last_year": int(merged.index.year.max()), "doy_mean_p10_p90": clim}


def export(config: dict, db: Database, log=print):
    out_dir = Path(config.get("export", {}).get("output_dir", "public/data"))
    (out_dir / "daily").mkdir(parents=True, exist_ok=True)
    priorities = config.get("export", {}).get("daily_priority", {})

    grouped = defaultdict(dict)  # (variable, location) -> {source: daily series}
    for source, variable, location in db.series_keys():
        s = daily_aggregate(db.fetch_series(variable, location, source), variable)
        if not s.empty:
            grouped[(variable, location)][source] = s

    latest = []
    for (variable, location), per_source in sorted(grouped.items()):
        merged = merge_by_priority(per_source, priorities.get(variable, []))
        if merged.empty:
            continue
        clim = climatology(merged)
        srcs = list(dict.fromkeys(merged["source"]))
        idx = {s: i for i, s in enumerate(srcs)}
        payload = {
            "variable": variable,
            "location": location,
            "unit": config["variables"].get(variable, {}).get("unit"),
            "sources": srcs,
            "data": [[d.strftime("%Y-%m-%d"), round(float(v), 4), idx[s]]
                     for d, v, s in zip(merged.index, merged["value"], merged["source"])],
            "climatology": clim,
        }
        (out_dir / "daily" / f"{variable}__{location}.json").write_text(json.dumps(payload, separators=(",", ":")))

        last_day = merged.index.max()
        last_val = float(merged.loc[last_day, "value"])
        anomaly = None
        if clim:
            doy = min(last_day.dayofyear, 365)
            m = clim["doy_mean_p10_p90"][doy - 1][1]
            anomaly = None if m is None else round(last_val - m, 4)
        latest.append({"variable": variable, "location": location, "date": last_day.strftime("%Y-%m-%d"),
                       "value": round(last_val, 4), "anomaly": anomaly,
                       "source": merged.loc[last_day, "source"]})

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variables": config.get("variables", {}),
        "areas": config.get("areas", {}),
        "points": config.get("points", {}),
        "sources": {k: {"description": v.get("description"), "product": v.get("product"),
                        "dataset": v.get("dataset_id") or v.get("short_name") or v.get("station_id")}
                    for k, v in config["sources"].items()},
        "series": [f"{v}__{l}" for v, l in sorted(grouped)],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "latest.json").write_text(json.dumps(latest, indent=1))
    log(f"Exported {len(grouped)} series to {out_dir}")
    return len(grouped)
