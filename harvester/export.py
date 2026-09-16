"""Export compact JSON files for the website dashboard.

Produces, under export.output_dir (default public/data):
  meta.json                          variables, locations, sources, generation time
  latest.json                        latest daily value per variable/location, with anomaly
  daily/<variable>__<location>.json  merged daily series, record statistics and day-of-year climatology
  dust_events.json                   Saharan dust episodes and what chlorophyll/light did around them

It also copies the dashboard page (the site/ folder) next to the data, so GitHub Pages
publishes page and data together.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .db import Database
from .stats import circular_mean_deg

SUM_VARS = {"precip", "precip_1h", "precip_3h", "precip_6h", "precip_12h", "precip_24h"}
MAX_VARS = {"wind_gust"}

# PAR from broadband shortwave radiation:
#   PAR fraction of incoming shortwave ~0.48, and 4.57 umol photons per joule of PAR.
#   daily mean W m-2 * 86400 s * 0.48 * 4.57e-6 mol/J  ->  einstein (mol photons) m-2 day-1
PAR_FRACTION = 0.48
PHOTONS_PER_JOULE = 4.57e-6
PAR_FACTOR = 86400 * PAR_FRACTION * PHOTONS_PER_JOULE   # ~0.1895


def par_from_shortwave(daily_mean_sw: pd.Series) -> pd.Series:
    """Daily PAR (E m-2 d-1) from daily mean shortwave radiation (W m-2)."""
    return daily_mean_sw * PAR_FACTOR


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


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
    frames = []
    taken = pd.DatetimeIndex([])
    for src in order:
        s = per_source[src]
        new = s[~s.index.isin(taken)]
        if not new.empty:
            frames.append(pd.DataFrame({"value": new.values, "source": src}, index=new.index))
            taken = taken.union(new.index)
    if not frames:
        return pd.DataFrame(columns=["value", "source"])
    return pd.concat(frames).sort_index()


def record_stats(merged: pd.DataFrame) -> dict | None:
    """Whole-record statistics: the horizontal reference lines on the full-record chart."""
    if merged.empty:
        return None
    v = merged["value"].astype(float)
    return {
        "n": int(v.size),
        "mean": _r(v.mean()),
        "sd": _r(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": _r(v.min()), "min_date": v.idxmin().strftime("%Y-%m-%d"),
        "max": _r(v.max()), "max_date": v.idxmax().strftime("%Y-%m-%d"),
        "first": merged.index.min().strftime("%Y-%m-%d"),
        "last": merged.index.max().strftime("%Y-%m-%d"),
    }


def climatology(merged: pd.DataFrame, min_years=3, half_window=7, circular=False):
    """Day-of-year statistics over all years, using a centred 15-day window wrapped round the year.

    Returns parallel arrays indexed by day of year 1..365: mean, sd, min, max, p10, p90, n.
    """
    if merged.empty:
        return None
    years = merged.index.year.nunique()
    if years < min_years:
        return None
    doy = np.minimum(merged.index.dayofyear.values, 365)
    vals = merged["value"].astype(float).values
    out = {k: [] for k in ("mean", "sd", "min", "max", "p10", "p90", "n")}
    for d in range(1, 366):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 365 - dist)
        sel = vals[dist <= half_window]
        sel = sel[np.isfinite(sel)]
        if sel.size < 5:
            for k in out:
                out[k].append(None if k != "n" else int(sel.size))
            continue
        if circular:
            out["mean"].append(_r(circular_mean_deg(sel), 1))
            for k in ("sd", "min", "max", "p10", "p90"):
                out[k].append(None)
        else:
            out["mean"].append(_r(sel.mean()))
            out["sd"].append(_r(sel.std(ddof=1)))
            out["min"].append(_r(sel.min()))
            out["max"].append(_r(sel.max()))
            out["p10"].append(_r(np.quantile(sel, 0.1)))
            out["p90"].append(_r(np.quantile(sel, 0.9)))
        out["n"].append(int(sel.size))
    return {"years": int(years), "first_year": int(merged.index.year.min()),
            "last_year": int(merged.index.year.max()), "window_days": 2 * half_window + 1, **out}


def dust_events(config: dict, merged_by_key: dict, log=print) -> dict:
    """Group days above the dust thresholds into episodes and compare response variables
    in the N days before an episode with the N days after it."""
    cfg = config.get("export", {}).get("dust", {})
    indicators = cfg.get("indicators", [])
    if not indicators:
        return {"episodes": [], "indicators": []}
    flagged: dict[pd.Timestamp, list] = defaultdict(list)
    used = []
    for ind in indicators:
        key = (ind["variable"], ind["location"])
        m = merged_by_key.get(key)
        if m is None or m.empty:
            continue
        s = m["value"].astype(float)
        mask = s > float(ind["threshold"])
        if ind.get("with"):
            w = ind["with"]
            other = merged_by_key.get((w["variable"], ind["location"]))
            if other is None:
                continue
            o = other["value"].astype(float).reindex(s.index)
            mask &= o < float(w["below"])
        for day, val in s[mask].items():
            flagged[day].append({"indicator": ind["variable"], "value": _r(val, 3)})
        used.append({**ind, "days_flagged": int(mask.sum())})

    days = sorted(flagged)
    episodes = []
    for day in days:
        if episodes and (day - episodes[-1]["_end"]).days <= 1:
            episodes[-1]["_end"] = day
            episodes[-1]["_days"].append(day)
        else:
            episodes.append({"_start": day, "_end": day, "_days": [day]})

    window = int(cfg.get("response_window_days", 7))
    responses = cfg.get("response", [])
    out = []
    for ep in episodes:
        start, end = ep["_start"], ep["_end"]
        peak = {}
        for d in ep["_days"]:
            for rec in flagged[d]:
                if rec["value"] is not None and rec["value"] > peak.get(rec["indicator"], -np.inf):
                    peak[rec["indicator"]] = rec["value"]
        resp = []
        for r in responses:
            m = merged_by_key.get((r["variable"], r["location"]))
            if m is None or m.empty:
                continue
            s = m["value"].astype(float)
            before = s[(s.index >= start - pd.Timedelta(days=window)) & (s.index < start)]
            after = s[(s.index > end) & (s.index <= end + pd.Timedelta(days=window))]
            b = before.mean() if before.size else np.nan
            a = after.mean() if after.size else np.nan
            resp.append({"variable": r["variable"], "location": r["location"],
                         "before_mean": _r(b), "after_mean": _r(a),
                         "n_before": int(before.size), "n_after": int(after.size),
                         "change_pct": _r((a - b) / b * 100, 1) if np.isfinite(a) and np.isfinite(b) and b else None})
        out.append({"start": start.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"),
                    "days": len(ep["_days"]), "peak": peak, "response": resp})
    return {"indicators": used, "response_window_days": window, "episodes": out}


def export(config: dict, db: Database, log=print):
    ecfg = config.get("export", {})
    out_dir = Path(ecfg.get("output_dir", "public/data"))
    (out_dir / "daily").mkdir(parents=True, exist_ok=True)
    priorities = ecfg.get("daily_priority", {})

    grouped = defaultdict(dict)  # (variable, location) -> {source: daily series}
    for source, variable, location in db.series_keys():
        series = db.fetch_series(variable, location, source)
        s = daily_aggregate(series, variable)
        if s.empty:
            continue
        grouped[(variable, location)][source] = s
        if variable == "sw_rad":
            # derived series kept separate from satellite PAR, so the two are never silently mixed
            df = pd.DataFrame(series, columns=["t", "v"])
            df["t"] = pd.to_datetime(df["t"])
            counts = df.groupby(df["t"].dt.normalize())["v"].count()
            complete = s[counts.reindex(s.index).fillna(0) >= 20]
            if not complete.empty:
                grouped[("par_era5", location)][source] = par_from_shortwave(complete)

    merged_by_key = {}
    latest = []
    for (variable, location), per_source in sorted(grouped.items()):
        merged = merge_by_priority(per_source, priorities.get(variable, []))
        if merged.empty:
            continue
        merged_by_key[(variable, location)] = merged
        clim = climatology(merged, circular=variable.endswith("_dir"))
        rec = record_stats(merged)
        srcs = list(dict.fromkeys(merged["source"]))
        idx = {s: i for i, s in enumerate(srcs)}
        payload = {
            "variable": variable,
            "location": location,
            "unit": config["variables"].get(variable, {}).get("unit"),
            "name": config["variables"].get(variable, {}).get("name", variable),
            "sources": srcs,
            "record": rec,
            "climatology": clim,
            "data": [[d.strftime("%Y-%m-%d"), round(float(v), 4), idx[s]]
                     for d, v, s in zip(merged.index, merged["value"], merged["source"])],
        }
        (out_dir / "daily" / f"{variable}__{location}.json").write_text(json.dumps(payload, separators=(",", ":")))

        last_day = merged.index.max()
        last_val = float(merged.loc[last_day, "value"])
        anomaly = z = mean = sd = None
        if clim and not variable.endswith("_dir"):
            d = min(last_day.dayofyear, 365) - 1
            mean, sd = clim["mean"][d], clim["sd"][d]
            if mean is not None:
                anomaly = _r(last_val - mean)
                z = _r((last_val - mean) / sd, 2) if sd else None
        latest.append({"variable": variable, "location": location, "date": last_day.strftime("%Y-%m-%d"),
                       "value": _r(last_val), "clim_mean": mean, "clim_sd": sd, "anomaly": anomaly, "z": z,
                       "source": merged.loc[last_day, "source"]})

    events = dust_events(config, merged_by_key, log=log)
    (out_dir / "dust_events.json").write_text(json.dumps(events, indent=1))

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "variables": config.get("variables", {}),
        "areas": config.get("areas", {}),
        "points": config.get("points", {}),
        "sources": {k: {"description": v.get("description"), "product": v.get("product"),
                        "dataset": v.get("dataset_id") or v.get("short_name") or v.get("station_id")}
                    for k, v in config["sources"].items()},
        "series": [f"{v}__{l}" for v, l in sorted(merged_by_key)],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "latest.json").write_text(json.dumps(latest, indent=1))

    # publish the dashboard page alongside the data
    site_dir = Path(ecfg.get("site_dir", Path(__file__).resolve().parents[1] / "site"))
    if site_dir.is_dir():
        for f in site_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, out_dir.parent / f.name)
    log(f"Exported {len(merged_by_key)} series and {len(events['episodes'])} dust episodes to {out_dir}")
    return len(merged_by_key)
