"""ERDDAP griddap: gridded observations served as CSV over HTTP.

Used for the Puertos del Estado HF radar in the Strait of Gibraltar, redistributed by EMODnet
Physics under CC-BY. This is the only *measured* current field the dashboard has: everything else
in the Strait is a model. The radar looks at the surface only, hourly, on a roughly 1 km grid.

Quality flags follow the EuroGOOS / Copernicus convention, where 1 means the value passed every
test. Anything else is dropped rather than averaged in, because a bad radar cell is not a small
error, it is a wrong direction.
"""
from __future__ import annotations

import csv
import io
import math
from datetime import date, datetime, timedelta

import numpy as np

from ..db import Observation
from .base import Source, SourceError

GOOD_FLAG = 1


def build_query(dataset: str, variables: list[str], start, end, bbox, depth=0.0) -> str:
    """griddap CSV query: every variable repeats the full [time][depth][lat][lon] bracket set."""
    lon_min, lat_min, lon_max, lat_max = bbox
    span = (f"[({start}):1:({end})][({depth}):1:({depth})]"
            f"[({lat_min}):1:({lat_max})][({lon_min}):1:({lon_max})]")
    return ",".join(f"{v}{span}" for v in variables)


def parse_griddap_csv(text: str) -> list[dict]:
    """griddap CSV -> rows. The first line is names, the second is units, then the data."""
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 3:
        return []
    header = [h.strip() for h in rows[0]]
    out = []
    for r in rows[2:]:                      # row 1 is the units line
        if len(r) != len(header):
            continue
        rec = {}
        for k, v in zip(header, r):
            v = v.strip()
            if k in ("time", "UTC"):
                rec[k] = v
            else:
                try:
                    rec[k] = float(v) if v not in ("", "NaN") else math.nan
                except ValueError:
                    rec[k] = math.nan
        out.append(rec)
    return out


def daily_vector_means(rows: list[dict], u_name: str, v_name: str, flag_names: list[str]):
    """Average the good cells for each day, as vectors.

    Currents are averaged as components and only then turned into speed and direction: averaging
    the speeds of a flooding and an ebbing hour would report a fast current where there is a slack
    one, and averaging compass directions across north is meaningless.
    """
    by_day: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        t = str(r.get("time", ""))[:10]
        if not t:
            continue
        u, v = r.get(u_name, math.nan), r.get(v_name, math.nan)
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        flags = [r.get(f) for f in flag_names if f in r]
        if flags and any(np.isfinite(f) and int(f) != GOOD_FLAG for f in flags):
            continue
        by_day.setdefault(t, []).append((u, v))
    out = {}
    for day, pairs in by_day.items():
        u = float(np.mean([p[0] for p in pairs]))
        v = float(np.mean([p[1] for p in pairs]))
        speed = math.hypot(u, v)
        # oceanographic convention: the direction the water is going TO
        direction = (math.degrees(math.atan2(u, v)) + 360) % 360
        out[day] = {"u": u, "v": v, "speed": speed, "dir": direction, "cells": len(pairs)}
    return out


class ErddapGrid(Source):
    """Daily means of a griddap dataset over an area, for a vector pair (u, v)."""

    def fetch(self, start: date, end: date):
        server = self.cfg["server"].rstrip("/")
        dataset = self.cfg["dataset_id"]
        u_name = self.cfg.get("u", "EWCT")
        v_name = self.cfg.get("v", "NSCT")
        flags = list(self.cfg.get("qc_variables") or [])
        codes = self.cfg.get("codes") or {}
        area_code = self.cfg.get("area") or next(iter(self.areas()))
        bbox = list(self.area(area_code)["bbox"])
        # the grid stops short of the box corners on some datasets, and griddap errors rather than
        # clamping, so pull the request inside the dataset's own limits when they are configured
        limits = self.cfg.get("grid_limits")
        if limits:
            lo_lon, lo_lat, hi_lon, hi_lat = limits
            bbox = [max(bbox[0], lo_lon), max(bbox[1], lo_lat),
                    min(bbox[2], hi_lon), min(bbox[3], hi_lat)]
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise SourceError(f"{self.code}: area {area_code} is outside the radar grid")

        out = []
        step = int(self.cfg.get("chunk_days", 10))
        day = start
        while day <= end:
            last = min(day + timedelta(days=step - 1), end)
            query = build_query(dataset, [u_name, v_name] + flags,
                                f"{day.isoformat()}T00:00:00Z", f"{last.isoformat()}T23:59:59Z",
                                bbox, float(self.cfg.get("depth", 0.0)))
            url = f"{server}/griddap/{dataset}.csv?{query}"
            r = self.http_get(url, timeout=300)
            if r.status_code in (403, 404):
                print(f"[{self.code}]   no data for {day}..{last} ({r.status_code})")
                day = last + timedelta(days=1)
                continue
            # ERDDAP answers "no rows in range" with a 404-ish message body rather than an error
            if "nRows = 0" in r.text or not r.text.strip():
                day = last + timedelta(days=1)
                continue
            means = daily_vector_means(parse_griddap_csv(r.text), u_name, v_name, flags)
            for d, m in sorted(means.items()):
                when = datetime.fromisoformat(f"{d}T00:00:00")
                for key, code in codes.items():
                    if key in m:
                        out.append(Observation(self.code, code, area_code, when, float(m[key]),
                                               n_valid=int(m["cells"]), n_total=int(m["cells"])))
            day = last + timedelta(days=1)
        return out
