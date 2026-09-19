"""Global Fishing Watch: apparent fishing effort from AIS, aggregated over an area.

"Apparent" is the operative word. GFW infers fishing from how a vessel moves, so the hours are a
model of behaviour, not a logbook. Vessels without AIS, or with it switched off, are invisible.
Read it as a picture of industrial activity that can be seen from space, and nothing more.

Licence: CC BY-NC 4.0. Non-commercial use only, and the attribution travels with the numbers, so
anywhere these appear has to credit Global Fishing Watch.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta

from ..db import Observation
from .base import Source, SourceError

ENDPOINT = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-fishing-effort:latest"


def clean_token(raw: str) -> str:
    """Whitespace and a pasted "Bearer " prefix are the two ways a good token arrives broken.

    A GitHub secret keeps whatever was in the clipboard, including the trailing newline you get from
    selecting a line in a browser, and "Bearer eyJ..." doubles up once we add our own prefix.
    """
    t = (raw or "").strip().strip('"').strip("'")
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    # A JWT is base64url text and two dots, with no spaces or line breaks in it. Copying a long
    # token out of a browser can fold it across lines, and the pieces then have to be joined back
    # up or the signature will not match.
    return "".join(t.split())


def token_shape(token: str, raw: str) -> str:
    """A description of the token that is safe to put in a log: never the token itself."""
    bits = [f"{len(token)} characters"]
    if raw != token:
        bits.append("had whitespace or a Bearer prefix, which was removed")
    bits.append("looks like a JWT" if token.startswith("eyJ") and token.count(".") == 2
                else "does NOT look like a JWT (a GFW API token starts with eyJ and has two dots)")
    return "; ".join(bits)


def bbox_ring(bbox) -> list:
    """A bbox as a closed polygon ring, anticlockwise from the south-west corner."""
    lon_min, lat_min, lon_max, lat_max = bbox
    return [[lon_min, lat_min], [lon_max, lat_min], [lon_max, lat_max],
            [lon_min, lat_max], [lon_min, lat_min]]


def bbox_geojson(bbox) -> str:
    """A bbox as the GeoJSON string form of the request body."""
    return json.dumps(feature_collection(bbox))


def feature_collection(bbox) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {},
         "geometry": {"type": "Polygon", "coordinates": [bbox_ring(bbox)]}}]}


def body_shapes(bbox) -> list:
    """The ways this endpoint has been documented to take a custom area, best guess first.

    The published example stringifies the GeoJSON, but the server answers "body malformed" to it,
    and their own clients send an object. Rather than spend a run per guess, the first request
    tries each shape until one is accepted, then remembers which for the rest of the harvest.
    """
    fc = feature_collection(bbox)
    return [
        ("object", {"geojson": fc}),
        ("string", {"geojson": json.dumps(fc)}),
        ("geometry", {"geojson": fc["features"][0]["geometry"]}),
        ("region", {"region": {"geojson": fc}}),
    ]


def daily_hours(entries, flag_key="flag") -> dict:
    """Report rows -> {date: {"hours": total, "vessels": n, "by_flag": {...}}}.

    Rows arrive one per day per group, so the same date appears several times and has to be summed.
    """
    out: dict[str, dict] = {}
    for row in entries or []:
        day = str(row.get("date") or "")[:10]
        if not day:
            continue
        hours = row.get("hours")
        if hours is None:
            continue
        rec = out.setdefault(day, {"hours": 0.0, "vessels": 0, "by_flag": {}})
        rec["hours"] += float(hours)
        rec["vessels"] += int(row.get("vessel_count") or row.get("detections") or 0)
        flag = row.get(flag_key)
        if flag:
            rec["by_flag"][flag] = rec["by_flag"].get(flag, 0.0) + float(hours)
    return out


class GfwFishingEffort(Source):
    """Daily apparent fishing hours inside an area."""

    required_env = ("GFW_API_TOKEN",)

    #: which body shape this API accepted, remembered after the first successful request
    _body_shape: str | None = None

    def fetch(self, start: date, end: date):
        raw = os.environ.get("GFW_API_TOKEN") or ""
        token = clean_token(raw)
        if not token:
            raise SourceError("GFW_API_TOKEN is not set")
        area_code = self.cfg.get("area") or next(iter(self.areas()))
        bbox = self.area(area_code)["bbox"]
        codes = self.cfg.get("codes") or {"hours": "fishing_hours"}
        step = int(self.cfg.get("chunk_days", 90))

        out = []
        day = start
        while day <= end:
            last = min(day + timedelta(days=step - 1), end)
            # One number per day for the whole area, so the report is spatially aggregated. The
            # grid resolution only means anything when it is not, and sending both is what the
            # server rejects with a 422.
            params = {
                "format": "JSON",
                "group-by": self.cfg.get("group_by", "FLAG"),
                "temporal-resolution": "DAILY",
                "spatial-aggregation": "true",
                "datasets[0]": self.cfg.get("dataset", DATASET),
                "date-range": f"{day.isoformat()},{last.isoformat()}",
            }
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            shapes = body_shapes(bbox)
            if GfwFishingEffort._body_shape:
                shapes = [s for s in shapes if s[0] == GfwFishingEffort._body_shape] or shapes
            # The endpoint runs one report per user at a time and answers 429 otherwise, so these
            # go one after another, never in parallel, and a 429 is worth waiting out rather than
            # failing the whole harvest.
            r = None
            for name, body in shapes:
                r = self.http_post(ENDPOINT, params=params, json=body, timeout=300, headers=headers)
                if r.status_code != 422:
                    if GfwFishingEffort._body_shape != name:
                        print(f"[{self.code}]   request body accepted as '{name}'")
                        GfwFishingEffort._body_shape = name
                    break
            if r.status_code in (401, 403):
                raise SourceError(
                    f"GFW rejected the token ({r.status_code}). The token in GFW_API_TOKEN is "
                    f"{token_shape(token, raw)}. Create a token at "
                    f"globalfishingwatch.org/our-apis/tokens and paste the long token itself, not "
                    f"the application name or an API key. Server said: {r.text[:160]}")
            if r.status_code == 404:
                print(f"[{self.code}]   no report for {day}..{last} (404)")
                day = last + timedelta(days=1)
                continue
            if r.status_code >= 400:
                # The body names the field it did not like, which is the only way to tell a bad
                # parameter from a bad date range without guessing.
                raise SourceError(f"GFW report failed ({r.status_code}) for "
                                  f"{day}..{last}: {r.text[:400]}")
            payload = r.json()
            entries = payload.get("entries")
            # entries is sometimes a list of lists, one per requested dataset
            if entries and isinstance(entries[0], list):
                entries = [row for group in entries for row in group]
            for d, rec in sorted(daily_hours(entries).items()):
                when = datetime.fromisoformat(f"{d}T00:00:00")
                for key, code in codes.items():
                    if key == "flags":
                        value = float(len(rec["by_flag"]))
                    elif key in rec:
                        value = float(rec[key])
                    else:
                        continue
                    out.append(Observation(self.code, code, area_code, when, value))
            day = last + timedelta(days=1)
        return out
