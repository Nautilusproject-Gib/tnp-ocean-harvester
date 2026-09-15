from __future__ import annotations

import os
from datetime import date, datetime

import requests

USER_AGENT = "TNP-Gibraltar-Ocean-Harvester/1.0 (+https://thenautilusproject.co)"


class SourceError(RuntimeError):
    pass


class Source:
    """Base class. Subclasses implement fetch(start, end) -> list[Observation]."""

    #: environment variables this source needs (checked before running)
    required_env: tuple[str, ...] = ()

    def __init__(self, code: str, cfg: dict, config: dict):
        self.code = code
        self.cfg = cfg
        self.config = config

    # -- coverage ----------------------------------------------------------
    def earliest(self) -> date | None:
        e = self.cfg.get("earliest")
        return date.fromisoformat(str(e)) if e else None

    def latest(self) -> date | None:
        l_ = self.cfg.get("latest")
        return date.fromisoformat(str(l_)) if l_ else None

    def missing_env(self) -> list[str]:
        return [k for k in self.required_env if not os.environ.get(k)]

    # -- helpers -----------------------------------------------------------
    def area(self, code):
        return self.config["areas"][code]

    def point(self, code):
        return self.config["points"][code]

    def areas(self):
        codes = self.cfg.get("areas") or list(self.config.get("areas", {}))
        return {c: self.config["areas"][c] for c in codes}

    def union_bbox(self, pad=0.05):
        boxes = [a["bbox"] for a in self.areas().values()]
        return (min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad,
                max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad)

    @staticmethod
    def http_get(url, params=None, timeout=120, retries=5, **kw):
        last = None
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=timeout,
                                 headers={"User-Agent": USER_AGENT}, **kw)
                if r.status_code in (403, 404):
                    return r
                r.raise_for_status()
                return r
            except requests.RequestException as e:  # pragma: no cover - network
                last = e
                import time
                time.sleep(30 * (attempt + 1))  # rides out brief 502s from busy APIs
        raise SourceError(f"GET {url} failed after {retries} attempts: {last}")

    def fetch(self, start: date, end: date):  # pragma: no cover - abstract
        raise NotImplementedError


def to_datetime(d) -> datetime:
    if isinstance(d, datetime):
        return d
    return datetime(d.year, d.month, d.day)
