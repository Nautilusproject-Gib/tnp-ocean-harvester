"""Decides what date range each source needs, runs it in chunks and records progress."""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from .db import Database
from .sources import build_source


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def date_chunks(start: date, end: date, days: int, reverse=False):
    """Inclusive [start, end] split into chunks of `days`."""
    days = max(1, int(days))
    chunks = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        chunks.append((cur, stop))
        cur = stop + timedelta(days=1)
    return list(reversed(chunks)) if reverse else chunks


def plan_update(cfg: dict, latest: datetime | None, today: date) -> tuple[date, date] | None:
    lookback = int(cfg.get("lookback_days", 5))
    window = cfg.get("rolling_window_days")
    if latest is not None:
        start = latest.date() - timedelta(days=lookback)
    else:
        start = today - timedelta(days=int(window or 30))
    if window:
        start = max(start, today - timedelta(days=int(window)))
    if cfg.get("earliest"):
        start = max(start, date.fromisoformat(str(cfg["earliest"])))
    end = today
    if cfg.get("latest"):
        end = min(end, date.fromisoformat(str(cfg["latest"])))
    return (start, end) if start <= end else None


def plan_backfill(cfg: dict, config: dict, floor: datetime | None, today: date,
                  start_override: date | None, end_override: date | None) -> tuple[date, date] | None:
    if cfg.get("rolling_window_days"):
        return None  # near-real-time products only keep recent data online
    lo = start_override or date.fromisoformat(str(config.get("backfill_start", "1980-01-01")))
    if cfg.get("earliest"):
        lo = max(lo, date.fromisoformat(str(cfg["earliest"])))
    hi = end_override or ((floor.date() - timedelta(days=1)) if floor else today)
    if cfg.get("latest"):
        hi = min(hi, date.fromisoformat(str(cfg["latest"])))
    return (lo, hi) if lo <= hi else None


@dataclass
class RunReport:
    ok: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    def summary(self):
        lines = []
        for s, msg in self.ok:
            lines.append(f"  OK      {s}: {msg}")
        for s, msg in self.skipped:
            lines.append(f"  SKIPPED {s}: {msg}")
        for s, msg in self.failed:
            lines.append(f"  FAILED  {s}: {msg}")
        return "\n".join(lines)


def run(config: dict, db: Database, mode: str = "update", only: list[str] | None = None,
        start: date | None = None, end: date | None = None, max_minutes: float | None = None,
        log=print) -> RunReport:
    report = RunReport()
    deadline = time.monotonic() + max_minutes * 60 if max_minutes else None
    today = utc_today()

    for code, cfg in config["sources"].items():
        if only and code not in only:
            continue
        if not cfg.get("enabled", True):
            report.skipped.append((code, "disabled in config"))
            continue
        src = build_source(code, config)
        missing = src.missing_env()
        if missing:
            report.skipped.append((code, f"missing credentials: {', '.join(missing)}"))
            log(f"[{code}] skipped - set {', '.join(missing)}")
            continue

        if mode == "update":
            rng = (start, end) if (start and end) else plan_update(cfg, db.latest_time(code), today)
            reverse = False
        else:
            rng = plan_backfill(cfg, config, db.backfill_floor(code), today, start, end)
            reverse = True
        if rng is None:
            report.skipped.append((code, "nothing to do"))
            continue

        chunk_days = int(cfg.get("chunk_days", 31))
        total_rows = 0
        failed = False
        for c_start, c_end in date_chunks(rng[0], rng[1], chunk_days, reverse=reverse):
            if deadline and time.monotonic() > deadline:
                log(f"[{code}] time budget reached; will resume next run")
                break
            t0 = datetime.now(timezone.utc)
            log(f"[{code}] {mode} {c_start} -> {c_end}")
            try:
                obs = src.fetch(c_start, c_end)
                n = db.upsert_observations(obs)
                total_rows += n
                db.log_run(code, mode, t0, datetime.now(timezone.utc), c_start, c_end, n, "ok")
                log(f"[{code}]   {n} rows")
            except Exception as e:
                db.log_run(code, mode, t0, datetime.now(timezone.utc), c_start, c_end, 0, "error",
                           f"{e}\n{traceback.format_exc()}")
                log(f"[{code}]   ERROR {e}")
                report.failed.append((code, f"{c_start}..{c_end}: {e}"))
                failed = True
                break  # do not leave gaps; the next run retries from here
        if not failed:
            report.ok.append((code, f"{total_rows} rows ({rng[0]} .. {rng[1]})"))
    return report
