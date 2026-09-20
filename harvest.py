#!/usr/bin/env python3
"""TNP Gibraltar Ocean Observatory - data harvester.

Usage:
  python harvest.py init                         create tables and load areas/sources/variables
  python harvest.py update [--source X ...]      fetch new data since the last run (daily job)
  python harvest.py backfill [--source X ...]    work backwards through the historical archive
         [--start 2000-01-01] [--end 2005-12-31] [--max-minutes 300]
  python harvest.py export                       write JSON files for the website dashboard
  python harvest.py status                       show what is in the database
  python harvest.py prune [--apply]              list (and with --apply delete) data config no longer asks for
  python harvest.py check                        show which sources have credentials set

Database: set DATABASE_URL (defaults to sqlite:///data/tnp_ocean.db).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

from harvester.db import Database
from harvester.export import export
from harvester.runner import run
from harvester.sources import build_source


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "update", "backfill", "export", "status", "check", "prune"])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--source", action="append", help="limit to one source (repeatable)")
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    ap.add_argument("--max-minutes", type=float, help="stop starting new chunks after this long")
    ap.add_argument("--database-url")
    ap.add_argument("--apply", action="store_true",
                    help="prune: actually delete, rather than only listing what would go")
    args = ap.parse_args(argv)

    config = load_config(args.config)

    if args.command == "check":
        for code in config["sources"]:
            src = build_source(code, config)
            missing = src.missing_env()
            state = "disabled" if not src.cfg.get("enabled", True) else ("ready" if not missing else
                                                                          "needs " + ", ".join(missing))
            print(f"{code:28s} {state}")
        return 0

    db = Database(args.database_url)
    try:
        return _run_command(args, config, db)
    finally:
        db.close()   # folds SQLite's write-ahead log back into the .db file before it is saved


def _run_command(args, config, db):
    db.init_schema()
    if args.command == "init":
        db.sync_metadata(config)
        print(f"Database ready ({db.dialect}).")
        return 0

    if args.command in ("update", "backfill"):
        db.sync_metadata(config)
        report = run(config, db, mode=args.command, only=args.source, start=args.start, end=args.end,
                     max_minutes=args.max_minutes)
        print("\nSummary\n" + report.summary())
        return 1 if report.failed else 0

    if args.command == "export":
        export(config, db)
        return 0

    if args.command == "prune":
        rows = db.orphans(config)
        if not rows:
            print("Nothing to prune: every series in the database is still asked for in config.yaml.")
            return 0
        print(f"{'source':26s} {'variable':18s} {'location':22s} {'rows':>9s}  why")
        for r in rows:
            print(f"{r['source']:26s} {r['variable']:18s} {r['location']:22s} "
                  f"{r['rows']:9d}  {r['reason']}")
        total = sum(r["rows"] for r in rows)
        held = db.row_count()
        print(f"\n{total:,} rows of {held:,} ({total / held:.1%}) are no longer asked for.")
        if not args.apply:
            print("Nothing was deleted. Run again with --apply to remove them.")
            return 0
        deleted = db.prune(rows)
        print(f"Deleted {deleted:,} rows. The database now holds {db.row_count():,}.")
        return 0

    if args.command == "status":
        rows = db.status()
        print(f"{'source':26s} {'variable':18s} {'location':22s} {'rows':>8s} {'valid':>8s}  first -> last")
        for source, variable, location, n, first, last, valid in rows:
            print(f"{source:26s} {variable:18s} {location:22s} {n:8d} {valid:8d}  {first} -> {last}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
