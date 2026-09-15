"""Database layer.

Works with SQLite (default, stdlib), PostgreSQL (psycopg 3) and
MySQL/MariaDB (PyMySQL). Choose with the DATABASE_URL environment variable:

    sqlite:///data/tnp_ocean.db
    postgresql://user:password@host:5432/dbname
    mysql://user:password@host:3306/dbname

All times are stored in UTC without a timezone suffix.
"""
from __future__ import annotations

import math
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

OBS_COLUMNS = (
    "source", "variable", "location", "obs_time",
    "val_mean", "val_median", "val_min", "val_max", "val_std",
    "n_valid", "n_total", "updated_at",
)
KEY_COLUMNS = ("source", "variable", "location", "obs_time")
VALUE_COLUMNS = tuple(c for c in OBS_COLUMNS if c not in KEY_COLUMNS)


@dataclass
class Observation:
    source: str
    variable: str
    location: str
    obs_time: datetime
    val_mean: float | None
    val_median: float | None = None
    val_min: float | None = None
    val_max: float | None = None
    val_std: float | None = None
    n_valid: int | None = None
    n_total: int | None = None


def _clean(v):
    """Convert numpy scalars and NaN to plain Python / None."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _ddl(dialect: str) -> list[str]:
    dt = "TIMESTAMP" if dialect == "postgresql" else ("DATETIME" if dialect == "mysql" else "TEXT")
    dbl = "DOUBLE PRECISION" if dialect == "postgresql" else ("DOUBLE" if dialect == "mysql" else "REAL")
    if dialect == "sqlite":
        auto = "INTEGER PRIMARY KEY AUTOINCREMENT"
    elif dialect == "postgresql":
        auto = "BIGSERIAL PRIMARY KEY"
    else:
        auto = "BIGINT AUTO_INCREMENT PRIMARY KEY"
    txt = "TEXT"
    engine = " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4" if dialect == "mysql" else ""
    return [
        f"""CREATE TABLE IF NOT EXISTS observations (
            source VARCHAR(64) NOT NULL,
            variable VARCHAR(32) NOT NULL,
            location VARCHAR(48) NOT NULL,
            obs_time {dt} NOT NULL,
            val_mean {dbl},
            val_median {dbl},
            val_min {dbl},
            val_max {dbl},
            val_std {dbl},
            n_valid INTEGER,
            n_total INTEGER,
            updated_at {dt},
            PRIMARY KEY (source, variable, location, obs_time)
        ){engine}""",
        f"""CREATE TABLE IF NOT EXISTS locations (
            code VARCHAR(48) PRIMARY KEY,
            name VARCHAR(128),
            kind VARCHAR(8),
            lat {dbl}, lon {dbl},
            lon_min {dbl}, lat_min {dbl}, lon_max {dbl}, lat_max {dbl}
        ){engine}""",
        f"""CREATE TABLE IF NOT EXISTS sources (
            code VARCHAR(64) PRIMARY KEY,
            kind VARCHAR(32),
            description {txt},
            product VARCHAR(128),
            dataset VARCHAR(128)
        ){engine}""",
        f"""CREATE TABLE IF NOT EXISTS variables (
            code VARCHAR(32) PRIMARY KEY,
            name VARCHAR(128),
            unit VARCHAR(32)
        ){engine}""",
        f"""CREATE TABLE IF NOT EXISTS harvest_runs (
            id {auto},
            source VARCHAR(64),
            mode VARCHAR(16),
            started_at {dt},
            finished_at {dt},
            range_start {dt},
            range_end {dt},
            n_rows INTEGER,
            status VARCHAR(16),
            message {txt}
        ){engine}""",
        # Speeds up "latest value" and dashboard queries
        "CREATE INDEX IF NOT EXISTS ix_obs_var_loc_time ON observations (variable, location, obs_time)"
        if dialect != "mysql" else None,
    ]


class Database:
    def __init__(self, url: str | None = None):
        self.url = url or os.environ.get("DATABASE_URL") or "sqlite:///data/tnp_ocean.db"
        parsed = urlparse(self.url)
        scheme = parsed.scheme.split("+")[0]
        if scheme in ("postgres", "postgresql"):
            self.dialect = "postgresql"
        elif scheme in ("mysql", "mariadb"):
            self.dialect = "mysql"
        elif scheme == "sqlite":
            self.dialect = "sqlite"
        else:
            raise ValueError(f"Unsupported DATABASE_URL scheme: {parsed.scheme}")
        self._parsed = parsed
        self.ph = "?" if self.dialect == "sqlite" else "%s"
        self.conn = self._connect()

    # ------------------------------------------------------------------ connect
    def _connect(self):
        p = self._parsed
        if self.dialect == "sqlite":
            path = self.url[len("sqlite:///"):]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path or ":memory:")
            conn.execute("PRAGMA journal_mode=WAL")
            return conn
        if self.dialect == "postgresql":
            import psycopg  # type: ignore
            return psycopg.connect(self.url.replace("postgres://", "postgresql://", 1))
        import pymysql  # type: ignore
        qs = parse_qs(p.query)
        kwargs = dict(
            host=p.hostname, port=p.port or 3306,
            user=unquote(p.username or ""), password=unquote(p.password or ""),
            database=p.path.lstrip("/"), charset="utf8mb4", autocommit=False,
        )
        if qs.get("ssl", ["false"])[0].lower() in ("1", "true", "required"):
            kwargs["ssl"] = {"ssl": {}}
        return pymysql.connect(**kwargs)

    @contextmanager
    def cursor(self):
        cur = self.conn.cursor()
        try:
            yield cur
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def _t(self, dt: datetime | None):
        if dt is None:
            return None
        if not isinstance(dt, datetime):  # plain date
            dt = datetime(dt.year, dt.month, dt.day)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        dt = dt.replace(microsecond=0)
        return dt.strftime("%Y-%m-%d %H:%M:%S") if self.dialect == "sqlite" else dt

    # ------------------------------------------------------------------ schema
    def init_schema(self):
        with self.cursor() as cur:
            for stmt in _ddl(self.dialect):
                if stmt:
                    cur.execute(stmt)
        if self.dialect == "mysql":
            with self.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.statistics "
                    "WHERE table_schema = DATABASE() AND table_name='observations' "
                    "AND index_name='ix_obs_var_loc_time'"
                )
                if cur.fetchone()[0] == 0:
                    cur.execute("CREATE INDEX ix_obs_var_loc_time ON observations (variable, location, obs_time)")

    def sync_metadata(self, config: dict):
        """Write areas, points, sources and variables from config.yaml."""
        loc_rows = []
        for code, a in config.get("areas", {}).items():
            lon_min, lat_min, lon_max, lat_max = a["bbox"]
            loc_rows.append((code, a["name"], "area", (lat_min + lat_max) / 2, (lon_min + lon_max) / 2,
                             lon_min, lat_min, lon_max, lat_max))
        for code, p in config.get("points", {}).items():
            loc_rows.append((code, p["name"], "point", p["lat"], p["lon"], None, None, None, None))
        self._upsert("locations", ("code", "name", "kind", "lat", "lon", "lon_min", "lat_min", "lon_max", "lat_max"),
                     ("code",), loc_rows)
        src_rows = [(code, s["type"], s.get("description"), s.get("product"),
                     s.get("dataset_id") or s.get("short_name") or s.get("station_id"))
                    for code, s in config.get("sources", {}).items()]
        self._upsert("sources", ("code", "kind", "description", "product", "dataset"), ("code",), src_rows)
        var_rows = [(code, v["name"], v["unit"]) for code, v in config.get("variables", {}).items()]
        self._upsert("variables", ("code", "name", "unit"), ("code",), var_rows)

    # ------------------------------------------------------------------ upsert
    def _upsert_sql(self, table, cols, keys):
        ph = ", ".join([self.ph] * len(cols))
        collist = ", ".join(cols)
        upd = [c for c in cols if c not in keys]
        if self.dialect == "mysql":
            sets = ", ".join(f"{c}=VALUES({c})" for c in upd)
            return f"INSERT INTO {table} ({collist}) VALUES ({ph}) ON DUPLICATE KEY UPDATE {sets}"
        sets = ", ".join(f"{c}=excluded.{c}" for c in upd)
        return f"INSERT INTO {table} ({collist}) VALUES ({ph}) ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {sets}"

    def _upsert(self, table, cols, keys, rows, batch=2000):
        if not rows:
            return 0
        sql = self._upsert_sql(table, cols, keys)
        n = 0
        with self.cursor() as cur:
            for i in range(0, len(rows), batch):
                chunk = [tuple(_clean(v) for v in r) for r in rows[i:i + batch]]
                cur.executemany(sql, chunk)
                n += len(chunk)
        return n

    def upsert_observations(self, observations: list[Observation]) -> int:
        now = self._t(datetime.now(timezone.utc))
        rows = []
        for o in observations:
            rows.append((o.source, o.variable, o.location, self._t(o.obs_time),
                         o.val_mean, o.val_median, o.val_min, o.val_max, o.val_std,
                         o.n_valid, o.n_total, now))
        return self._upsert("observations", OBS_COLUMNS, KEY_COLUMNS, rows)

    # ------------------------------------------------------------------ queries
    def latest_time(self, source: str) -> datetime | None:
        with self.cursor() as cur:
            cur.execute(f"SELECT MAX(obs_time) FROM observations WHERE source = {self.ph}", (source,))
            v = cur.fetchone()[0]
        if v is None:
            return None
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

    def earliest_time(self, source: str) -> datetime | None:
        with self.cursor() as cur:
            cur.execute(f"SELECT MIN(obs_time) FROM observations WHERE source = {self.ph}", (source,))
            v = cur.fetchone()[0]
        if v is None:
            return None
        return datetime.fromisoformat(v) if isinstance(v, str) else v

    def log_run(self, source, mode, started, finished, range_start, range_end, n_rows, status, message=""):
        sql = (f"INSERT INTO harvest_runs (source, mode, started_at, finished_at, range_start, range_end, n_rows, status, message) "
               f"VALUES ({', '.join([self.ph] * 9)})")
        with self.cursor() as cur:
            cur.execute(sql, (source, mode, self._t(started), self._t(finished), self._t(range_start),
                              self._t(range_end), n_rows, status, (message or "")[:4000]))

    def backfill_floor(self, source: str) -> datetime | None:
        """Earliest date already covered by a successful backfill chunk."""
        with self.cursor() as cur:
            cur.execute(f"SELECT MIN(range_start) FROM harvest_runs WHERE source={self.ph} "
                        f"AND mode='backfill' AND status='ok'", (source,))
            v = cur.fetchone()[0]
        if v is None:
            return None
        return datetime.fromisoformat(v) if isinstance(v, str) else v

    def status(self):
        with self.cursor() as cur:
            cur.execute(
                "SELECT source, variable, location, COUNT(*), MIN(obs_time), MAX(obs_time), "
                "SUM(CASE WHEN val_mean IS NULL THEN 0 ELSE 1 END) "
                "FROM observations GROUP BY source, variable, location ORDER BY source, variable, location"
            )
            return cur.fetchall()

    def fetch_series(self, variable: str, location: str, source: str):
        with self.cursor() as cur:
            cur.execute(
                f"SELECT obs_time, val_mean FROM observations WHERE variable={self.ph} AND location={self.ph} "
                f"AND source={self.ph} AND val_mean IS NOT NULL ORDER BY obs_time",
                (variable, location, source),
            )
            rows = cur.fetchall()
        return [(datetime.fromisoformat(t) if isinstance(t, str) else t, v) for t, v in rows]

    def series_keys(self):
        with self.cursor() as cur:
            cur.execute("SELECT DISTINCT source, variable, location FROM observations")
            return cur.fetchall()

    def close(self):
        self.conn.close()
