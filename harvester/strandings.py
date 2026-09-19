"""TNP's strandings log: the record of truth for animals found dead, stranded or entangled.

This is TNP's own data, kept by hand in a spreadsheet, so the reader is forgiving by design. It
finds its own header row, tolerates trailing spaces and inconsistent wording, and never discards a
record it cannot fully understand: an entry with a species and no location is still a stranding.

Nothing here is published at full precision. Locations are place names as written, and the public
summary carries counts and dates rather than who found the animal.
"""
from __future__ import annotations

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd

# Columns as the log writes them, and the names we use. Matching ignores case, spaces and colons,
# because "Collected by:" and "Collected By" are the same column in anyone's book.
COLUMN_ALIASES = {
    "date": "date", "location": "location", "species": "species", "condition": "condition",
    "collectedby": "collected_by", "cod": "cause", "causeofdeath": "cause",
    "takento": "taken_to", "report": "report", "datereceived": "date_received",
}

# Condition wording varies with whoever filled the row in; these are the states that matter.
CONDITION_RULES = (
    ("alive", r"alive|released|rescued|live\b"),
    ("fresh", r"fresh|recently dead|recent"),
    ("moderate", r"moderate"),
    ("decomposed", r"decompos|carcass|skeletal"),
    ("dead", r"dead|deceased|euthanis|euthaniz"),
)

# What kind of animal, from the name as written. Order matters: the first rule that matches wins.
GROUP_RULES = (
    ("cetacean", r"dolphin|whale|orca|porpoise|cetacean"),
    ("turtle", r"turtle"),
    ("shark or ray", r"shark|ray\b|skate|dogfish|spurdog|tope|catshark"),
    ("seabird", r"gull|gannet|cormorant|shearwater|razorbill|puffin|tern|petrel|guillemot|"
                r"kittiwake|auk|booby|shag"),
    ("fish", r"tuna|sunfish|mola|eel|sardinell|triggerfish|seahorse|bass|bream|fish"),
    ("other bird", r"vulture|eagle|swift|martin|partridge|patridge|falcon|kestrel|heron|stork|swan|"
                   r"duck|pigeon|swallow|hawk|owl"),
    ("other mammal", r"otter|bat\b|fox|monkey|macaque"),
)

MARINE_GROUPS = ("cetacean", "turtle", "shark or ray", "seabird", "fish")


class StrandingsError(RuntimeError):
    """The log could not be read. Never fatal: the rest of the dashboard does not depend on it."""


def _key(name) -> str:
    return re.sub(r"[^a-z]", "", str(name).lower())


def find_header(rows: list) -> int | None:
    """The row that holds the column names, wherever the spreadsheet happens to start.

    The log has empty columns to the left of the table and could gain a title row at any time, so
    the header is found by looking for one, not assumed to be the first row.
    """
    for i, row in enumerate(rows[:20]):
        keys = {_key(c) for c in row if c is not None}
        if "date" in keys and "species" in keys:
            return i
    return None


def read_log(data, sheet: str | None = None) -> pd.DataFrame:
    """A strandings log (.xlsx bytes, or CSV text) as a frame with our column names."""
    if isinstance(data, (bytes, bytearray)) and data[:2] == b"PK":      # xlsx is a zip
        try:
            import openpyxl
        except ImportError as e:      # pragma: no cover - depends on the install
            raise StrandingsError("reading the log needs openpyxl (add it to requirements.txt)") from e
        wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb[wb.sheetnames[0]]
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
    else:
        # csv, not pandas: a spreadsheet saved as CSV has ragged rows, and read_csv treats the
        # first long line as an error rather than as a row with an extra cell.
        import csv as _csv
        text = data.decode("utf-8-sig") if isinstance(data, (bytes, bytearray)) else data
        rows = [list(r) for r in _csv.reader(io.StringIO(text))]
    head = find_header(rows)
    if head is None:
        return pd.DataFrame(columns=["date", "species"])
    header = [_key(c) for c in rows[head]]
    names, seen = [], 0
    for h in header:
        mapped = COLUMN_ALIASES.get(h)
        if mapped is None:
            seen += 1
            mapped = f"unused_{seen}"
        names.append(mapped)
    body = [r + [None] * (len(names) - len(r)) for r in rows[head + 1:]]
    df = pd.DataFrame([r[:len(names)] for r in body], columns=names)
    return df.loc[:, ~df.columns.str.startswith("unused_")]


def _text(s: pd.Series) -> pd.Series:
    """A column as tidy text. Empty cells arrive as None, NaN or the strings those print as."""
    out = s.where(s.notna(), "").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return out.mask(out.str.lower().isin(["nan", "none", "nat", "-", "n/a"]), "")


def split_count(species) -> tuple:
    """("Razorbills x 2") -> ("Razorbill", 2). A count hidden in the name is still a count."""
    species = "" if species is None else str(species)
    m = re.search(r"\bx\s*(\d+)\b", species, flags=re.I)
    n = int(m.group(1)) if m else 1
    name = re.sub(r"\bx\s*\d+\b", "", species, flags=re.I).strip(" ,")
    if n > 1:
        name = re.sub(r"s$", "", name)          # "Razorbills x 2" is two Razorbills
    return name, n


def classify(species, rules=GROUP_RULES) -> str:
    low = str(species or "").lower()
    for group, pattern in rules:
        if re.search(pattern, low):
            return group
    return "unknown"


def condition_state(text) -> str:
    low = str(text or "").lower()
    for state, pattern in CONDITION_RULES:
        if re.search(pattern, low):
            return state
    return "unknown"


def repair_swapped_columns(out: pd.DataFrame, rules=GROUP_RULES) -> pd.DataFrame:
    """Put a row right when the species was typed into the location column, or the pair swapped.

    It happens in any hand-kept log, and the animal is the part worth saving: a row reading
    location "Common Dolphin", species "Ocean Village" is a common dolphin at Ocean Village.
    """
    species_like = out["species"].map(lambda v: classify(v, rules) != "unknown")
    location_like = out["location"].map(lambda v: classify(v, rules) != "unknown")
    swap = location_like & ~species_like
    if swap.any():
        loc, sp = out.loc[swap, "location"].copy(), out.loc[swap, "species"].copy()
        out.loc[swap, "species"], out.loc[swap, "location"] = loc, sp
    return out


def clean(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Normalise the log: dates, names, counts, animal group and condition."""
    cfg = config or {}
    if df.empty:
        return pd.DataFrame(columns=["date", "species", "group", "count"])
    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df.get("date"), errors="coerce", dayfirst=True)
    for col in ("location", "species", "condition", "cause", "taken_to", "report"):
        out[col] = _text(df[col]) if col in df.columns else ""
    # a row with no date and no species is a blank line in the spreadsheet, not a record
    out = out[(out["species"] != "") | out["date"].notna()]
    pairs = [split_count(s) for s in out["species"]]
    out["species"] = [p[0] for p in pairs]
    out["count"] = [p[1] for p in pairs]
    rules = tuple((g["group"], g["match"]) for g in cfg.get("group_rules", [])) or GROUP_RULES
    out = repair_swapped_columns(out, rules)
    out["group"] = [classify(s, rules) for s in out["species"]]
    out.loc[out["species"] == "", "species"] = "Unknown"
    out["marine"] = out["group"].isin(cfg.get("marine_groups") or MARINE_GROUPS)
    out["state"] = [condition_state(c) for c in out["condition"]]
    out["year"] = out["date"].dt.year
    return out.sort_values("date").reset_index(drop=True)


def summary(records: pd.DataFrame, recent_days: int = 365, today=None) -> dict:
    """The public summary: counts and dates, no finders' names, no free text."""
    if records.empty:
        return {"records": 0, "animals": 0, "by_year": {}, "by_group": {}, "recent": []}
    today = pd.Timestamp(today or datetime.utcnow().date())
    dated = records[records["date"].notna()]
    recent = dated[dated["date"] > today - pd.Timedelta(days=recent_days)]
    return {
        "records": int(len(records)),
        "animals": int(records["count"].sum()),
        "first": dated["date"].min().strftime("%Y-%m-%d") if len(dated) else None,
        "last": dated["date"].max().strftime("%Y-%m-%d") if len(dated) else None,
        "by_year": {str(int(y)): int(n) for y, n in
                    dated.groupby(dated["date"].dt.year)["count"].sum().items()},
        "by_group": {g: int(n) for g, n in
                     records.groupby("group")["count"].sum().sort_values(ascending=False).items()},
        "by_state": {s: int(n) for s, n in records.groupby("state")["count"].sum().items()},
        "by_month": [int(dated.loc[dated["date"].dt.month == m, "count"].sum())
                     for m in range(1, 13)],
        "marine_records": int(records.loc[records["marine"], "count"].sum()),
        "recent_days": recent_days,
        "recent_count": int(recent["count"].sum()),
        "top_species": {s: int(n) for s, n in
                        records.groupby("species")["count"].sum()
                        .sort_values(ascending=False).head(8).items()},
        "recent": [{"date": r.date.strftime("%Y-%m-%d"), "species": r.species, "group": r.group,
                    "location": r.location or None, "state": r.state, "count": int(r.count)}
                   for r in recent.sort_values("date", ascending=False).head(20).itertuples()],
    }
