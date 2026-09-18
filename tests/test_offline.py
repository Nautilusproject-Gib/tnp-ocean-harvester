"""Offline tests: parsers, statistics, planning, database upserts and export.

Run with:  python -m unittest discover -s tests -v
No network, no credentials and no xarray needed.
"""
import json
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harvester.db import Database, Observation  # noqa: E402
from harvester.export import daily_aggregate, export, merge_by_priority  # noqa: E402
from harvester.runner import date_chunks, plan_backfill, plan_update, run  # noqa: E402
from harvester.sources.stations import (  # noqa: E402
    parse_aeronet_daily, parse_isd_csv, parse_isd_precip, parse_isd_wnd, parse_openmeteo_hourly)
from harvester.stats import area_stats, circular_mean_deg, nearest_valid_cell  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

ISD_SAMPLE = '''"STATION","DATE","SOURCE","LATITUDE","LONGITUDE","ELEVATION","NAME","REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","WND","CIG","VIS","TMP","DEW","SLP","AA1","OC1"
"08495099999","2024-03-01T05:50:00","4","36.15","-5.349","4.6","GIBRALTAR, GI","FM-15","99999","V020","090,1,N,0082,1","99999,9,9,N","010000,1,9,9","+0155,1","+0120,1","99999,9","",""
"08495099999","2024-03-01T06:00:00","4","36.15","-5.349","4.6","GIBRALTAR, GI","FM-12","99999","V020","100,1,N,0090,1","99999,9,9,N","010000,1,9,9","+0160,1","+0120,1","10182,1","12,0004,9,1","0150,1"
"08495099999","2024-03-01T06:50:00","4","36.15","-5.349","4.6","GIBRALTAR, GI","FM-15","99999","V020","999,9,C,0000,1","99999,9,9,N","010000,1,9,9","+9999,9","+0120,1","99999,9","",""
"08495099999","2024-03-01T07:50:00","4","36.15","-5.349","4.6","GIBRALTAR, GI","FM-15","99999","V020","270,1,N,9999,9","99999,9,9,N","010000,1,9,9","+0170,3","+0120,1","99999,9","",""
'''

AERONET_SAMPLE = """AERONET Version 3;
Malaga
Version 3: AOD Level 1.5
The following data are automatically cloud cleared...
Contact: PI=...
AERONET_Site,Date(dd:mm:yyyy),Time(hh:mm:ss),Day_of_Year,AOD_500nm,440-870_Angstrom_Exponent
Malaga,01:07:2024,00:00:00,183,0.412,0.21
Malaga,02:07:2024,00:00:00,184,-999.,-999.
"""


class TestParsers(unittest.TestCase):
    def test_wnd(self):
        self.assertEqual(parse_isd_wnd("090,1,N,0082,1"), (90.0, 8.2))
        self.assertEqual(parse_isd_wnd("999,9,C,0000,1"), (None, 0.0))
        self.assertEqual(parse_isd_wnd("270,1,N,9999,9"), (270.0, None))
        self.assertEqual(parse_isd_wnd("270,3,N,0050,3"), (None, None))  # erroneous QC

    def test_precip(self):
        self.assertEqual(parse_isd_precip("12,0004,9,1"), (12, 0.4))
        self.assertEqual(parse_isd_precip("99,9999,9,9"), (None, None))

    def test_isd_csv(self):
        obs = parse_isd_csv(ISD_SAMPLE, "ncei_isd_lxgb", "gibraltar_airport")
        d = {(o.obs_time, o.variable): o.val_mean for o in obs}
        t6 = datetime(2024, 3, 1, 6)
        # 05:50 and 06:00 both round to 06:00; the later report wins
        self.assertAlmostEqual(d[(t6, "wind_speed")], 9.0)
        self.assertAlmostEqual(d[(t6, "air_temp")], 16.0)
        self.assertAlmostEqual(d[(t6, "precip_12h")], 0.4)
        self.assertAlmostEqual(d[(t6, "wind_gust")], 15.0)
        self.assertAlmostEqual(d[(t6, "slp")], 1018.2)
        t7 = datetime(2024, 3, 1, 7)
        self.assertEqual(d[(t7, "wind_speed")], 0.0)  # calm
        self.assertNotIn((t7, "air_temp"), d)          # missing temperature
        t8 = datetime(2024, 3, 1, 8)
        self.assertNotIn((t8, "air_temp"), d)          # QC flag 3 rejected
        self.assertNotIn((t8, "wind_speed"), d)

    def test_openmeteo(self):
        js = {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                         "wind_speed_10m": [5.1, None], "precipitation": [0.0, 1.2]}}
        obs = parse_openmeteo_hourly(js, {"wind_speed_10m": "wind_speed", "precipitation": "precip"}, "s", "p")
        self.assertEqual(len(obs), 3)

    def test_aeronet(self):
        obs = parse_aeronet_daily(AERONET_SAMPLE, "aeronet")
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].location, "aeronet_malaga")
        self.assertIn("aeronet_malaga", CONFIG["points"])
        self.assertEqual({o.variable for o in obs}, {"aeronet_aod_500", "aeronet_ae_440_870"})


class TestStats(unittest.TestCase):
    def test_area_stats(self):
        a = np.array([[1.0, np.nan], [3.0, 5.0]])
        s = area_stats(a)
        self.assertEqual(s["n_valid"], 3)
        self.assertEqual(s["n_total"], 4)
        self.assertAlmostEqual(s["val_mean"], 3.0)
        self.assertEqual(area_stats(np.full((2, 2), np.nan))["val_mean"], None)

    def test_nearest_valid(self):
        lats = np.array([36.0, 36.1, 36.2])
        lons = np.array([-5.4, -5.3, -5.2])
        valid = np.array([[True, True, True], [False, False, True], [True, True, True]])
        i, j, dist = nearest_valid_cell(lats, lons, valid, 36.1, -5.3, max_km=50)
        self.assertNotEqual((i, j), (1, 1))
        self.assertIsNone(nearest_valid_cell(lats, lons, np.zeros((3, 3), bool), 36.1, -5.3))

    def test_circular(self):
        self.assertAlmostEqual(circular_mean_deg([350, 10]) % 360, 0.0, places=6)


class TestPlanning(unittest.TestCase):
    def test_chunks(self):
        c = date_chunks(date(2020, 1, 1), date(2020, 1, 10), 4)
        self.assertEqual(c, [(date(2020, 1, 1), date(2020, 1, 4)), (date(2020, 1, 5), date(2020, 1, 8)),
                             (date(2020, 1, 9), date(2020, 1, 10))])
        self.assertEqual(date_chunks(date(2020, 1, 1), date(2020, 1, 10), 4, reverse=True)[0][1], date(2020, 1, 10))

    def test_update_plan(self):
        today = date(2026, 9, 15)
        cfg = CONFIG["sources"]["cmems_med_chl_nrt"]
        self.assertEqual(plan_update(cfg, None, today), (date(2026, 8, 16), today))
        self.assertEqual(plan_update(cfg, datetime(2026, 9, 14), today), (date(2026, 9, 7), today))
        seawifs = CONFIG["sources"]["nasa_par_seawifs"]
        self.assertIsNone(plan_update(seawifs, None, today))

    def test_backfill_plan(self):
        today = date(2026, 9, 15)
        self.assertIsNone(plan_backfill(CONFIG["sources"]["cmems_med_chl_nrt"], CONFIG, None, today, None, None))
        rng = plan_backfill(CONFIG["sources"]["cmems_med_chl_my"], CONFIG, datetime(2020, 1, 1), today, None, None)
        self.assertEqual(rng, (date(1997, 9, 16), date(2019, 12, 31)))


class FakeSource:
    """Injects deterministic data in place of a network source."""
    calls = []

    def __init__(self, code, cfg, config):
        self.code, self.cfg, self.config = code, cfg, config

    def missing_env(self):
        return []

    def fetch(self, start, end):
        FakeSource.calls.append((start, end))
        out = []
        d = start
        while d <= end:
            t = datetime(d.year, d.month, d.day)
            doy = d.timetuple().tm_yday
            out.append(Observation(self.code, "sst", "bay_of_gibraltar", t, 17 + 4 * np.sin(doy / 58.0),
                                   n_valid=10, n_total=12))
            d += timedelta(days=1)
        return out


class TestDatabaseAndExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(f"sqlite:///{self.tmp.name}/t.db")
        self.db.init_schema()
        self.db.init_schema()  # idempotent
        self.db.sync_metadata(CONFIG)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_upsert_idempotent(self):
        t = datetime(2024, 1, 1)
        self.db.upsert_observations([Observation("s", "sst", "bay_of_gibraltar", t, 16.0, n_valid=1, n_total=1)])
        self.db.upsert_observations([Observation("s", "sst", "bay_of_gibraltar", t, np.float64(16.5), n_valid=1, n_total=1),
                                     Observation("s", "sst", "bay_of_gibraltar", t + timedelta(days=1), float("nan"),
                                                 n_valid=0, n_total=1)])
        rows = self.db.status()
        self.assertEqual(rows[0][3], 2)   # two rows
        self.assertEqual(rows[0][6], 1)   # one with a value
        self.assertEqual(self.db.fetch_series("sst", "bay_of_gibraltar", "s")[0][1], 16.5)
        self.assertEqual(self.db.latest_time("s"), t + timedelta(days=1))

    def test_run_backfill_resume_and_export(self):
        import harvester.runner as runner_mod
        cfg = {**CONFIG, "sources": {"fake_rep": {"type": "fake", "enabled": True, "earliest": "2018-01-01",
                                                  "chunk_days": 400, "lookback_days": 3},
                                     "fake_nrt": {"type": "fake", "enabled": True, "rolling_window_days": 20}},
               "export": {"output_dir": f"{self.tmp.name}/public", "daily_priority": {"sst": ["fake_rep", "fake_nrt"]}}}
        orig = runner_mod.build_source
        runner_mod.build_source = lambda code, config: FakeSource(code, config["sources"][code], config)
        try:
            FakeSource.calls.clear()
            rep = run(cfg, self.db, mode="backfill", end=date(2022, 12, 31), log=lambda *a: None)
            self.assertFalse(rep.failed)
            self.assertEqual(FakeSource.calls[0][1], date(2022, 12, 31))   # newest chunk first
            self.assertEqual(FakeSource.calls[-1][0], date(2018, 1, 1))
            # a second backfill has nothing older to do
            FakeSource.calls.clear()
            run(cfg, self.db, mode="backfill", only=["fake_rep"], log=lambda *a: None)
            self.assertTrue(all(c[1] <= date(2017, 12, 31) for c in FakeSource.calls) or not FakeSource.calls)
            run(cfg, self.db, mode="update", log=lambda *a: None)
        finally:
            runner_mod.build_source = orig

        n = export(cfg, self.db, log=lambda *a: None)
        self.assertEqual(n, 1)
        payload = json.loads(Path(f"{self.tmp.name}/public/daily/sst__bay_of_gibraltar.json").read_text())
        self.assertEqual(payload["unit"], "degC")
        self.assertIsNotNone(payload["climatology"])
        self.assertEqual(len(payload["climatology"]["mean"]), 365)
        self.assertEqual(len(payload["climatology"]["sd"]), 365)
        rec = payload["record"]
        self.assertLessEqual(rec["min"], rec["mean"])
        self.assertLessEqual(rec["mean"], rec["max"])
        dates = [d[0] for d in payload["data"]]
        self.assertEqual(len(dates), len(set(dates)))
        latest = json.loads(Path(f"{self.tmp.name}/public/latest.json").read_text())
        self.assertEqual(latest[0]["variable"], "sst")

    def test_par_from_era5(self):
        base = datetime(2024, 6, 21)
        self.db.upsert_observations([Observation("openmeteo_era5", "sw_rad", "gibraltar_airport",
                                                 base + timedelta(hours=h), 300.0, n_valid=1, n_total=1)
                                     for h in range(24)])
        cfg = {**CONFIG, "export": {"output_dir": f"{self.tmp.name}/pub2", "daily_priority": {}}}
        export(cfg, self.db, log=lambda *a: None)
        payload = json.loads(Path(f"{self.tmp.name}/pub2/daily/par_era5__gibraltar_airport.json").read_text())
        self.assertAlmostEqual(payload["data"][0][1], 300 * 86400 * 0.48 * 4.57e-6, places=3)  # ~56.9
        self.assertEqual(payload["unit"], "einstein m-2 day-1")

    def test_dust_episodes(self):
        obs = []
        for i in range(30):
            day = datetime(2025, 3, 1) + timedelta(days=i)
            dust = 120.0 if 10 <= i <= 12 else 10.0          # one three-day episode
            obs += [Observation("openmeteo_cams_dust", "dust_sfc", "gibraltar_airport", day + timedelta(hours=h),
                                dust, n_valid=1, n_total=1) for h in range(24)]
            chl = 0.3 if i < 13 else 0.6                       # chlorophyll doubles afterwards
            obs.append(Observation("cmems_med_chl_my", "chl", "gibraltar_20km", day, chl, n_valid=50, n_total=60))
        self.db.upsert_observations(obs)
        cfg = {**CONFIG, "export": {**CONFIG["export"], "output_dir": f"{self.tmp.name}/pub3"}}
        export(cfg, self.db, log=lambda *a: None)
        ev = json.loads(Path(f"{self.tmp.name}/pub3/dust_events.json").read_text())
        self.assertEqual(len(ev["episodes"]), 1)
        ep = ev["episodes"][0]
        self.assertEqual((ep["start"], ep["end"], ep["days"]), ("2025-03-11", "2025-03-13", 3))
        chl = next(r for r in ep["response"] if r["variable"] == "chl")
        self.assertAlmostEqual(chl["change_pct"], 100.0)
        latest = json.loads(Path(f"{self.tmp.name}/pub3/latest.json").read_text())
        self.assertTrue(any(l["variable"] == "chl" for l in latest))

    def test_bad_pixels_are_ignored(self):
        from harvester.stats import mask_valid
        a = mask_valid(np.array([[0.3, 2.9e22], [0.5, -5.0]]), [0.01, 100])
        self.assertEqual(area_stats(a)["n_valid"], 2)
        self.assertAlmostEqual(area_stats(a)["val_mean"], 0.4)
        a = mask_valid(np.array([1.0, 50.0]), None, attrs={"valid_min": 0.0, "valid_max": 10.0})
        self.assertTrue(np.isnan(a[1]))

    def test_clean_and_median_export(self):
        day = datetime(2004, 10, 1)
        self.db.upsert_observations([
            Observation("cmems_med_chl_my", "chl", "gibraltar_20km", day, 2.9e22, val_median=0.41,
                        val_min=0.1, val_max=2.9e22, val_std=1e21, n_valid=40, n_total=60),
            Observation("cmems_med_chl_my", "chl", "gibraltar_20km", day + timedelta(days=1), 0.5, val_median=0.45,
                        val_min=0.2, val_max=1.1, val_std=0.2, n_valid=40, n_total=60)])
        cfg = {**CONFIG, "export": {**CONFIG["export"], "output_dir": f"{self.tmp.name}/pub4"}}
        export(cfg, self.db, log=lambda *a: None)
        payload = json.loads(Path(f"{self.tmp.name}/pub4/daily/chl__gibraltar_20km.json").read_text())
        self.assertEqual([d[1] for d in payload["data"]], [0.41, 0.45])     # medians, bad mean gone
        self.assertLess(payload["record"]["max"], 1)
        self.assertEqual(payload["statistic"], "median")

    def test_source_options(self):
        obs = []
        for i in range(10):
            t = datetime(2024, 1, 1, 12) + timedelta(days=i)
            obs.append(Observation("openmeteo_era5", "wind_speed", "gibraltar_airport", t, 5.0, n_valid=1, n_total=1))
            if i < 5:
                obs.append(Observation("ncei_isd_lxgb", "wind_speed", "gibraltar_airport", t, 3.0, n_valid=1, n_total=1))
        self.db.upsert_observations(obs)
        cfg = {**CONFIG, "export": {**CONFIG["export"], "output_dir": f"{self.tmp.name}/pub5"}}
        export(cfg, self.db, log=lambda *a: None)
        d = Path(f"{self.tmp.name}/pub5/daily")
        combined = json.loads((d / "wind_speed__gibraltar_airport.json").read_text())
        self.assertEqual([o["source"] for o in combined["source_options"]], ["openmeteo_era5", "ncei_isd_lxgb"])
        self.assertEqual({r[1] for r in combined["data"]}, {5.0})              # ERA5 preferred for wind
        station = json.loads((d / "wind_speed__gibraltar_airport__ncei_isd_lxgb.json").read_text())
        self.assertEqual(len(station["data"]), 5)
        self.assertEqual(station["source_labels"], ["Gibraltar Airport station (NOAA)"])

    def test_daily_aggregation(self):
        base = datetime(2024, 1, 1)
        hourly = [(base + timedelta(hours=h), 1.0) for h in range(24)]
        self.assertEqual(daily_aggregate(hourly, "precip").iloc[0], 24.0)
        self.assertTrue(daily_aggregate(hourly[:10], "precip").empty)  # incomplete day
        dirs = [(base, 350.0), (base + timedelta(hours=1), 10.0)]
        self.assertAlmostEqual(daily_aggregate(dirs, "wind_dir").iloc[0] % 360, 0.0, places=6)
        import pandas as pd
        a = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
        b = pd.Series([9.0, 9.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        m = merge_by_priority({"b": b, "a": a}, ["a", "b"])
        self.assertEqual(list(m["value"]), [1.0, 2.0, 9.0])
        self.assertEqual(list(m["source"]), ["a", "a", "b"])


class FakeDA:
    """Just enough of an xarray DataArray for the Copernicus source code."""
    def __init__(self, values, dims, attrs=None):
        self.values, self.dims, self.attrs = np.asarray(values, dtype="float64"), tuple(dims), attrs or {}
    def isel(self, sel):
        (d, i), = sel.items()
        ax = self.dims.index(d)
        return FakeDA(np.take(self.values, i, axis=ax), [x for x in self.dims if x != d], self.attrs)
    def transpose(self, *dims):
        return FakeDA(np.transpose(self.values, [self.dims.index(d) for d in dims]), dims, self.attrs)


class FakeDS:
    def __init__(self, coords, data_vars):
        self.coords, self.data_vars = coords, data_vars
        self.dims = list(coords)
    def __getitem__(self, k):
        if k in self.coords:
            return FakeDA(self.coords[k], [k]) if k != "time" else type("T", (), {"values": self.coords[k]})()
        return self.data_vars[k]
    def close(self):
        pass


class CopernicusTests(unittest.TestCase):
    def test_variable_specs_and_resolution(self):
        from harvester.sources.copernicus import resolve_variable, variable_specs
        specs = variable_specs({"variables": {"CHL": "chl", "DIATO": {"code": "diatoms", "match": "diatom"},
                                              "dissic": {"code": "dic", "scale": 1000}}})
        self.assertEqual([s["code"] for s in specs], ["chl", "diatoms", "dic"])
        self.assertEqual(specs[2]["scale"], 1000)
        ds_vars = {"chl": {}, "DIATOMS_CHL": {"long_name": "Mass concentration of diatoms"}}
        self.assertEqual(resolve_variable(ds_vars, specs[0]), "chl")
        self.assertEqual(resolve_variable({"Chl": {}}, specs[0]), "Chl")
        self.assertEqual(resolve_variable(ds_vars, specs[1]), "DIATOMS_CHL")
        self.assertIsNone(resolve_variable(ds_vars, specs[2]))

    def test_current_stats_direction(self):
        from harvester.sources.copernicus import current_stats
        stats, d = current_stats([[1.0, 1.0], [np.nan, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
        self.assertAlmostEqual(d, 90.0)                      # flowing east
        self.assertEqual(stats["n_valid"], 3)
        self.assertAlmostEqual(stats["val_mean"], 1.0)
        _, d = current_stats([[0.0]], [[-0.5]])
        self.assertAlmostEqual(d, 180.0)                     # flowing south
        _, d = current_stats([[-0.3]], [[0.0]])
        self.assertAlmostEqual(d, 270.0)                     # flowing west

    def _fake(self, data_vars, n_days=3):
        import pandas as pd
        lats = np.arange(35.80, 36.45, 0.042)
        lons = np.arange(-5.80, -4.45, 0.042)
        times = pd.date_range("2024-06-01", periods=n_days, freq="D").values
        dv = {k: FakeDA(fn(len(times), len(lats), len(lons)), ["time", "depth", "latitude", "longitude"], attrs)
              for k, (fn, attrs) in data_vars.items()}
        return FakeDS({"time": times, "latitude": lats, "longitude": lons, "depth": np.array([1.0, 3.0])}, dv)

    def test_currents_derived_from_model_grid(self):
        from harvester.sources.copernicus import CopernicusGrid
        ds = self._fake({"uo": (lambda t, y, x: np.full((t, 2, y, x), 0.3), {"units": "m s-1"}),
                         "vo": (lambda t, y, x: np.full((t, 2, y, x), 0.4), {"units": "m s-1"})})
        src = CopernicusGrid("cmems_med_cur_my", CONFIG["sources"]["cmems_med_cur_my"], CONFIG)
        src._open = lambda *a, **k: ds
        obs = src.fetch(date(2024, 6, 1), date(2024, 6, 3))
        speed = [o for o in obs if o.variable == "current_speed"]
        dirs = [o for o in obs if o.variable == "current_dir"]
        self.assertEqual(len(speed), 3 * len(CONFIG["areas"]))
        self.assertAlmostEqual(speed[0].val_mean, 0.5)
        self.assertAlmostEqual(dirs[0].val_mean, np.degrees(np.arctan2(0.3, 0.4)))

    def test_scale_and_valid_range(self):
        from harvester.sources.copernicus import CopernicusGrid
        ds = self._fake({"ph": (lambda t, y, x: np.full((t, 2, y, x), 8.1), {}),
                         "dissic": (lambda t, y, x: np.full((t, 2, y, x), 2.3), {"units": "mol m-3"}),
                         "talk": (lambda t, y, x: np.full((t, 2, y, x), 99.0), {})}, n_days=1)
        src = CopernicusGrid("cmems_med_car_my", CONFIG["sources"]["cmems_med_car_my"], CONFIG)
        src._open = lambda *a, **k: ds
        obs = {o.variable: o for o in src.fetch(date(2024, 6, 1), date(2024, 6, 1)) if o.location == "gibraltar_20km"}
        self.assertAlmostEqual(obs["dic"].val_mean, 2300.0)
        self.assertAlmostEqual(obs["ph"].val_mean, 8.1)
        self.assertEqual(obs["alkalinity"].n_valid, 0)        # 99 000 mmol m-3 is impossible, so masked

    def test_new_sources_are_consistent(self):
        from harvester.sources.copernicus import spec_codes, variable_specs
        for v, meta in CONFIG["variables"].items():
            self.assertLessEqual(set(meta), {"group", "name", "unit", "valid_range", "daily_statistic"},
                                 f"{v}: stray keys (unquoted comma in the name?)")
        for code, cfg in CONFIG["sources"].items():
            if not cfg["type"].startswith("copernicus"):
                continue
            codes = [c for s in variable_specs(cfg) for c in spec_codes(s)]
            if cfg.get("derive") == "currents":
                codes = ["current_speed", "current_dir"]
            for v in codes:
                self.assertIn(v, CONFIG["variables"], f"{code}: {v} missing from variables")
                self.assertIn(CONFIG["variables"][v]["group"], ("physical", "chemical", "biological"))
        for v, srcs in CONFIG["export"]["daily_priority"].items():
            for s in srcs:
                self.assertIn(s, CONFIG["sources"], f"priority for {v} names unknown source {s}")


class MarineHeatwaveTests(unittest.TestCase):
    def _sst(self, years=range(1991, 2021), noise=0.3, seed=1):
        import pandas as pd
        idx = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
        rng = np.random.default_rng(seed)
        doy = idx.dayofyear.values
        vals = 18.5 + 3.5 * np.cos(2 * np.pi * (doy - 232) / 365.25) + rng.normal(0, noise, len(idx))
        return pd.Series(vals, index=idx)

    def test_detects_event_joins_gaps_and_categorises(self):
        import pandas as pd
        from harvester.export import marine_heatwaves
        s = self._sst()
        s.loc["2018-07-01":"2018-07-06"] += 3.0          # 6 days
        s.loc["2018-07-09":"2018-07-15"] += 3.0          # 7 days after a 2-day gap: joined
        s.loc["2019-08-01":"2019-08-03"] += 3.0          # only 3 days: not a heatwave
        res = marine_heatwaves(s, baseline=(1991, 2020))
        ev = [e for e in res["events"] if e["start"].startswith("2018-07")]
        self.assertEqual(len(ev), 1)
        self.assertEqual((ev[0]["start"], ev[0]["end"], ev[0]["days"]), ("2018-07-01", "2018-07-15", 15))
        self.assertGreaterEqual(ev[0]["category"], 3)    # +3 C against a threshold about 0.4 C above normal
        self.assertFalse(any(e["start"].startswith("2019-08") for e in res["events"]))
        self.assertEqual(res["baseline"], [1991, 2020])
        self.assertEqual(len(res["thresh"]), 366)
        self.assertTrue(all(t > m for t, m in zip(res["thresh"], res["seas"])))
        # random noise alone rarely makes 5-day runs above the 90th percentile
        self.assertLess(len(res["events"]), 40)

    def test_status_ongoing_and_short_record(self):
        from harvester.export import marine_heatwaves
        s = self._sst(years=range(2015, 2025))           # shorter than the baseline: uses all years
        s.iloc[-8:] += 2.5
        res = marine_heatwaves(s, baseline=(1991, 2020))
        self.assertEqual(res["baseline"], [2015, 2024])
        self.assertEqual(res["status"]["state"], "heatwave")
        self.assertEqual(res["status"]["days"], 8)
        s2 = self._sst(years=range(2015, 2025))
        s2.iloc[-2:] += 2.5
        self.assertEqual(marine_heatwaves(s2)["status"]["state"], "warm")


class DepthProfileTests(unittest.TestCase):
    def _profile_ds(self, n_days=2):
        import pandas as pd
        lats = np.arange(35.80, 36.45, 0.042)
        lons = np.arange(-5.80, -4.45, 0.042)
        # model 'elevation' axis: negative, deepest first, as in the Copernicus Med datasets
        elev = -np.array([300.0, 200.0, 150.0, 100.0, 50.0, 30.0, 19.7, 10.5, 1.0])
        times = pd.date_range("2024-06-01", periods=n_days, freq="D").values
        depth = np.abs(elev)
        temp = 22.0 - 0.03 * depth                         # 22 C at the surface, cooling with depth
        sal = 36.4 + 2.2 * np.clip(depth / 200.0, 0, 1)    # 37.5 reached at 100 m
        shape = (len(times), len(elev), len(lats), len(lons))
        T = np.broadcast_to(temp[None, :, None, None], shape).copy()
        S = np.broadcast_to(sal[None, :, None, None], shape).copy()
        T[:, :4, :, :5] = np.nan                            # shallow seabed in the first columns
        return FakeDS({"time": times, "elevation": elev, "latitude": lats, "longitude": lons},
                      {"thetao": FakeDA(T, ["time", "elevation", "latitude", "longitude"], {"units": "degrees_C"}),
                       "so": FakeDA(S, ["time", "elevation", "latitude", "longitude"], {"units": "1e-3"})})

    def test_isoline_depth(self):
        from harvester.sources.copernicus import isoline_depth
        z = np.array([1.0, 50, 100, 200])
        prof = np.array([[36.5, 37.0, 37.4, 38.0], [37.6, 37.8, 38, 38], [36.0, 36.1, np.nan, np.nan]])
        out = isoline_depth(prof, z, 37.5)
        self.assertAlmostEqual(out[0], 100 + 0.1 / 0.6 * 100, places=5)
        self.assertEqual(out[1], 0.0)
        self.assertTrue(np.isnan(out[2]))

    def test_temperature_at_depth_and_interface(self):
        from harvester.sources.copernicus import CopernicusGrid
        ds = self._profile_ds()
        t = CopernicusGrid("cmems_med_temp_my", CONFIG["sources"]["cmems_med_temp_my"], CONFIG)
        t._open = lambda *a, **k: ds
        obs = {(o.variable, o.location): o for o in t.fetch(date(2024, 6, 1), date(2024, 6, 1))}
        self.assertAlmostEqual(obs[("temp_surface", "gibraltar_20km")].val_mean, 22.0 - 0.03, places=4)
        self.assertAlmostEqual(obs[("temp_10m", "gibraltar_20km")].val_mean, 22.0 - 0.03 * 10.5, places=4)
        self.assertAlmostEqual(obs[("temp_20m", "gibraltar_20km")].val_mean, 22.0 - 0.03 * 19.7, places=4)
        self.assertAlmostEqual(obs[("temp_100m", "western_alboran")].val_mean, 22.0 - 3.0, places=4)
        s = CopernicusGrid("cmems_med_salprof_my", CONFIG["sources"]["cmems_med_salprof_my"], CONFIG)
        s._open = lambda *a, **k: ds
        obs = {(o.variable, o.location): o for o in s.fetch(date(2024, 6, 1), date(2024, 6, 1))}
        self.assertAlmostEqual(obs[("interface_depth", "strait_of_gibraltar")].val_mean, 100.0, places=3)
        self.assertAlmostEqual(obs[("salinity_200m", "strait_of_gibraltar")].val_mean, 38.6, places=4)

    def test_plan_update_with_forecast_days(self):
        cfg = {"lookback_days": 3, "forecast_days": 5}
        today = date(2026, 9, 17)
        start, end = plan_update(cfg, datetime(2026, 9, 22), today)       # forecast rows already stored
        self.assertEqual((start, end), (date(2026, 9, 14), date(2026, 9, 22)))


class DerivedProductTests(unittest.TestCase):
    def test_upwelling_index_sign_and_events(self):
        import pandas as pd
        from harvester.derived import daily_upwelling_index, ekman_upwelling_index, upwelling_events
        west = ekman_upwelling_index([8.0], [250.0], 70, 36.5)[0]      # WSW wind blowing along the coast
        east = ekman_upwelling_index([8.0], [70.0], 70, 36.5)[0]       # Levanter
        across = ekman_upwelling_index([8.0], [340.0], 70, 36.5)[0]    # blowing across the coast
        self.assertGreater(west, 800)
        self.assertAlmostEqual(east, -west)
        self.assertLess(abs(across), 1e-6)
        hours = pd.date_range("2024-07-01", periods=24 * 10, freq="h")
        speed = [(t, 9.0 if 3 <= t.day <= 5 else 3.0) for t in hours]
        direc = [(t, 270.0) for t in hours]
        daily = daily_upwelling_index(speed, direc, 70, 36.5)
        self.assertEqual(len(daily), 10)
        events, status = upwelling_events(daily, threshold=500, min_days=2)
        self.assertEqual([(e["start"], e["end"]) for e in events], [("2024-07-03", "2024-07-05")])
        self.assertEqual(status["state"], "none")

    def test_seabed_light(self):
        import pandas as pd
        from harvester.derived import kd_par_from_kd490, light_at_depths
        idx = pd.date_range("2024-06-01", periods=3, freq="D")
        par = pd.Series([50.0, 50.0, np.nan], index=idx)
        kd = pd.Series([0.05, 0.10, 0.1], index=idx)
        out = light_at_depths(par, kd, [10])
        self.assertEqual(len(out[10]), 2)
        kp = 0.0864 + 0.884 * 0.05 - 0.00137 / 0.05
        self.assertAlmostEqual(kd_par_from_kd490(0.05), kp)
        self.assertAlmostEqual(out[10].iloc[0], 50 * np.exp(-kp * 10))
        self.assertLess(out[10].iloc[1], out[10].iloc[0])          # murkier water, less light

    def test_bloom_detection_with_gaps_and_triggers(self):
        import pandas as pd
        from harvester.derived import attach_triggers, detect_blooms
        idx = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        rng = np.random.default_rng(3)
        chl = 10 ** (np.log10(0.3) + 0.2 * np.cos(2 * np.pi * (idx.dayofyear - 75) / 365) + rng.normal(0, 0.08, len(idx)))
        s = pd.Series(chl, index=idx)
        s.loc["2024-10-10":"2024-10-20"] *= 4
        s = s[rng.random(len(s)) > 0.35]                         # cloud gaps
        s = s.drop(pd.Timestamp("2024-10-14"), errors="ignore").drop(pd.Timestamp("2024-10-15"), errors="ignore")
        res = detect_blooms(s)
        ev = [e for e in res["events"] if e["start"].startswith("2024-10")]
        self.assertEqual(len(ev), 1)
        self.assertLessEqual(ev[0]["start"], "2024-10-12")
        self.assertGreaterEqual(ev[0]["end"], "2024-10-18")
        self.assertGreater(ev[0]["peak_ratio"], 3)
        attach_triggers(ev, {"upwelling": [{"start": "2024-10-02", "end": "2024-10-06"}],
                             "dust": [{"start": "2024-08-01", "end": "2024-08-02"}]})
        self.assertEqual([t["type"] for t in ev[0]["triggers"]], ["upwelling"])
        self.assertEqual(len(res["normal"]), 366)

    def test_tide_fit_prediction_and_surge(self):
        import pandas as pd
        from harvester.derived import EPOCH, fit_tide, tide_analysis
        idx = pd.date_range("2023-01-01", "2024-12-31 23:00", freq="h")
        t = ((idx - EPOCH) / pd.Timedelta(hours=1)).values
        tide = 0.32 * np.cos(np.radians(28.9841042 * t) - 1.1) + 0.11 * np.cos(np.radians(30.0 * t) - 0.4) \
            + 0.04 * np.cos(np.radians(15.0410686 * t) - 2.0)
        level = 1.8 + tide + np.random.default_rng(1).normal(0, 0.02, len(t))
        s = pd.Series(level, index=idx)
        s.loc["2024-12-20":"2024-12-22"] += 0.25                  # a storm surge
        fit = fit_tide(s[s.index.year == 2023])
        self.assertAlmostEqual(fit["constituents"]["M2"]["amp"], 0.32, places=2)
        self.assertAlmostEqual(fit["constituents"]["S2"]["amp"], 0.11, places=2)
        self.assertAlmostEqual(fit["mean"], 1.8, places=2)
        level_d, surge_d, payload = tide_analysis(s, datetime(2024, 12, 31, 12), predict_days=3, datum="msl")
        self.assertGreater(surge_d.loc["2024-12-21"], 0.2)
        self.assertLess(abs(surge_d.loc["2024-06-01"]), 0.05)
        highs = [e for e in payload["extremes"] if e["type"] == "high"]
        self.assertGreaterEqual(len(highs), 5)                    # about two high waters a day
        self.assertTrue(all(0.15 < e["height"] < 0.5 for e in highs))      # neap to spring high waters
        times = [pd.Timestamp(e["time"]) for e in payload["extremes"]]
        gaps = np.diff([x.value for x in times]) / 3.6e12
        self.assertTrue(all(5 < g < 7.5 for g in gaps))          # semidiurnal: high and low about 6 h apart

    def test_chart_datum_heights(self):
        """Heights should be given above chart datum by default, as printed tide tables do."""
        import pandas as pd
        from harvester.derived import EPOCH, tide_analysis
        idx = pd.date_range("2013-01-01", "2024-12-31 23:00", freq="h")
        t = ((idx - EPOCH) / pd.Timedelta(hours=1)).values
        amps = {28.9841042: 0.35, 30.0: 0.11, 28.4397295: 0.07, 15.0410686: 0.04, 13.9430356: 0.03}
        tide = sum(a * np.cos(np.radians(w * t) - 1.0) for w, a in amps.items())
        s = pd.Series(1.9 + tide, index=idx)
        _, _, chart = tide_analysis(s, datetime(2024, 12, 20, 12), predict_days=6, datum="chart", sample_offset_minutes=0)
        _, _, msl = tide_analysis(s, datetime(2024, 12, 20, 12), predict_days=6, datum="msl", sample_offset_minutes=0)
        z0 = chart["chart_datum_below_msl"]
        self.assertAlmostEqual(z0, sum(amps.values()), delta=0.06)      # close to the lowest astronomical tide
        self.assertEqual(chart["mean_sea_level"], z0)
        self.assertTrue(all(e["height"] >= 0 for e in chart["extremes"]))
        lows = [e["height"] for e in chart["extremes"] if e["type"] == "low"]
        self.assertLess(min(lows), 0.1)                                 # lowest tides sit close to the datum
        for a1, b1 in zip(chart["extremes"], msl["extremes"]):
            self.assertAlmostEqual(a1["height"] - b1["height"], z0, delta=0.02)
            self.assertEqual(a1["time"], b1["time"])                    # only the reference level changes
        # a published datum can be forced
        _, _, forced = tide_analysis(s, datetime(2024, 12, 20, 12), predict_days=2, chart_datum_offset_m=0.48,
                                     sample_offset_minutes=0)
        self.assertEqual(forced["chart_datum_below_msl"], 0.48)

    def test_hourly_means_are_shifted_to_the_middle_of_the_hour(self):
        """Hourly means are labelled at the start of the hour, so predictions must be shifted 30 minutes."""
        import pandas as pd
        from harvester.derived import EPOCH, tide_analysis
        from harvester.sources.stations import parse_ioc_sealevel
        idx = pd.date_range("2025-09-01", "2026-09-16 05:00", freq="5min")
        t = ((idx - EPOCH) / pd.Timedelta(hours=1)).values
        v = 1.0 + 0.35 * np.cos(np.radians(28.9841042 * t) - 1.0) + 0.12 * np.cos(np.radians(30.0 * t) - 0.4)
        true = pd.Series(v, index=idx)
        hourly = parse_ioc_sealevel([{"slevel": float(x), "stime": ts.strftime("%Y-%m-%d %H:%M:%S"), "sensor": "rad"}
                                     for ts, x in zip(idx, v)], min_per_hour=10)
        self.assertEqual(hourly.index[0].minute, 0)
        def errors(offset):
            _, _, pl = tide_analysis(hourly, datetime(2026, 9, 14), predict_days=1, sample_offset_minutes=offset)
            out = []
            for e in pl["extremes"]:
                tp = pd.Timestamp(e["time"].replace("Z", ""))
                w = true[(true.index > tp - timedelta(hours=3)) & (true.index < tp + timedelta(hours=3))]
                if len(w) < 60:
                    continue
                peak = w.idxmax() if e["type"] == "high" else w.idxmin()
                out.append((tp - peak).total_seconds() / 60)
            return out
        self.assertTrue(all(-32 < e < -24 for e in errors(0)), errors(0))       # the bug: half an hour early
        self.assertTrue(all(abs(e) <= 6 for e in errors(30)), errors(30))       # fixed

    def test_ioc_parser(self):
        from harvester.sources.stations import parse_ioc_sealevel
        recs = []
        for m in range(120):
            t = datetime(2026, 9, 15, 0, 0) + timedelta(minutes=m)
            recs.append({"slevel": 0.6 + 0.001 * m, "stime": t.strftime("%Y-%m-%d %H:%M:%S"), "sensor": "rad"})
            recs.append({"slevel": 5.0, "stime": t.strftime("%Y-%m-%d %H:%M:%S"), "sensor": "pr1"})
            recs.append({"slevel": 0.01, "stime": t.strftime("%Y-%m-%d %H:%M:%S"), "sensor": "pr2"})
        recs[40]["slevel"] = 3.0                                 # a spike
        h = parse_ioc_sealevel(recs)
        self.assertEqual(len(h), 2)
        self.assertAlmostEqual(h.iloc[0], 0.6 + 0.001 * 29.5, delta=0.002)     # radar chosen over pr1

    def test_ioc_station_fallback(self):
        from harvester.sources.stations import IocSeaLevel
        src = IocSeaLevel("ioc_gibraltar", CONFIG["sources"]["ioc_gibraltar"], CONFIG)
        asked = []
        def fake_get(code, cur, stop):
            asked.append(code)
            if code == "gibr3":
                return []
            return [{"slevel": 1.0, "stime": (datetime(cur.year, cur.month, cur.day) + timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"),
                     "sensor": "rad"} for m in range(0, 60)]
        src._get = fake_get
        src.cfg = {**src.cfg, "pause_seconds": 0}
        obs = src.fetch(date(2012, 3, 1), date(2012, 3, 1))
        self.assertEqual(asked, ["gibr3", "gibr"])
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].location, "gibraltar_tide_gauge")


class ExportNewProductsTests(unittest.TestCase):
    def test_forecast_split_upwelling_light_blooms_tides(self):
        import pandas as pd
        tmp = tempfile.TemporaryDirectory()
        db = Database(f"sqlite:///{tmp.name}/t.db"); db.init_schema()
        today = datetime(2026, 9, 17)
        rng = np.random.default_rng(0)
        obs = []
        days = pd.date_range("2020-01-01", today, freq="D")
        for d in days:
            dd = d.to_pydatetime()
            seas = np.cos(2 * np.pi * (d.dayofyear - 232) / 365)
            obs.append(Observation("cmems_med_sst_rep", "sst", "gibraltar_20km", dd, 19 + 3 * seas + rng.normal(0, .3), n_valid=9, n_total=9))
            obs.append(Observation("cmems_med_temp_my", "temp_surface", "gibraltar_20km", dd, 18.5 + 3 * seas, n_valid=9, n_total=9))
            obs.append(Observation("nasa_par_modisa", "par", "gibraltar_20km", dd, 40.0, n_valid=4, n_total=4))
            obs.append(Observation("cmems_med_kd490_my", "kd490", "gibraltar_20km", dd, 0.06, val_median=0.06, n_valid=9, n_total=9))
            obs.append(Observation("cmems_med_chl_my", "chl", "gibraltar_20km", dd, 0.3, val_median=0.3 * (1 + rng.normal(0, .1)), n_valid=9, n_total=9))
        for k in range(1, 6):                                       # model forecast days
            d = today + timedelta(days=k)
            obs.append(Observation("cmems_med_temp_anfc", "temp_surface", "gibraltar_20km", d, 22.0, n_valid=9, n_total=9))
        for h in range(-24 * 20, 24 * 5):                           # hourly wind incl. forecast hours
            t = today + timedelta(hours=h)
            src = "openmeteo_era5" if h < -24 * 5 else "openmeteo_forecast"
            obs.append(Observation(src, "wind_speed", "gibraltar_airport", t, 9.0, n_valid=1, n_total=1))
            obs.append(Observation(src, "wind_dir", "gibraltar_airport", t, 260.0, n_valid=1, n_total=1))
        idx = pd.date_range(today - timedelta(days=120), today - timedelta(hours=1), freq="h")
        tt = ((idx - pd.Timestamp("2000-01-01")) / pd.Timedelta(hours=1)).values
        for t, v in zip(idx, 1.0 + 0.4 * np.cos(np.radians(28.9841042 * tt))):
            obs.append(Observation("ioc_gibraltar", "sea_level_hourly", "gibraltar_tide_gauge", t.to_pydatetime(), float(v), n_valid=1, n_total=1))
        db.upsert_observations(obs)
        cfg = {**CONFIG, "export": {**CONFIG["export"], "output_dir": f"{tmp.name}/pub", "site_dir": f"{tmp.name}/nosite"}}
        export(cfg, db, log=lambda *a: None, today=today.date())
        out = Path(f"{tmp.name}/pub")
        daily = json.loads((out / "daily" / "temp_surface__gibraltar_20km.json").read_text())
        self.assertLessEqual(daily["data"][-1][0], "2026-09-17")               # no future days in the record
        fc = json.loads((out / "forecast" / "sst__gibraltar_20km.json").read_text())
        self.assertEqual(len(fc["data"]), 5)
        self.assertAlmostEqual(fc["bias_offset"], 0.5, delta=0.3)              # satellite runs ~0.5 C warmer
        self.assertAlmostEqual(fc["data"][0][1], 22.0 + fc["bias_offset"], places=3)
        up = json.loads((out / "upwelling.json").read_text())
        self.assertEqual(up["status"]["state"], "upwelling")
        self.assertEqual(len(up["forecast"]), 4)                             # 18-21 Sept: whole forecast days
        self.assertTrue((out / "daily" / "par_10m__gibraltar_20km.json").exists())
        self.assertTrue((out / "daily" / "upwelling_index__gibraltar_airport.json").exists())
        self.assertIn("gibraltar_20km", json.loads((out / "blooms.json").read_text()))
        tides = json.loads((out / "tides.json").read_text())
        self.assertGreater(len(tides["extremes"]), 10)
        self.assertTrue((out / "daily" / "surge__gibraltar_tide_gauge.json").exists())
        meta = json.loads((out / "meta.json").read_text())
        self.assertIn("sst__gibraltar_20km", meta["forecasts"])
        db.close()
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
