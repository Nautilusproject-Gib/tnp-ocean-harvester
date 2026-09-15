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
        self.assertEqual(len(payload["climatology"]["doy_mean_p10_p90"]), 365)
        dates = [d[0] for d in payload["data"]]
        self.assertEqual(len(dates), len(set(dates)))
        latest = json.loads(Path(f"{self.tmp.name}/public/latest.json").read_text())
        self.assertEqual(latest[0]["variable"], "sst")

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


if __name__ == "__main__":
    unittest.main()
