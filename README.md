# TNP Gibraltar Ocean Observatory: data harvester

This harvester collects satellite, model and station data for the waters around Gibraltar, stores it in a database and exports compact JSON for a dashboard on The Nautilus Project website.

It runs every morning on GitHub Actions. A separate backfill workflow works back through each archive as far as that source goes.

## What it collects

| Variable | Source | Resolution | Record starts | Account |
|---|---|---|---|---|
| Chlorophyll-a (CHL) | Copernicus Marine, Med ocean colour L3 (MY `009_143` + NRT `009_141`) | 1 km, daily | Sept 1997 | Copernicus Marine (free) |
| KD490 | Copernicus Marine, Med transparency L3 (MY + NRT) | 1 km, daily | Sept 1997 | Copernicus Marine |
| SST (foundation, L4 gap-free) | Copernicus Marine, Med SST reprocessed `010_021` + NRT `010_004` | 0.05° / 0.01°, daily | 1982 | Copernicus Marine |
| PAR | NASA OB.DAAC L3 mapped: SeaWiFS, MODIS-Aqua, PACE OCI | 9 km / 4 km, daily | Sept 1997 | NASA Earthdata (free) |
| Waves (Hs, direction, Tm-10, Tp) | Copernicus Marine, Med waves reanalysis `006_012` + analysis/forecast `006_017` | 4.2 km, hourly | 1985 | Copernicus Marine |
| Wind, gusts, air temp, pressure, rain | NOAA NCEI ISD, Gibraltar Airport LXGB (08495) | station, hourly | 1973 | none |
| Wind, gusts, rain, air temp (gap-free) | ERA5 reanalysis via Open-Meteo | ~25 km, hourly | 1940 | none |
| Saharan dust: surface dust, AOD, PM10 | CAMS forecasts via Open-Meteo air quality API | 11-45 km, hourly | Aug 2022 | none |
| Saharan dust: dust AOD, total AOD | CAMS global reanalysis EAC4 (Atmosphere Data Store) | ~80 km, 3-hourly | 2003 | ADS (free) |
| Aerosol optical depth, Angstrom exponent | NASA AERONET v3 Level 1.5, Málaga site | station, daily | site record | none |

Satellite values are stored as statistics over five boxes set in `config.yaml`: Bay of Gibraltar, Gibraltar Eastside, Strait of Gibraltar, Western Alboran and a 20 km box around the Rock. Each record holds the mean, median, min, max, standard deviation and the number of valid (cloud-free) pixels. Model and station series are stored at points: Europa Point offshore, the bay centre and the airport.

To add a metric later (SPM, Secchi depth, currents, oxygen...), add a source block to `config.yaml`. Most Copernicus products need only a new `dataset_id` and variable name.

## One-time setup

1. **Create free accounts**
   - Copernicus Marine: https://data.marine.copernicus.eu/register
   - NASA Earthdata: https://urs.earthdata.nasa.gov/users/new
   - Copernicus Atmosphere Data Store: https://ads.atmosphere.copernicus.eu. Open the *CAMS global reanalysis (EAC4)* dataset page, accept the licence, and copy your API token from your profile page.
2. **Create a GitHub repository** (for example `tnp-ocean-harvester`) and upload this folder.
3. **Add secrets** under *Settings → Secrets and variables → Actions*:
   - `COPERNICUSMARINE_SERVICE_USERNAME`, `COPERNICUSMARINE_SERVICE_PASSWORD`
   - `EARTHDATA_USERNAME`, `EARTHDATA_PASSWORD`
   - `ADS_API_KEY`
   - `DATABASE_URL` (optional, see below)
4. **Turn on GitHub Pages**: *Settings → Pages → Source: GitHub Actions*. The dashboard JSON is then published at `https://<account>.github.io/<repo>/data/`.
5. **Run it**: *Actions → Daily harvest → Run workflow*. Then run *Historical backfill* and repeat it (or uncomment its schedule) until `status` shows full records.

Any source without credentials is skipped with a note, so ISD, ERA5, CAMS via Open-Meteo and AERONET work before the accounts exist.

## Where the data lives

The harvester stores data in whichever database `DATABASE_URL` points to:

```
sqlite:///data/tnp_ocean.db                     (default)
mysql://user:password@host:3306/dbname          (typical WordPress / cPanel hosting)
postgresql://user:password@host:5432/dbname     (Supabase, Neon, a VPS)
```

**Without a `DATABASE_URL` secret**, SQLite is kept in the GitHub Actions cache between runs and the website reads the published JSON. That works well for a start. Treat it as temporary, though: GitHub removes caches that go unused for 7 days.

**With the website's own MySQL database**: many shared hosts block connections from outside their servers. In cPanel, check *Remote MySQL*. If you can't allow GitHub's IP ranges there, use a hosted Postgres database (Supabase and Neon both have free tiers) or stay on the JSON route.

`sql/` contains the schema for MySQL and PostgreSQL. The harvester also creates the tables itself on first run.

## Tables

- `observations`: one row per source × variable × location × time, holding `val_mean` (the value), `val_median`, `val_min`, `val_max`, `val_std`, `n_valid` and `n_total`. Times are UTC.
- `locations`, `sources`, `variables`: descriptions taken from `config.yaml`.
- `harvest_runs`: log of every chunk fetched, with status and error messages. Backfill resume points come from here.

Near real time and reprocessed versions of a product are stored side by side under different source codes. The export picks the best available one for each day, using the `export.daily_priority` order in `config.yaml`.

## Dashboard JSON

`python harvest.py export` writes:

- `meta.json`: variables, units, areas, points, sources
- `latest.json`: most recent daily value per variable and location, plus its anomaly against the day-of-year climatology
- `daily/<variable>__<location>.json`: the full daily series (`[date, value, source index]`) and a 365-day climatology with mean, 10th and 90th percentiles

Hourly values are reduced to daily means. The exceptions are rain (daily total, only for complete days), gusts (daily maximum) and directions (circular mean).

## Running locally

```bash
pip install -r requirements.txt
python harvest.py check          # which sources have credentials
python harvest.py update         # last few weeks, or since the previous run
python harvest.py backfill --source openmeteo_era5 --start 2015-01-01
python harvest.py export
python harvest.py status
python -m unittest discover -s tests
```

On Windows, set credentials with `set NAME=value` in Command Prompt or `$env:NAME="value"` in PowerShell before running.

## Notes and known limits

- **Not yet tested against the live services.** The parsers, statistics, database and export code pass the offline tests. The network calls follow each provider's documentation, and the first Actions run is where any wrong variable name or dataset version will show up. Check the log of that first run.
- ISD is being replaced by NOAA's GHCNh. It still updates, but the station source will need switching at some point.
- MODIS-Aqua is near the end of its mission. PACE carries PAR forward from 2024, and `daily_priority` prefers it.
- Copernicus Marine NRT ocean colour only keeps a few weeks online. The daily run picks it up, and the multi-year dataset replaces it once reprocessed.
- The Gibraltar Government air quality network (gibraltarairquality.gi) has local PM10 back to 2005, a good ground-truth check for dust events. It has no API, only a download form, so it isn't harvested yet.
- The CAMS values from Open-Meteo are model forecasts. EAC4 is a reanalysis. They are stored under separate sources and shouldn't be mixed in one trend line.

## Attribution (show on the dashboard)

- Generated using E.U. Copernicus Marine Service Information
- NASA Ocean Biology Processing Group, OB.DAAC (SeaWiFS, MODIS-Aqua, PACE OCI)
- NOAA National Centers for Environmental Information, Integrated Surface Database
- Weather data by Open-Meteo.com (CC BY 4.0); ERA5 by the Copernicus Climate Change Service
- Copernicus Atmosphere Monitoring Service (CAMS) information
- AERONET: thank the PI of the Málaga site, as AERONET's data policy asks
