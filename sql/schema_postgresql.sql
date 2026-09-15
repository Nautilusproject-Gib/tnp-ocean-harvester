-- TNP Gibraltar Ocean Observatory schema (postgresql).
-- The harvester creates these automatically; this file is for reference or manual setup.

CREATE TABLE IF NOT EXISTS observations (
    source VARCHAR(64) NOT NULL,
    variable VARCHAR(32) NOT NULL,
    location VARCHAR(48) NOT NULL,
    obs_time TIMESTAMP NOT NULL,
    val_mean DOUBLE PRECISION,
    val_median DOUBLE PRECISION,
    val_min DOUBLE PRECISION,
    val_max DOUBLE PRECISION,
    val_std DOUBLE PRECISION,
    n_valid INTEGER,
    n_total INTEGER,
    updated_at TIMESTAMP,
    PRIMARY KEY (source, variable, location, obs_time)
);

CREATE TABLE IF NOT EXISTS locations (
    code VARCHAR(48) PRIMARY KEY,
    name VARCHAR(128),
    kind VARCHAR(8),
    lat DOUBLE PRECISION, lon DOUBLE PRECISION,
    lon_min DOUBLE PRECISION, lat_min DOUBLE PRECISION, lon_max DOUBLE PRECISION, lat_max DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS sources (
    code VARCHAR(64) PRIMARY KEY,
    kind VARCHAR(32),
    description TEXT,
    product VARCHAR(128),
    dataset VARCHAR(128)
);

CREATE TABLE IF NOT EXISTS variables (
    code VARCHAR(32) PRIMARY KEY,
    name VARCHAR(128),
    unit VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS harvest_runs (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(64),
    mode VARCHAR(16),
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    range_start TIMESTAMP,
    range_end TIMESTAMP,
    n_rows INTEGER,
    status VARCHAR(16),
    message TEXT
);

CREATE INDEX IF NOT EXISTS ix_obs_var_loc_time ON observations (variable, location, obs_time);
