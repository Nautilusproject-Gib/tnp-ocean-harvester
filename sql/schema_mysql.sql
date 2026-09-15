-- TNP Gibraltar Ocean Observatory schema (mysql).
-- The harvester creates these automatically; this file is for reference or manual setup.

CREATE TABLE IF NOT EXISTS observations (
    source VARCHAR(64) NOT NULL,
    variable VARCHAR(32) NOT NULL,
    location VARCHAR(48) NOT NULL,
    obs_time DATETIME NOT NULL,
    val_mean DOUBLE,
    val_median DOUBLE,
    val_min DOUBLE,
    val_max DOUBLE,
    val_std DOUBLE,
    n_valid INTEGER,
    n_total INTEGER,
    updated_at DATETIME,
    PRIMARY KEY (source, variable, location, obs_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS locations (
    code VARCHAR(48) PRIMARY KEY,
    name VARCHAR(128),
    kind VARCHAR(8),
    lat DOUBLE, lon DOUBLE,
    lon_min DOUBLE, lat_min DOUBLE, lon_max DOUBLE, lat_max DOUBLE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sources (
    code VARCHAR(64) PRIMARY KEY,
    kind VARCHAR(32),
    description TEXT,
    product VARCHAR(128),
    dataset VARCHAR(128)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS variables (
    code VARCHAR(32) PRIMARY KEY,
    name VARCHAR(128),
    unit VARCHAR(32)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS harvest_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    source VARCHAR(64),
    mode VARCHAR(16),
    started_at DATETIME,
    finished_at DATETIME,
    range_start DATETIME,
    range_end DATETIME,
    n_rows INTEGER,
    status VARCHAR(16),
    message TEXT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE INDEX ix_obs_var_loc_time ON observations (variable, location, obs_time);
