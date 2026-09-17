from .base import Source, SourceError
from .cams import CamsEac4
from .copernicus import CopernicusGrid, CopernicusPoint
from .nasa import NasaL3m
from .stations import (Aeronet, IocSeaLevel, NceiIsd, OpenMeteoAirQuality, OpenMeteoArchive,
                       OpenMeteoForecast)

SOURCE_TYPES = {
    "copernicus_grid": CopernicusGrid,
    "copernicus_point": CopernicusPoint,
    "nasa_l3m": NasaL3m,
    "ncei_isd": NceiIsd,
    "openmeteo_archive": OpenMeteoArchive,
    "openmeteo_airquality": OpenMeteoAirQuality,
    "cams_eac4": CamsEac4,
    "aeronet": Aeronet,
    "openmeteo_forecast": OpenMeteoForecast,
    "ioc_sealevel": IocSeaLevel,
}


def build_source(code: str, config: dict) -> Source:
    cfg = config["sources"][code]
    try:
        cls = SOURCE_TYPES[cfg["type"]]
    except KeyError:
        raise SourceError(f"Unknown source type '{cfg.get('type')}' for {code}")
    return cls(code, cfg, config)


__all__ = ["SOURCE_TYPES", "build_source", "Source", "SourceError"]
