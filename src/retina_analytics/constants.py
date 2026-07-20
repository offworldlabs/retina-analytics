"""Shared constants and helpers for node analytics."""

import math

C_KM_US = 0.299792458   # speed of light km/μs
R_EARTH = 6371.0         # Earth radius km

# Yagi antenna spec
YAGI_BEAM_WIDTH_DEG = 41.0   # typical 40-42° half-power beamwidth
YAGI_MAX_RANGE_KM = 50.0


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from (lat1,lon1) to (lat2,lon2) in degrees [0,360)."""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return math.degrees(math.atan2(x, y)) % 360.0


def resolve_beam_azimuth_deg(config, rx_lat, rx_lon, tx_lat, tx_lon):
    """Resolve a node's beam azimuth (deg).

    Honours an explicit ``config['beam_azimuth_deg']`` (aimed ring-Yagi) when it
    parses to a float; otherwise points broadside — perpendicular to the RX→TX
    baseline — to maximise cross-coverage. ``config`` is node-supplied, so the
    parse is guarded and falls back to broadside on bad input.
    """
    explicit_az = config.get("beam_azimuth_deg")
    try:
        explicit_az = float(explicit_az) if explicit_az is not None else None
    except (TypeError, ValueError):
        explicit_az = None
    if explicit_az is not None:
        return explicit_az
    return (bearing_deg(rx_lat, rx_lon, tx_lat, tx_lon) + 90.0) % 360.0
