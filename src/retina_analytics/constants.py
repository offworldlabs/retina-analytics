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


def bistatic_range_limit_km(psi_deg: float, baseline_km: float,
                            max_bistatic_km: float) -> float:
    """How far a bistatic node can see at angle *psi_deg* off its RX→TX baseline.

    A node's detection limit is a *differential* range — the extra path the echo
    travels over the direct one, R_tx + R_rx − L.  The locus of constant
    differential range Δ is an ellipse with foci at TX and RX, and its polar
    radius measured from the RX focus is

        r(ψ) = Δ(Δ + 2L) / (2·[(L + Δ) − L·cos ψ])

    with ψ the angle from the RX→TX direction.  The two extremes are worth
    committing to memory, because a circle of radius Δ gets both wrong:

        ψ = 0    (toward TX)     r = Δ/2 + L
        ψ = 180° (away from TX)  r = Δ/2        — independent of the baseline

    At Δ = 60 km the true limit away from the transmitter is 30 km, not 60: a
    circle over-reaches by 2× in radius and 4× in area exactly where false
    pairings are cheapest to form.  Toward a distant tower it under-reaches
    instead — a 43 km baseline genuinely reaches 73 km.

    L → 0 collapses the ellipse to a circle of radius Δ/2, which is the correct
    monostatic limit (there R_tx = R_rx = r, so 2r = Δ), so a co-sited TX needs
    no special case.
    """
    d = float(max_bistatic_km)
    lb = max(float(baseline_km), 0.0)
    denom = 2.0 * ((lb + d) - lb * math.cos(math.radians(psi_deg)))
    if denom <= 0.0:  # only reachable for Δ ≤ 0, which is not a detection limit
        return 0.0
    return d * (d + 2.0 * lb) / denom


def bistatic_max_radius_km(baseline_km: float, max_bistatic_km: float) -> float:
    """Largest RX-relative range the footprint reaches, i.e. r(ψ=0) = Δ/2 + L.

    Used where a single conservative radius is needed — bounding-box and
    do-these-nodes-overlap pre-filters — so those never prune a region the
    ellipse actually covers.
    """
    return float(max_bistatic_km) / 2.0 + max(float(baseline_km), 0.0)


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
    if explicit_az is not None and not math.isfinite(explicit_az):
        explicit_az = None
    if explicit_az is not None:
        return explicit_az
    return (bearing_deg(rx_lat, rx_lon, tx_lat, tx_lon) + 90.0) % 360.0
