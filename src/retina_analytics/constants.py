"""Shared constants and helpers for node analytics."""

import math

C_KM_US = 0.299792458
C_KM_S = 299792.458  # speed of light, km/s   # speed of light km/μs
R_EARTH = 6371.0  # Earth radius km

# One kilometres-per-degree, derived rather than typed.  Four different values
# were in circulation across this library and the backend — 111.0, 111.32,
# 111.320 and R_EARTH·π/180 — for the same quantity, so a position dead-reckoned
# in one module and compared in another disagreed by up to 0.3%.  Deriving it
# from R_EARTH also means distance and dead-reckoning cannot drift apart.
KM_PER_DEG_LAT = R_EARTH * math.pi / 180.0  # 111.1949…
M_PER_DEG_LAT = KM_PER_DEG_LAT * 1000.0

# Yagi antenna spec
YAGI_BEAM_WIDTH_DEG = 42.0  # half-power beamwidth of the fleet Yagis
YAGI_MAX_RANGE_KM = 50.0


def km_per_deg_lon(lat_deg: float) -> float:
    """Kilometres per degree of longitude at *lat_deg*.

    The guard is a divide-by-zero floor, not a physical clamp: some callers
    previously used ``max(0.1, cos)``, which only differs above 84.3° latitude
    and would silently misplace a target there rather than admit it cannot.
    """
    return KM_PER_DEG_LAT * max(math.cos(math.radians(lat_deg)), 1e-6)


def enu_km(ref_lat, ref_lon, lat, lon) -> tuple[float, float]:
    """(east_km, north_km) of a point relative to a reference, flat-earth."""
    return ((lon - ref_lon) * km_per_deg_lon(ref_lat), (lat - ref_lat) * KM_PER_DEG_LAT)


def offset_latlon(lat, lon, east_km, north_km) -> tuple[float, float]:
    """Move a position by an east/north offset in km.  Inverse of enu_km."""
    return (lat + north_km / KM_PER_DEG_LAT, lon + east_km / km_per_deg_lon(lat))


def offset_latlon_m(lat, lon, east_m, north_m) -> tuple[float, float]:
    """offset_latlon in metres — the shape dead-reckoning callers want."""
    return offset_latlon(lat, lon, east_m / 1000.0, north_m / 1000.0)


def bistatic_differential_km(tx_lat, tx_lon, rx_lat, rx_lon, target_lat, target_lon) -> float:
    """Differential range R_tx + R_rx − L for a target, in km.

    This is what the delay measures and what sets received power, so it — not
    the RX-relative range — is the quantity every bistatic gate should compare
    against.  Ground-projected: altitude is a second-order correction that the
    callers of this helper do not have.
    """
    r_tx = haversine_km(tx_lat, tx_lon, target_lat, target_lon)
    r_rx = haversine_km(rx_lat, rx_lon, target_lat, target_lon)
    baseline = haversine_km(tx_lat, tx_lon, rx_lat, rx_lon)
    return r_tx + r_rx - baseline


def bistatic_delay_us(tx_lat, tx_lon, rx_lat, rx_lon, target_lat, target_lon) -> float:
    """Bistatic delay for a target, in microseconds."""
    return bistatic_differential_km(tx_lat, tx_lon, rx_lat, rx_lon, target_lat, target_lon) / C_KM_US


def point_in_beam(
    lat,
    lon,
    *,
    rx_lat,
    rx_lon,
    beam_azimuth_deg,
    beam_width_deg,
    max_range_km,
    tx_lat=None,
    tx_lon=None,
    max_bistatic_range_km=None,
) -> bool:
    """Is a target inside a node's detection area?

    Three rules, which were separately re-derived in four places before this
    existed and had drifted apart in each:

    1. **Range.** If the node declares a bistatic limit and its TX is known,
       the target must be within it on *differential* range — the ellipse, not
       a circle about the RX.  Otherwise fall back to the monostatic circle.
    2. **Bearing.** Within half the beam width of the aim, measured as a signed
       wrap so 359° and 1° are 2° apart.
    3. **Aim.** ``beam_azimuth_deg=None`` means omnidirectional and skips (2)
       entirely; callers wanting the broadside default should resolve it with
       resolve_beam_azimuth_deg first, which is the rule the rest of the system
       already uses.

    Note this is a *ground-projected* test.  association.compute_overlap_zone
    deliberately does not use it: it gates on the exact per-grid-point delay it
    has already computed, which accounts for altitude, and it applies a looser
    radius first so the empirical coverage prior can tighten it afterwards.
    """
    if max_bistatic_range_km and tx_lat is not None and tx_lon is not None:
        reach = bistatic_differential_km(tx_lat, tx_lon, rx_lat, rx_lon, lat, lon)
        if reach > float(max_bistatic_range_km):
            return False
    elif haversine_km(rx_lat, rx_lon, lat, lon) > float(max_range_km):
        return False

    if beam_azimuth_deg is None:
        return True
    brg = bearing_deg(rx_lat, rx_lon, lat, lon)
    off = abs((brg - float(beam_azimuth_deg) + 180.0) % 360.0 - 180.0)
    return off <= float(beam_width_deg) / 2.0


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from (lat1,lon1) to (lat2,lon2) in degrees [0,360)."""
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0


def bistatic_range_limit_km(psi_deg: float, baseline_km: float, max_bistatic_km: float) -> float:
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


def resolve_beam_width_deg(config):
    """Resolve a node's beam width (deg), defaulting to the nominal Yagi spec.

    ``beam_width_deg`` is nullable on the node wire contract, and the config dict
    is built field by field from the node row, so an unset width arrives *present
    and None* rather than absent.  ``dict.get`` with a default therefore never
    fires, and the None reaches the half-width divisions in ``_point_in_beam``
    and ``point_in_detection_area`` as a TypeError.  A null resolves to
    YAGI_BEAM_WIDTH_DEG because every node in the fleet carries the same Yagi:
    the width is a documented hardware spec that retina-gui simply never asks the
    owner for, not a value invented to paper over the gap.

    ``config`` is node-supplied, so the parse is guarded the same way as
    resolve_beam_azimuth_deg and falls back to the nominal on bad input.
    """
    width = config.get("beam_width_deg")
    try:
        width = float(width) if width is not None else None
    except (TypeError, ValueError):
        width = None
    if width is not None and not math.isfinite(width):
        width = None
    return width if width is not None else YAGI_BEAM_WIDTH_DEG
