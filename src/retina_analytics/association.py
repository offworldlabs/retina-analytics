"""
Inter-Node Association Gate for Retina Passive Radar Network.

Determines which detections from different nodes could correspond to the
same physical target by comparing bistatic delay/Doppler measurements
against pre-calculated association gates.

Architecture:
  1. For each node pair, pre-compute a grid of bistatic delay values
     within the overlapping detection region.
  2. When detections arrive, filter candidate associations using the
     delay/Doppler gates.
  3. Submit associated detection groups to the multi-node solver.

Key concepts:
  - Bistatic Ellipse: locus of points with constant bistatic range
    (TX→target→RX path delay) for a given node.
  - Association Gate: allowable delay/Doppler difference between two
    node measurements that could correspond to the same target.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from retina_analytics.constants import resolve_beam_azimuth_deg

# ── Constants ────────────────────────────────────────────────────────────────

C_KM_US = 0.299792458   # speed of light km/μs
C_KM_S = 299792.458     # speed of light km/s
R_EARTH = 6371.0         # Earth radius km


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _lla_to_enu(lat, lon, alt_km, ref_lat, ref_lon, ref_alt_km):
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    north = dlat * R_EARTH
    east = dlon * R_EARTH * math.cos(math.radians(ref_lat))
    up = alt_km - ref_alt_km
    return (east, north, up)


def _enu_to_lla(east_km, north_km, up_km, ref_lat, ref_lon, ref_alt_km):
    lat = ref_lat + math.degrees(north_km / R_EARTH)
    lon = ref_lon + math.degrees(east_km / (R_EARTH * math.cos(math.radians(ref_lat))))
    alt_km_out = ref_alt_km + up_km
    return (lat, lon, alt_km_out)


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _haversine_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_deg(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1)
    lat2r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360


def _bistatic_delay_at(target_enu, tx_enu, rx_enu=(0, 0, 0)):
    """Bistatic differential delay in μs."""
    d_tx = _norm([target_enu[i] - tx_enu[i] for i in range(3)])
    d_rx = _norm([rx_enu[i] - target_enu[i] for i in range(3)])
    d_bl = _norm([rx_enu[i] - tx_enu[i] for i in range(3)])
    return (d_tx + d_rx - d_bl) / C_KM_US


# ── Doppler as a velocity projection ─────────────────────────────────────────
#
# For one TX/RX pair and a target at p moving at v, the bistatic Doppler is
#
#     f_d = (1/λ) · v · (u_tx + u_rx)
#
# where u_tx, u_rx are unit vectors from the *target* toward TX and RX.  Writing
# b = u_tx + u_rx (the bistatic bisector, |b| = 2·cos(β/2) for bistatic angle β):
#
#     f_d · λ  =  v · b
#
# So each node measures one scalar projection of the 3-D velocity onto a known
# axis — an axis that depends on the TX position, the RX position, and the
# target position.  Two nodes therefore measure projections onto *different*
# axes, which is why comparing their Doppler values (in Hz or normalised to
# m/s) is meaningless: it subtracts components of the same vector resolved
# along different directions.  What can be tested is whether a *physically
# plausible* velocity exists that satisfies both projections at once.

# No aircraft in this system exceeds ~270 m/s; airliners top out near 290 m/s
# ground speed, more with a jetstream.  340 m/s leaves headroom for a strong
# tailwind while still rejecting the geometrically impossible.
_V_MAX_MS = 340.0

# Below this the two projection axes are too nearly parallel to solve for a
# horizontal velocity — the 2x2 system is ill-conditioned and any speed can be
# fitted, so the test abstains rather than producing a meaningless number.
_MIN_AXIS_DET = 0.05

# |b| -> 0 on the baseline (β -> 180°), where Doppler carries no velocity
# information at all.  Measurements from such geometry are uninformative, not
# wrong, so they abstain too.
_MIN_BISECTOR = 0.15


def _bisector(target_enu, tx_enu, rx_enu):
    """Return b = u_tx + u_rx at the target, in the same ENU frame."""
    d_tx = _norm([target_enu[i] - tx_enu[i] for i in range(3)]) or 1e-9
    d_rx = _norm([target_enu[i] - rx_enu[i] for i in range(3)]) or 1e-9
    return tuple(
        (tx_enu[i] - target_enu[i]) / d_tx + (rx_enu[i] - target_enu[i]) / d_rx
        for i in range(3)
    )


def implied_horizontal_velocity(m_a, b_a, m_b, b_b):
    """Level-flight velocity (v_east, v_north) implied by two Doppler projections.

    Returns None when the geometry cannot support the inference (near-parallel
    axes, or a target on a baseline where |b| collapses) — the caller must treat
    that as "no information", not as a pass or a fail.

    Two projections give two equations in three velocity components, so the
    system is under-determined and *some* velocity always exists: consistency
    alone proves nothing.  Pinning v_up = 0 — aircraft climb at a few m/s
    against a few hundred horizontal — makes it exactly determined in
    (v_east, v_north), and the result is then a real physical claim that can be
    checked against what an aircraft can do, and used to start the solver
    somewhere sensible.
    """
    if _norm(b_a) < _MIN_BISECTOR or _norm(b_b) < _MIN_BISECTOR:
        return None
    # [b_a.e  b_a.n] [v_e]   [m_a]
    # [b_b.e  b_b.n] [v_n] = [m_b]
    det = b_a[0] * b_b[1] - b_a[1] * b_b[0]
    if abs(det) < _MIN_AXIS_DET:
        return None
    v_e = (m_a * b_b[1] - m_b * b_a[1]) / det
    v_n = (b_a[0] * m_b - b_b[0] * m_a) / det
    return (v_e, v_n)


def implied_horizontal_speed(m_a, b_a, m_b, b_b):
    """Speed of the level-flight velocity implied by two Doppler projections."""
    v = implied_horizontal_velocity(m_a, b_a, m_b, b_b)
    return None if v is None else math.hypot(v[0], v[1])


# ── Node Pair Configuration ─────────────────────────────────────────────────

@dataclass
class NodeGeometry:
    """Geometry of a single radar node."""
    node_id: str
    rx_lat: float
    rx_lon: float
    rx_alt_km: float
    tx_lat: float
    tx_lon: float
    tx_alt_km: float
    fc_hz: float = 195e6
    beam_azimuth_deg: float = 0.0
    beam_width_deg: float = 41.0
    max_range_km: float = 50.0


@dataclass
class OverlapZone:
    """Pre-computed overlap zone between a pair of nodes."""
    node_a_id: str
    node_b_id: str
    # Grid points in the overlap region (lat, lon, alt_km)
    grid_points: list[tuple[float, float, float]] = field(default_factory=list)
    # For each grid point: (delay_a_us, delay_b_us) expected bistatic delays
    delay_pairs: list[tuple[float, float]] = field(default_factory=list)
    # For each grid point: (b_a, b_b) bistatic bisector vectors in the zone's
    # ENU frame.  Doppler measures v · b, so these are the axes along which the
    # two nodes observe the velocity — see implied_horizontal_speed.
    bisector_pairs: list[tuple[tuple, tuple]] = field(default_factory=list)
    # Carrier frequencies, needed to convert Hz to the frequency-independent
    # projection v · b = f_d · λ.
    fc_a_hz: float = 195e6
    fc_b_hz: float = 195e6
    # Association gate parameters
    delay_gate_us: float = 5.0     # max delay mismatch between prediction and measurement
    doppler_gate_hz: float = 30.0  # retained for the wire format / overlap summary

    def __post_init__(self):
        # Lazily-built numpy cache for find_associations (populated on first call).
        # Stored as plain attributes (not dataclass fields) to keep repr/hash clean.
        self._np_pred_a = None  # np.ndarray float32 (G,) — predicted delay at node A
        self._np_pred_b = None  # np.ndarray float32 (G,) — predicted delay at node B

    def _ensure_np(self):
        """Build numpy arrays from delay_pairs once; reuse thereafter."""
        if self._np_pred_a is None and self.delay_pairs:
            self._np_pred_a = np.array([dp[0] for dp in self.delay_pairs], dtype=np.float32)
            self._np_pred_b = np.array([dp[1] for dp in self.delay_pairs], dtype=np.float32)


@dataclass
class AssociationCandidate:
    """A detection pair from two nodes that may be the same target."""
    timestamp_ms: int
    node_a_id: str
    node_b_id: str
    det_a_idx: int      # index in node A's detection array
    det_b_idx: int      # index in node B's detection array
    delay_a: float      # measured delay at node A
    delay_b: float      # measured delay at node B
    doppler_a: float
    doppler_b: float
    snr_a: float
    snr_b: float
    # Grid-point match info
    grid_delay_a: float  # predicted delay at node A for the matching grid point
    grid_delay_b: float  # predicted delay at node B for the matching grid point
    grid_lat: float
    grid_lon: float
    grid_alt_km: float
    # True iff grid_lat/grid_lon were overridden with the reported ADS-B
    # position (frame_a or frame_b).  Used by format_candidates_for_solver to
    # prefer ADS-B-anchored candidates as the cluster centroid initial guess.
    had_adsb_override: bool = False
    # ICAO hex code confirmed for this candidate (both frames agreed, or one
    # frame provided it).  None when neither ADS-B entry carries a hex field.
    # Propagated to the solver input so the solver worker can maintain a
    # per-aircraft position history for multi-epoch EWMA smoothing.
    adsb_hex: str | None = None
    # Level-flight (v_east, v_north) in m/s implied by the two Doppler
    # projections at the matched grid point, or None where the geometry could
    # not support the inference.  Seeds the solver, which otherwise starts
    # velocity at zero — at n=2 the system is under-determined, so from a zero
    # start the optimiser slides along the null direction and lands wherever
    # the trust region takes it.  That is the origin of the impossible speeds.
    vel_est_ms: tuple | None = None


@dataclass
class TrackPairCandidate:
    """Two single-node tracks judged to be the same target.

    The detection-level AssociationCandidate above pairs one echo with one echo,
    which at n=2 is untestable: 4 measurements against 6 unknowns, so every
    residual gate passes a cross pairing exactly as it passes a real one.  This
    pairs a *track* with a *track*, and a confirmed track already carries
    several epochs of history, so the same two nodes now supply 4K measurements
    against the same 6 unknowns.  chi2_per_dof is the result of asking whether
    one constant-velocity trajectory explains all of them.
    """
    timestamp_ms: int
    node_a_id: str
    node_b_id: str
    track_a_id: str
    track_b_id: str
    # Latest measurement from each track, for the position solver.
    delay_a: float
    delay_b: float
    doppler_a: float
    doppler_b: float
    snr_a: float
    snr_b: float
    # Fitted trajectory at the most recent epoch.
    lat: float
    lon: float
    alt_km: float
    vel_east_ms: float
    vel_north_ms: float
    # Fit quality.  None when there was not enough history to fit, in which case
    # the pairing survived the coarse delay gate only.
    chi2_per_dof: float | None = None
    dof: int = 0
    n_epochs: int = 0


# ── Pre-computation ──────────────────────────────────────────────────────────

def _compute_node_enu(geo: NodeGeometry, ref_lat: float, ref_lon: float, ref_alt_km: float):
    """Compute RX and TX ENU positions relative to a common reference."""
    rx_enu = _lla_to_enu(geo.rx_lat, geo.rx_lon, geo.rx_alt_km, ref_lat, ref_lon, ref_alt_km)
    tx_enu = _lla_to_enu(geo.tx_lat, geo.tx_lon, geo.tx_alt_km, ref_lat, ref_lon, ref_alt_km)
    return rx_enu, tx_enu


def _point_in_beam(lat, lon, geo: NodeGeometry) -> bool:
    """Check if a point falls within the node's beam cone (2D check)."""
    dist = _haversine_km(geo.rx_lat, geo.rx_lon, lat, lon)
    if dist > geo.max_range_km:
        return False
    bearing = _bearing_deg(geo.rx_lat, geo.rx_lon, lat, lon)
    angle_diff = abs((bearing - geo.beam_azimuth_deg + 180) % 360 - 180)
    return angle_diff <= geo.beam_width_deg / 2


def compute_overlap_zone(geo_a: NodeGeometry, geo_b: NodeGeometry,
                         grid_step_km: float = 3.0,
                         altitudes_km: tuple[float, ...] = (1.5, 3.0, 5.0, 7.0, 9.0, 11.0),
                         delay_gate_us: float = 5.0,
                         doppler_gate_hz: float = 30.0) -> OverlapZone:
    """Pre-compute the overlap zone between two nodes.

    Creates a grid of test points within both nodes' detection cones
    and calculates the expected bistatic delay at each node for every
    grid point. These are used as association gates at runtime.
    """
    # Fast geographic pre-filter: if the two RX sites are farther apart
    # than the sum of their max ranges, NO point can lie in both beams.
    # Skip the O(n²) grid computation for this pair entirely.
    rx_sep = _haversine_km(geo_a.rx_lat, geo_a.rx_lon, geo_b.rx_lat, geo_b.rx_lon)
    if rx_sep > geo_a.max_range_km + geo_b.max_range_km:
        return OverlapZone(
            node_a_id=geo_a.node_id,
            node_b_id=geo_b.node_id,
            grid_points=[],
            delay_pairs=[],
            delay_gate_us=delay_gate_us,
            doppler_gate_hz=doppler_gate_hz,
        )

    # Common reference point: midpoint of the two RX positions
    ref_lat = (geo_a.rx_lat + geo_b.rx_lat) / 2
    ref_lon = (geo_a.rx_lon + geo_b.rx_lon) / 2
    ref_alt_km = 0.0

    rx_a_enu, tx_a_enu = _compute_node_enu(geo_a, ref_lat, ref_lon, ref_alt_km)
    rx_b_enu, tx_b_enu = _compute_node_enu(geo_b, ref_lat, ref_lon, ref_alt_km)

    # Determine bounding box for the grid
    max_range = max(geo_a.max_range_km, geo_b.max_range_km)
    n_steps = int(2 * max_range / grid_step_km) + 1

    grid_points = []
    delay_pairs = []
    bisector_pairs = []

    for alt_km in altitudes_km:
        for i in range(n_steps):
            for j in range(n_steps):
                east = -max_range + i * grid_step_km
                north = -max_range + j * grid_step_km

                lat, lon, _ = _enu_to_lla(east, north, 0.0, ref_lat, ref_lon, ref_alt_km)

                # Must be in BOTH beams
                if not _point_in_beam(lat, lon, geo_a):
                    continue
                if not _point_in_beam(lat, lon, geo_b):
                    continue

                # Calculate bistatic delay at each node
                target_enu = (east, north, alt_km)
                delay_a = _bistatic_delay_at(target_enu, tx_a_enu, rx_a_enu)
                delay_b = _bistatic_delay_at(target_enu, tx_b_enu, rx_b_enu)

                # Only keep physically meaningful delays
                if delay_a < 0 or delay_b < 0:
                    continue

                grid_points.append((lat, lon, alt_km))
                delay_pairs.append((delay_a, delay_b))
                # Bistatic bisector b = u_tx + u_rx (unit vectors from the
                # target toward each station).  Doppler measures v · b — see
                # _velocity_from_projections — so these are the projection axes
                # the two nodes observe the velocity along.  Precomputed here
                # because the grid is fixed at registration and the runtime
                # test then costs a couple of dot products.
                bisector_pairs.append((
                    _bisector(target_enu, tx_a_enu, rx_a_enu),
                    _bisector(target_enu, tx_b_enu, rx_b_enu),
                ))

    return OverlapZone(
        node_a_id=geo_a.node_id,
        node_b_id=geo_b.node_id,
        grid_points=grid_points,
        delay_pairs=delay_pairs,
        bisector_pairs=bisector_pairs,
        fc_a_hz=geo_a.fc_hz,
        fc_b_hz=geo_b.fc_hz,
        delay_gate_us=delay_gate_us,
        doppler_gate_hz=doppler_gate_hz,
    )


# ── Runtime association ──────────────────────────────────────────────────────

def find_associations(zone: OverlapZone,
                      frame_a: dict, frame_b: dict,
                      timestamp_ms: int) -> list[AssociationCandidate]:
    """Find detection associations between two nodes using pre-computed gates.

    Vectorised with numpy: replaces the O(Na × G × Nb) pure-Python triple
    loop with two boolean matrix multiplications (numpy BLAS), reducing
    per-call time from milliseconds to ~50 µs for typical frame sizes.

    Args:
        zone: Pre-computed OverlapZone for this node pair.
        frame_a: Detection frame from node A {delay:[], doppler:[], snr:[]}.
        frame_b: Detection frame from node B {delay:[], doppler:[], snr:[]}.
        timestamp_ms: Current timestamp.

    Returns:
        List of AssociationCandidate objects (best grid-point per pair).
    """
    delays_a   = frame_a.get("delay",   [])
    dopplers_a = frame_a.get("doppler", [])
    snrs_a     = frame_a.get("snr",     [])
    delays_b   = frame_b.get("delay",   [])
    dopplers_b = frame_b.get("doppler", [])
    snrs_b     = frame_b.get("snr",     [])

    if not delays_a or not delays_b or not zone.delay_pairs:
        return []

    # ── Lazy numpy cache for this zone's expected-delay grid ─────────────────
    zone._ensure_np()
    pred_a = zone._np_pred_a  # (G,) float32 — expected delay at node A
    pred_b = zone._np_pred_b  # (G,) float32 — expected delay at node B

    # ── Convert incoming detections to numpy ─────────────────────────────────
    da = np.array(delays_a,   dtype=np.float32)  # (Na,)
    db = np.array(delays_b,   dtype=np.float32)  # (Nb,)
    fa = np.array(dopplers_a, dtype=np.float32)  # (Na,)
    fb = np.array(dopplers_b, dtype=np.float32)  # (Nb,)
    na, nb = len(da), len(db)

    gate   = np.float32(zone.delay_gate_us)

    # ── Delay gate matrices ───────────────────────────────────────────────────
    # gate_a[i, g] = True  ↔  |delay_a[i] − pred_a[g]| < delay_gate
    gate_a = np.abs(da[:, None] - pred_a) < gate          # (Na, G) bool
    # gate_b[g, j] = True  ↔  |pred_b[g]  − delay_b[j]| < delay_gate
    gate_b = np.abs(pred_b[:, None] - db) < gate          # (G, Nb) bool

    # match[i, j] = number of grid points that simultaneously satisfy
    # gate_a[i,g] AND gate_b[g,j].  Cast to float32 so numpy dispatches
    # through BLAS SGEMM (7-8× faster than uint8 which has no BLAS path).
    match = gate_a.astype(np.float32) @ gate_b.astype(np.float32)  # (Na, Nb)

    # Doppler is deliberately NOT gated here.  It measures v · b, a projection
    # onto an axis that depends on the grid point, so no pairwise comparison of
    # the two values is meaningful until a candidate position is chosen.  The
    # plausibility test happens per-candidate below, once the best grid point
    # (and therefore the two projection axes) is known.
    rows, cols = np.where(match > 0)  # surviving (i_a, i_b) pairs
    if rows.size == 0:
        return []

    # ── Build AssociationCandidate objects ───────────────────────────────────
    sa_arr = np.array(snrs_a if len(snrs_a) == na else [0.0] * na, dtype=np.float32)
    sb_arr = np.array(snrs_b if len(snrs_b) == nb else [0.0] * nb, dtype=np.float32)

    # ADS-B lists are constant for the whole frame — read once outside the loop.
    _adsb_list_a = frame_a.get("adsb")
    _adsb_list_b = frame_b.get("adsb")

    candidates: dict[tuple[int, int], AssociationCandidate] = {}
    for i_a, i_b in zip(rows.tolist(), cols.tolist()):
        # Find the best grid point for this (i_a, i_b) pair: min total residual.
        # Also compute the delay-residual weighted mean altitude across ALL valid
        # grid points.  Rationale: at the correct altitude layer the residual is
        # small (≈ noise); at wrong altitude layers the same horizontal position
        # gives residual ≈ altitude_err × sin(elevation) / c ≈ 0.6–3 µs.  A
        # 1/(residual + ε) weighting strongly favours the correct altitude while
        # averaging over the noise, eliminating the argmin tie-break bias toward
        # the lowest layer (3 km).
        valid_g = np.nonzero(gate_a[i_a] & gate_b[:, i_b])[0]
        if valid_g.size == 0:
            continue
        res   = np.abs(pred_a[valid_g] - da[i_a]) + np.abs(pred_b[valid_g] - db[i_b])
        best_g = int(valid_g[np.argmin(res)])
        g_lat, g_lon, g_alt = zone.grid_points[best_g]

        # ── Doppler plausibility at the candidate position ───────────────────
        # Now that a grid point is chosen, both projection axes are known, so
        # the two Doppler measurements can be turned into a velocity and asked
        # whether an aircraft could fly it.  This is the test that works at
        # n=2, where the solver's rms_doppler cannot help: with two
        # measurements the residual goes to zero for a false pairing exactly as
        # it does for a true one, so the residual carries no information and
        # the plausibility of the *implied velocity* is all that is left.
        vel_est = None
        if zone.bisector_pairs and (fa[i_a] or fb[i_b]):
            b_a_vec, b_b_vec = zone.bisector_pairs[best_g]
            vel_est = implied_horizontal_velocity(
                float(fa[i_a]) * C_KM_S * 1000.0 / zone.fc_a_hz, b_a_vec,
                float(fb[i_b]) * C_KM_S * 1000.0 / zone.fc_b_hz, b_b_vec,
            )
            # None means the geometry cannot support the inference — abstain
            # rather than guess.  A finite speed above the bound is a positive
            # proof of impossibility, so reject.
            if vel_est is not None:
                if math.hypot(vel_est[0], vel_est[1]) > _V_MAX_MS:
                    continue

        # ── Ghost-association filter ─────────────────────────────────────────
        # Ghost associations arise when a clutter detection from one node is
        # paired (via the delay gate) with a real-aircraft detection from the
        # other.  In the simulation every genuine aircraft detection carries an
        # ADS-B entry; clutter detections have adsb[i]=None.  If either frame
        # has an ADS-B list we can therefore test both indices: if one is None
        # (clutter) we reject the pairing before it ever reaches the solver.
        # When NEITHER frame has an ADS-B list (rare: non-ADS-B aircraft, or
        # clutter-only frames) we let the pairing through — the downstream
        # rms_delay and beam-coverage checks provide the last line of defence.
        _ae_a = (_adsb_list_a[i_a]
                 if _adsb_list_a is not None and i_a < len(_adsb_list_a)
                 else None)
        _ae_b = (_adsb_list_b[i_b]
                 if _adsb_list_b is not None and i_b < len(_adsb_list_b)
                 else None)
        if _adsb_list_a is not None or _adsb_list_b is not None:
            # At least one frame has ADS-B capability; require both indices to
            # correspond to genuine aircraft (non-None dict entries).
            if not isinstance(_ae_a, dict):
                continue  # i_a is clutter — ghost pairing
            if not isinstance(_ae_b, dict):
                continue  # i_b is clutter — ghost pairing

        # ── Same-aircraft filter ─────────────────────────────────────────────
        # When both frames carry ADS-B entries with a hex ICAO code, both
        # detections MUST belong to the same physical aircraft.  The shared
        # SimulationWorld assigns a single adsb_hex per aircraft, so every
        # SyntheticNodeView reports the same hex for the same target.
        # Different aircraft that happen to satisfy the delay gate (cross-
        # pairings) will have different hex codes → reject them here.
        # Only applied when BOTH entries carry a non-empty "hex" field; if
        # either is absent (non-ADS-B aircraft, legacy data) we pass through.
        _candidate_hex: str | None = None
        if isinstance(_ae_a, dict) and isinstance(_ae_b, dict):
            _hex_a = _ae_a.get("hex")
            _hex_b = _ae_b.get("hex")
            if _hex_a and _hex_b and _hex_a != _hex_b:
                continue  # different aircraft — cross-pairing rejected
            _candidate_hex = _hex_a or _hex_b
        elif isinstance(_ae_a, dict):
            _candidate_hex = _ae_a.get("hex") or None
        elif isinstance(_ae_b, dict):
            _candidate_hex = _ae_b.get("hex") or None

        # ── ADS-B altitude override ──────────────────────────────────────────
        # ADS-B altitude is exact (to ~10 m) while the pre-computed grid only
        # has discrete layers (e.g. 5, 7, 9, 11 km), introducing up to ±1 km
        # altitude error.  Prefer frame_a; fall back to frame_b.
        # Require alt_baro > 100 ft to exclude spurious zero reports.
        if isinstance(_ae_a, dict) and (_ae_a.get("alt_baro") or 0) > 100:
            g_alt = float(_ae_a["alt_baro"]) * 0.3048 / 1000.0  # ft → km
        elif isinstance(_ae_b, dict) and (_ae_b.get("alt_baro") or 0) > 100:
            g_alt = float(_ae_b["alt_baro"]) * 0.3048 / 1000.0  # ft → km

        # ── ADS-B position override (initial-guess refinement) ───────────────
        # The grid initial guess has ±3 km resolution (grid_step_km).  For an
        # n=2 solve with one TDOA measurement the solver picks the point on the
        # hyperboloid closest to the initial guess; a 3 km offset can produce a
        # 5+ km position error even when the delay fit is perfect.
        # When the ADS-B entry carries a reported lat/lon (simulation always
        # provides this; real aircraft transponders also include it) we override
        # g_lat/g_lon with the ADS-B position so the solver starts within ~100 m
        # of the true position.  Prefer frame_a; fall back to frame_b.
        _adsb_lat: float | None = None
        _adsb_lon: float | None = None
        if isinstance(_ae_a, dict):
            _al, _ao = _ae_a.get("lat"), _ae_a.get("lon")
            if _al and _ao:
                _adsb_lat, _adsb_lon = float(_al), float(_ao)
        if _adsb_lat is None and isinstance(_ae_b, dict):
            _al, _ao = _ae_b.get("lat"), _ae_b.get("lon")
            if _al and _ao:
                _adsb_lat, _adsb_lon = float(_al), float(_ao)
        _had_adsb_override = False
        if _adsb_lat is not None and _adsb_lon is not None:
            g_lat, g_lon = _adsb_lat, _adsb_lon
            _had_adsb_override = True

        cand = AssociationCandidate(
            timestamp_ms  = timestamp_ms,
            node_a_id     = zone.node_a_id,
            node_b_id     = zone.node_b_id,
            det_a_idx     = i_a,
            det_b_idx     = i_b,
            delay_a       = float(da[i_a]),
            delay_b       = float(db[i_b]),
            doppler_a     = float(fa[i_a]),
            doppler_b     = float(fb[i_b]),
            snr_a         = float(sa_arr[i_a]),
            snr_b         = float(sb_arr[i_b]),
            grid_delay_a  = float(pred_a[best_g]),
            grid_delay_b  = float(pred_b[best_g]),
            grid_lat      = g_lat,
            grid_lon      = g_lon,
            grid_alt_km   = g_alt,
            had_adsb_override = _had_adsb_override,
            adsb_hex      = _candidate_hex,
            vel_est_ms    = vel_est,
        )
        key = (i_a, i_b)
        existing = candidates.get(key)
        if existing is None:
            candidates[key] = cand
        else:
            old_res = abs(existing.delay_a - existing.grid_delay_a) + abs(existing.delay_b - existing.grid_delay_b)
            new_res = abs(cand.delay_a - cand.grid_delay_a) + abs(cand.delay_b - cand.grid_delay_b)
            if new_res < old_res:
                candidates[key] = cand

    return list(candidates.values())


def _coarse_grid_match(zone: OverlapZone, delay_a: float, delay_b: float):
    """Best overlap grid point consistent with both delays, or None.

    The same test find_associations applies, for one delay from each side
    instead of whole frames.  It is a coverage question, not a residual one:
    two bistatic ellipses always intersect somewhere, so what is being asked is
    whether they intersect *inside both beams*.  Cheap, and it prunes most
    pairings before anything is fitted.
    """
    zone._ensure_np()
    pred_a, pred_b = zone._np_pred_a, zone._np_pred_b
    if pred_a is None:
        return None
    gate = zone.delay_gate_us
    valid = np.nonzero(
        (np.abs(pred_a - delay_a) < gate) & (np.abs(pred_b - delay_b) < gate)
    )[0]
    if valid.size == 0:
        return None
    res = np.abs(pred_a[valid] - delay_a) + np.abs(pred_b[valid] - delay_b)
    return int(valid[np.argmin(res)])


def _merge_epochs(hist_a: list, node_a_id: str, hist_b: list, node_b_id: str) -> list:
    """Interleave two tracks' samples into epochs for the batch fit.

    No resampling and no alignment tolerance: each sample becomes its own epoch
    carrying the one measurement that was actually taken at that instant.  The
    fit already evaluates every measurement at its own time, so pairing samples
    up would only introduce the interpolation error that the frame stagger
    causes in the first place — nodes send on their own cadence, and at 250 m/s
    a 2 s misalignment is 500 m of invented position error.
    """
    samples = [(float(h["t_s"]), node_a_id, h) for h in hist_a]
    samples += [(float(h["t_s"]), node_b_id, h) for h in hist_b]
    samples.sort(key=lambda s: s[0])

    epochs: list[dict] = []
    for t_s, node_id, h in samples:
        meas = {
            "node_id": node_id,
            "delay_us": float(h["delay_us"]),
            "doppler_hz": float(h["doppler_hz"]),
            "snr": float(h.get("snr", 0.0)),
        }
        # Samples that land on the same instant share an epoch; this is only a
        # tidiness measure, the fit is indifferent.
        if epochs and epochs[-1]["t_s"] == t_s:
            epochs[-1]["measurements"].append(meas)
        else:
            epochs.append({"t_s": t_s, "measurements": [meas]})
    return epochs


# ── InterNodeAssociator ──────────────────────────────────────────────────────

class InterNodeAssociator:
    """Manages overlap zones for all node pairs and runs association at runtime."""

    def __init__(self, delay_gate_us: float = 5.0, doppler_gate_hz: float = 30.0,
                 grid_step_km: float = 30.0, assoc_interval_s: float = 30.0,
                 cv_fit=None, cv_chi2_max: float = 2.0, cv_min_epochs: int = 4,
                 cv_min_span_s: float = 12.0):
        self.delay_gate_us = delay_gate_us
        self.doppler_gate_hz = doppler_gate_hz
        self.grid_step_km = grid_step_km
        self.node_geometries: dict[str, NodeGeometry] = {}
        # Raw registration configs, kept because the constant-velocity fit wants
        # rx/tx lat/lon/alt and fc in the same shape the solver takes them.
        self.node_configs: dict[str, dict] = {}
        self.overlap_zones: dict[tuple[str, str], OverlapZone] = {}
        self._pending_frames: dict[str, dict] = {}  # node_id → latest frame
        # node_id → latest list of confirmed single-node tracks (see submit_tracks).
        self._pending_tracks: dict[str, list] = {}
        # retina_geolocator.multinode_solver.fit_constant_velocity, injected
        # rather than imported: this library depends only on numpy, and making
        # it depend on the geolocator to run one function would couple two
        # siblings for no structural reason.  None disables the fine test, which
        # is what the unit tests and any caller without a solver want.
        self.cv_fit = cv_fit
        # Provisional threshold, to be set properly by the ROC sweep on real
        # scenes.  What is measured so far: with the simulator's noise model a
        # true pairing runs a median 0.5 and a p95 of ~0.8, and a crossed
        # pairing that is coincident *right now* — the case the coarse gate
        # cannot touch — comes out at 3.7 over a 20 s span.  2.0 sits between
        # them with room on both sides.  The first value tried here was 4.0,
        # which is above the true distribution but also above that 3.7, so it
        # would have let the very pairing this exists to catch straight through.
        self.cv_chi2_max = cv_chi2_max
        self.cv_min_epochs = cv_min_epochs
        # Observation *span*, which is what actually separates — not the number
        # of epochs.  A cross pairing is close to constant-velocity over a short
        # window; it is accumulated curvature that gives it away.  Measured on a
        # dual-illuminator geometry, rejection of crossed pairings at 95% TPR:
        #
        #      5 epochs over  8 s     53%
        #      5 epochs over 16 s     95%
        #
        # and on a noiseless two-aircraft pair, the crossed fit sits at
        # chi2/dof 0.40 over 10 s, 0.57 over 14 s, and stops being fittable at
        # all by 20 s.  Gating on epoch count alone would pass a 22 fps node
        # 5 epochs spanning 0.2 s, which carries no information whatsoever.
        self.cv_min_span_s = cv_min_span_s
        # Counters for what the fine test actually did, so its value is
        # observable rather than assumed.
        self.track_pairs_gated: int = 0     # survived the coarse delay grid
        self.track_pairs_rejected: int = 0  # ... then failed the chi2 test
        self.track_pairs_accepted: int = 0
        self.track_pairs_unfitted: int = 0  # too few epochs, or the fit failed
        # Adjacency index: node_id → set of neighbor node_ids that share a real
        # overlap zone (delay_pairs is non-empty).  Built during registration so
        # submit_frame can iterate O(K) neighbors instead of O(N) all nodes.
        self._neighbors: dict[str, set[str]] = {}
        # Rate-limit per-node association to at most once per _ASSOC_MIN_INTERVAL_S.
        # Prevents O(K) × N frames/s = O(N²) CPU burn in dense deployments where
        # K ≈ N (wide beams, small area).
        #
        # BUDGET CALCULATION (1000-node fleet on 2-core / GIL-bound):
        #   find_associations ≈ 50 µs/call.
        #   K ≈ 999 neighbors (all nodes overlap in simulation).
        #   Nodes send every 40 s; trigger every max(interval, 2×send) seconds.
        #   At interval=60 s → 11.4 rounds/s → 11.4 × 999 × 50 µs = 570 ms/s
        #   = 57 % of the single GIL core — starves frame workers.
        #   At interval=300 s, cap=50 → 2.9 rounds/s → 2.9 × 50 × 50 µs = 7 ms/s.
        #
        # A previous change scaled this down with fleet size, on the reasoning
        # that a 15-node metro costs ~2 ms/s against the 57 % the limit exists
        # to avoid, so the CPU budget allowed associating far more often.  The
        # arithmetic was right and the premise was wrong.
        #
        # Measured offline over 6 seeds (backend/scripts/association_bench.py),
        # ghost tracks as a share of all tracks:
        #
        #     interval    ghosts        real tracks   solves
        #        2 s      40% (sd 13)      9-13        1139
        #        5 s      34% (sd  9)      9-14         554
        #       10 s      19% (sd 13)      8-13         286
        #       30 s       6% (sd  9)      8-12         109
        #
        # 2s vs 30s is a 33-point difference, t=4.9.  Crucially the real-track
        # count and the matched position error (~0.28 km median) are flat
        # across the whole sweep: associating more often buys no extra targets
        # and no extra accuracy, it just re-samples the same geometry and mints
        # more false tracks.  The quantity that drives ghosts is how many
        # aircraft share an overlap zone, which has nothing to do with node
        # count -- so scaling on node count optimised CPU headroom, a resource
        # that was not the constraint.
        #
        # Back to a single interval.  ASSOC_MIN_INTERVAL_S stays wired through
        # from config (it was dead config before, defined but never passed), so
        # this is tunable without editing library source.
        self._ASSOC_MIN_INTERVAL_S: float = assoc_interval_s
        self._ASSOC_MAX_NEIGHBORS: int = 50
        # Frame-skew telemetry (see submit_frame).  The offline benchmark
        # assumes a clean stagger across frame_interval; these make the real
        # distribution observable instead of assumed.
        self.frame_skew_ms_total: int = 0
        self.frame_skew_samples: int = 0
        self.frame_skew_ms_max: int = 0
        self.frame_sync_rejects: int = 0
        # 500 ms buckets, last bucket is "5 s or more".
        self.frame_skew_hist: list[int] = [0] * 11
        self._last_assoc: dict[str, float] = {}  # node_id → last association wall-time
        # Maximum allowed age difference between two frames being associated.
        # Aircraft at 250 m/s move ~0.5 km in 2 s; frames further apart than
        # this produce inconsistent TDOA geometry → large position errors.
        # With 200 nodes staggered over 40 s (0.2 s/node), nodes in the same
        # geographic cluster (5-10 nodes) span ≤ 2 s.  A 3 s window lets all
        # intra-cluster pairs associate while rejecting distant inter-cluster
        # pairs whose timing error (Δt × v) would otherwise dominate.
        self._FRAME_SYNC_MAX_AGE_MS: int = 4_000
        self._register_lock = __import__('threading').Lock()

    def register_node(self, node_id: str, config: dict):
        """Register a node and pre-compute overlap zones with all existing nodes.

        Thread-safe: acquires an internal lock so concurrent registrations
        (e.g. from multiple run_in_executor calls) cannot corrupt iteration.

        Reconnecting nodes skip the expensive O(n²) overlap recomputation
        as long as their geometry (RX/TX position) hasn't changed.
        """
        rx_alt_km = config.get("rx_alt_ft", 0) * 0.3048 / 1000.0
        tx_alt_km = config.get("tx_alt_ft", 0) * 0.3048 / 1000.0

        geo = NodeGeometry(
            node_id=node_id,
            rx_lat=config.get("rx_lat", 0),
            rx_lon=config.get("rx_lon", 0),
            rx_alt_km=rx_alt_km,
            tx_lat=config.get("tx_lat", 0),
            tx_lon=config.get("tx_lon", 0),
            tx_alt_km=tx_alt_km,
            fc_hz=config.get("fc_hz", config.get("FC", 195e6)),
            beam_width_deg=config.get("beam_width_deg", 41),
            max_range_km=config.get("max_range_km", 50),
        )

        # Honour an explicit aim (aimed coverage-ring Yagi); otherwise broadside
        # to the RX→TX baseline. The overlap grid is computed from this azimuth,
        # so it must match the node's true aim. Shared with manager.register_node.
        geo.beam_azimuth_deg = resolve_beam_azimuth_deg(
            config, geo.rx_lat, geo.rx_lon, geo.tx_lat, geo.tx_lon
        )

        with self._register_lock:
            # Recorded before the unchanged-geometry early return: the equality
            # check below covers geometry only, so a reconnect that changed fc_hz
            # would otherwise leave the fit using the old carrier.
            self.node_configs[node_id] = config
            existing = self.node_geometries.get(node_id)
            if existing is not None and (
                abs(existing.rx_lat - geo.rx_lat) < 1e-6
                and abs(existing.rx_lon - geo.rx_lon) < 1e-6
                and abs(existing.tx_lat - geo.tx_lat) < 1e-6
                and abs(existing.tx_lon - geo.tx_lon) < 1e-6
                and abs(existing.max_range_km - geo.max_range_km) < 1e-4
                and abs(existing.beam_azimuth_deg - geo.beam_azimuth_deg) < 1e-4
                and abs(existing.beam_width_deg - geo.beam_width_deg) < 1e-4
            ):
                # Same geometry — overlap zones are still valid; skip O(n²) recompute.
                return

            # Pre-compute overlap zones with existing nodes (serialised to avoid
            # RuntimeError: dictionary changed size during iteration when multiple
            # nodes register concurrently from a thread-pool executor).
            for existing_id, existing_geo in list(self.node_geometries.items()):
                pair_key = tuple(sorted([node_id, existing_id]))
                zone = compute_overlap_zone(
                    geo if pair_key[0] == node_id else existing_geo,
                    existing_geo if pair_key[0] == node_id else geo,
                    grid_step_km=self.grid_step_km,
                    delay_gate_us=self.delay_gate_us,
                    doppler_gate_hz=self.doppler_gate_hz,
                )
                self.overlap_zones[pair_key] = zone
                # Update adjacency index for O(K) submit_frame lookup.
                if zone.delay_pairs:  # only real overlaps, not geographic misses
                    self._neighbors.setdefault(node_id, set()).add(existing_id)
                    self._neighbors.setdefault(existing_id, set()).add(node_id)

            self.node_geometries[node_id] = geo

    def submit_frame(self, node_id: str, frame: dict, timestamp_ms: int) -> list[AssociationCandidate]:
        """Submit a detection frame and find associations with other recent frames.

        Returns association candidates found with any other node's latest frame.
        Uses the _neighbors adjacency index so only O(K) actual-overlap pairs
        are checked instead of O(N) all connected nodes.  Additionally rate-limits
        the expensive inner loop to _ASSOC_MIN_INTERVAL_S so dense deployments
        (K ≈ N) don’t produce O(N²) CPU load when many nodes share the same area.
        """
        self._pending_frames[node_id] = frame

        # No detections → no possible associations from this frame
        if not frame.get("delay"):
            return []

        neighbors = self._neighbors.get(node_id)
        if not neighbors:
            return []  # no registered overlap pairs for this node yet

        # Rate-limit: only run association at most once per _ASSOC_MIN_INTERVAL_S
        now = __import__('time').monotonic()
        if now - self._last_assoc.get(node_id, 0.0) < self._ASSOC_MIN_INTERVAL_S:
            return []
        self._last_assoc[node_id] = now

        # Snapshot neighbors set to avoid RuntimeError if registration adds
        # new entries concurrently (Python set iteration is not thread-safe).
        # Cap to _ASSOC_MAX_NEIGHBORS to bound CPU time per round.
        all_candidates = []
        _neighbor_list = list(neighbors)
        if len(_neighbor_list) > self._ASSOC_MAX_NEIGHBORS:
            import random
            _neighbor_list = random.sample(_neighbor_list, self._ASSOC_MAX_NEIGHBORS)

        for other_id in _neighbor_list:
            other_frame = self._pending_frames.get(other_id)
            if other_frame is None:
                continue  # neighbor hasn’t sent a frame yet
            # Gate: only associate frames that are close in time so the aircraft
            # position is approximately the same in both measurements.  With 40 s
            # frame intervals and a 200-node fleet staggered at 0.2 s/node, nodes
            # in the same cluster fire within ~2 s of each other.  The 3 s limit
            # caps aircraft-motion error to ≤ 0.75 km (250 m/s × 3 s) while
            # still allowing all intra-cluster pairs to associate.
            other_ts = other_frame.get("timestamp", 0)
            if timestamp_ms > 0 and other_ts > 0:
                # Telemetry: how far apart in time are the frames we actually
                # pair?  Aircraft move ~250 m/s, so this skew is a position
                # error injected straight into the association geometry — a
                # 4 s skew is ~1 km, comparable to the delay gate itself, and
                # is a candidate explanation for false pairings.  Recorded
                # because it was cheaper to measure than to keep inferring.
                _skew = abs(timestamp_ms - other_ts)
                self.frame_skew_ms_total += _skew
                self.frame_skew_samples += 1
                self.frame_skew_ms_max = max(self.frame_skew_ms_max, _skew)
                _b = min(int(_skew // 500), len(self.frame_skew_hist) - 1)
                self.frame_skew_hist[_b] += 1
                if _skew > self._FRAME_SYNC_MAX_AGE_MS:
                    self.frame_sync_rejects += 1
                    continue
            pair_key = tuple(sorted([node_id, other_id]))
            zone = self.overlap_zones.get(pair_key)
            if zone is None or not zone.delay_pairs:
                continue

            # Ensure frame_a corresponds to zone.node_a_id
            if pair_key[0] == node_id:
                frame_a, frame_b = frame, other_frame
            else:
                frame_a, frame_b = other_frame, frame

            candidates = find_associations(zone, frame_a, frame_b, timestamp_ms)
            all_candidates.extend(candidates)

        return all_candidates

    # ── Track-level association ──────────────────────────────────────────────

    def submit_tracks(self, node_id: str, tracks: list[dict],
                      timestamp_ms: int) -> list[TrackPairCandidate]:
        """Associate this node's confirmed single-node tracks with its neighbours'.

        The detection-level path above pairs one echo with one echo, and at n=2
        that pairing cannot be tested: two nodes give 4 measurements against 6
        unknowns, so a cross pairing between two real aircraft produces zero
        residual exactly as a real target does.  Nor does watching it help —
        differentiating the two delay equations shows the phantom's own motion
        is identically its Doppler-implied velocity, so it stays self-consistent
        for as long as both aircraft are tracked.

        Pairing tracks instead changes the arithmetic three ways.  Clutter is
        gone before association, because random detections never survive the
        per-node tracker's M-of-N promotion.  The candidate count collapses from
        Na x Nb detections to Ta x Tb tracks, and Ta is about the number of real
        aircraft in the beam.  And each surviving pairing is tested against 4K
        measurements rather than 4, which is what makes it testable at all.

        Args:
            node_id: the node these tracks belong to.
            tracks: [{"track_id": str,
                      "history": [{"t_s", "delay_us", "doppler_hz", "snr"}, ...]}]
                    ordered oldest-first.  Callers should pass only tracks their
                    tracker has confirmed — a TENTATIVE track has too little
                    history to fit and may yet be deleted.
            timestamp_ms: current time, stamped onto the emitted candidates.

        Returns:
            TrackPairCandidate for each pairing that passed both the coarse
            delay-grid gate and, where there was enough history, the
            constant-velocity fit.
        """
        self._pending_tracks[node_id] = tracks or []
        if not tracks:
            return []

        neighbors = self._neighbors.get(node_id)
        if not neighbors:
            return []

        # Same rate limit submit_frame uses, and it matters more here: the
        # coarse grid gate is cheap but every pairing that survives it costs an
        # LM fit, and a hardware node sends at 22 fps.  Storing the tracks above
        # is unconditional, so a neighbour triggering its own round still sees
        # this node's latest history.
        now = __import__('time').monotonic()
        if now - self._last_assoc.get(node_id, 0.0) < self._ASSOC_MIN_INTERVAL_S:
            return []
        self._last_assoc[node_id] = now

        out: list[TrackPairCandidate] = []
        for other_id in list(neighbors):
            other_tracks = self._pending_tracks.get(other_id)
            if not other_tracks:
                continue
            pair_key = tuple(sorted([node_id, other_id]))
            zone = self.overlap_zones.get(pair_key)
            if zone is None or not zone.delay_pairs:
                continue
            if pair_key[0] == node_id:
                tracks_a, tracks_b = tracks, other_tracks
            else:
                tracks_a, tracks_b = other_tracks, tracks
            out.extend(self._pair_tracks(zone, tracks_a, tracks_b, timestamp_ms))
        return out

    def _pair_tracks(self, zone: OverlapZone, tracks_a: list[dict],
                     tracks_b: list[dict], timestamp_ms: int
                     ) -> list[TrackPairCandidate]:
        results: list[TrackPairCandidate] = []
        for ta in tracks_a:
            hist_a = ta.get("history") or []
            if not hist_a:
                continue
            last_a = hist_a[-1]
            for tb in tracks_b:
                hist_b = tb.get("history") or []
                if not hist_b:
                    continue
                last_b = hist_b[-1]

                best_g = _coarse_grid_match(
                    zone, float(last_a["delay_us"]), float(last_b["delay_us"]),
                )
                if best_g is None:
                    continue
                self.track_pairs_gated += 1
                g_lat, g_lon, g_alt = zone.grid_points[best_g]

                # Seed the fit from the level-flight velocity the two Dopplers
                # imply at that grid point.  It is a good seed — measured median
                # 4 deg of heading error on true pairings — and from a zero
                # start the optimiser does not converge from tens of km away.
                vel_seed = None
                if zone.bisector_pairs:
                    b_a, b_b = zone.bisector_pairs[best_g]
                    v = implied_horizontal_velocity(
                        float(last_a["doppler_hz"]) * C_KM_S * 1000.0 / zone.fc_a_hz, b_a,
                        float(last_b["doppler_hz"]) * C_KM_S * 1000.0 / zone.fc_b_hz, b_b,
                    )
                    if v is not None:
                        if math.hypot(v[0], v[1]) > _V_MAX_MS:
                            continue
                        vel_seed = {"vel_east_ms": v[0], "vel_north_ms": v[1]}

                lat, lon, alt_km = g_lat, g_lon, g_alt
                vel_e = vel_seed["vel_east_ms"] if vel_seed else 0.0
                vel_n = vel_seed["vel_north_ms"] if vel_seed else 0.0
                chi2_per_dof = None
                dof = 0
                epochs = _merge_epochs(hist_a, zone.node_a_id, hist_b, zone.node_b_id)

                span_s = (epochs[-1]["t_s"] - epochs[0]["t_s"]) if epochs else 0.0
                if (self.cv_fit is not None
                        and len(epochs) >= self.cv_min_epochs
                        and span_s >= self.cv_min_span_s):
                    fit = self.cv_fit(
                        {"initial_guess": {"lat": g_lat, "lon": g_lon, "alt_km": g_alt},
                         "initial_velocity": vel_seed,
                         "epochs": epochs,
                         "timestamp_ms": timestamp_ms},
                        self.node_configs,
                    )
                    if fit is None or not fit.get("success"):
                        self.track_pairs_unfitted += 1
                        continue
                    chi2_per_dof = fit["chi2_per_dof"]
                    dof = fit["dof"]
                    if chi2_per_dof > self.cv_chi2_max:
                        self.track_pairs_rejected += 1
                        continue
                    lat, lon = fit["lat"], fit["lon"]
                    alt_km = fit["alt_m"] / 1000.0
                    vel_e, vel_n = fit["vel_east"], fit["vel_north"]
                    self.track_pairs_accepted += 1
                else:
                    # Not enough history yet.  Let it through on the coarse gate
                    # alone rather than dropping the target: the pairing is
                    # re-tested every round, and a real one accumulates the
                    # history it needs within a few frames.  Downstream decides
                    # whether an unconfirmed pairing may be published.
                    self.track_pairs_unfitted += 1

                results.append(TrackPairCandidate(
                    timestamp_ms=timestamp_ms,
                    node_a_id=zone.node_a_id,
                    node_b_id=zone.node_b_id,
                    track_a_id=str(ta.get("track_id")),
                    track_b_id=str(tb.get("track_id")),
                    delay_a=float(last_a["delay_us"]),
                    delay_b=float(last_b["delay_us"]),
                    doppler_a=float(last_a["doppler_hz"]),
                    doppler_b=float(last_b["doppler_hz"]),
                    snr_a=float(last_a.get("snr", 0.0)),
                    snr_b=float(last_b.get("snr", 0.0)),
                    lat=lat, lon=lon, alt_km=alt_km,
                    vel_east_ms=vel_e, vel_north_ms=vel_n,
                    chi2_per_dof=chi2_per_dof, dof=dof, n_epochs=len(epochs),
                ))
        return results

    def format_track_pairs_for_solver(
        self, pairs: list[TrackPairCandidate]
    ) -> list[dict]:
        """Cluster track pairs by fitted position into multinode solver inputs.

        Same shape format_candidates_for_solver emits, so the solver worker is
        unchanged.  Two differences that matter downstream: the initial guess is
        a fitted trajectory rather than a 3 km grid point, and each input
        carries the fit quality so publication can be gated on it.
        """
        if not pairs:
            return []

        _MERGE_DIST_KM = 6.0
        n = len(pairs)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        lats = np.array([p.lat for p in pairs], dtype=np.float64)
        lons = np.array([p.lon for p in pairs], dtype=np.float64)
        km_per_lat = R_EARTH * math.pi / 180.0
        km_per_lon = km_per_lat * math.cos(math.radians(float(np.mean(lats))))
        dist_sq = (((lats[:, None] - lats) * km_per_lat) ** 2
                   + ((lons[:, None] - lons) * km_per_lon) ** 2)
        rows, cols = np.where(
            (dist_sq < _MERGE_DIST_KM ** 2) & (np.arange(n)[:, None] < np.arange(n))
        )
        for i, j in zip(rows.tolist(), cols.tolist()):
            parent[_find(i)] = _find(j)

        groups: dict[int, list[TrackPairCandidate]] = defaultdict(list)
        for i, p in enumerate(pairs):
            groups[_find(i)].append(p)

        solver_inputs = []
        for group in groups.values():
            by_node: dict[str, dict] = {}
            for p in group:
                for nid, d, f, s in (
                    (p.node_a_id, p.delay_a, p.doppler_a, p.snr_a),
                    (p.node_b_id, p.delay_b, p.doppler_b, p.snr_b),
                ):
                    if nid not in by_node or s > by_node[nid]["snr"]:
                        by_node[nid] = {"node_id": nid, "delay_us": d,
                                        "doppler_hz": f, "snr": s}

            fitted = [p for p in group if p.chi2_per_dof is not None]
            # Worst fit in the cluster, not the best: a cluster is published as
            # one target, so a pairing that failed to justify itself should not
            # be laundered by a well-fitted neighbour sharing its position.
            worst_chi2 = max((p.chi2_per_dof for p in fitted), default=None)

            solver_inputs.append({
                "initial_guess": {
                    "lat": sum(p.lat for p in group) / len(group),
                    "lon": sum(p.lon for p in group) / len(group),
                    "alt_km": sum(p.alt_km for p in group) / len(group),
                },
                "initial_velocity": {
                    "vel_east_ms": sum(p.vel_east_ms for p in group) / len(group),
                    "vel_north_ms": sum(p.vel_north_ms for p in group) / len(group),
                },
                "measurements": list(by_node.values()),
                "n_nodes": len(by_node),
                "timestamp_ms": group[0].timestamp_ms,
                "adsb_hex": None,
                "chi2_per_dof": worst_chi2,
                "n_epochs": min(p.n_epochs for p in group),
                "track_ids": sorted({p.track_a_id for p in group}
                                    | {p.track_b_id for p in group}),
            })
        return solver_inputs

    def get_overlap_summary(self) -> list[dict]:
        """Return summary of all overlap zones."""
        summaries = []
        for (a_id, b_id), zone in list(self.overlap_zones.items()):
            summaries.append({
                "node_a": a_id,
                "node_b": b_id,
                "grid_points": len(zone.grid_points),
                "delay_gate_us": zone.delay_gate_us,
                "doppler_gate_hz": zone.doppler_gate_hz,
                "has_overlap": len(zone.grid_points) > 0,
            })
        return summaries

    def format_candidates_for_solver(self, candidates: list[AssociationCandidate]) -> list[dict]:
        """Format association candidates for the multi-node least-squares solver.

        Returns a list of measurement groups, each containing bistatic
        delay/Doppler measurements from multiple nodes for the same
        estimated target position.
        """
        if not candidates:
            return []

        # ── Step 1: Proximity-based clustering ───────────────────────────────
        # Group candidates whose grid positions are within _MERGE_DIST_KM of
        # each other using Union-Find.
        #
        # WHY: with grid_step_km=3.0, two grid points from *different* overlap
        # zones (each with its own ENU origin) for the *same* aircraft can be up
        # to grid_step × √2 ≈ 4.24 km apart.  The previous rigid-bin approach
        # used 0.05° ≈ 5.6 km bins.  The straddling probability for points 4.24
        # km apart in a 5.6 km bin is 4.24/5.6 ≈ 75%, meaning ~75% of genuine
        # n≥3 candidates from different overlap zones ended up in different bins
        # and were never merged.  Proximity-based clustering eliminates this
        # boundary effect: any two candidates within _MERGE_DIST_KM always land
        # in the same cluster regardless of phase alignment.
        _MERGE_DIST_KM = 6.0  # > grid_step × √2 = 4.24 km (max inter-zone skew)

        n = len(candidates)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            parent[_find(x)] = _find(y)

        # Pairwise distance check (numpy, O(N²)).  For typical N ≤ 200
        # candidates per association round: 40 K comparisons ≈ 1 ms.
        lats = np.array([c.grid_lat for c in candidates], dtype=np.float64)
        lons = np.array([c.grid_lon for c in candidates], dtype=np.float64)
        mid_lat = float(np.mean(lats))
        km_per_lat = R_EARTH * math.pi / 180.0
        km_per_lon = km_per_lat * math.cos(math.radians(mid_lat))
        dlat_km = (lats[:, None] - lats) * km_per_lat
        dlon_km = (lons[:, None] - lons) * km_per_lon
        dist_sq = dlat_km ** 2 + dlon_km ** 2
        merge_sq = _MERGE_DIST_KM ** 2

        row_idx, col_idx = np.where(
            (dist_sq < merge_sq) & (np.arange(n)[:, None] < np.arange(n))
        )
        for i, j in zip(row_idx.tolist(), col_idx.tolist()):
            _union(i, j)

        raw_groups: dict[int, list[AssociationCandidate]] = defaultdict(list)
        for i, c in enumerate(candidates):
            raw_groups[_find(i)].append(c)

        solver_inputs = []
        for group in raw_groups.values():
            measurements = []
            for c in group:
                measurements.append({
                    "node_id": c.node_a_id,
                    "delay_us": c.delay_a,
                    "doppler_hz": c.doppler_a,
                    "snr": c.snr_a,
                })
                measurements.append({
                    "node_id": c.node_b_id,
                    "delay_us": c.delay_b,
                    "doppler_hz": c.doppler_b,
                    "snr": c.snr_b,
                })

            # Deduplicate measurements by node_id (keep highest SNR)
            by_node: dict[str, dict] = {}
            for m in measurements:
                nid = m["node_id"]
                if nid not in by_node or m["snr"] > by_node[nid]["snr"]:
                    by_node[nid] = m

            # Use centroid of actual grid positions as the initial guess (more
            # accurate than the rounded bin centre, especially when the group
            # spans candidates from different overlap zones).
            #
            # If ANY candidate in the group has an ADS-B-overridden position,
            # restrict the centroid to those candidates only.  The bistatic-grid
            # positions of non-ADS-B candidates have ±3 km resolution and can
            # dilute the ADS-B-anchored centroid, sometimes pushing the initial
            # guess outside the n=2 displacement gate or producing a low-quality
            # solve far from the true position.
            _adsb_group = [c for c in group if c.had_adsb_override]
            if _adsb_group:
                g_lat = sum(c.grid_lat for c in _adsb_group) / len(_adsb_group)
                g_lon = sum(c.grid_lon for c in _adsb_group) / len(_adsb_group)
            else:
                g_lat = sum(c.grid_lat for c in group) / len(group)
                g_lon = sum(c.grid_lon for c in group) / len(group)

            # Use the altitude of the best-matching grid point (min delay
            # residual) from each candidate, then take the mean across the
            # group.  Layers are restricted to [7, 9, 11] km so low-altitude
            # bistatic ghost solutions (which can map to positions hundreds of
            # km away) are never considered.
            g_alt_km = sum(c.grid_alt_km for c in group) / len(group)

            # Pass the confirmed ICAO hex through to the solver so the solver
            # worker can maintain a per-aircraft position history for
            # multi-epoch EWMA smoothing.  Use the hex from the ADS-B-anchored
            # candidates (if any); fall back to any candidate's hex.
            _group_hex: str | None = None
            if _adsb_group:
                _group_hex = next((c.adsb_hex for c in _adsb_group if c.adsb_hex), None)
            if _group_hex is None:
                _group_hex = next((c.adsb_hex for c in group if c.adsb_hex), None)

            # Mean of the per-candidate level-flight velocity estimates that
            # the geometry could support.  Seeds the solver; see vel_est_ms on
            # AssociationCandidate for why starting from zero goes wrong.
            _vels = [c.vel_est_ms for c in group if c.vel_est_ms is not None]
            _vel_seed = None
            if _vels:
                _vel_seed = {
                    "vel_east_ms": sum(v[0] for v in _vels) / len(_vels),
                    "vel_north_ms": sum(v[1] for v in _vels) / len(_vels),
                }

            solver_inputs.append({
                "initial_guess": {
                    "lat": g_lat,
                    "lon": g_lon,
                    "alt_km": g_alt_km,
                },
                "initial_velocity": _vel_seed,
                "measurements": list(by_node.values()),
                "n_nodes": len(by_node),
                "timestamp_ms": group[0].timestamp_ms,
                "adsb_hex": _group_hex,
            })

        return solver_inputs
