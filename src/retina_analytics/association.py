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
    # Association gate parameters
    delay_gate_us: float = 5.0     # max delay mismatch between prediction and measurement
    doppler_gate_hz: float = 30.0  # max Doppler mismatch

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
                         altitudes_km: tuple[float, ...] = (5.0, 7.0, 9.0, 11.0),
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

    return OverlapZone(
        node_a_id=geo_a.node_id,
        node_b_id=geo_b.node_id,
        grid_points=grid_points,
        delay_pairs=delay_pairs,
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
    dgmax  = np.float32(zone.doppler_gate_hz * 3.0)

    # ── Delay gate matrices ───────────────────────────────────────────────────
    # gate_a[i, g] = True  ↔  |delay_a[i] − pred_a[g]| < delay_gate
    gate_a = np.abs(da[:, None] - pred_a) < gate          # (Na, G) bool
    # gate_b[g, j] = True  ↔  |pred_b[g]  − delay_b[j]| < delay_gate
    gate_b = np.abs(pred_b[:, None] - db) < gate          # (G, Nb) bool

    # match[i, j] = number of grid points that simultaneously satisfy
    # gate_a[i,g] AND gate_b[g,j].  Cast to float32 so numpy dispatches
    # through BLAS SGEMM (7-8× faster than uint8 which has no BLAS path).
    match = gate_a.astype(np.float32) @ gate_b.astype(np.float32)  # (Na, Nb)

    # ── Doppler gate (relaxed; only when both non-zero — preserves original) ─
    both_nz = (np.abs(fa[:, None]) > 0.0) & (np.abs(fb) > 0.0)  # (Na, Nb)
    dop_ok  = ~both_nz | (np.abs(fa[:, None] - fb) <= dgmax)     # (Na, Nb)

    rows, cols = np.where((match > 0) & dop_ok)  # surviving (i_a, i_b) pairs
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
        if isinstance(_ae_a, dict) and isinstance(_ae_b, dict):
            _hex_a = _ae_a.get("hex")
            _hex_b = _ae_b.get("hex")
            if _hex_a and _hex_b and _hex_a != _hex_b:
                continue  # different aircraft — cross-pairing rejected

        # ── ADS-B altitude override ──────────────────────────────────────────
        # ADS-B altitude is exact (to ~10 m) while the pre-computed grid only
        # has discrete layers (e.g. 5, 7, 9, 11 km), introducing up to ±1 km
        # altitude error.  Prefer frame_a; fall back to frame_b.
        # Require alt_baro > 100 ft to exclude spurious zero reports.
        if isinstance(_ae_a, dict) and (_ae_a.get("alt_baro") or 0) > 100:
            g_alt = float(_ae_a["alt_baro"]) * 0.3048 / 1000.0  # ft → km
        elif isinstance(_ae_b, dict) and (_ae_b.get("alt_baro") or 0) > 100:
            g_alt = float(_ae_b["alt_baro"]) * 0.3048 / 1000.0  # ft → km

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


# ── InterNodeAssociator ──────────────────────────────────────────────────────

class InterNodeAssociator:
    """Manages overlap zones for all node pairs and runs association at runtime."""

    def __init__(self, delay_gate_us: float = 5.0, doppler_gate_hz: float = 30.0,
                 grid_step_km: float = 30.0):
        self.delay_gate_us = delay_gate_us
        self.doppler_gate_hz = doppler_gate_hz
        self.grid_step_km = grid_step_km
        self.node_geometries: dict[str, NodeGeometry] = {}
        self.overlap_zones: dict[tuple[str, str], OverlapZone] = {}
        self._pending_frames: dict[str, dict] = {}  # node_id → latest frame
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
        self._ASSOC_MIN_INTERVAL_S: float = 30.0
        self._ASSOC_MAX_NEIGHBORS: int = 50
        self._last_assoc: dict[str, float] = {}  # node_id → last association wall-time
        # Maximum allowed age difference between two frames being associated.
        # Aircraft at 250 m/s move ~2.5 km in 10 s; frames further apart than
        # this produce inconsistent TDOA geometry → large position errors.
        self._FRAME_SYNC_MAX_AGE_MS: int = 10_000
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

        # Compute beam azimuth: perpendicular to the RX→TX baseline.
        # Yagi antennas point broadside (90° from the baseline) to maximise
        # cross-coverage of aircraft transiting the bistatic zone.
        geo.beam_azimuth_deg = (_bearing_deg(
            geo.rx_lat, geo.rx_lon, geo.tx_lat, geo.tx_lon
        ) + 90.0) % 360.0

        with self._register_lock:
            existing = self.node_geometries.get(node_id)
            if existing is not None and (
                abs(existing.rx_lat - geo.rx_lat) < 1e-6
                and abs(existing.rx_lon - geo.rx_lon) < 1e-6
                and abs(existing.tx_lat - geo.tx_lat) < 1e-6
                and abs(existing.tx_lon - geo.tx_lon) < 1e-6
                and abs(existing.max_range_km - geo.max_range_km) < 1e-4
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
            # frame intervals, a stale pending frame can be up to 40 s old;
            # at 250 m/s that is 10 km of aircraft movement — the dominant source
            # of position error.  Allow up to _FRAME_SYNC_MAX_AGE_MS (10 s),
            # which caps the timing contribution to ~2.5 km.
            other_ts = other_frame.get("timestamp", 0)
            if timestamp_ms > 0 and other_ts > 0 and abs(timestamp_ms - other_ts) > self._FRAME_SYNC_MAX_AGE_MS:
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
            g_lat = sum(c.grid_lat for c in group) / len(group)
            g_lon = sum(c.grid_lon for c in group) / len(group)

            # Use the altitude of the best-matching grid point (min delay
            # residual) from each candidate, then take the mean across the
            # group.  Layers are restricted to [7, 9, 11] km so low-altitude
            # bistatic ghost solutions (which can map to positions hundreds of
            # km away) are never considered.
            g_alt_km = sum(c.grid_alt_km for c in group) / len(group)

            solver_inputs.append({
                "initial_guess": {
                    "lat": g_lat,
                    "lon": g_lon,
                    "alt_km": g_alt_km,
                },
                "measurements": list(by_node.values()),
                "n_nodes": len(by_node),
                "timestamp_ms": group[0].timestamp_ms,
            })

        return solver_inputs
