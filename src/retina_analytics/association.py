"""
Inter-Node Association Gate for Retina Passive Radar Network.

Decides which observations from different nodes describe the same target, by
testing bistatic delay and Doppler against pre-computed per-pair gates.

Architecture:
  1. For each node pair, pre-compute a grid of bistatic delay values within the
     overlapping detection region (compute_overlap_zone).
  2. Each node submits its *confirmed single-node tracks* (submit_tracks).
     Pairings are gated on the coarse delay grid, then tested by fitting one
     constant-velocity trajectory to both tracks' whole observation windows.
  3. Surviving pairings are clustered by position and handed to the multi-node
     solver (format_track_pairs_for_solver).

Tracks rather than detections, because at n=2 a single epoch gives four
measurements against six unknowns: a cross pairing between two real aircraft
produces zero residual exactly as a real target does, and no first-order test
separates them.  Over K epochs it is 4K against 6, which is testable.  The
superseded detection-level path is preserved in detection_association.py as the
benchmark's A/B baseline.

The constant-velocity fit is injected (cv_fit), not imported — this library
depends only on numpy.  Production injects None and defers the fit to the
solver worker, so the chi2 gate and the hypothesis selection below run only
where a fit is supplied, which today means the offline benchmark.

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

from retina_analytics.constants import (
    C_KM_US,
    KM_PER_DEG_LAT,
    R_EARTH,
    bearing_deg as _bearing_deg,
    bistatic_max_radius_km,
    haversine_km as _haversine_km,
    km_per_deg_lon,
    resolve_beam_azimuth_deg,
)
from retina_analytics.empirical_coverage import OBSERVED_LIMIT_MARGIN

# ── Constants ────────────────────────────────────────────────────────────────

C_KM_S = 299792.458     # speed of light km/s


# ── Geometry helpers ─────────────────────────────────────────────────────────
#
# Distance and bearing come from constants.py; this module used to carry
# byte-identical copies.  What remains here is genuinely local: an ENU frame
# (the grid search works in metres from a reference, not in degrees) and the
# delay/bisector maths that operates inside it.

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
    # Differential-range limit (R_tx + R_rx − baseline), which is what a
    # bistatic receiver is actually bounded by.  None keeps the legacy
    # monostatic circle of radius max_range_km, so a node that does not declare
    # one gates exactly as before.
    max_bistatic_range_km: float | None = None
    # bearing (deg) -> furthest this node has been *observed* to detect, or
    # None where there is not enough evidence.  Shrink-only: it may tighten the
    # theoretical footprint, never extend it.  See _point_in_beam.
    coverage_limit: object = None

    @property
    def baseline_km(self) -> float:
        return _haversine_km(self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon)

    @property
    def footprint_radius_km(self) -> float:
        """Largest RX-relative range this node's footprint reaches."""
        if self.max_bistatic_range_km is None:
            return self.max_range_km
        return bistatic_max_radius_km(self.baseline_km, self.max_bistatic_range_km)


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
    # The measurements the constant-velocity fit needs, in
    # fit_constant_velocity's input shape.  Carried so the fit can run
    # somewhere other than here: it is an ~86 ms LM solve and submit_tracks
    # runs in the frame worker, where that cost is frame latency.  None once
    # the fit has run, inline or downstream.
    epochs: list | None = None


# ── Pre-computation ──────────────────────────────────────────────────────────

def _compute_node_enu(geo: NodeGeometry, ref_lat: float, ref_lon: float, ref_alt_km: float):
    """Compute RX and TX ENU positions relative to a common reference."""
    rx_enu = _lla_to_enu(geo.rx_lat, geo.rx_lon, geo.rx_alt_km, ref_lat, ref_lon, ref_alt_km)
    tx_enu = _lla_to_enu(geo.tx_lat, geo.tx_lon, geo.tx_alt_km, ref_lat, ref_lon, ref_alt_km)
    return rx_enu, tx_enu


def _point_in_beam(lat, lon, geo: NodeGeometry) -> bool:
    """Check if a point falls within the node's beam sector (2D check).

    Bearing only.  The *range* limit is applied in compute_overlap_zone against
    the exact bistatic delay it already computes for the grid point, which
    accounts for altitude and needs no closed form.  Splitting them this way
    keeps the cheap angular test first, where it prunes most of the bounding
    box, and leaves the range test where the geometry is already in hand.

    geo.footprint_radius_km bounds how far the sector can extend, so callers
    that need a radius before any delay exists still have one.
    """
    dist = _haversine_km(geo.rx_lat, geo.rx_lon, lat, lon)
    if dist > geo.footprint_radius_km:
        return False
    bearing = _bearing_deg(geo.rx_lat, geo.rx_lon, lat, lon)
    angle_diff = abs((bearing - geo.beam_azimuth_deg + 180) % 360 - 180)
    if angle_diff > geo.beam_width_deg / 2:
        return False

    # Shrink-only empirical prior.  Where a node has been observed often enough
    # on this bearing to say something, the grid is pulled in to what it has
    # actually detected — a node aimed at a ridge does not cover the far side of
    # it, however good the geometry looks.
    #
    # Only ever tightens.  The polygon is fed from ADS-B fixes alone, so it
    # grows where cooperative traffic flies and stays empty elsewhere; an empty
    # bearing means nobody flew there, not that the node is deaf.  Reading it as
    # an upper bound would carve blind spots into good footprints, so
    # observed_limit_km abstains below its evidence floor and this test lets an
    # abstention through untouched.
    if geo.coverage_limit is not None:
        observed = geo.coverage_limit(bearing)
        if observed is not None and dist > observed * OBSERVED_LIMIT_MARGIN:
            return False
    return True


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
    # Fast geographic pre-filter: if the two RX sites are farther apart than the
    # sum of their footprint radii, NO point can lie in both beams, so the
    # O(n²) grid is skipped entirely.  The radius has to be the *largest* the
    # footprint reaches — Δ/2 + L toward the transmitter, not Δ — or a pair
    # whose overlap lies out along a long baseline gets pruned before it is
    # ever looked at.
    rx_sep = _haversine_km(geo_a.rx_lat, geo_a.rx_lon, geo_b.rx_lat, geo_b.rx_lon)
    if rx_sep > geo_a.footprint_radius_km + geo_b.footprint_radius_km:
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

    # Determine bounding box for the grid.  Sized on the footprint radius, so a
    # node aimed along a long baseline still gets its far lobe enumerated.
    max_range = max(geo_a.footprint_radius_km, geo_b.footprint_radius_km)
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

                # Range limit, applied on the differential range the node is
                # actually bounded by rather than on distance from the RX.  The
                # delay is already in hand and is exact in 3-D, so this needs no
                # closed form and costs nothing.
                #
                # The two are not close: at Δ = 60 km the footprint reaches only
                # 30 km directly away from the transmitter, against the 60 km
                # circle this used to allow — 4× the area, in the half where
                # cross-aircraft pairings are cheapest to form.  Toward a distant
                # tower it runs the other way, a 43 km baseline genuinely
                # reaching 73 km.
                if (geo_a.max_bistatic_range_km is not None
                        and delay_a * C_KM_US > geo_a.max_bistatic_range_km):
                    continue
                if (geo_b.max_bistatic_range_km is not None
                        and delay_b * C_KM_US > geo_b.max_bistatic_range_km):
                    continue

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
def _batch_grid_match(zone: OverlapZone, delays_a: list, delays_b: list) -> dict:
    """Best overlap grid point for every (a, b) delay pairing that has one.

    Returns {(i_a, i_b): best_grid_index}; absent keys had no common point.

    Vectorised for the same reason find_associations is: the gate is a
    (Ta x G) x (G x Tb) boolean contraction, which numpy dispatches to one BLAS
    matmul, and doing it per pairing instead costs Ta x Tb full scans of a grid
    that runs to ~16 000 points.  A first version of this function did exactly
    that, and it put the live frame queue at 92% depth with the frame processor
    21 s behind a 6 frame/s feed — the offline bench never showed it because it
    runs unthrottled on one core with no queue to fall behind.

    The test itself is a coverage question, not a residual one: two bistatic
    ellipses always intersect somewhere, so what is being asked is whether they
    intersect *inside both beams*.
    """
    zone._ensure_np()
    pred_a, pred_b = zone._np_pred_a, zone._np_pred_b
    if pred_a is None or not delays_a or not delays_b:
        return {}

    da = np.asarray(delays_a, dtype=np.float32)
    db = np.asarray(delays_b, dtype=np.float32)
    gate = np.float32(zone.delay_gate_us)

    gate_a = np.abs(da[:, None] - pred_a) < gate      # (Ta, G)
    gate_b = np.abs(pred_b[:, None] - db) < gate      # (G, Tb)
    # float32 so numpy takes the SGEMM path; uint8 has no BLAS kernel.
    match = gate_a.astype(np.float32) @ gate_b.astype(np.float32)

    out: dict = {}
    for i_a, i_b in zip(*np.where(match > 0)):
        i_a, i_b = int(i_a), int(i_b)
        valid = np.nonzero(gate_a[i_a] & gate_b[:, i_b])[0]
        if valid.size == 0:
            continue
        res = (np.abs(pred_a[valid] - da[i_a])
               + np.abs(pred_b[valid] - db[i_b]))
        out[(i_a, i_b)] = int(valid[np.argmin(res)])
    return out


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
                 cv_min_span_s: float = 12.0, cv_exclusive: bool = True,
                 coverage_provider=None, max_neighbors: int = 50,
                 max_pairs_per_round: int = 64):
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
        # Hypothesis selection: within one node pair, one track belongs to at
        # most one pairing, awarded to the lowest chi2.  See _pair_tracks
        # stage 2 for why this exists and why greedy-on-chi2 is safe where two
        # earlier exclusivity schemes were not.
        self.cv_exclusive = cv_exclusive
        # node_id -> (bearing -> observed limit km | None), or None for no
        # constraint.  Injected the same way cv_fit is, so this library keeps
        # knowing nothing about NodeAnalyticsManager.
        self.coverage_provider = coverage_provider
        # The constraint digest each node's grids were last built under, so a
        # polygon that tightens later can be detected and the grids rebuilt.
        self._coverage_digests: dict[str, tuple] = {}
        # Counters for what the fine test actually did, so its value is
        # observable rather than assumed.
        self.track_pairs_gated: int = 0       # survived the coarse delay grid
        self.track_pairs_rejected: int = 0    # ... then failed the chi2 test
        self.track_pairs_accepted: int = 0
        self.track_pairs_unfitted: int = 0    # too few epochs, or the fit failed
        self.track_pairs_superseded: int = 0  # lost their tracks to a better fit
        self.track_pairs_deferred: int = 0    # rounds cut short by a budget (once per round)
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
        # Neighbours visited per association round.  submit_frame had this cap
        # from the start; submit_tracks did not, so the live path was the
        # uncapped one and ASSOC_MAX_NEIGHBORS was dead config — the same shape
        # of bug as assoc_interval_s, which was also defined and never passed.
        self._ASSOC_MAX_NEIGHBORS: int = max_neighbors
        # Where the round stopped last time, per node.  Set iteration order is
        # fixed for the life of the process, so cutting the list off at the cap
        # would visit the same neighbours every round and starve the tail
        # permanently — which would make the "nothing is lost, only deferred"
        # claim in submit_tracks false.  A rotating cursor makes it true.
        self._neighbor_cursor: dict[str, int] = {}
        # Constant-velocity fits attempted per association *round*, across all
        # of a node's neighbours.  Each fit is a real LM solve over both tracks'
        # whole history and costs ~86 ms measured; when it runs inline this
        # number times 86 ms is how long one round blocks the frame queue.
        # Unbounded, staging went to 92% queue depth with the processor 21 s
        # behind a 6 frame/s feed and the map went empty.
        #
        # Per *node pair* would not bound it: a node has up to
        # _ASSOC_MAX_NEIGHBORS of them, so a 12-fit pair budget is really a
        # 600-fit round.
        self._MAX_FITS_PER_ROUND: int = 8
        # Pairings emitted per round when the fit is *deferred* (cv_fit=None,
        # which is production).  This used to be governed by the fit budget,
        # which was only decremented inside the fit branch — so with no fit
        # running the budget never depleted and the truncation acted as a fixed
        # cap of 8 pairings per node pair, discarding candidates to bound a cost
        # that was not being paid.
        #
        # It still needs a bound: each surviving pairing costs a _merge_epochs
        # over both histories, a TrackPairCandidate carrying those epochs, and a
        # slot on the solver queue.  Removing the cap would convert a bounded
        # frame path into a solver-queue overflow, which fails more quietly.
        self._MAX_PAIRS_PER_ROUND: int = max_pairs_per_round
        self._last_assoc: dict[str, float] = {}  # node_id → last association wall-time
        # Maximum allowed age difference between two frames being associated.
        # Aircraft at 250 m/s move ~0.5 km in 2 s; frames further apart than
        # this produce inconsistent TDOA geometry → large position errors.
        # Read only by the detection path (detection_association.py); the
        # frame-skew counters that reported on it moved there with it, having
        # been permanently zero here since the track path took over.
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
            max_bistatic_range_km=config.get("max_bistatic_range_km"),
            coverage_limit=(self.coverage_provider(node_id)
                            if self.coverage_provider else None),
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
                # Changing the range *rule* reshapes the footprint from a circle
                # on the RX to an ellipse with foci at RX and TX, so the cached
                # grid is no longer describing the same region.
                and existing.max_bistatic_range_km == geo.max_bistatic_range_km
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

    def _reset_for_tests(self) -> None:
        """Clear every per-node store, cursor and counter.  Test isolation
        only — production removal goes through unregister_node."""
        with self._register_lock:
            self.node_geometries.clear()
            self.node_configs.clear()
            self._pending_frames.clear()
            self._pending_tracks.clear()
            self.overlap_zones.clear()
            self._neighbors.clear()
            self._last_assoc.clear()
            self._neighbor_cursor.clear()
            for name in (
                "track_pairs_gated", "track_pairs_rejected",
                "track_pairs_accepted", "track_pairs_unfitted",
                "track_pairs_superseded", "track_pairs_deferred",
            ):
                setattr(self, name, 0)

    def unregister_node(self, node_id: str) -> int:
        """Drop a node and every overlap zone it participates in.

        register_node only ever adds, so without this a node that leaves the
        fleet keeps its geometry, its grids against every neighbour, and its
        adjacency entries until the process restarts — and each round still
        walks them.  Returns the number of overlap zones removed.
        """
        with self._register_lock:
            self.node_geometries.pop(node_id, None)
            self.node_configs.pop(node_id, None)
            self._pending_frames.pop(node_id, None)
            self._pending_tracks.pop(node_id, None)
            # Rate-limit timestamp and rotation cursor too: a fleet
            # regeneration reuses node ids, and a stale _last_assoc
            # suppressed the replacement node's first association round
            # while a stale cursor started its rotation at a leftover
            # offset.
            self._last_assoc.pop(node_id, None)
            self._neighbor_cursor.pop(node_id, None)

            stale_pairs = [k for k in self.overlap_zones if node_id in k]
            for key in stale_pairs:
                del self.overlap_zones[key]

            for other in self._neighbors.pop(node_id, set()):
                self._neighbors.get(other, set()).discard(node_id)
        return len(stale_pairs)

    def rebuild_zones_for(self, node_id: str) -> int:
        """Recompute this node's overlap grids against its current coverage.

        Grids are built once at registration, so a polygon that tightens
        afterwards does not retroactively tighten them — the constraint would
        only ever take effect on a restart.  Callers detect the change with
        coverage_digest and call this.

        Returns the number of pairs rebuilt.  Costs one grid computation per
        neighbour, so it is meant for a background cadence, not the frame path.
        """
        geo = self.node_geometries.get(node_id)
        if geo is None:
            return 0
        if self.coverage_provider is not None:
            geo.coverage_limit = self.coverage_provider(node_id)
        rebuilt = 0
        with self._register_lock:
            for other_id, other_geo in list(self.node_geometries.items()):
                if other_id == node_id:
                    continue
                pair_key = tuple(sorted([node_id, other_id]))
                a, b = ((geo, other_geo) if pair_key[0] == node_id
                        else (other_geo, geo))
                zone = compute_overlap_zone(
                    a, b,
                    grid_step_km=self.grid_step_km,
                    delay_gate_us=self.delay_gate_us,
                    doppler_gate_hz=self.doppler_gate_hz,
                )
                self.overlap_zones[pair_key] = zone
                # Adjacency follows: a grid that has tightened to nothing is no
                # longer a neighbour, and leaving it in place would keep pairing
                # frames against an empty zone every round.
                if zone.delay_pairs:
                    self._neighbors.setdefault(node_id, set()).add(other_id)
                    self._neighbors.setdefault(other_id, set()).add(node_id)
                else:
                    self._neighbors.get(node_id, set()).discard(other_id)
                    self._neighbors.get(other_id, set()).discard(node_id)
                rebuilt += 1
        return rebuilt

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

        # Visit at most _ASSOC_MAX_NEIGHBORS, starting where the last round
        # stopped.  Sorted so the rotation is over a stable sequence — a set's
        # iteration order is fixed but arbitrary, and slicing it unrotated would
        # hand the same neighbours every round and never reach the tail.
        ordered_neighbors = sorted(neighbors)
        n_total = len(ordered_neighbors)
        cap = min(self._ASSOC_MAX_NEIGHBORS, n_total)
        start = self._neighbor_cursor.get(node_id, 0) % n_total
        visit = [ordered_neighbors[(start + i) % n_total] for i in range(cap)]
        if cap < n_total:
            # Advance the cursor for the capped fleet, but do NOT count this
            # as a deferral: nothing has been deferred yet, and the counter
            # (documented as "rounds that deferred work") was incrementing on
            # every rotated round — and a second time below when the budget
            # also ran out in the same round.
            self._neighbor_cursor[node_id] = (start + cap) % n_total

        out: list[TrackPairCandidate] = []
        # Two budgets, because they bound different costs.  fits bounds inline
        # LM solves; pairs bounds candidates handed to the solver worker when
        # the fit is deferred.  Merging them was the bug: the fit budget was
        # decremented only inside the fit branch, so on the deferred path it
        # never depleted and its cap silently applied per node pair instead.
        budget = {"fits": self._MAX_FITS_PER_ROUND,
                  "pairs": self._MAX_PAIRS_PER_ROUND}
        for other_id in visit:
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
            emitted = self._pair_tracks(zone, tracks_a, tracks_b, timestamp_ms,
                                        budget)
            out.extend(emitted)
            budget["pairs"] -= len(emitted)
            if budget["fits"] <= 0 or budget["pairs"] <= 0:
                # Round exhausted.  Remaining neighbours are picked up next
                # round from the cursor above, so nothing is lost, only
                # deferred — which is only true because the cursor rotates.
                # This is the one place work is actually deferred, so it is
                # the one place the counter increments: once per such round.
                self._neighbor_cursor[node_id] = (
                    (start + visit.index(other_id) + 1) % n_total)
                self.track_pairs_deferred += 1
                break
        return out

    def _pair_tracks(self, zone: OverlapZone, tracks_a: list[dict],
                     tracks_b: list[dict], timestamp_ms: int,
                     budget: dict | None = None
                     ) -> list[TrackPairCandidate]:
        # ── Stage 1: enumerate and score every hypothesis ────────────────────
        # Each surviving (track_a, track_b) is a hypothesis about the world:
        # "these two echo histories are one aircraft."  They are scored here and
        # judged against each other in stage 2 — not one at a time, because the
        # hypotheses are not independent: one track is one aircraft, so two
        # pairings sharing a track are mutually exclusive.
        fitted: list[TrackPairCandidate] = []
        held: list[TrackPairCandidate] = []   # too little span to score yet

        # One BLAS contraction for every pairing, rather than a full-grid scan
        # each — see _batch_grid_match.  This runs in the frame worker, so its
        # cost is frame latency.
        usable_a = [t for t in tracks_a if t.get("history")]
        usable_b = [t for t in tracks_b if t.get("history")]
        matches = _batch_grid_match(
            zone,
            [float(t["history"][-1]["delay_us"]) for t in usable_a],
            [float(t["history"][-1]["delay_us"]) for t in usable_b],
        )
        if not matches:
            return []

        # Ordering by coarse delay residual and capping the count keeps one
        # association round bounded regardless of how many aircraft share the
        # zone; the survivors are re-tested every round, so a genuine pairing
        # that loses a crowded round is deferred rather than dropped.
        #
        # Which cap applies depends on what the round is paying for.  Fitting
        # inline, it is the LM budget.  Deferring (cv_fit=None, i.e. production)
        # no fit runs here at all, so bounding on the fit budget would discard
        # candidates for a cost nobody incurs — the pair budget is the real one,
        # and it is an order of magnitude larger.
        _b = budget or {}
        limit = (_b.get("fits", self._MAX_FITS_PER_ROUND)
                 if self.cv_fit is not None
                 else _b.get("pairs", self._MAX_PAIRS_PER_ROUND))
        ordered = sorted(
            matches.items(),
            key=lambda kv: abs(zone._np_pred_a[kv[1]]
                               - float(usable_a[kv[0][0]]["history"][-1]["delay_us"]))
            + abs(zone._np_pred_b[kv[1]]
                  - float(usable_b[kv[0][1]]["history"][-1]["delay_us"])),
        )[: max(limit, 0)]

        for (i_a, i_b), best_g in ordered:
            ta, tb = usable_a[i_a], usable_b[i_b]
            hist_a, hist_b = ta["history"], tb["history"]
            last_a, last_b = hist_a[-1], hist_b[-1]
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
                # Seed only — never a rejection here, unlike the detection
                # path.  The implied-speed test measured 0% power against
                # real cross pairings (their speeds are 27-298 m/s), and at
                # a 3 km grid point on a possibly-wrong altitude layer the
                # distorted bisectors make it false-alarm against true ones
                # (observed: it silently dropped a genuine pairing in the
                # three-node test).  The fit is the arbiter now; an absurd
                # implied velocity just means no seed.
                if v is not None and math.hypot(v[0], v[1]) <= _V_MAX_MS:
                    vel_seed = {"vel_east_ms": v[0], "vel_north_ms": v[1]}

            lat, lon, alt_km = g_lat, g_lon, g_alt
            vel_e = vel_seed["vel_east_ms"] if vel_seed else 0.0
            vel_n = vel_seed["vel_north_ms"] if vel_seed else 0.0
            chi2_per_dof = None
            dof = 0
            epochs = _merge_epochs(hist_a, zone.node_a_id, hist_b, zone.node_b_id)

            # The span that must clear cv_min_span_s is the *shorter
            # track's own*, not the merged window's.  Discrimination comes
            # from curvature accumulated by both trajectories; a 2-sample
            # track welded onto a 20 s one fits at chi2/dof ~0.04 because
            # its 4 residuals can be absorbed by sliding the velocity along
            # the long track's unobservable direction — the least-tested
            # hypothesis would then outscore the most-tested one in
            # selection, which is exactly backwards.
            span_a = float(hist_a[-1]["t_s"]) - float(hist_a[0]["t_s"])
            span_b = float(hist_b[-1]["t_s"]) - float(hist_b[0]["t_s"])
            if (self.cv_fit is not None
                    and len(epochs) >= self.cv_min_epochs
                    and min(span_a, span_b) >= self.cv_min_span_s):
                fit = self.cv_fit(
                    {"initial_guess": {"lat": g_lat, "lon": g_lon, "alt_km": g_alt},
                     "initial_velocity": vel_seed,
                     "epochs": epochs,
                     "timestamp_ms": timestamp_ms},
                    self.node_configs,
                )
                if budget is not None:
                    budget["fits"] -= 1
                if fit is None or not fit.get("success"):
                    self.track_pairs_unfitted += 1
                    continue
                chi2_per_dof = fit["chi2_per_dof"]
                dof = fit["dof"]
                lat, lon = fit["lat"], fit["lon"]
                alt_km = fit["alt_m"] / 1000.0
                vel_e, vel_n = fit["vel_east"], fit["vel_north"]
                deferred_epochs = None
            else:
                # No inline fit.  Two reasons, and they want different things.
                #
                # Too little history: held rather than dropped, since the
                # pairing is re-tested every round and a real one accumulates
                # the span it needs within a few frames.
                #
                # No cv_fit injected: the caller wants the fit run elsewhere,
                # so the epochs travel with the candidate and the solver worker
                # does it on its own threads.  That is the production shape —
                # an 86 ms solve does not belong on the frame path.
                self.track_pairs_unfitted += 1
                deferred_epochs = (
                    epochs
                    if (self.cv_fit is None
                        and len(epochs) >= self.cv_min_epochs
                        and min(span_a, span_b) >= self.cv_min_span_s)
                    else None
                )

            cand = TrackPairCandidate(
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
                epochs=deferred_epochs,
            )
            (fitted if chi2_per_dof is not None else held).append(cand)

        # ── Stage 2: hypothesis selection ────────────────────────────────────
        # The absolute threshold alone is blind early: over a 4 s window a
        # crossed pairing fits at chi2/dof ~1.4, under any bar that keeps real
        # targets.  But selection does not need the crossed hypothesis to fail
        # the bar — only the true one to beat it, and ~0.5 beats 1.4 at any
        # span.  So exclusivity discriminates exactly where the threshold
        # cannot.
        #
        # Greedy in ascending chi2.  Two earlier exclusivity attempts failed —
        # ranking by cluster size (reverted; contaminated big clusters starved
        # true pairs) and assignment on the delay residual (recall of true
        # pairings fell to 64%, the cost being ~0 for every pairing at n=2).
        # Both failures were the score, not the idea: chi2 from the CV fit is
        # the first non-degenerate cost this competition has had.
        #
        # Claim-then-vet: a winner claims its tracks even when it fails the
        # threshold.  If the *best* explanation of two tracks is implausible, a
        # worse one must not inherit them.
        #
        # Exclusivity is per node pair only.  A 3-node target legitimately
        # pairs its A-track with a B-track and a C-track; those matrices are
        # separate.
        results: list[TrackPairCandidate] = []
        claimed_a: set[str] = set()
        claimed_b: set[str] = set()
        if self.cv_exclusive:
            fitted.sort(key=lambda c: c.chi2_per_dof)
        for c in fitted:
            if self.cv_exclusive and (
                    c.track_a_id in claimed_a or c.track_b_id in claimed_b):
                self.track_pairs_superseded += 1
                continue
            claimed_a.add(c.track_a_id)
            claimed_b.add(c.track_b_id)
            if c.chi2_per_dof <= self.cv_chi2_max:
                self.track_pairs_accepted += 1
                results.append(c)
            else:
                self.track_pairs_rejected += 1
        for c in held:
            # A held pairing has no score, so it cannot claim anything — and a
            # track already explained by a scored winner does not get to seed a
            # second target on no evidence.
            if self.cv_exclusive and (
                    c.track_a_id in claimed_a or c.track_b_id in claimed_b):
                self.track_pairs_superseded += 1
                continue
            results.append(c)
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
        km_per_lat = KM_PER_DEG_LAT
        km_per_lon = km_per_deg_lon(float(np.mean(lats)))
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
                # Present only when the fit has not run yet: the solver worker
                # runs it there, on its own threads and behind its own queue,
                # instead of on the frame path.  Taken from the pairing with the
                # most history, which is the best-conditioned in the cluster.
                "cv_epochs": max((p.epochs for p in group if p.epochs),
                                 key=len, default=None),
                "track_pair_ids": sorted(
                    {(p.track_a_id, p.track_b_id) for p in group}
                )[:1],
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
