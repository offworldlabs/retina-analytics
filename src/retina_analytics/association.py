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

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from retina_analytics.constants import (
    C_KM_S,
    C_KM_US,
    KM_PER_DEG_LAT,
    R_EARTH,
    bistatic_max_radius_km,
    has_full_geometry,
    km_per_deg_lon,
    offset_latlon_m,
    resolve_beam_azimuth_deg,
    resolve_beam_width_deg,
)
from retina_analytics.constants import (
    bearing_deg as _bearing_deg,
)
from retina_analytics.constants import (
    haversine_km as _haversine_km,
)
from retina_analytics.empirical_coverage import OBSERVED_LIMIT_MARGIN

_logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Top-down claiming (ASSOC_CLAIM_MODE) — see _claim_round.  Predicted-vs-
# measured gates, deliberately different from the bottom-up delay_gate_us
# (which compares two *measurements* of the same instant): a claim compares a
# measurement against a dead-reckoned projection of a published position.
#
# Wider than delay_gate_us=5.0: delay sensitivity is ~3-6.7 us/km and a
# published dark position carries 0.5-2.5 km of its own error, plus EWMA lag
# and DR drift on top.
CLAIM_DELAY_GATE_US = 10.0
# Doppler IS gated here, unlike the bottom-up path — a global track has a
# velocity to project, where a bare tracklet does not.  f_d error ~
# Delta(v.b)*fc/c ~ 10-20 Hz for a 10-15 m/s velocity error at 100-200 MHz.
CLAIM_DOPPLER_GATE_HZ = 25.0
# Maximum age of the global track's own last solve before a projection is no
# longer trusted to dead-reckon forward.
CLAIM_MAX_DR_AGE_S = 30.0
# Eligibility is an OR of the two below — the ghost guard.  A track qualifies
# either by corroboration (enough distinct nodes already agree) or by
# persistence (it has survived enough rounds that a one-shot phantom would
# already have expired) — see claim_eligible.
CLAIM_ELIGIBLE_MIN_N_NODES = 3
CLAIM_ELIGIBLE_MIN_SOLVE_COUNT = 2
# Round cost bound: caps how many global tracks one claim round projects,
# regardless of how many are live.  See _claim_round's cost comment.
CLAIM_MAX_GLOBAL_TRACKS = 200

# ADS-B seeding (ADSB_SEED_MODE) — see _adsb_seed_round.  Same predicted-vs-
# measured shape as the CLAIM_* gates above: trust the node's own tag, but
# verify it against geometry before it can pull a tracklet out of dark
# solving.  Values match CLAIM_DELAY_GATE_US / CLAIM_DOPPLER_GATE_HZ — the
# comparison is the identical "measurement vs. dead-reckoned published
# position" shape, just sourced from a live ADS-B fix instead of a solved
# global track.
ADSB_SEED_DELAY_GATE_US = 10.0
ADSB_SEED_DOPPLER_GATE_HZ = 25.0
# Maximum |age| between the tracklet's newest epoch and the ADS-B fix,
# dead-reckoned across the gap in EITHER direction — unlike
# CLAIM_MAX_DR_AGE_S (a published dark solve, always behind the tracklet),
# a live ADS-B fix is usually *fresher* than the newest tracklet epoch:
# synthetic nodes frame at ~40 s, so the fix routinely leads.  45 s covers
# one frame interval plus margin.
ADSB_SEED_MAX_DR_AGE_S = 45.0

# ── Geometry helpers ─────────────────────────────────────────────────────────
#
# Distance and bearing come from constants.py; this module used to carry
# byte-identical copies.  What remains here is genuinely local: an ENU frame
# (the grid search works in metres from a reference, not in degrees) and the
# delay/bisector maths that operates inside it.


def _lla_to_enu(lat, lon, alt_km, ref_lat, ref_lon, ref_alt_km, cos_ref=None):
    """LLA to ENU metres-from-reference.

    ``cos_ref`` is ``cos(radians(ref_lat))``, the one per-reference constant in
    here, passed in by callers that already hold it for a fixed reference (see
    NodeGeometry._rx_frame).  Left None it is computed as before, and the
    arithmetic below is written in the same order either way, so a memoised
    caller gets bit-identical output rather than merely close output.
    """
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    north = dlat * R_EARTH
    if cos_ref is None:
        cos_ref = math.cos(math.radians(ref_lat))
    east = dlon * R_EARTH * cos_ref
    up = alt_km - ref_alt_km
    return (east, north, up)


def _enu_to_lla(east_km, north_km, up_km, ref_lat, ref_lon, ref_alt_km):
    lat = ref_lat + math.degrees(north_km / R_EARTH)
    lon = ref_lon + math.degrees(east_km / (R_EARTH * math.cos(math.radians(ref_lat))))
    alt_km_out = ref_alt_km + up_km
    return (lat, lon, alt_km_out)


def _norm(v):
    """Euclidean length of a 3-vector.

    Every caller in this module passes exactly three components (ENU
    displacements and bisectors), so this unpacks rather than folding over an
    arbitrary length: the generator-in-sum it replaces built a throwaway
    iterator per call on a path that runs once per grid point per node.
    """
    x, y, z = v
    return math.sqrt(x * x + y * y + z * z)


def _ranges_to_stations(target_enu, tx_enu, rx_enu):
    """(d_tx, d_rx): target→TX and target→RX ranges, in the given ENU frame.

    The delay and the bisector both need this pair, so a caller wanting both
    (predict_observation) computes it once here instead of twice.
    """
    d_tx = _norm((target_enu[0] - tx_enu[0], target_enu[1] - tx_enu[1], target_enu[2] - tx_enu[2]))
    d_rx = _norm((target_enu[0] - rx_enu[0], target_enu[1] - rx_enu[1], target_enu[2] - rx_enu[2]))
    return d_tx, d_rx


def _delay_from_ranges(d_tx, d_rx, d_bl):
    """Bistatic differential delay in μs from ranges that are already in hand."""
    return (d_tx + d_rx - d_bl) / C_KM_US


def _bistatic_delay_at(target_enu, tx_enu, rx_enu=(0, 0, 0), d_bl=None):
    """Bistatic differential delay in μs.

    ``d_bl`` is the RX→TX baseline length in the same frame — a per-node
    constant that a caller holding one (see NodeGeometry._rx_frame) passes in
    rather than having it re-normed on every call.
    """
    d_tx, d_rx = _ranges_to_stations(target_enu, tx_enu, rx_enu)
    if d_bl is None:
        d_bl = _norm((rx_enu[0] - tx_enu[0], rx_enu[1] - tx_enu[1], rx_enu[2] - tx_enu[2]))
    return _delay_from_ranges(d_tx, d_rx, d_bl)


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


def _bisector_from_ranges(target_enu, tx_enu, rx_enu, d_tx, d_rx):
    """b = u_tx + u_rx at the target, from ranges that are already in hand."""
    d_tx = d_tx or 1e-9
    d_rx = d_rx or 1e-9
    return (
        (tx_enu[0] - target_enu[0]) / d_tx + (rx_enu[0] - target_enu[0]) / d_rx,
        (tx_enu[1] - target_enu[1]) / d_tx + (rx_enu[1] - target_enu[1]) / d_rx,
        (tx_enu[2] - target_enu[2]) / d_tx + (rx_enu[2] - target_enu[2]) / d_rx,
    )


def _bisector(target_enu, tx_enu, rx_enu):
    """Return b = u_tx + u_rx at the target, in the same ENU frame."""
    d_tx, d_rx = _ranges_to_stations(target_enu, tx_enu, rx_enu)
    return _bisector_from_ranges(target_enu, tx_enu, rx_enu, d_tx, d_rx)


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
    # The node's EmpiricalCoverageState (learned FOV), or None.  Injected via
    # fov_provider — active mode only, so off/shadow grids are byte-identical
    # to before this existed.  When set, _point_in_beam's fov branch REPLACES
    # the theoretical bearing+range checks below it entirely; see
    # _point_in_beam.
    fov: object = None

    @property
    def baseline_km(self) -> float:
        """RX→TX baseline.

        Memoised on the coordinate tuple, so a geometry whose position is
        rewritten in place still recomputes.  footprint_radius_km reads this
        once per grid point per node, which made a per-call haversine the
        single largest cost in compute_overlap_zone (33% of a node rebuild)
        for a value that is constant across the whole grid.
        """
        key = (self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon)
        cached = self.__dict__.get("_baseline_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        value = _haversine_km(*key)
        self.__dict__["_baseline_cache"] = (key, value)
        return value

    @property
    def _rx_frame(self) -> tuple[tuple[float, float, float], float, float]:
        """ENU constants of the frame anchored at this node's own RX.

        Returns (tx_enu, baseline_len_km, cos_rx_lat).  In that frame the RX is
        the origin by construction, so the only station position left to derive
        is the TX — and it, its distance from the origin, and the longitude
        scale factor are all fixed by the node's own coordinates, not by
        whatever target is being predicted.  predict_observation re-derived all
        three on every call.

        Memoised on the six coordinate fields the derivation reads, with the
        same key-tuple pattern baseline_km uses, so a geometry whose position
        is rewritten in place still recomputes rather than serving a stale
        frame.
        """
        key = (self.rx_lat, self.rx_lon, self.rx_alt_km, self.tx_lat, self.tx_lon, self.tx_alt_km)
        cached = self.__dict__.get("_rx_frame_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        cos_rx = math.cos(math.radians(self.rx_lat))
        tx_enu = _lla_to_enu(self.tx_lat, self.tx_lon, self.tx_alt_km, self.rx_lat, self.rx_lon, self.rx_alt_km, cos_rx)
        # RX is the origin here, so |RX − TX| is just |tx_enu|.
        value = (tx_enu, _norm(tx_enu), cos_rx)
        self.__dict__["_rx_frame_cache"] = (key, value)
        return value

    @property
    def footprint_radius_km(self) -> float:
        """Largest RX-relative range this node's footprint reaches.

        _point_in_beam reads this once (twice under FOV_MODE) for every grid
        point of every node, for a value that is constant across the whole
        grid, so the branch that derives something is memoised on exactly the
        fields it reads — the bistatic limit and the four coordinates
        baseline_km is itself keyed on — with baseline_km's key-tuple pattern,
        so an in-place geometry rewrite still recomputes.

        The monostatic branch returns before any of that: it is a bare field
        read, and building a key to cache it costs more than it saves.
        """
        if self.max_bistatic_range_km is None:
            return self.max_range_km
        key = (self.max_bistatic_range_km, self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon)
        cached = self.__dict__.get("_footprint_cache")
        if cached is not None and cached[0] == key:
            return cached[1]
        value = bistatic_max_radius_km(self.baseline_km, self.max_bistatic_range_km)
        self.__dict__["_footprint_cache"] = (key, value)
        return value

    @property
    def effective_radius_km(self) -> float:
        """footprint_radius_km, widened to the learned FOV's reach when it
        exceeds the theoretical footprint.

        The learned FOV only ever widens past the theoretical wedge (see
        empirical_coverage.limit_km) — a bin can open outside it and extend
        past _reach_at on positives — so a bounding box or prefilter sized on
        footprint_radius_km alone would silently truncate exactly the region
        FOV_MODE exists to add.  Used by compute_overlap_zone's pair
        prefilter and grid bounding box.

        Deliberately NOT memoised, unlike the two properties above: fov is a
        live EmpiricalCoverageState that keeps learning, so max_limit_km()
        moves without any field of this dataclass changing and there is no key
        that would invalidate on it.  It inherits footprint_radius_km's memo
        for the half that does have one.
        """
        radius = self.footprint_radius_km
        if self.fov is not None:
            radius = max(radius, self.fov.max_limit_km())
        return radius


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
    delay_gate_us: float = 5.0  # max delay mismatch between prediction and measurement
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

    The detection-level AssociationCandidate (now in detection_association.py)
    pairs one echo with one echo,
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


def predict_observation(
    geo: NodeGeometry,
    lat: float,
    lon: float,
    alt_km: float,
    vel_east_ms: float,
    vel_north_ms: float,
    vel_up_ms: float = 0.0,
) -> tuple[float, float]:
    """Predicted (delay_us, doppler_hz) this node would observe for a target
    at (lat, lon, alt_km) moving at (vel_east_ms, vel_north_ms, vel_up_ms).

    ENU anchored at the node's own RX (matches _compute_node_enu / the test
    fixtures' convention of rx_enu=(0,0,0)).  doppler_hz = (v . b) *
    geo.fc_hz / (C_KM_S * 1000.0) is the exact inverse of the m/s conversion
    applied to a *measured* Doppler at the velocity-seed step above
    (m = f_d * lambda, lambda = c/fc): here f_d = (v.b)/lambda, so a claim's
    predicted Doppler and a tracklet's measured Doppler are directly
    comparable in Hz.  Positive is closing, matching _bisector's convention
    (b points from the target toward TX and RX, so v.b > 0 shrinks the
    bistatic path).
    """
    # The ENU reference IS this node's own RX, so rx_enu is the origin by
    # construction and everything else about the frame is a per-node constant —
    # see NodeGeometry._rx_frame, which memoises it.  The delay and the
    # bisector then share one computation of the two target→station ranges
    # instead of taking them twice.
    tx_enu, d_bl, cos_rx = geo._rx_frame
    rx_enu = (0.0, 0.0, 0.0)
    target_enu = _lla_to_enu(lat, lon, alt_km, geo.rx_lat, geo.rx_lon, geo.rx_alt_km, cos_rx)
    d_tx, d_rx = _ranges_to_stations(target_enu, tx_enu, rx_enu)
    delay_us = _delay_from_ranges(d_tx, d_rx, d_bl)
    b = _bisector_from_ranges(target_enu, tx_enu, rx_enu, d_tx, d_rx)
    v_dot_b = vel_east_ms * b[0] + vel_north_ms * b[1] + vel_up_ms * b[2]
    doppler_hz = v_dot_b * geo.fc_hz / (C_KM_S * 1000.0)
    return delay_us, doppler_hz


def associate_detections_to_adsb(
    geo: NodeGeometry,
    delays_us: list,
    dopplers_hz: list,
    adsb_states: dict[str, dict],
    frame_ts_ms: int,
    *,
    delay_gate_us: float = ADSB_SEED_DELAY_GATE_US,
    doppler_gate_hz: float = ADSB_SEED_DOPPLER_GATE_HZ,
    max_age_s: float = ADSB_SEED_MAX_DR_AGE_S,
) -> list | None:
    """Backend-side ADS-B correlation for a node with no receiver of its own.

    Builds the index-aligned per-detection adsb list a real node's own
    correlation would attach to frame["adsb"] — {hex, lat, lon, alt_baro,
    gs, track, flight} or None per detection index — from the region's
    ADS-B picture instead.

    Returns None when nothing could be produced: no state is fresh enough
    (abs(frame_ts - fix_ts) <= max_age_s) to dead-reckon, or none of the
    fresh ones gates against any detection.  None is load-bearing — it means
    "no ADS-B capability exercised this frame", distinct from an assigned
    empty list, and the caller (process_one_frame) must never overwrite a
    node-supplied list with it and must leave the frame untouched when it
    sees one.

    Per fresh state: dead-reckon to frame_ts_ms, predict_observation,
    residual against every detection, gate, score with the same
    d_res/gate + f_res/gate formula _adsb_seed_round uses.  Assignment is
    global greedy one-to-one — every surviving (detection, hex) pair sorted
    by score ascending, assigned first-come so neither a detection nor a
    hex is claimed twice.  Attaches the state's REPORTED lat/lon (not the
    dead-reckoned position), matching what a real node attaches.
    """
    if not delays_us:
        return None

    frame_ts_s = frame_ts_ms / 1000.0
    fresh = {
        h: st
        for h, st in adsb_states.items()
        if st.get("lat") is not None
        and st.get("lon") is not None
        and abs(frame_ts_s - st.get("timestamp_ms", 0) / 1000.0) <= max_age_s
    }
    if not fresh:
        return None

    candidates = []
    for hexn, st in fresh.items():
        dt = frame_ts_s - st.get("timestamp_ms", 0) / 1000.0
        dr_lat, dr_lon = offset_latlon_m(
            st["lat"],
            st["lon"],
            east_m=st.get("vel_east", 0.0) * dt,
            north_m=st.get("vel_north", 0.0) * dt,
        )
        pred_delay, pred_doppler = predict_observation(
            geo,
            dr_lat,
            dr_lon,
            st.get("alt_m", 0.0) / 1000.0,
            st.get("vel_east", 0.0),
            st.get("vel_north", 0.0),
        )
        for i, (d, f) in enumerate(zip(delays_us, dopplers_hz)):
            d_res = abs(pred_delay - float(d))
            f_res = abs(pred_doppler - float(f))
            if d_res > delay_gate_us or f_res > doppler_gate_hz:
                continue
            candidates.append((d_res / delay_gate_us + f_res / doppler_gate_hz, i, hexn, st))
    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    assigned_det: set = set()
    assigned_hex: set = set()
    out: list = [None] * len(delays_us)
    for _score, i, hexn, st in candidates:
        if i in assigned_det or hexn in assigned_hex:
            continue
        assigned_det.add(i)
        assigned_hex.add(hexn)
        out[i] = {
            "hex": hexn,
            "lat": st["lat"],
            "lon": st["lon"],
            "alt_baro": st.get("alt_baro"),
            "gs": st.get("gs"),
            "track": st.get("track"),
            "flight": st.get("flight"),
        }
    return out


def claim_eligible(track: dict) -> bool:
    """Ghost guard for top-down claiming: eligible by corroboration OR by
    persistence, never by a single unconfirmed solve.

    A track qualifying only by n_nodes has enough independent nodes that the
    bottom-up solve is already trustworthy; one qualifying only by
    solve_count has survived enough rounds that a one-shot mirror/wrong-frame
    ghost would already have been superseded or expired.  Either is enough —
    the OR is deliberate, not a placeholder for AND.
    """
    return (
        track.get("n_nodes", 0) >= CLAIM_ELIGIBLE_MIN_N_NODES
        or track.get("solve_count", 0) >= CLAIM_ELIGIBLE_MIN_SOLVE_COUNT
    )


def _point_in_beam(lat, lon, geo: NodeGeometry) -> bool:
    """Check if a point falls within the node's beam sector (2D check).

    Bearing only.  The *range* limit is applied in compute_overlap_zone against
    the exact bistatic delay it already computes for the grid point, which
    accounts for altitude and needs no closed form.  Splitting them this way
    keeps the cheap angular test first, where it prunes most of the bounding
    box, and leaves the range test where the geometry is already in hand.

    geo.footprint_radius_km bounds how far the sector can extend, so callers
    that need a radius before any delay exists still have one.

    geo.fov (FOV_MODE active only — see InterNodeAssociator's fov_provider)
    REPLACES every check below it, rather than adding to them: the learned
    FOV already embeds the theoretical prior for young bins (_bin_open opens
    inside the theoretical wedge unconditionally), so falling through to the
    theoretical checks afterward would double-apply the same prior AND
    re-impose the shrink-only ceiling this mode exists to lift.
    """
    dist = _haversine_km(geo.rx_lat, geo.rx_lon, lat, lon)
    if geo.fov is not None:
        if dist > max(geo.footprint_radius_km, geo.fov.max_limit_km()):
            return False
        bearing = _bearing_deg(geo.rx_lat, geo.rx_lon, lat, lon)
        return geo.fov.contains(bearing, dist)
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


def compute_overlap_zone(
    geo_a: NodeGeometry,
    geo_b: NodeGeometry,
    grid_step_km: float = 3.0,
    altitudes_km: tuple[float, ...] = (1.5, 3.0, 5.0, 7.0, 9.0, 11.0),
    delay_gate_us: float = 5.0,
    doppler_gate_hz: float = 30.0,
) -> OverlapZone:
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
    # ever looked at.  effective_radius_km additionally widens to the learned
    # FOV's reach (FOV_MODE active) so a bin that opened past the theoretical
    # wedge is not pruned before compute_overlap_zone ever gets to test it.
    rx_sep = _haversine_km(geo_a.rx_lat, geo_a.rx_lon, geo_b.rx_lat, geo_b.rx_lon)
    if rx_sep > geo_a.effective_radius_km + geo_b.effective_radius_km:
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

    # Determine bounding box for the grid.  Sized on the footprint radius (or
    # the learned FOV's reach if wider — see effective_radius_km), so a node
    # aimed along a long baseline, or one whose learned FOV opened past its
    # theoretical wedge, still gets its far lobe enumerated.
    max_range = max(geo_a.effective_radius_km, geo_b.effective_radius_km)
    n_steps = int(2 * max_range / grid_step_km) + 1

    grid_points = []
    delay_pairs = []
    bisector_pairs = []

    # The both-beams test is 2-D — _point_in_beam takes (lat, lon) only, and
    # the lat/lon of a column depends on (east, north) alone — so it is
    # resolved once here rather than re-derived identically for each of the
    # six altitudes.  It dominated the rebuild (87% of a node rebuild's time,
    # 855k calls where 143k distinct columns exist), and the altitude loop
    # below now runs over the survivors, which on this fleet is a small
    # fraction of the bounding box.  Column order is preserved and altitude
    # stays outermost, so the emitted grid is identical to the triple loop's.
    columns = []
    for i in range(n_steps):
        east = -max_range + i * grid_step_km
        for j in range(n_steps):
            north = -max_range + j * grid_step_km

            lat, lon, _ = _enu_to_lla(east, north, 0.0, ref_lat, ref_lon, ref_alt_km)

            # Must be in BOTH beams
            if not _point_in_beam(lat, lon, geo_a):
                continue
            if not _point_in_beam(lat, lon, geo_b):
                continue
            columns.append((east, north, lat, lon))

    if not columns:
        return OverlapZone(
            node_a_id=geo_a.node_id,
            node_b_id=geo_b.node_id,
            grid_points=[],
            delay_pairs=[],
            fc_a_hz=geo_a.fc_hz,
            fc_b_hz=geo_b.fc_hz,
            delay_gate_us=delay_gate_us,
            doppler_gate_hz=doppler_gate_hz,
        )

    # Each node's RX→TX baseline length is fixed for the whole grid, so it is
    # normed once here rather than inside _bistatic_delay_at at every one of
    # the ~16 000 points.
    d_bl_a = _norm((rx_a_enu[0] - tx_a_enu[0], rx_a_enu[1] - tx_a_enu[1], rx_a_enu[2] - tx_a_enu[2]))
    d_bl_b = _norm((rx_b_enu[0] - tx_b_enu[0], rx_b_enu[1] - tx_b_enu[1], rx_b_enu[2] - tx_b_enu[2]))

    for alt_km in altitudes_km:
        for east, north, lat, lon in columns:
            # Calculate bistatic delay at each node
            target_enu = (east, north, alt_km)
            delay_a = _bistatic_delay_at(target_enu, tx_a_enu, rx_a_enu, d_bl_a)
            delay_b = _bistatic_delay_at(target_enu, tx_b_enu, rx_b_enu, d_bl_b)

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
            if geo_a.max_bistatic_range_km is not None and delay_a * C_KM_US > geo_a.max_bistatic_range_km:
                continue
            if geo_b.max_bistatic_range_km is not None and delay_b * C_KM_US > geo_b.max_bistatic_range_km:
                continue

            # Only keep physically meaningful delays
            if delay_a < 0 or delay_b < 0:
                continue

            grid_points.append((lat, lon, alt_km))
            delay_pairs.append((delay_a, delay_b))
            # Bistatic bisector b = u_tx + u_rx (unit vectors from the
            # target toward each station).  Doppler measures v · b — see
            # implied_horizontal_velocity — so these are the projection axes
            # the two nodes observe the velocity along.  Precomputed here
            # because the grid is fixed at registration and the runtime
            # test then costs a couple of dot products.
            bisector_pairs.append(
                (
                    _bisector(target_enu, tx_a_enu, rx_a_enu),
                    _bisector(target_enu, tx_b_enu, rx_b_enu),
                )
            )

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

    Vectorised for the same reason detection_association.find_associations
    is: the gate is a
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

    gate_a = np.abs(da[:, None] - pred_a) < gate  # (Ta, G)
    gate_b = np.abs(pred_b[:, None] - db) < gate  # (G, Tb)
    # float32 so numpy takes the SGEMM path; uint8 has no BLAS kernel.
    match = gate_a.astype(np.float32) @ gate_b.astype(np.float32)

    out: dict = {}
    for i_a, i_b in zip(*np.where(match > 0)):
        i_a, i_b = int(i_a), int(i_b)
        valid = np.nonzero(gate_a[i_a] & gate_b[:, i_b])[0]
        if valid.size == 0:
            continue
        res = np.abs(pred_a[valid] - da[i_a]) + np.abs(pred_b[valid] - db[i_b])
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


def _coord(config: dict, key: str) -> float:
    """A latitude/longitude from a config, absent or explicitly null reading 0.0.

    `config.get(key, 0)` is not enough: a v1 registration may carry the key with
    a null value, and None then reaches the geodesy as a float.
    """
    return float(config.get(key) or 0.0)


def _merge_epochs_multi(histories: list[tuple[str, list[dict]]]) -> list:
    """N-node generalisation of _merge_epochs, same output shape.

    Used to build cv_epochs for a claim's anchored solver input, which may
    span 2 or more matched nodes — _merge_epochs is fixed at exactly two.
    Same no-resampling rule applies: see _merge_epochs.
    """
    samples = []
    for node_id, hist in histories:
        samples += [(float(h["t_s"]), node_id, h) for h in hist]
    samples.sort(key=lambda s: s[0])

    epochs: list[dict] = []
    for t_s, node_id, h in samples:
        meas = {
            "node_id": node_id,
            "delay_us": float(h["delay_us"]),
            "doppler_hz": float(h["doppler_hz"]),
            "snr": float(h.get("snr", 0.0)),
        }
        if epochs and epochs[-1]["t_s"] == t_s:
            epochs[-1]["measurements"].append(meas)
        else:
            epochs.append({"t_s": t_s, "measurements": [meas]})
    return epochs


@dataclass
class AssociationRound:
    """Everything one submit_tracks_round call produced.

    pairs is the bottom-up result submit_tracks always returned.
    anchored_inputs and claims are the top-down claiming stage's output —
    empty unless claim_mode != "off" and a provider is set; anchored_inputs
    is additionally forced empty in shadow mode (see submit_tracks_round).
    claims carries per-claim debug records ({anchor_key, node_id, track_id,
    d_res_us, f_res_hz, dr_age_s, won}) regardless of mode, for the shadow
    DEBUG log and tests — it is never filtered by mode.
    adsb_inputs is the ADS-B seeding stage's output (ADSB_SEED_MODE) — same
    solver-input shape as anchored_inputs but carrying a non-None adsb_hex
    and no anchor_key; empty unless adsb_seed_mode != "off" and a provider
    is set, and additionally forced empty in shadow mode (see
    submit_tracks_round).
    """

    pairs: list = field(default_factory=list)
    anchored_inputs: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    adsb_inputs: list = field(default_factory=list)


# ── InterNodeAssociator ──────────────────────────────────────────────────────


class InterNodeAssociator:
    """Manages overlap zones for all node pairs and runs association at runtime."""

    # grid_step_km default matches what production ships (ASSOC_GRID_STEP_KM
    # = 3.0) so a bare construction exercises the real geometry.  It was 30.0
    # — a grid 10x coarser and ~100x sparser than any deployment, and the
    # merge-distance rationale in detection_association assumes 3.0.
    def __init__(
        self,
        delay_gate_us: float = 5.0,
        doppler_gate_hz: float = 30.0,
        grid_step_km: float = 3.0,
        assoc_interval_s: float = 30.0,
        cv_fit=None,
        cv_chi2_max: float = 2.0,
        cv_min_epochs: int = 4,
        cv_min_span_s: float = 12.0,
        cv_exclusive: bool = True,
        coverage_provider=None,
        max_neighbors: int = 50,
        max_pairs_per_round: int = 64,
        *,
        fov_provider=None,
        claim_mode: str = "off",
        claim_delay_gate_us: float = CLAIM_DELAY_GATE_US,
        claim_doppler_gate_hz: float = CLAIM_DOPPLER_GATE_HZ,
        claim_max_dr_age_s: float = CLAIM_MAX_DR_AGE_S,
        global_track_provider=None,
        adsb_seed_mode: str = "off",
        adsb_seed_delay_gate_us: float = ADSB_SEED_DELAY_GATE_US,
        adsb_seed_doppler_gate_hz: float = ADSB_SEED_DOPPLER_GATE_HZ,
        adsb_seed_max_dr_age_s: float = ADSB_SEED_MAX_DR_AGE_S,
        adsb_provider=None,
        node_world_provider=None,
    ):
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
        # INLINE-MODE ONLY: with cv_fit=None (production) nothing here ever
        # scores a pairing, so cv_chi2_max and cv_exclusive are inert — the
        # live n=2 threshold is the solver worker's.  Only the offline bench's
        # --cv-fit inline mode exercises this.
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
        # node_id -> EmpiricalCoverageState | None, the learned FOV.  Backend
        # passes this ONLY in FOV_MODE active (core/state.py), so off/shadow
        # construct this associator with fov_provider=None and every geo.fov
        # stays None — grids identical to before FOV_MODE existed.  Kept
        # separate from coverage_provider (a bearing -> limit callable): the
        # fov branch in _point_in_beam needs the whole state (contains,
        # max_limit_km), not a single-bearing lookup.
        self.fov_provider = fov_provider
        # The constraint digest each node's grids were last built under, so a
        # polygon that tightens later can be detected and the grids rebuilt.
        # Counters for what the fine test actually did, so its value is
        # observable rather than assumed.
        self.track_pairs_gated: int = 0  # survived the coarse delay grid
        self.track_pairs_rejected: int = 0  # ... then failed the chi2 test
        self.track_pairs_accepted: int = 0
        self.track_pairs_unfitted: int = 0  # too few epochs, or the fit failed
        self.track_pairs_superseded: int = 0  # lost their tracks to a better fit
        self.track_pairs_deferred: int = 0  # rounds cut short by a budget (once per round)
        # Adjacency index: node_id → set of neighbor node_ids that share a real
        # overlap zone (delay_pairs is non-empty).  Built during registration so
        # submit_frame can iterate O(K) neighbors instead of O(N) all nodes.
        self._neighbors: dict[str, set[str]] = {}
        # Rate-limit per-node association to at most once per _ASSOC_MIN_INTERVAL_S.
        # Prevents O(K) × N frames/s = O(N²) CPU burn in dense deployments where
        # K ≈ N (wide beams, small area).
        #
        # BUDGET CALCULATION (1000-node fleet on 2-core / GIL-bound):
        #   detection_association.find_associations ≈ 50 µs/call.
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
        self._register_lock = __import__("threading").Lock()

        # ── Top-down claiming (ASSOC_CLAIM_MODE) ──────────────────────────
        # off/shadow/active, following the SOLVER_CONSENSUS_MODE precedent —
        # an unrecognised value falls back to "off" rather than raising, so a
        # typo'd env var degrades to today's bottom-up-only behaviour instead
        # of taking the pipeline down.
        self.claim_mode: str = claim_mode if claim_mode in ("off", "shadow", "active") else "off"
        self.claim_delay_gate_us = claim_delay_gate_us
        self.claim_doppler_gate_hz = claim_doppler_gate_hz
        self.claim_max_dr_age_s = claim_max_dr_age_s
        # Injected the same way cv_fit/coverage_provider are: this library
        # keeps knowing nothing about state.multinode_tracks.  Provider
        # contract — an iterable of dicts, each:
        #   {key, lat, lon, alt_m, vel_east, vel_north, vel_up?,
        #    timestamp_ms, n_nodes, solve_count}
        # The LIB applies claim_eligible, the DR-age cap, and the
        # CLAIM_MAX_GLOBAL_TRACKS truncation — the provider stays a dumb
        # snapshot (list(state.multinode_tracks.items())) so the offline
        # bench measures the shipped filtering rather than a backend-side
        # pre-filter the bench cannot see.
        self.global_track_provider = global_track_provider
        # Counters — plain += like track_pairs_*, added to _reset_for_tests.
        self.claim_rounds: int = 0
        self.claims_matched: int = 0
        self.claim_conflicts: int = 0
        self.anchored_inputs_emitted: int = 0
        self.tracklets_excluded: int = 0

        # ── ADS-B seeding (ADSB_SEED_MODE) ─────────────────────────────────
        # off/shadow/active, ASSOC_CLAIM_MODE precedent — an unrecognised
        # value falls back to "off" rather than raising.
        self.adsb_seed_mode: str = adsb_seed_mode if adsb_seed_mode in ("off", "shadow", "active") else "off"
        self.adsb_seed_delay_gate_us = adsb_seed_delay_gate_us
        self.adsb_seed_doppler_gate_hz = adsb_seed_doppler_gate_hz
        self.adsb_seed_max_dr_age_s = adsb_seed_max_dr_age_s
        # Injected the same way cv_fit/global_track_provider are: this
        # library keeps knowing nothing about state.adsb_aircraft.  Provider
        # contract — a callable returning dict[str, dict] keyed by
        # LOWERCASE ICAO hex, each value:
        #   {hex, lat, lon, alt_m, vel_east, vel_north, timestamp_ms,
        #    alt_baro, gs, track, flight}
        # (alt_m metres, vel_* m/s, timestamp_ms epoch ms of the fix;
        # alt_baro/gs/track/flight are the raw feed fields, carried so
        # associate_detections_to_adsb can attach entries in the same shape
        # nodes themselves report.)  The provider stays a dumb snapshot —
        # the LIB applies freshness and gating, so the offline bench
        # measures shipped filtering.
        self.adsb_provider = adsb_provider
        # node_id → "sim" | "real" | None.  The provider's cache mixes
        # simulated transponders with real traffic over one footprint, and a
        # node's echoes can only ever be of aircraft from its own world — so
        # a tagged state from the other world is verified-looking noise (a
        # hex collision, a mis-tag) whatever its residuals say.  Optional and
        # fail-open in both directions: no provider, or an unknown world on
        # either side, changes nothing.
        self.node_world_provider = node_world_provider
        # Counters — plain += like the claiming ones above.
        self.adsb_seed_rounds: int = 0
        self.adsb_tracklets_tagged: int = 0
        self.adsb_seed_no_state: int = 0
        self.adsb_seed_gate_rejects: int = 0
        self.adsb_seed_world_rejects: int = 0
        self.adsb_tracklets_excluded: int = 0
        self.adsb_inputs_emitted: int = 0

    def register_node(self, node_id: str, config: dict):
        """Register a node and pre-compute overlap zones with all existing nodes.

        Thread-safe: acquires an internal lock so concurrent registrations
        (e.g. from multiple run_in_executor calls) cannot corrupt iteration.

        Reconnecting nodes skip the expensive O(n²) overlap recomputation
        as long as their geometry (RX/TX position) hasn't changed.

        A node whose config lacks either end of the bistatic pair is
        registered but takes no part in overlap: see has_full_geometry.
        """
        positioned = has_full_geometry(config)
        rx_alt_km = (config.get("rx_alt_ft") or 0) * 0.3048 / 1000.0
        tx_alt_km = (config.get("tx_alt_ft") or 0) * 0.3048 / 1000.0

        geo = NodeGeometry(
            node_id=node_id,
            rx_lat=_coord(config, "rx_lat"),
            rx_lon=_coord(config, "rx_lon"),
            rx_alt_km=rx_alt_km,
            tx_lat=_coord(config, "tx_lat"),
            tx_lon=_coord(config, "tx_lon"),
            tx_alt_km=tx_alt_km,
            fc_hz=config.get("fc_hz", config.get("FC", 195e6)),
            beam_width_deg=resolve_beam_width_deg(config),
            max_range_km=config.get("max_range_km", 50),
            max_bistatic_range_km=config.get("max_bistatic_range_km"),
            coverage_limit=(self.coverage_provider(node_id) if self.coverage_provider else None),
            fov=(self.fov_provider(node_id) if self.fov_provider else None),
        )

        # Honour an explicit aim (aimed coverage-ring Yagi); otherwise broadside
        # to the RX→TX baseline. The overlap grid is computed from this azimuth,
        # so it must match the node's true aim. Shared with manager.register_node.
        geo.beam_azimuth_deg = resolve_beam_azimuth_deg(config, geo.rx_lat, geo.rx_lon, geo.tx_lat, geo.tx_lon)

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

            if not positioned:
                # Nothing to pair against, and nothing that was paired stays
                # valid: a node that re-registers without its position has lost
                # the geometry its zones were built from.
                self._drop_zones_for(node_id)
                self.node_geometries[node_id] = geo
                return

            # Pre-compute overlap zones with existing nodes (serialised to avoid
            # RuntimeError: dictionary changed size during iteration when multiple
            # nodes register concurrently from a thread-pool executor).
            for existing_id, existing_geo in list(self.node_geometries.items()):
                # node_configs[node_id] already holds the new config (rewritten
                # above), so _is_positioned(node_id) cannot be used to skip this
                # node's own stale entry here.
                if existing_id == node_id or not self._is_positioned(existing_id):
                    continue
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
                else:
                    # A relocated node may no longer overlap a former
                    # neighbour; leaving the adjacency in place would occupy a
                    # slot in the capped neighbour rotation forever.
                    self._neighbors.get(node_id, set()).discard(existing_id)
                    self._neighbors.get(existing_id, set()).discard(node_id)

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
                "track_pairs_gated",
                "track_pairs_rejected",
                "track_pairs_accepted",
                "track_pairs_unfitted",
                "track_pairs_superseded",
                "track_pairs_deferred",
                "claim_rounds",
                "claims_matched",
                "claim_conflicts",
                "anchored_inputs_emitted",
                "tracklets_excluded",
                "adsb_seed_rounds",
                "adsb_tracklets_tagged",
                "adsb_seed_no_state",
                "adsb_seed_gate_rejects",
                "adsb_seed_world_rejects",
                "adsb_tracklets_excluded",
                "adsb_inputs_emitted",
            ):
                setattr(self, name, 0)

    def _is_positioned(self, node_id: str) -> bool:
        """Whether a registered node has both ends of its geometry to pair against."""
        return has_full_geometry(self.node_configs.get(node_id, {}))

    def _drop_zones_for(self, node_id: str) -> int:
        """Remove every overlap zone and adjacency entry naming this node.

        Caller holds _register_lock.  Unlike unregister_node the node itself
        stays registered; only its pairings go.
        """
        stale_pairs = [k for k in self.overlap_zones if node_id in k]
        for key in stale_pairs:
            del self.overlap_zones[key]
        for other in self._neighbors.pop(node_id, set()):
            self._neighbors.get(other, set()).discard(node_id)
        return len(stale_pairs)

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

            return self._drop_zones_for(node_id)

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
        # An unpositioned node has no zones to rebuild, and rebuilding would put
        # back exactly the dense grids registration declined to compute.
        if not self._is_positioned(node_id):
            return 0
        if self.coverage_provider is not None:
            geo.coverage_limit = self.coverage_provider(node_id)
        if self.fov_provider is not None:
            geo.fov = self.fov_provider(node_id)
        rebuilt = 0
        with self._register_lock:
            for other_id, other_geo in list(self.node_geometries.items()):
                if other_id == node_id or not self._is_positioned(other_id):
                    continue
                pair_key = tuple(sorted([node_id, other_id]))
                a, b = (geo, other_geo) if pair_key[0] == node_id else (other_geo, geo)
                zone = compute_overlap_zone(
                    a,
                    b,
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

    def _adsb_seed_round(
        self, node_id: str, tracks: list[dict], neighbor_ids: list[str], timestamp_ms: int
    ) -> tuple[dict[str, set], list[dict]]:
        """Verify each round-node tracklet's own ADS-B tag against geometry,
        then group verified tracklets by hex into same-aircraft seeded
        solver inputs.

        Returns (tagged_ids_by_node, adsb_inputs).  Round-node assembly is
        identical to _claim_round's: the triggering node's `tracks` plus
        registered neighbours' `_pending_tracks` snapshots — a tag is
        verified in exactly the same node-local geometry a claim is.

        Fail-open throughout: a missing/stale ADS-B fix or a residual
        outside gate leaves the tracklet untouched (not tagged, not
        grouped) rather than trusting the node's tag on faith — trusting it
        unverified is exactly the failure mode this stage exists to catch
        (a stale or swapped tag pulling a dark target out of solving).
        """
        states = self.adsb_provider()
        if not states:
            return {}, []

        _norm_hex = lambda h: (h or "").strip().lower()  # noqa: E731 — local helper

        round_nodes: dict[str, list[dict]] = {}
        if node_id in self.node_geometries:
            round_nodes[node_id] = tracks
        for nid in neighbor_ids:
            if nid in self.node_geometries:
                round_nodes[nid] = self._pending_tracks.get(nid) or []

        # Which world each round node's echoes come from, resolved once per
        # round.  A state tagged with the other world is no verification
        # target for that node's tracklet, whatever its residuals say — the
        # cache is keyed by bare hex, so a simulated hex colliding with a
        # real one hands the wrong world's fix to the tag check.  None on
        # either side (no provider, unknown node, untagged state) disables
        # the comparison, never the tracklet.
        world_of: dict[str, object] = {}
        if self.node_world_provider is not None:
            for nid in round_nodes:
                world_of[nid] = self.node_world_provider(nid)

        # Best-scoring match per (hex, node_id) — step 4: two tracklets at
        # one node both verifying against the same hex, the loser is NOT
        # tagged (it may be a coincidentally-gating distinct target).
        best_by_hex_node: dict[tuple, dict] = {}

        for nid, views in round_nodes.items():
            geo = self.node_geometries[nid]
            for view in views:
                raw_hex = view.get("adsb_hex")
                if not raw_hex:
                    continue
                self.adsb_tracklets_tagged += 1
                hist = view.get("history")
                if not hist:
                    # Mirrors _claim_round's silent guard — confirmed views
                    # always carry history, this is defensive only.
                    continue
                last = hist[-1]
                hexn = _norm_hex(raw_hex)
                st = states.get(hexn)
                if st is None or st.get("lat") is None or st.get("lon") is None:
                    self.adsb_seed_no_state += 1
                    continue
                st_world = st.get("world")
                node_world = world_of.get(nid)
                if st_world is not None and node_world is not None and st_world != node_world:
                    self.adsb_seed_world_rejects += 1
                    continue
                dt = float(last["t_s"]) - st.get("timestamp_ms", 0) / 1000.0
                if abs(dt) > self.adsb_seed_max_dr_age_s:
                    self.adsb_seed_no_state += 1
                    continue

                dr_lat, dr_lon = offset_latlon_m(
                    st["lat"],
                    st["lon"],
                    east_m=st.get("vel_east", 0.0) * dt,
                    north_m=st.get("vel_north", 0.0) * dt,
                )
                pred_delay, pred_doppler = predict_observation(
                    geo,
                    dr_lat,
                    dr_lon,
                    st.get("alt_m", 0.0) / 1000.0,
                    st.get("vel_east", 0.0),
                    st.get("vel_north", 0.0),
                )
                d_res = abs(pred_delay - float(last["delay_us"]))
                f_res = abs(pred_doppler - float(last["doppler_hz"]))
                if d_res > self.adsb_seed_delay_gate_us or f_res > self.adsb_seed_doppler_gate_hz:
                    self.adsb_seed_gate_rejects += 1
                    continue

                rec = {
                    "node_id": nid,
                    "track_id": str(view.get("track_id")),
                    "hist": hist,
                    "last": last,
                    "_st": st,
                    "score": (d_res / self.adsb_seed_delay_gate_us + f_res / self.adsb_seed_doppler_gate_hz),
                }
                key = (hexn, nid)
                cur = best_by_hex_node.get(key)
                if cur is None or rec["score"] < cur["score"]:
                    best_by_hex_node[key] = rec

        # Every surviving (hex, node) match is tagged — including
        # single-node hexes.  Exclusion is deliberate even when no seeded
        # input can form: a verified lit tracklet's lit-dark cross-pairings
        # are exactly the ghosts this stage removes, and the aircraft
        # itself stays displayed via the live ADS-B feed.
        tagged_ids_by_node: dict[str, set] = defaultdict(set)
        matches_by_hex: dict[str, list[dict]] = defaultdict(list)
        for (hexn, nid), rec in best_by_hex_node.items():
            tagged_ids_by_node[nid].add(rec["track_id"])
            matches_by_hex[hexn].append(rec)

        # Seeded input per hex with >= 2 matched nodes AND the triggering
        # node among them — same emission-dedup rule as anchored inputs
        # (see _claim_round).
        adsb_inputs: list[dict] = []
        for hexn, matches in matches_by_hex.items():
            matched_node_ids = {m["node_id"] for m in matches}
            if len(matched_node_ids) < 2 or node_id not in matched_node_ids:
                continue
            newest = max(matches, key=lambda m: float(m["last"]["t_s"]))
            st = newest["_st"]
            dt_newest = float(newest["last"]["t_s"]) - st.get("timestamp_ms", 0) / 1000.0
            guess_lat, guess_lon = offset_latlon_m(
                st["lat"],
                st["lon"],
                east_m=st.get("vel_east", 0.0) * dt_newest,
                north_m=st.get("vel_north", 0.0) * dt_newest,
            )
            cv_epochs = _merge_epochs_multi([(m["node_id"], m["hist"]) for m in matches])
            by_snr = sorted(
                matches,
                key=lambda m: float(m["last"].get("snr", 0.0)),
                reverse=True,
            )
            track_ids_by_node: dict[str, list] = defaultdict(list)
            for m in matches:
                track_ids_by_node[m["node_id"]].append(m["track_id"])

            adsb_inputs.append(
                {
                    "initial_guess": {
                        "lat": guess_lat,
                        "lon": guess_lon,
                        "alt_km": st.get("alt_m", 0.0) / 1000.0,
                    },
                    "initial_velocity": {
                        "vel_east_ms": st.get("vel_east", 0.0),
                        "vel_north_ms": st.get("vel_north", 0.0),
                    },
                    "measurements": [
                        {
                            "node_id": m["node_id"],
                            "delay_us": float(m["last"]["delay_us"]),
                            "doppler_hz": float(m["last"]["doppler_hz"]),
                            "snr": float(m["last"].get("snr", 0.0)),
                        }
                        for m in matches
                    ],
                    "n_nodes": len(matched_node_ids),
                    "timestamp_ms": timestamp_ms,
                    "adsb_hex": hexn,
                    "chi2_per_dof": None,
                    "n_epochs": len(cv_epochs),
                    # REQUIRED, same reason as anchored_inputs: without
                    # cv_epochs an n=2 seeded input dies permanently at the
                    # solver's confirmation gate (_resolve_cv_fit -> None ->
                    # n2_unconfirmed).
                    "cv_epochs": cv_epochs,
                    "track_pair_ids": (
                        [tuple(sorted([by_snr[0]["track_id"], by_snr[1]["track_id"]]))] if len(by_snr) >= 2 else []
                    ),
                    "track_ids": sorted({m["track_id"] for m in matches}),
                    "track_ids_by_node": {nid: sorted(ids) for nid, ids in track_ids_by_node.items()},
                }
            )
        self.adsb_inputs_emitted += len(adsb_inputs)
        return dict(tagged_ids_by_node), adsb_inputs

    def _claim_round(
        self,
        node_id: str,
        tracks: list[dict],
        neighbor_ids: list[str],
        timestamp_ms: int,
        exclude_ids_by_node: dict[str, set] | None = None,
    ) -> tuple[dict[str, set], list[dict], list[dict]]:
        """Top-down claiming: project eligible global tracks into this
        round's node set and match them against local tracklet views.

        Returns (claimed_ids_by_node, anchored_inputs, claim_records).
        claimed_ids_by_node maps node_id -> set of track_id claimed at that
        node this round; the caller decides whether that exclusion actually
        applies (active mode only).

        exclude_ids_by_node (node_id -> set of track_id, typically ADS-B
        seeding's tagged set) removes views from candidacy entirely before
        any global is tested against them — an already-verified-lit
        tracklet must never be claimed by a dark global (identity theft),
        so it produces no claim record at all here, won or lost.

        Cost: <= CLAIM_MAX_GLOBAL_TRACKS x (1+|neighbor_ids|) x
        tracklets/node predict_observation calls, each a few dot products —
        measured <1 ms/round at observed scale, and it runs once per node
        per _ASSOC_MIN_INTERVAL_S (30 s in production), same cadence as the
        bottom-up round it rides alongside.
        """
        provider = self.global_track_provider
        if provider is None:
            return {}, [], []

        globals_ = sorted(
            (g for g in provider() if claim_eligible(g)),
            key=lambda g: g.get("timestamp_ms", 0),
            reverse=True,
        )[:CLAIM_MAX_GLOBAL_TRACKS]
        if not globals_:
            return {}, [], []

        # Round node set: the triggering node plus this round's neighbours,
        # restricted to nodes actually registered with a geometry — an
        # unregistered neighbour has nothing to predict an observation with.
        # Views come from the same unlocked _pending_tracks read the
        # bottom-up pairing below already does (latest-wins, no lock: a
        # neighbour's frame worker may be mid-write, and the next round sees
        # whatever lands).
        round_nodes: dict[str, list[dict]] = {}
        if node_id in self.node_geometries:
            round_nodes[node_id] = tracks
        for nid in neighbor_ids:
            if nid in self.node_geometries:
                round_nodes[nid] = self._pending_tracks.get(nid) or []

        candidates: list[dict] = []
        for g in globals_:
            g_key = g.get("key")
            g_ts_s = g.get("timestamp_ms", 0) / 1000.0
            for nid, views in round_nodes.items():
                geo = self.node_geometries[nid]
                for view in views:
                    hist = view.get("history")
                    if not hist:
                        continue
                    if exclude_ids_by_node and str(view.get("track_id")) in (exclude_ids_by_node.get(nid) or ()):
                        continue
                    last = hist[-1]
                    dt = float(last["t_s"]) - g_ts_s
                    if not (0.0 <= dt <= self.claim_max_dr_age_s):
                        continue
                    dr_lat, dr_lon = offset_latlon_m(
                        g["lat"],
                        g["lon"],
                        east_m=g.get("vel_east", 0.0) * dt,
                        north_m=g.get("vel_north", 0.0) * dt,
                    )
                    pred_delay, pred_doppler = predict_observation(
                        geo,
                        dr_lat,
                        dr_lon,
                        g.get("alt_m", 0.0) / 1000.0,
                        g.get("vel_east", 0.0),
                        g.get("vel_north", 0.0),
                        g.get("vel_up", 0.0),
                    )
                    d_res = abs(pred_delay - float(last["delay_us"]))
                    f_res = abs(pred_doppler - float(last["doppler_hz"]))
                    if d_res > self.claim_delay_gate_us or f_res > self.claim_doppler_gate_hz:
                        continue
                    candidates.append(
                        {
                            "anchor_key": g_key,
                            "node_id": nid,
                            "track_id": str(view.get("track_id")),
                            "d_res_us": d_res,
                            "f_res_hz": f_res,
                            "dr_age_s": dt,
                            "score": (d_res / self.claim_delay_gate_us + f_res / self.claim_doppler_gate_hz),
                            "_g": g,
                            "_hist": hist,
                            "_last": last,
                        }
                    )

        # One-to-one selection, greedy ascending score — same shape as
        # _pair_tracks stage 2.  Two exclusivity sets rather than one: a
        # tracklet may be claimed by at most one global (claimed_ids_by_node),
        # AND a (global, node) slot may be filled by at most one tracklet
        # (claimed_global_node) — otherwise an ambiguous node with two
        # tracklets both gating against the same global could hand it two
        # "matched nodes" that are really the same node.
        candidates.sort(key=lambda c: c["score"])
        claimed_ids_by_node: dict[str, set] = defaultdict(set)
        claimed_global_node: set[tuple] = set()
        matches_by_global: dict[str, list[dict]] = defaultdict(list)
        claim_records: list[dict] = []

        for c in candidates:
            slot = (c["anchor_key"], c["node_id"])
            won = c["track_id"] not in claimed_ids_by_node[c["node_id"]] and slot not in claimed_global_node
            if won:
                claimed_ids_by_node[c["node_id"]].add(c["track_id"])
                claimed_global_node.add(slot)
                matches_by_global[c["anchor_key"]].append(c)
                self.claims_matched += 1
            else:
                self.claim_conflicts += 1
            claim_records.append(
                {
                    "anchor_key": c["anchor_key"],
                    "node_id": c["node_id"],
                    "track_id": c["track_id"],
                    "d_res_us": c["d_res_us"],
                    "f_res_hz": c["f_res_hz"],
                    "dr_age_s": c["dr_age_s"],
                    "won": won,
                }
            )

        # Anchored input per global with >= 2 matched nodes AND the
        # triggering node among them.  The trigger-must-match rule dedups
        # emission to ~one anchored input per matched node per round: every
        # matched node eventually triggers its own round and emits once,
        # rather than every node re-emitting the same anchor every round.
        anchored_inputs: list[dict] = []
        for g_key, matches in matches_by_global.items():
            matched_node_ids = {m["node_id"] for m in matches}
            if len(matched_node_ids) < 2 or node_id not in matched_node_ids:
                continue
            g = matches[0]["_g"]
            g_ts_s = g.get("timestamp_ms", 0) / 1000.0
            newest = max(matches, key=lambda m: float(m["_last"]["t_s"]))
            dt_newest = float(newest["_last"]["t_s"]) - g_ts_s
            guess_lat, guess_lon = offset_latlon_m(
                g["lat"],
                g["lon"],
                east_m=g.get("vel_east", 0.0) * dt_newest,
                north_m=g.get("vel_north", 0.0) * dt_newest,
            )
            cv_epochs = _merge_epochs_multi([(m["node_id"], m["_hist"]) for m in matches])
            by_snr = sorted(
                matches,
                key=lambda m: float(m["_last"].get("snr", 0.0)),
                reverse=True,
            )
            track_ids_by_node: dict[str, list] = defaultdict(list)
            for m in matches:
                track_ids_by_node[m["node_id"]].append(m["track_id"])

            anchored_inputs.append(
                {
                    "initial_guess": {
                        "lat": guess_lat,
                        "lon": guess_lon,
                        "alt_km": g.get("alt_m", 0.0) / 1000.0,
                    },
                    "initial_velocity": {
                        "vel_east_ms": g.get("vel_east", 0.0),
                        "vel_north_ms": g.get("vel_north", 0.0),
                    },
                    "measurements": [
                        {
                            "node_id": m["node_id"],
                            "delay_us": float(m["_last"]["delay_us"]),
                            "doppler_hz": float(m["_last"]["doppler_hz"]),
                            "snr": float(m["_last"].get("snr", 0.0)),
                        }
                        for m in matches
                    ],
                    "n_nodes": len(matched_node_ids),
                    "timestamp_ms": timestamp_ms,
                    "adsb_hex": None,
                    "chi2_per_dof": None,
                    "n_epochs": len(cv_epochs),
                    # REQUIRED: without cv_epochs an anchored n=2 input dies
                    # permanently at the solver's confirmation gate
                    # (_resolve_cv_fit -> None -> n2_unconfirmed), because it has
                    # no epochs to fit and chi2_per_dof is already None here.
                    "cv_epochs": cv_epochs,
                    # Participates in _claim_track_pair arbitration exactly like
                    # a bottom-up pairing's track_pair_ids — the two best-SNR
                    # matched nodes, truncated the same way
                    # format_track_pairs_for_solver truncates a cluster's.
                    "track_pair_ids": (
                        [tuple(sorted([by_snr[0]["track_id"], by_snr[1]["track_id"]]))] if len(by_snr) >= 2 else []
                    ),
                    "track_ids": sorted({m["track_id"] for m in matches}),
                    "track_ids_by_node": {nid: sorted(ids) for nid, ids in track_ids_by_node.items()},
                    "anchor_key": g_key,
                }
            )
        self.anchored_inputs_emitted += len(anchored_inputs)
        return dict(claimed_ids_by_node), anchored_inputs, claim_records

    def submit_tracks_round(self, node_id: str, tracks: list[dict], timestamp_ms: int) -> AssociationRound:
        """Associate this node's confirmed single-node tracks with its neighbours'.

        The detection-level path (now in detection_association.py) pairs one echo
        with one echo, and at n=2
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
            AssociationRound: pairs is a TrackPairCandidate for each pairing
            that passed both the coarse delay-grid gate and, where there was
            enough history, the constant-velocity fit.  anchored_inputs and
            claims are the top-down claiming stage's output — see
            _claim_round and submit_tracks_round's claim-mode handling below.
        """
        self._pending_tracks[node_id] = tracks or []
        if not tracks:
            return AssociationRound()

        neighbors = self._neighbors.get(node_id)
        if not neighbors:
            return AssociationRound()

        # Same rate limit submit_frame uses, and it matters more here: the
        # coarse grid gate is cheap but every pairing that survives it costs an
        # LM fit, and a hardware node sends at 22 fps.  Storing the tracks above
        # is unconditional, so a neighbour triggering its own round still sees
        # this node's latest history.
        now = __import__("time").monotonic()
        if now - self._last_assoc.get(node_id, 0.0) < self._ASSOC_MIN_INTERVAL_S:
            return AssociationRound()
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

        # ADS-B seeding rides this same round, and runs BEFORE claiming: its
        # tagged set is what keeps a dark global from claiming an already
        # verified-lit tracklet below.  Always computed and counted when a
        # mode is configured and a provider is set, same shadow discipline
        # as claiming — inertness is enforced AFTER counting.
        adsb_tagged_by_node: dict[str, set] = {}
        adsb_inputs: list[dict] = []
        if self.adsb_seed_mode != "off" and self.adsb_provider is not None:
            self.adsb_seed_rounds += 1
            adsb_tagged_by_node, adsb_inputs = self._adsb_seed_round(node_id, tracks, visit, timestamp_ms)
            _logger.debug(
                "adsb seed round node=%s mode=%s tagged=%d inputs=%d gate_rejects=%d",
                node_id,
                self.adsb_seed_mode,
                self.adsb_tracklets_tagged,
                len(adsb_inputs),
                self.adsb_seed_gate_rejects,
            )
            if self.adsb_seed_mode != "active":
                # Shadow: computed and counted above, but must not influence
                # anything downstream — bottom-up exclusion, claiming's
                # exclude_ids_by_node, and the returned inputs all see {}/[].
                adsb_tagged_by_node, adsb_inputs = {}, []

        # Top-down claiming rides this same round.  Always computed and
        # counted when a mode is configured and a provider is set — shadow's
        # inertness is enforced AFTER counting, not by skipping the work,
        # because the counters (and the DEBUG line below) are the signal
        # shadow exists to produce.
        claimed_by_node: dict[str, set] = {}
        anchored: list[dict] = []
        claim_records: list[dict] = []
        if self.claim_mode != "off" and self.global_track_provider is not None:
            self.claim_rounds += 1
            claimed_by_node, anchored, claim_records = self._claim_round(
                node_id, tracks, visit, timestamp_ms, exclude_ids_by_node=adsb_tagged_by_node
            )
            if claim_records:
                won = [c for c in claim_records if c["won"]]
                d_med = f_med = None
                if won:
                    d_sorted = sorted(c["d_res_us"] for c in won)
                    f_sorted = sorted(c["f_res_hz"] for c in won)
                    d_med, f_med = d_sorted[len(d_sorted) // 2], f_sorted[len(f_sorted) // 2]
                _logger.debug(
                    "claim round node=%s mode=%s matched=%d conflicts=%d "
                    "anchored_inputs=%d d_res_med_us=%s f_res_med_hz=%s",
                    node_id,
                    self.claim_mode,
                    len(won),
                    len(claim_records) - len(won),
                    len(anchored),
                    d_med,
                    f_med,
                )
            if self.claim_mode != "active":
                # Shadow: claims computed and counted above, but must not
                # influence anything downstream — pairs identical to off,
                # no anchored inputs returned.
                claimed_by_node, anchored = {}, []

        out: list[TrackPairCandidate] = []
        # Two budgets, because they bound different costs.  fits bounds inline
        # LM solves; pairs bounds candidates handed to the solver worker when
        # the fit is deferred.  Merging them was the bug: the fit budget was
        # decremented only inside the fit branch, so on the deferred path it
        # never depleted and its cap silently applied per node pair instead.
        budget = {"fits": self._MAX_FITS_PER_ROUND, "pairs": self._MAX_PAIRS_PER_ROUND}
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
            if claimed_by_node:
                # Active-mode exclusivity, filtered round-locally.  This view
                # (tracks_a/tracks_b) is a local copy from _pending_tracks —
                # never the stored list itself — so a wrong claim's blast
                # radius is bounded to this one round: the next round
                # resubmits the un-mutated tracker views.
                tracks_a = self._exclude_claimed(pair_key[0], tracks_a, claimed_by_node)
                tracks_b = self._exclude_claimed(pair_key[1], tracks_b, claimed_by_node)
            if adsb_tagged_by_node:
                # Same blast-radius argument as the claimed-tracklet filter
                # above: round-local list copies only, _pending_tracks is
                # never mutated.
                tracks_a = self._exclude_tagged(pair_key[0], tracks_a, adsb_tagged_by_node)
                tracks_b = self._exclude_tagged(pair_key[1], tracks_b, adsb_tagged_by_node)
            emitted = self._pair_tracks(zone, tracks_a, tracks_b, timestamp_ms, budget)
            out.extend(emitted)
            budget["pairs"] -= len(emitted)
            if budget["fits"] <= 0 or budget["pairs"] <= 0:
                # Round exhausted.  Remaining neighbours are picked up next
                # round from the cursor above, so nothing is lost, only
                # deferred — which is only true because the cursor rotates.
                # This is the one place work is actually deferred, so it is
                # the one place the counter increments: once per such round.
                self._neighbor_cursor[node_id] = (start + visit.index(other_id) + 1) % n_total
                self.track_pairs_deferred += 1
                break
        return AssociationRound(pairs=out, anchored_inputs=anchored, claims=claim_records, adsb_inputs=adsb_inputs)

    def _exclude_claimed(self, node_for: str, trk_list: list[dict], claimed_by_node: dict[str, set]) -> list[dict]:
        """Drop this round's claimed tracklets from one side of a pairing.

        Round-local only — trk_list is the view submit_tracks_round already
        holds locally (tracks for the triggering node, or a
        _pending_tracks.get() snapshot for a neighbour), never mutated.
        """
        claimed = claimed_by_node.get(node_for)
        if not claimed:
            return trk_list
        filtered = [t for t in trk_list if str(t.get("track_id")) not in claimed]
        self.tracklets_excluded += len(trk_list) - len(filtered)
        return filtered

    def _exclude_tagged(self, node_for: str, trk_list: list[dict], tagged_by_node: dict[str, set]) -> list[dict]:
        """Drop this round's ADS-B-verified tracklets from one side of a
        pairing.  Same body/blast-radius as _exclude_claimed — see there."""
        tagged = tagged_by_node.get(node_for)
        if not tagged:
            return trk_list
        filtered = [t for t in trk_list if str(t.get("track_id")) not in tagged]
        self.adsb_tracklets_excluded += len(trk_list) - len(filtered)
        return filtered

    def submit_tracks(self, node_id: str, tracks: list[dict], timestamp_ms: int) -> list[TrackPairCandidate]:
        """Backward-compatible wrapper over submit_tracks_round.

        Returns just the bottom-up pairs, the shape every existing caller
        (production's frame_processor pre-migration, lib tests, the bench's
        baseline paths) expects.  Callers that need anchored_inputs/claims
        use submit_tracks_round directly.
        """
        return self.submit_tracks_round(node_id, tracks, timestamp_ms).pairs

    def _pair_tracks(
        self,
        zone: OverlapZone,
        tracks_a: list[dict],
        tracks_b: list[dict],
        timestamp_ms: int,
        budget: dict | None = None,
    ) -> list[TrackPairCandidate]:
        # ── Stage 1: enumerate and score every hypothesis ────────────────────
        # Each surviving (track_a, track_b) is a hypothesis about the world:
        # "these two echo histories are one aircraft."  They are scored here and
        # judged against each other in stage 2 — not one at a time, because the
        # hypotheses are not independent: one track is one aircraft, so two
        # pairings sharing a track are mutually exclusive.
        fitted: list[TrackPairCandidate] = []
        held: list[TrackPairCandidate] = []  # too little span to score yet

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
        limit = (
            _b.get("fits", self._MAX_FITS_PER_ROUND)
            if self.cv_fit is not None
            else _b.get("pairs", self._MAX_PAIRS_PER_ROUND)
        )
        ordered = sorted(
            matches.items(),
            key=lambda kv: (
                abs(zone._np_pred_a[kv[1]] - float(usable_a[kv[0][0]]["history"][-1]["delay_us"]))
                + abs(zone._np_pred_b[kv[1]] - float(usable_b[kv[0][1]]["history"][-1]["delay_us"]))
            ),
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
                    float(last_a["doppler_hz"]) * C_KM_S * 1000.0 / zone.fc_a_hz,
                    b_a,
                    float(last_b["doppler_hz"]) * C_KM_S * 1000.0 / zone.fc_b_hz,
                    b_b,
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
            if (
                self.cv_fit is not None
                and len(epochs) >= self.cv_min_epochs
                and min(span_a, span_b) >= self.cv_min_span_s
            ):
                fit = self.cv_fit(
                    {
                        "initial_guess": {"lat": g_lat, "lon": g_lon, "alt_km": g_alt},
                        "initial_velocity": vel_seed,
                        "epochs": epochs,
                        "timestamp_ms": timestamp_ms,
                    },
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
                    if (
                        self.cv_fit is None
                        and len(epochs) >= self.cv_min_epochs
                        and min(span_a, span_b) >= self.cv_min_span_s
                    )
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
                lat=lat,
                lon=lon,
                alt_km=alt_km,
                vel_east_ms=vel_e,
                vel_north_ms=vel_n,
                chi2_per_dof=chi2_per_dof,
                dof=dof,
                n_epochs=len(epochs),
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
            if self.cv_exclusive and (c.track_a_id in claimed_a or c.track_b_id in claimed_b):
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
            if self.cv_exclusive and (c.track_a_id in claimed_a or c.track_b_id in claimed_b):
                self.track_pairs_superseded += 1
                continue
            results.append(c)
        return results

    def format_track_pairs_for_solver(self, pairs: list[TrackPairCandidate]) -> list[dict]:
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
        dist_sq = ((lats[:, None] - lats) * km_per_lat) ** 2 + ((lons[:, None] - lons) * km_per_lon) ** 2
        rows, cols = np.where((dist_sq < _MERGE_DIST_KM**2) & (np.arange(n)[:, None] < np.arange(n)))
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
                        by_node[nid] = {"node_id": nid, "delay_us": d, "doppler_hz": f, "snr": s}

            # Per-node track id sets, so a downstream trim (dropping one
            # contaminated node's measurement and re-solving) can also drop
            # exactly that node's source tracks from the published solve's
            # provenance instead of carrying the whole cluster's track_ids.
            track_ids_by_node: dict[str, set] = defaultdict(set)
            for p in group:
                track_ids_by_node[p.node_a_id].add(p.track_a_id)
                track_ids_by_node[p.node_b_id].add(p.track_b_id)

            fitted = [p for p in group if p.chi2_per_dof is not None]
            # Worst fit in the cluster, not the best: a cluster is published as
            # one target, so a pairing that failed to justify itself should not
            # be laundered by a well-fitted neighbour sharing its position.
            worst_chi2 = max((p.chi2_per_dof for p in fitted), default=None)

            solver_inputs.append(
                {
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
                    "cv_epochs": max((p.epochs for p in group if p.epochs), key=len, default=None),
                    "track_pair_ids": sorted({(p.track_a_id, p.track_b_id) for p in group})[:1],
                    "track_ids": sorted({p.track_a_id for p in group} | {p.track_b_id for p in group}),
                    "track_ids_by_node": {nid: sorted(ids) for nid, ids in track_ids_by_node.items()},
                }
            )
        return solver_inputs

    def get_overlap_summary(self) -> list[dict]:
        """Return summary of all overlap zones."""
        summaries = []
        for (a_id, b_id), zone in list(self.overlap_zones.items()):
            summaries.append(
                {
                    "node_a": a_id,
                    "node_b": b_id,
                    "grid_points": len(zone.grid_points),
                    "delay_gate_us": zone.delay_gate_us,
                    "doppler_gate_hz": zone.doppler_gate_hz,
                    "has_overlap": len(zone.grid_points) > 0,
                }
            )
        return summaries
