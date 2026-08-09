"""Empirical detection-area characterisation built from known-position calibration points.

Instead of assuming a fixed Yagi-like antenna lobe, this module accumulates
ground-truth target positions (from ADS-B or multinode-solver solutions) that a
node has positively detected, then derives a smoothed coverage polygon that
reflects the node's *actual* detection area as observed over time.

Algorithm
---------
1. Each confirmed detection is projected from the RX site into (bearing, range)
   polar coordinates and accumulated in one of N_BINS angular bins (5°/bin).
2. Per bin, the robust range estimate is the 85th-percentile of observed ranges
   (enough samples sit below a farther outlier so we use P85, not max).
3. Bins with no observations are filled by angular-linear interpolation between
   the nearest filled neighbours on each side, with a conservative discount
   (30 %) applied for estimated coverage that we haven't actually seen yet.
4. A circular rolling average (window = 3 bins) smooths the resulting vector.
5. Polygon vertices are computed at each bin centre and returned as [[lat, lon]].

The polygon is only returned once at least MIN_POINTS calibration points have
been recorded; below that the frontend falls back to the theoretical Yagi sector.

Learned FOV (FOV_MODE, schema 3)
---------------------------------
The methods above (observed_limit_km / constraint_digest / to_polygon without
use_learned_wedge) are the shrink-only prior: it may only tighten the
theoretical wedge, never widen it, and abstains below _MIN_BIN_POINTS_TO_CONSTRAIN.
That is deliberately kept — it is the off/shadow-mode API and every off-mode
caller still reads it unchanged.

_bin_open / limit_km / contains / wedge_state / fov_digest are a second,
independent read surface over the same accumulated bins, for FOV_MODE
shadow/active callers only.  The semantics differ on purpose: the theoretical
wedge is a PRIOR, not a hard boundary — a bin outside it opens on
FOV_OPEN_MIN_POINTS positive detections (broaden fast), and the resulting
limit only ever shrinks on sustained *negative* evidence (a tracked target
that disappeared while still predicted detectable), never on mere absence of
traffic.  See record_disappearance / _neg_cap_km.
"""

import json
import math
import os
import time

from retina_analytics.constants import (
    YAGI_BEAM_WIDTH_DEG,
    YAGI_MAX_RANGE_KM,
    bearing_deg,
    bistatic_range_limit_km,
    haversine_km,
    offset_latlon,
)

N_BINS = 72          # 5 ° per bin  (360 / 5 = 72)
_DEG_PER_BIN = 360.0 / N_BINS
_MAX_PER_BIN = 200   # cap per-bin history to prevent unbounded RAM growth
MIN_POINTS = 20      # minimum calibration points before emitting a polygon

# Calibration points a bin needs before its P85 is allowed to *constrain*
# association rather than merely be drawn.  Below this the bin is one or two
# aircraft passing through, which says where traffic flew, not where the node
# can see — and since the constraint only ever tightens, a premature one is a
# blind spot rather than a wrong shape.
_MIN_BIN_POINTS_TO_CONSTRAIN = 10

# Headroom on the observed P85 before it is treated as a limit.  ADS-B traffic
# does not fly to the edge of a footprint, so the furthest *observed* return is
# a lower bound on reach, never an upper one.
OBSERVED_LIMIT_MARGIN = 1.25

# What the accumulated bins *mean*.  Bump this whenever the calibration input
# changes in a way that makes previously-stored points incomparable with new
# ones — the numbers stay well-formed, so nothing else would notice.
#
#   1: positions came from the multinode solver, including unverified n=2
#      solves.  Blind, 55-85% of those are ghosts a median 20+ km from any
#      aircraft, so a v1 polygon describes where the solver *thought* targets
#      were, not where the node can see.
#   2: reported ADS-B positions only.
#   3: adds the learned-FOV fields (prior_azimuth_deg/prior_width_deg,
#      per-bin last-positive timestamps, negative-event history).  A v2 file
#      has none of these, so under FOV_MODE every bin would silently start
#      "closed" outside its theoretical wedge regardless of how much v2
#      evidence it actually carried — auto-discarding it at register is a
#      clean migration with no operator action, the same reasoning v1->v2
#      used.
#   4: positives are detection-backed, not track-backed.  v3 recorders
#      stamped positives for tracks coasting on ADS-B enrichment (up to 15 s
#      past the last real detection, painting every departure path) and for
#      every node contributing to a published solve (so a ghost publish
#      tagged with a real hex opened bins at another aircraft's position —
#      under an active FOV gate, a feedback loop; staging 2026-08-09 opened
#      out-of-wedge bins on 15/29 synthetic nodes in ~25 min).  A v3 polygon
#      is therefore shaped partly by places the node never detected anything
#      and cannot be repaired in place.
#   5: positives additionally require the newest associated detection to be
#      ADS-B-tagged with the track's own hex.  v4 trusted track identity: a
#      track that swaps onto an untagged (dark) target keeps its old hex —
#      the tracker's swap debounce only advances on tagged mismatches — and
#      recorded the departed aircraft's live position for as long as it kept
#      associating, painting contiguous out-of-wedge fans (staging
#      2026-08-09, post-v4).
#
# Production and staging mount coverage_data as a named volume that survives
# rebuilds, so without this a stale-schema polygon would be served indefinitely
# with no operator action to prompt it.
CALIBRATION_SCHEMA = 5

# ── Learned FOV (FOV_MODE shadow/active) ────────────────────────────────────
#
# Positives (bin ± 1) needed to OPEN a bin that sits outside the theoretical
# wedge.  Small on purpose — the whole point is to broaden fast off real
# detections instead of waiting on a shrink-only prior.  1 would let a single
# mis-matched ADS-B hex (wrong-aircraft correlation) open a bin permanently;
# 3 independent detections at the same bearing is a real pattern.
FOV_OPEN_MIN_POINTS = 3

# Percentile used for the learned-FOV range extension.  P85 (the shrink-only
# prior's percentile) exists to be robust to a handful of outliers when the
# only thing at stake is how far a *tightening* reaches; here the far
# detections are exactly the evidence a widening claim rests on, and P85
# would discard the top 15% of it.  Extension additionally requires
# _MIN_BIN_POINTS_TO_CONSTRAIN (10) points, same floor as the shrink-only
# prior, so a thin bin cannot swing the reach off one lucky detection.
FOV_EXTEND_PCTL = 95

# Negative-evidence shrink.  A single disappearance is not evidence — a target
# coasts, a frame drops, ADS-B glitches — so shrinking needs a *pattern*.
FOV_NEG_EVENTS_TO_SHRINK = 3
FOV_NEG_MAX_PER_BIN = 20   # FIFO cap, mirrors _MAX_PER_BIN's role for positives
# Events must span at least this long. One bad minute of interference (a
# co-channel transmitter, a maintenance window) cannot close a bin; sustained
# absence over minutes is what distinguishes an obstruction from noise.
FOV_NEG_MIN_SPAN_S = 600.0
# The shrink lands short of the nearest disappearance range, not AT it — the
# target was predicted detectable there and wasn't, so the true edge is at or
# inside that range.
FOV_NEG_CAP_MARGIN = 0.95
# A "closed" bin's limit is 0.0 km, not "nothing near the RX" — add_point
# itself never accumulates evidence inside 0.5 km (too close to be
# informative), so without this floor the node's own immediate vicinity would
# read as permanently unreachable.  contains() floors every bin at this
# radius regardless of state.
FOV_CLOSED_EPSILON_KM = 1.0

# Admit-clamp multiplier for nodes WITHOUT a declared bistatic limit.  Their
# max_range_km is a config guess (radar3's is the 140 km default), not a
# physical bound the way a declared bistatic ellipse is — range_clamp_mult
# (2.0) stays for bistatic-declared nodes because that footprint is physical
# and a detection past 2x it really is implausible.  A monostatic guess
# deserves more room before a real detection is thrown away as "too far".
FOV_CLAMP_MULT_MONOSTATIC = 4.0


def _bin_for_bearing(bearing_deg: float) -> int:
    return int(bearing_deg / _DEG_PER_BIN) % N_BINS


def _bearing_and_range(rx_lat: float, rx_lon: float,
                       lat: float, lon: float) -> tuple[float, float]:
    """Return (bearing °, range_km) from RX to target.

    Spherical, matching the rest of the system.  This was flat-earth while the
    association gate read the resulting bins with the spherical bearing, so a
    point could be *filed* under one bin and *looked up* under its neighbour.
    The mismatch is small — 0.27° at this latitude, about 2% of points crossing
    a 5° boundary — but it was real, it is the cause named in observed_limit_km's
    docstring, and it also made the clamp in add_point mix models: the bearing
    came from here flat, and _reach_at derived the TX bearing spherically.

    Persisted state is not re-binned, because it cannot be: to_dict stores
    ranges per bin, not the points they came from.  Nothing needs re-binning
    though — old points already carry this mismatch relative to the query, new
    ones do not, and bins age out at _MAX_PER_BIN.  The state is therefore
    monotonically self-healing and CALIBRATION_SCHEMA is deliberately NOT
    bumped: a bump would discard every node's polygon (days of cooperative
    traffic each, since a bin needs _MIN_BIN_POINTS_TO_CONSTRAIN to say
    anything) to correct an error that then averages through a P85, a 3-bin
    rolling mean and a 1.25x margin.
    """
    return (bearing_deg(rx_lat, rx_lon, lat, lon),
            haversine_km(rx_lat, rx_lon, lat, lon))


def _percentile(values: list[float], pct: float) -> float:
    """pct-th percentile of a non-empty list (sorted() — the input is untouched)."""
    s = sorted(values)
    idx = min(int(len(s) * pct / 100.0), len(s) - 1)
    return s[idx]


def _p85(values: list[float]) -> float:
    """85th-percentile — the shrink-only prior's percentile.  See _percentile."""
    return _percentile(values, 85)


class EmpiricalCoverageState:
    """Accumulates calibration points and derives a smoothed detection polygon."""

    def __init__(self, rx_lat: float, rx_lon: float,
                 max_range_km: float | None = None,
                 range_clamp_mult: float = 2.0,
                 tx_lat: float | None = None, tx_lon: float | None = None,
                 prior_azimuth_deg: float | None = None,
                 prior_width_deg: float | None = None):
        self.rx_lat = rx_lat
        self.rx_lon = rx_lon
        # Transmitter position, so the clamp can follow the ellipse the node is
        # actually bounded by rather than a circle on the receiver.  The two
        # differ by 2x in radius directly away from the transmitter, which is
        # precisely where a circle would let a mis-attributed detection through.
        self.tx_lat = tx_lat
        self.tx_lon = tx_lon
        # Detections beyond the admit clamp (see add_point) are rejected as
        # mis-attributed. None (e.g. states loaded from disk) falls back to
        # YAGI_MAX_RANGE_KM at use-time rather than disabling the bound.
        self.max_range_km = max_range_km
        # The bistatic limit this polygon was accumulated under, if any.  Not
        # used for clamping — recorded so NodeAnalyticsManager.register_node can
        # tell whether a persisted polygon was built under the same range rule
        # and discard it when the rule changes (the footprint goes from a circle
        # on the RX to an ellipse with foci at RX and TX).
        self.max_bistatic_range_km: float | None = None
        # Which calibration input these bins were accumulated from; see
        # CALIBRATION_SCHEMA.  A freshly-constructed state is current by
        # definition — from_dict is where an old one declares itself.
        self.schema = CALIBRATION_SCHEMA
        self.range_clamp_mult = range_clamp_mult
        # Theoretical prior for the learned FOV: node_beam_params semantics,
        # resolved once by the caller (NodeAnalyticsManager._register_node_locked
        # mirrors geo.node_beam_params — explicit aim -> resolved; elif TX known
        # -> broadside+90; else None=omni) and updated in place on every
        # reconnect, never invalidating accumulated bins.  None azimuth means
        # every bin starts inside the theoretical wedge — an omni node has no
        # direction to be wrong about.  Unused by the shrink-only prior API
        # (observed_limit_km / constraint_digest / to_polygon without
        # use_learned_wedge); only _bin_open/limit_km/wedge_state read it.
        self.prior_azimuth_deg = prior_azimuth_deg
        self.prior_width_deg = prior_width_deg
        # Per-bin list of observed ranges (km).  List, not array — no numpy dep.
        self._bins: list[list[float]] = [[] for _ in range(N_BINS)]
        # Per-bin wall-clock time of the last accepted positive (add_point).
        # Negative events older than this are superseded — see _neg_cap_km.
        self._bin_last_pos_ts: list[float] = [0.0] * N_BINS
        # Per-bin FIFO of (range_km, ts) disappearance events — see
        # record_disappearance / _neg_cap_km.
        self._neg_events: list[list[tuple[float, float]]] = [[] for _ in range(N_BINS)]
        # max_limit_km() cache — O(N_BINS), read by both the pair prefilter and
        # the grid bounding box at registration/rebuild time.  Invalidated by
        # add_point and record_disappearance, the only two things that can
        # move it.
        self._max_limit_cache: float | None = None

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def _reach_at(self, bearing_deg_: float) -> float:
        """How far the node can see on this bearing, before any margin.

        The bistatic ellipse when the node declares a differential limit and its
        transmitter is known; otherwise the legacy circle.
        """
        if (self.max_bistatic_range_km is not None
                and self.tx_lat is not None and self.tx_lon is not None):
            baseline = haversine_km(self.rx_lat, self.rx_lon,
                                    self.tx_lat, self.tx_lon)
            to_tx = bearing_deg(self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon)
            psi = abs((bearing_deg_ - to_tx + 180.0) % 360.0 - 180.0)
            return bistatic_range_limit_km(psi, baseline, self.max_bistatic_range_km)
        return self.max_range_km if self.max_range_km is not None else YAGI_MAX_RANGE_KM

    def _admit_mult(self) -> float:
        """Admit-clamp multiplier on _reach_at for add_point / record_disappearance.

        range_clamp_mult (2.0) for a node with a declared bistatic limit — that
        footprint is physical, so 2x it really is implausible.  A node with no
        declared limit is scored against a config guess (max_range_km), which
        deserves more slack before a real detection is thrown away — see
        FOV_CLAMP_MULT_MONOSTATIC.
        """
        return self.range_clamp_mult if self.max_bistatic_range_km else FOV_CLAMP_MULT_MONOSTATIC

    def add_point(self, lat: float, lon: float, ts: float | None = None) -> None:
        """Record one calibration point (known target position).

        ts defaults to wall-clock time; the learned-FOV shrink logic
        (_neg_cap_km) needs it to tell whether a positive supersedes an
        earlier disappearance, so callers replaying history may supply it.
        """
        bearing, range_km = _bearing_and_range(self.rx_lat, self.rx_lon, lat, lon)
        if range_km < 0.5:
            return  # too close — not informative
        if range_km > self._reach_at(bearing) * self._admit_mult():
            return  # implausibly far — mis-attributed detection
        i = _bin_for_bearing(bearing)
        b = self._bins[i]
        b.append(range_km)
        if len(b) > _MAX_PER_BIN:
            del b[0]  # drop oldest
        self._bin_last_pos_ts[i] = ts if ts is not None else time.time()
        self._max_limit_cache = None

    def record_disappearance(self, lat: float, lon: float, ts: float | None = None) -> bool:
        """Record one negative-evidence event: a target predicted detectable at
        (lat, lon) that this node failed to detect.

        Same admission rule as add_point (too-close / implausibly-far are
        rejected the same way) — a disappearance too close to be informative,
        or too far to plausibly be this node's target, says nothing about this
        node's FOV.  Returns whether the event was actually recorded, so the
        backend disappearance detector's per-cycle cap counts only what
        landed.  This alone never shrinks anything: see _neg_cap_km for the
        (count, span, supersession) rule that turns accumulated events into a
        cap.
        """
        bearing, range_km = _bearing_and_range(self.rx_lat, self.rx_lon, lat, lon)
        if range_km < 0.5:
            return False
        if range_km > self._reach_at(bearing) * self._admit_mult():
            return False
        i = _bin_for_bearing(bearing)
        events = self._neg_events[i]
        events.append((range_km, ts if ts is not None else time.time()))
        if len(events) > FOV_NEG_MAX_PER_BIN:
            del events[0]  # drop oldest
        self._max_limit_cache = None
        return True

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_points(self) -> int:
        return sum(len(b) for b in self._bins)

    @property
    def n_filled_bins(self) -> int:
        return sum(1 for b in self._bins if b)

    # ── Learned FOV (FOV_MODE shadow/active) ─────────────────────────────────

    def _in_theoretical_wedge(self, i: int) -> bool:
        """Is bin i inside the theoretical prior (or is the prior omni)?"""
        if self.prior_azimuth_deg is None:
            return True
        centre = i * _DEG_PER_BIN
        diff = abs((centre - self.prior_azimuth_deg + 180.0) % 360.0 - 180.0)
        half = (self.prior_width_deg if self.prior_width_deg is not None
                else YAGI_BEAM_WIDTH_DEG) / 2.0
        return diff <= half

    def _bin_open(self, i: int) -> bool:
        """Is bin i part of the node's admitted learned FOV?

        True inside the theoretical wedge (or under an omni prior — see
        _in_theoretical_wedge), OR when positives at bin i and its two
        neighbours reach FOV_OPEN_MIN_POINTS.  The neighbour widening mirrors
        observed_limit_km's — bins are 5° wide, so a detection just inside a
        boundary is filed one bin over from where a query at the boundary
        reads it.
        """
        if self._in_theoretical_wedge(i):
            return True
        pos = sum(len(self._bins[(i + off) % N_BINS]) for off in (-1, 0, 1))
        return pos >= FOV_OPEN_MIN_POINTS

    def _neg_cap_km(self, i: int) -> float | None:
        """Negative-evidence range cap for bin i, or None if none applies.

        Only events strictly newer than the bin's last accepted positive
        count — a fresh positive supersedes every earlier disappearance at
        that bearing (broaden fast, shrink slow: a bin that reopens does not
        stay haunted by evidence that predates the reopening).  Shrinking
        needs >= FOV_NEG_EVENTS_TO_SHRINK events spanning >=
        FOV_NEG_MIN_SPAN_S seconds, so one bad minute of interference cannot
        close a bin — only a sustained pattern of a target going undetected
        where it should have been seen does.
        """
        last_pos_ts = self._bin_last_pos_ts[i]
        events = [(r, ts) for r, ts in self._neg_events[i] if ts > last_pos_ts]
        if len(events) < FOV_NEG_EVENTS_TO_SHRINK:
            return None
        span_s = max(ts for _, ts in events) - min(ts for _, ts in events)
        if span_s < FOV_NEG_MIN_SPAN_S:
            return None
        # The shrink lands short of the nearest disappearance range, not at
        # it: the target was predicted detectable there and was not seen, so
        # the true edge is at or inside that range.
        return min(r for r, _ in events) * FOV_NEG_CAP_MARGIN

    def limit_km(self, bearing_deg_: float) -> float:
        """How far the learned FOV reaches on this bearing (km).

        0.0 when the bin is not open (_bin_open) — closed, not merely
        unconstrained.  Otherwise the theoretical reach, widened past it once
        the bin itself (not neighbours — the extension is a claim about THIS
        bin's own detections) accumulates >= _MIN_BIN_POINTS_TO_CONSTRAIN
        positives, to P95 x OBSERVED_LIMIT_MARGIN if that exceeds the
        theoretical value.  An active negative cap (_neg_cap_km) narrows the
        result afterward — the ONLY thing that ever pulls this below the
        theoretical reach: absence never shrinks, only a tracked target that
        disappeared while still predicted detectable does.
        """
        i = _bin_for_bearing(bearing_deg_ % 360.0)
        if not self._bin_open(i):
            return 0.0
        # The bin CENTRE, not the left edge: the whole bin shares one limit,
        # and the centre is the representative bearing for the ellipse —
        # to_polygon samples the same way.
        bearing_i = (i + 0.5) * _DEG_PER_BIN
        reach = self._reach_at(bearing_i)
        b = self._bins[i]
        if len(b) >= _MIN_BIN_POINTS_TO_CONSTRAIN:
            reach = max(reach, _percentile(b, FOV_EXTEND_PCTL) * OBSERVED_LIMIT_MARGIN)
        cap = self._neg_cap_km(i)
        if cap is not None:
            reach = min(reach, cap)
        return reach

    def contains(self, bearing_deg_: float, range_km: float) -> bool:
        """Is (bearing, range) inside the node's learned FOV?

        Floored at FOV_CLOSED_EPSILON_KM regardless of bin state: add_point
        never accumulates evidence inside 0.5 km (too close to be
        informative), so without this floor a closed bin (limit 0.0) would
        make the node's own immediate vicinity permanently unreachable.
        """
        return range_km <= max(self.limit_km(bearing_deg_), FOV_CLOSED_EPSILON_KM)

    def wedge_state(self, bearing_deg_: float) -> str:
        """"closed" (not open — limit_km is 0.0), "observed" (open AND backed
        by >= _MIN_BIN_POINTS_TO_CONSTRAIN of this bin's own detections), or
        "prior" (open only via the theoretical wedge/omni, not yet backed by
        enough of its own evidence to extend past it)."""
        i = _bin_for_bearing(bearing_deg_ % 360.0)
        if not self._bin_open(i):
            return "closed"
        if len(self._bins[i]) >= _MIN_BIN_POINTS_TO_CONSTRAIN:
            return "observed"
        return "prior"

    def max_limit_km(self) -> float:
        """Largest learned-FOV reach across all bins.

        Used for the pair-overlap prefilter's distance test and the grid
        bounding box (see association.py) — both need a single worst-case
        radius before any per-bearing test can run.  Cached: it is
        O(N_BINS) and both call sites run at grid-build time; invalidated by
        add_point and record_disappearance, the only two things that can
        move it.  A stale cache would silently truncate grids to a radius
        the node has since outgrown.
        """
        if self._max_limit_cache is None:
            self._max_limit_cache = max(
                (self.limit_km(i * _DEG_PER_BIN) for i in range(N_BINS)),
                default=0.0,
            )
        return self._max_limit_cache

    def fov_digest(self) -> tuple:
        """Per-bin (state_code, rounded limit) — cheap change detection for
        FOV_MODE, the fov analogue of constraint_digest.

        state_code is compared exactly by the caller (_quantize_digest): a
        bin opening (state moving "closed" -> "prior"/"observed") must
        trigger a grid rebuild even when its rounded limit does not move in
        the same tick, because a closed bin excludes association entirely
        while an open one — however small its limit — does not.
        """
        out = []
        for i in range(N_BINS):
            bearing_i = i * _DEG_PER_BIN
            out.append((self.wedge_state(bearing_i), round(self.limit_km(bearing_i), 1)))
        return tuple(out)

    # ── Polygon generation ────────────────────────────────────────────────────

    def to_polygon(self, min_points: int = MIN_POINTS,
                   beam_azimuth_deg: float | None = None,
                   beam_width_deg: float | None = None,
                   max_range_km: float | None = None,
                   use_learned_wedge: bool = False) -> list[list[float]] | None:
        """Return a closed polygon [[lat, lon], …] or None if insufficient data.

        When *beam_azimuth_deg* and *beam_width_deg* are provided the polygon
        is constrained to the beam sector (a pie-slice shape starting and ending
        at the RX position).  Bins outside the sector are zeroed so the
        interpolation step never bleeds coverage into directions the antenna
        physically cannot observe.

        use_learned_wedge=True replaces both the in-beam test and the per-bin
        range with the learned FOV (_bin_open / limit_km) and ignores
        beam_azimuth_deg/beam_width_deg/max_range_km — the learned wedge
        already encodes which bins are admitted and how far each reaches, so a
        separate theoretical constraint would only reintroduce the shrink-only
        prior this mode replaces.
        """
        if self.n_points < min_points:
            return None

        # --- Determine which bins fall inside the beam sector -----------------
        if use_learned_wedge:
            def _in_beam(bin_idx: int) -> bool:
                return self._bin_open(bin_idx)
        elif beam_azimuth_deg is not None and beam_width_deg is not None:
            half = beam_width_deg / 2.0
            def _in_beam(bin_idx: int) -> bool:
                centre = bin_idx * _DEG_PER_BIN
                diff = (centre - beam_azimuth_deg + 180.0) % 360.0 - 180.0
                return abs(diff) <= half
        else:
            _in_beam = lambda _: True  # noqa: E731 — no constraint

        # Step 1: robust range per bin (P85, or 0 if empty / outside beam),
        # clamped so one mis-attributed far detection can't fling a vertex.
        # Clamp per bearing, not per node: the footprint is an ellipse when the
        # node is bounded by differential range, so a single radius would both
        # over-clamp toward the transmitter and under-clamp away from it.
        # An explicit max_range_km argument still overrides, for callers that
        # want the legacy circle.
        ranges: list[float] = []
        for i, b in enumerate(self._bins):
            if not _in_beam(i):
                ranges.append(0.0)
                continue
            if use_learned_wedge:
                # Unclipped from the theoretical sector: limit_km already
                # carries whatever extension/negative cap applies, and
                # re-clamping it against range_clamp_mult would silently
                # reintroduce the shrink-only prior this mode replaces.
                ranges.append(self.limit_km(i * _DEG_PER_BIN))
                continue
            bearing_i = i * _DEG_PER_BIN
            clamp = (max_range_km if max_range_km is not None
                     else self._reach_at(bearing_i))
            r = _p85(b) if b else 0.0
            ranges.append(min(r, clamp * self.range_clamp_mult))

        # Step 2: fill empty *in-beam* bins by angular interpolation from neighbours
        for i in range(N_BINS):
            if ranges[i] > 0.0 or not _in_beam(i):
                continue
            left_dist, left_val = None, None
            for j in range(1, N_BINS):
                ni = (i - j) % N_BINS
                if not _in_beam(ni):
                    break  # stop at beam edge
                lv = ranges[ni]
                if lv > 0.0:
                    left_dist, left_val = j, lv
                    break
            right_dist, right_val = None, None
            for j in range(1, N_BINS):
                ni = (i + j) % N_BINS
                if not _in_beam(ni):
                    break  # stop at beam edge
                rv = ranges[ni]
                if rv > 0.0:
                    right_dist, right_val = j, rv
                    break

            if left_val is None and right_val is None:
                continue
            elif left_val is None:
                est = right_val
                gap = right_dist
            elif right_val is None:
                est = left_val
                gap = left_dist
            else:
                total = left_dist + right_dist
                est = (left_val * right_dist + right_val * left_dist) / total
                gap = max(left_dist, right_dist)

            discount = max(0.70, 1.0 - 0.10 * gap)
            ranges[i] = est * discount

        # Step 3: rolling smooth (window = 3), only among in-beam bins
        smoothed = list(ranges)
        for i in range(N_BINS):
            if not _in_beam(i):
                continue
            prev_i = (i - 1) % N_BINS
            next_i = (i + 1) % N_BINS
            vals = [ranges[i]]
            if _in_beam(prev_i):
                vals.append(ranges[prev_i])
            if _in_beam(next_i):
                vals.append(ranges[next_i])
            smoothed[i] = sum(vals) / len(vals)

        # Step 4: generate sector polygon (pie-slice shape)
        polygon: list[list[float]] = []

        # Start at RX (sector tip)
        polygon.append([round(self.rx_lat, 5), round(self.rx_lon, 5)])

        # Emit vertices in angular order around the beam centre so the sector
        # boundary is monotonic. Bin-INDEX order self-intersects (bow-tie) when
        # the beam straddles north, because the in-beam bins wrap 71→0.
        in_beam_bins = [i for i in range(N_BINS) if _in_beam(i)]
        if beam_azimuth_deg is not None and beam_width_deg is not None:
            in_beam_bins.sort(
                key=lambda i: ((i * _DEG_PER_BIN - beam_azimuth_deg + 180.0) % 360.0) - 180.0
            )
        for i in in_beam_bins:
            r_km = smoothed[i]
            if r_km < 0.1:
                r_km = 0.1
            # Emission must use the inverse of the model the bins were filed
            # under.  While _bearing_and_range was flat and this stayed flat the
            # two agreed; changing one alone would have replaced the old
            # mismatch with a new one, between a bin and the vertex drawn for it.
            bearing_rad = math.radians(i * _DEG_PER_BIN)
            lat, lon = offset_latlon(
                self.rx_lat, self.rx_lon,
                east_km=r_km * math.sin(bearing_rad),
                north_km=r_km * math.cos(bearing_rad),
            )
            polygon.append([round(lat, 5), round(lon, 5)])

        # Close back to RX
        polygon.append([round(self.rx_lat, 5), round(self.rx_lon, 5)])

        if len(polygon) < 4:
            return None
        return polygon

    # ── Shrink-only prior ────────────────────────────────────────────────────

    def observed_limit_km(self, bearing_deg_: float) -> float | None:
        """Furthest this node has been *seen* to detect on this bearing.

        None where there is not enough evidence to say anything, which callers
        must read as "no constraint" rather than "no coverage".  That asymmetry
        is the whole design: the polygon is fed only from ADS-B fixes, so it
        grows where cooperative traffic flies and stays empty elsewhere.  An
        empty bearing means nobody has flown there, not that the node is deaf,
        and a constraint derived from it would carve a blind spot into a
        perfectly good footprint.

        So this may only ever *tighten* the theoretical ellipse, never extend
        it, and it abstains below _MIN_BIN_POINTS_TO_CONSTRAIN.

        Answers from the bin *and its two neighbours*, taking the most
        permissive.  The original reason was a formula mismatch — add_point
        binned with a flat-earth bearing while the association gate queried
        with a spherical one — and that is now fixed; both are spherical.

        The widening stays, for two reasons that never depended on it.  Bins
        are 5 deg wide, so a query at 4.9 deg still reads a bin whose evidence
        mostly sits at 5.1: quantisation is inherent at +-2.5 deg, an order of
        magnitude larger than the 0.27 deg the formulas differed by.  And it
        doubles as a sparsity smoother, letting a bin one sample short of
        _MIN_BIN_POINTS_TO_CONSTRAIN borrow its neighbour's evidence.

        Removing it is strictly tightening, and the failure mode of tightening
        is a blind spot — a real detection gated out of association, which is
        silent and shows up only as unattributed recall loss.  If it is to go,
        it should go on its own with its own measurement.
        """
        i = _bin_for_bearing(bearing_deg_ % 360.0)
        best = None
        for off in (-1, 0, 1):
            b = self._bins[(i + off) % N_BINS]
            if len(b) < _MIN_BIN_POINTS_TO_CONSTRAIN:
                continue
            v = _p85(b)
            if best is None or v > best:
                best = v
        return best

    def constraint_digest(self) -> tuple:
        """Per-bin observed limits, rounded — cheap change detection.

        Overlap grids are built once at registration, so a polygon that tightens
        later does not retroactively tighten them.  Callers compare this against
        the digest a grid was built under and rebuild when it moves.
        """
        out = []
        for i in range(N_BINS):
            b = self._bins[i]
            out.append(round(_p85(b), 1)
                       if len(b) >= _MIN_BIN_POINTS_TO_CONSTRAIN else None)
        return tuple(out)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "rx_lat": self.rx_lat,
            "rx_lon": self.rx_lon,
            "max_range_km": self.max_range_km,
            "max_bistatic_range_km": self.max_bistatic_range_km,
            "tx_lat": self.tx_lat,
            "tx_lon": self.tx_lon,
            "schema": self.schema,
            "range_clamp_mult": self.range_clamp_mult,
            "prior_azimuth_deg": self.prior_azimuth_deg,
            "prior_width_deg": self.prior_width_deg,
            "bins": [b[:] for b in self._bins],
            "bin_last_pos_ts": list(self._bin_last_pos_ts),
            "neg_events": [[list(ev) for ev in evs] for evs in self._neg_events],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmpiricalCoverageState":
        obj = cls(rx_lat=d["rx_lat"], rx_lon=d["rx_lon"],
                  max_range_km=d.get("max_range_km"),
                  tx_lat=d.get("tx_lat"), tx_lon=d.get("tx_lon"),
                  prior_azimuth_deg=d.get("prior_azimuth_deg"),
                  prior_width_deg=d.get("prior_width_deg"))
        obj.max_bistatic_range_km = d.get("max_bistatic_range_km")
        # Round-trip the clamp: dropping it silently reset a customised
        # multiplier to the 2.0 default on every restart.
        if d.get("range_clamp_mult") is not None:
            obj.range_clamp_mult = float(d["range_clamp_mult"])
        # Absent means v1 — written before the field existed, i.e. accumulated
        # from solver positions.
        obj.schema = d.get("schema", 1)
        for i, b in enumerate(d.get("bins", [])):
            if i < N_BINS:
                obj._bins[i] = list(b)
        bin_last_pos_ts = d.get("bin_last_pos_ts")
        if bin_last_pos_ts:
            for i, ts in enumerate(bin_last_pos_ts):
                if i < N_BINS:
                    obj._bin_last_pos_ts[i] = float(ts)
        neg_events = d.get("neg_events")
        if neg_events:
            for i, evs in enumerate(neg_events):
                if i < N_BINS:
                    obj._neg_events[i] = [(float(r), float(ts)) for r, ts in evs]
        return obj

    def save_to_file(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp, path)

    @classmethod
    def load_from_file(cls, path: str) -> "EmpiricalCoverageState":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
