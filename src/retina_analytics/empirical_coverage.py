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
"""

import json
import math
import os

from retina_analytics.constants import (
    YAGI_MAX_RANGE_KM,
    bearing_deg,
    bistatic_range_limit_km,
    haversine_km,
    offset_latlon,
)

N_BINS = 72  # 5 ° per bin  (360 / 5 = 72)
_DEG_PER_BIN = 360.0 / N_BINS
_MAX_PER_BIN = 200  # cap per-bin history to prevent unbounded RAM growth
MIN_POINTS = 20  # minimum calibration points before emitting a polygon

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
#
# Production and staging mount coverage_data as a named volume that survives
# rebuilds, so without this a v1 polygon would be served indefinitely with no
# operator action to prompt it.
CALIBRATION_SCHEMA = 2


def _bin_for_bearing(bearing_deg: float) -> int:
    return int(bearing_deg / _DEG_PER_BIN) % N_BINS


def _bearing_and_range(rx_lat: float, rx_lon: float, lat: float, lon: float) -> tuple[float, float]:
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
    return (bearing_deg(rx_lat, rx_lon, lat, lon), haversine_km(rx_lat, rx_lon, lat, lon))


def _p85(values: list[float]) -> float:
    """85th-percentile of a non-empty list (sorted() — the input is untouched)."""
    s = sorted(values)
    idx = min(int(len(s) * 0.85), len(s) - 1)
    return s[idx]


class EmpiricalCoverageState:
    """Accumulates calibration points and derives a smoothed detection polygon."""

    def __init__(
        self,
        rx_lat: float,
        rx_lon: float,
        max_range_km: float | None = None,
        range_clamp_mult: float = 2.0,
        tx_lat: float | None = None,
        tx_lon: float | None = None,
    ):
        self.rx_lat = rx_lat
        self.rx_lon = rx_lon
        # Transmitter position, so the clamp can follow the ellipse the node is
        # actually bounded by rather than a circle on the receiver.  The two
        # differ by 2x in radius directly away from the transmitter, which is
        # precisely where a circle would let a mis-attributed detection through.
        self.tx_lat = tx_lat
        self.tx_lon = tx_lon
        # Detections beyond range_clamp_mult × max_range_km are rejected as
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
        # Per-bin list of observed ranges (km).  List, not array — no numpy dep.
        self._bins: list[list[float]] = [[] for _ in range(N_BINS)]

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def _reach_at(self, bearing_deg_: float) -> float:
        """How far the node can see on this bearing, before any margin.

        The bistatic ellipse when the node declares a differential limit and its
        transmitter is known; otherwise the legacy circle.
        """
        if self.max_bistatic_range_km is not None and self.tx_lat is not None and self.tx_lon is not None:
            baseline = haversine_km(self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon)
            to_tx = bearing_deg(self.rx_lat, self.rx_lon, self.tx_lat, self.tx_lon)
            psi = abs((bearing_deg_ - to_tx + 180.0) % 360.0 - 180.0)
            return bistatic_range_limit_km(psi, baseline, self.max_bistatic_range_km)
        return self.max_range_km if self.max_range_km is not None else YAGI_MAX_RANGE_KM

    def add_point(self, lat: float, lon: float) -> None:
        """Record one calibration point (known target position)."""
        bearing, range_km = _bearing_and_range(self.rx_lat, self.rx_lon, lat, lon)
        if range_km < 0.5:
            return  # too close — not informative
        if range_km > self._reach_at(bearing) * self.range_clamp_mult:
            return  # implausibly far — mis-attributed detection
        b = self._bins[_bin_for_bearing(bearing)]
        b.append(range_km)
        if len(b) > _MAX_PER_BIN:
            del b[0]  # drop oldest

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_points(self) -> int:
        return sum(len(b) for b in self._bins)

    @property
    def n_filled_bins(self) -> int:
        return sum(1 for b in self._bins if b)

    # ── Polygon generation ────────────────────────────────────────────────────

    def to_polygon(
        self,
        min_points: int = MIN_POINTS,
        beam_azimuth_deg: float | None = None,
        beam_width_deg: float | None = None,
        max_range_km: float | None = None,
    ) -> list[list[float]] | None:
        """Return a closed polygon [[lat, lon], …] or None if insufficient data.

        When *beam_azimuth_deg* and *beam_width_deg* are provided the polygon
        is constrained to the beam sector (a pie-slice shape starting and ending
        at the RX position).  Bins outside the sector are zeroed so the
        interpolation step never bleeds coverage into directions the antenna
        physically cannot observe.
        """
        if self.n_points < min_points:
            return None

        # --- Determine which bins fall inside the beam sector -----------------
        if beam_azimuth_deg is not None and beam_width_deg is not None:
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
            bearing_i = i * _DEG_PER_BIN
            clamp = max_range_km if max_range_km is not None else self._reach_at(bearing_i)
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
            in_beam_bins.sort(key=lambda i: ((i * _DEG_PER_BIN - beam_azimuth_deg + 180.0) % 360.0) - 180.0)
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
                self.rx_lat,
                self.rx_lon,
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
            out.append(round(_p85(b), 1) if len(b) >= _MIN_BIN_POINTS_TO_CONSTRAIN else None)
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
            "bins": [b[:] for b in self._bins],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmpiricalCoverageState":
        obj = cls(
            rx_lat=d["rx_lat"],
            rx_lon=d["rx_lon"],
            max_range_km=d.get("max_range_km"),
            tx_lat=d.get("tx_lat"),
            tx_lon=d.get("tx_lon"),
        )
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
        return obj

    def save_to_file(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp, path)

    @classmethod
    def load_from_file(cls, path: str) -> "EmpiricalCoverageState":
        with open(path) as f:
            return cls.from_dict(json.load(f))
