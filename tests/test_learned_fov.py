"""Learned FOV (FOV_MODE) — schema/serialisation, prior resolution, polygon,
and cache-invalidation surface.

The per-bearing learning behaviour (open/extend/shrink) lives in
test_coverage_prior.py's TestAsymmetricLearning, next to the shrink-only
prior it replaces for FOV_MODE callers.  This file covers the surrounding
machinery: what makes a persisted state current (schema), how a node's
theoretical prior is derived (mirrors geo.node_beam_params without importing
it), and the two read surfaces that consume the learned state (to_polygon,
max_limit_km).
"""

import math

import pytest

from retina_analytics.constants import KM_PER_DEG_LAT
from retina_analytics.empirical_coverage import (
    CALIBRATION_SCHEMA,
    EmpiricalCoverageState,
)
from retina_analytics.manager import NodeAnalyticsManager

_RX_LAT, _RX_LON = 34.85, -82.40
_TX_LAT, _TX_LON = 35.236, -82.40   # ~43 km due north


def _at_bearing(bearing_deg, range_km, rx_lat=_RX_LAT, rx_lon=_RX_LON):
    rad = math.radians(bearing_deg)
    return (rx_lat + range_km * math.cos(rad) / KM_PER_DEG_LAT,
            rx_lon + range_km * math.sin(rad) / (KM_PER_DEG_LAT * math.cos(math.radians(rx_lat))))


# ── Schema ───────────────────────────────────────────────────────────────────

class TestSchemaRoundTrip:
    # The exact schema number is pinned (with the bump-rationale ledger) in
    # test_coverage_binning.py; here only currency matters.
    def test_a_fresh_state_is_current_schema(self):
        ec = EmpiricalCoverageState(rx_lat=_RX_LAT, rx_lon=_RX_LON)
        assert ec.schema == CALIBRATION_SCHEMA

    def test_prior_and_learned_fields_survive_to_dict_from_dict(self):
        ec = EmpiricalCoverageState(
            rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50.0,
            prior_azimuth_deg=90.0, prior_width_deg=40.0,
        )
        for _ in range(12):
            ec.add_point(*_at_bearing(90.0, 20.0), ts=1000.0)
        for dt in (0.0, 300.0, 650.0):
            ec.record_disappearance(*_at_bearing(200.0, 15.0), ts=2000.0 + dt)

        d = ec.to_dict()
        assert d["schema"] == CALIBRATION_SCHEMA
        restored = EmpiricalCoverageState.from_dict(d)

        assert restored.schema == CALIBRATION_SCHEMA
        assert restored.prior_azimuth_deg == 90.0
        assert restored.prior_width_deg == 40.0
        assert restored.n_points == ec.n_points
        assert restored.fov_digest() == ec.fov_digest()
        # bin_pos_ts (schema 6) round-trips in lockstep with bins — same
        # per-bin length, same values, so bin i's k-th range and k-th
        # timestamp still describe the same positive after the round trip.
        assert restored._bin_pos_ts == ec._bin_pos_ts
        for i in range(72):
            assert len(restored._bin_pos_ts[i]) == len(restored._bins[i])

    def test_a_file_without_the_new_fields_reads_back_as_v1(self):
        """A schema-2 file predates prior_azimuth_deg/bin_last_pos_ts/
        neg_events entirely — from_dict must not choke on their absence, and
        the discard happens at register (see TestV2DiscardAtRegister), not
        here."""
        d = {"rx_lat": _RX_LAT, "rx_lon": _RX_LON, "max_range_km": 50.0,
             "schema": 2, "bins": [[] for _ in range(72)]}
        restored = EmpiricalCoverageState.from_dict(d)
        assert restored.schema == 2
        assert restored.prior_azimuth_deg is None
        assert restored._bin_last_pos_ts == [0.0] * 72
        assert restored._neg_events == [[] for _ in range(72)]

    def test_a_v5_shaped_file_with_no_bin_pos_ts_gets_aligned_zero_filled_lists(self):
        """A schema-5 file predates bin_pos_ts entirely — from_dict must not
        choke on its absence, and must not leave _bin_pos_ts shorter than
        _bins: that would break the 1:1 lockstep _bin_open's span rule and
        add_point's FIFO eviction both rely on.  The discard-at-register
        happens separately (TestV5DiscardAtRegister below), not here —
        from_dict on its own must still be safe against any input."""
        d = {"rx_lat": _RX_LAT, "rx_lon": _RX_LON, "max_range_km": 50.0,
             "schema": 5, "bins": [[1.0, 2.0, 3.0]] + [[] for _ in range(71)]}
        restored = EmpiricalCoverageState.from_dict(d)
        assert restored.schema == 5
        assert "bin_pos_ts" not in d   # the fixture really is v5-shaped
        for i in range(72):
            assert len(restored._bin_pos_ts[i]) == len(restored._bins[i])
        assert restored._bin_pos_ts[0] == [0.0, 0.0, 0.0]


class TestV2DiscardAtRegister:
    """The manager auto-discards a stale-schema polygon at register (same
    machinery that already discarded v1 for v2) — clean migration, no
    operator action."""

    def test_a_schema_2_polygon_is_rebuilt_on_registration(self):
        cfg = dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON,
                  max_range_km=50)
        m = NodeAnalyticsManager()
        m.register_node("N", cfg)
        ec = m.empirical_coverages["N"]
        for i in range(30):
            ec.add_point(_RX_LAT + 0.05 + i * 1e-4, _RX_LON + 0.05)
        assert ec.n_points == 30

        ec.schema = 2   # as if loaded from a pre-FOV file
        m.register_node("N", cfg)   # same RX, same range rule

        assert m.empirical_coverages["N"].n_points == 0
        assert m.empirical_coverages["N"].schema == CALIBRATION_SCHEMA

    def test_current_schema_survives_reregistration(self):
        cfg = dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON,
                  max_range_km=50)
        m = NodeAnalyticsManager()
        m.register_node("N", cfg)
        ec = m.empirical_coverages["N"]
        for i in range(30):
            ec.add_point(_RX_LAT + 0.05 + i * 1e-4, _RX_LON + 0.05)
        m.register_node("N", cfg)
        assert m.empirical_coverages["N"].n_points == 30


class TestV5DiscardAtRegister:
    """Schema 6 (bin_pos_ts + the out-of-wedge open-span rule) discards a v5
    polygon on the same footing as every prior bump: a v5 file has no
    per-bin positive timestamps at all, so its bins could never legitimately
    open an out-of-wedge bin under the new rule — and separately, v5's
    recorder stamped up to 5 s of post-exit live-fix smear into every bin it
    touched (staging 2026-08-10, 32/32 directional nodes).  Neither defect
    is fixable by reinterpreting the stored ranges, so the migration is a
    clean rebuild, identical in mechanism to TestV2DiscardAtRegister above."""

    def test_a_schema_5_polygon_is_rebuilt_on_registration(self):
        cfg = dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON,
                  max_range_km=50)
        m = NodeAnalyticsManager()
        m.register_node("N", cfg)
        ec = m.empirical_coverages["N"]
        for i in range(30):
            ec.add_point(_RX_LAT + 0.05 + i * 1e-4, _RX_LON + 0.05)
        assert ec.n_points == 30

        ec.schema = 5   # as if loaded from a pre-bin_pos_ts file
        m.register_node("N", cfg)   # same RX, same range rule

        assert m.empirical_coverages["N"].n_points == 0
        assert m.empirical_coverages["N"].schema == CALIBRATION_SCHEMA
        # Reregistration-preserves-current-schema is already pinned by
        # TestV2DiscardAtRegister.test_current_schema_survives_reregistration
        # — the machinery is schema-number-agnostic, so it is not repeated
        # per bump.


# ── Prior resolution (node_beam_params semantics) ────────────────────────────

class TestPriorResolution:
    def test_explicit_azimuth_is_resolved(self):
        m = NodeAnalyticsManager()
        m.register_node("N", dict(rx_lat=_RX_LAT, rx_lon=_RX_LON,
                                  tx_lat=_TX_LAT, tx_lon=_TX_LON,
                                  beam_azimuth_deg=123.0, max_range_km=50))
        ec = m.empirical_coverages["N"]
        assert ec.prior_azimuth_deg == 123.0

    def test_unaimed_with_known_tx_falls_back_to_broadside(self):
        from retina_analytics.constants import bearing_deg
        m = NodeAnalyticsManager()
        m.register_node("N", dict(rx_lat=_RX_LAT, rx_lon=_RX_LON,
                                  tx_lat=_TX_LAT, tx_lon=_TX_LON,
                                  max_range_km=50))
        ec = m.empirical_coverages["N"]
        expected = (bearing_deg(_RX_LAT, _RX_LON, _TX_LAT, _TX_LON) + 90.0) % 360.0
        assert ec.prior_azimuth_deg == pytest.approx(expected, abs=1e-6)

    def test_unaimed_with_no_tx_is_omni(self):
        """No declared aim and no TX to derive broadside from -> None, the
        radar3 case: this is where the invented-broadside kill path stops
        being invented — every bearing starts inside the prior."""
        m = NodeAnalyticsManager()
        m.register_node("N", dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50))
        ec = m.empirical_coverages["N"]
        assert ec.prior_azimuth_deg is None

    def test_prior_updates_in_place_on_reconnect_without_losing_calibration(self):
        cfg = dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON,
                  max_range_km=50)
        m = NodeAnalyticsManager()
        m.register_node("N", cfg)
        ec = m.empirical_coverages["N"]
        for i in range(30):
            ec.add_point(_RX_LAT + 0.05 + i * 1e-4, _RX_LON + 0.05)
        pts = ec.n_points

        # Reconnect with a newly-declared explicit aim — same RX, same range
        # rule, so the accumulated bins must survive; only the prior moves.
        m.register_node("N", {**cfg, "beam_azimuth_deg": 200.0})
        ec2 = m.empirical_coverages["N"]
        assert ec2 is ec   # same object — never recreated for a prior change
        assert ec2.n_points == pts
        assert ec2.prior_azimuth_deg == 200.0


# ── to_polygon(use_learned_wedge=True) ──────────────────────────────────────

class TestLearnedWedgePolygon:
    def _state(self):
        # Omni prior (no TX) so every bin starts open — isolates the
        # negative-cap clipping this test is about from the wedge test.
        return EmpiricalCoverageState(rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50.0)

    def test_excludes_bins_that_never_opened(self):
        """A narrow theoretical prior with no positives outside it leaves
        most bins closed; the learned-wedge polygon must not draw them."""
        ec = EmpiricalCoverageState(
            rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50.0,
            prior_azimuth_deg=0.0, prior_width_deg=20.0,
        )
        for i in range(25):
            ec.add_point(*_at_bearing(i * (20.0 / 25) - 10.0, 20.0))
        poly = ec.to_polygon(use_learned_wedge=True)
        assert poly is not None
        # RX-start + RX-close + only the ~4-5 bins inside a 20 deg wedge.
        assert len(poly) < 20

    def test_extension_is_not_capped_by_range_clamp_mult(self):
        """limit_km's P95 extension is the "unclipped from the theoretical
        sector" property to_polygon(use_learned_wedge=True) relies on: the
        legacy to_polygon clamps P85 to theoretical_reach * range_clamp_mult,
        which would silently re-impose a ceiling the learned FOV is meant to
        widen past."""
        ec = self._state()   # max_range_km=50, range_clamp_mult=2.0 (default)
        for _ in range(12):
            ec.add_point(*_at_bearing(92.5, 150.0))   # within the 4x monostatic admit clamp
        assert ec.limit_km(92.5) > 50.0 * ec.range_clamp_mult

    def test_a_capped_bin_pulls_the_polygon_in_at_that_bearing(self):
        """Wiring check: to_polygon(use_learned_wedge=True) actually reads
        the negative cap, not a stale P85 snapshot."""
        ec = self._state()
        # Positives stamped BEFORE the synthetic event timestamps below —
        # a fresh positive supersedes older events, so wall-clock-now
        # positives would (correctly) neutralise 1e6-epoch events.
        for _ in range(12):
            ec.add_point(*_at_bearing(92.5, 20.0), ts=999_000.0)
        from retina_analytics.empirical_coverage import N_BINS, _bearing_and_range, _bin_for_bearing

        def _vertex_range_at(poly, bearing):
            in_beam = [i for i in range(N_BINS) if ec._bin_open(i)]
            idx = in_beam.index(_bin_for_bearing(bearing)) + 1
            _, r = _bearing_and_range(_RX_LAT, _RX_LON, *poly[idx])
            return r

        before_poly = ec.to_polygon(use_learned_wedge=True, min_points=1)
        before = _vertex_range_at(before_poly, 92.5)

        for dt in (0.0, 300.0, 650.0):
            ec.record_disappearance(*_at_bearing(92.5, 12.0), ts=1_000_000.0 + dt)
        after_poly = ec.to_polygon(use_learned_wedge=True, min_points=1)
        after = _vertex_range_at(after_poly, 92.5)

        assert after < before

    def test_closed_bins_are_excluded_even_with_min_points_satisfied(self):
        ec = EmpiricalCoverageState(
            rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50.0,
            prior_azimuth_deg=0.0, prior_width_deg=10.0,
        )
        for i in range(25):
            ec.add_point(*_at_bearing(i * (10.0 / 25) - 5.0, 20.0))
        assert ec.n_points >= 20
        poly = ec.to_polygon(use_learned_wedge=True)
        assert poly is not None
        from retina_analytics.empirical_coverage import N_BINS
        open_bins = sum(1 for i in range(N_BINS) if ec._bin_open(i))
        assert open_bins < N_BINS   # most bearings never opened
        # RX-start + RX-close + exactly the open bins.
        assert len(poly) == open_bins + 2


# ── max_limit_km cache invalidation ──────────────────────────────────────────

class TestMaxLimitKmCache:
    def test_add_point_invalidates_the_cache(self):
        ec = EmpiricalCoverageState(rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50.0)
        before = ec.max_limit_km()
        for _ in range(12):
            ec.add_point(*_at_bearing(45.0, 45.0))   # extends past theoretical
        after = ec.max_limit_km()
        assert after > before

    def test_a_read_between_writes_is_cached_not_recomputed(self):
        ec = EmpiricalCoverageState(rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50.0)
        first = ec.max_limit_km()
        second = ec.max_limit_km()
        assert first == second
        assert ec._max_limit_cache == first

    def test_record_disappearance_invalidates_the_cache(self):
        ec = EmpiricalCoverageState(
            rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50.0,
        )
        for _ in range(12):
            ec.add_point(*_at_bearing(47.5, 45.0), ts=999_000.0)
        before = ec.max_limit_km()
        for dt in (0.0, 300.0, 650.0):
            ec.record_disappearance(*_at_bearing(47.5, 10.0), ts=1_000_000.0 + dt)
        after = ec.max_limit_km()
        assert after < before
