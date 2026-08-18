"""The empirical coverage prior: bistatic clamp, and shrink-only constraint.

The polygon is fed from ADS-B fixes alone, so it grows where cooperative
traffic flies and stays empty elsewhere.  That asymmetry is the whole safety
argument for these tests: an empty bearing means nobody flew there, not that
the node is deaf, so the observed limit may only ever *tighten* the theoretical
footprint.  Read as an upper bound it would carve blind spots into perfectly
good coverage.
"""

import pytest

from retina_analytics.association import (
    NodeGeometry,
    _point_in_beam,
    compute_overlap_zone,
)
from retina_analytics.constants import KM_PER_DEG_LAT, bistatic_range_limit_km
from retina_analytics.empirical_coverage import (
    _MIN_BIN_POINTS_TO_CONSTRAIN,
    OBSERVED_LIMIT_MARGIN,
    EmpiricalCoverageState,
)

_RX_LAT, _RX_LON = 34.85, -82.40
_TX_LAT, _TX_LON = 35.236, -82.40  # 43 km due north
_DELTA = 60.0


def _state(bistatic=_DELTA, tx=(_TX_LAT, _TX_LON)):
    ec = EmpiricalCoverageState(
        rx_lat=_RX_LAT,
        rx_lon=_RX_LON,
        max_range_km=_DELTA,
        tx_lat=tx[0] if tx else None,
        tx_lon=tx[1] if tx else None,
    )
    ec.max_bistatic_range_km = bistatic
    return ec


def _at_bearing(bearing_deg, range_km):
    """A lat/lon that many km from the RX on that bearing (flat approximation)."""
    import math

    rad = math.radians(bearing_deg)
    return (
        _RX_LAT + range_km * math.cos(rad) / KM_PER_DEG_LAT,
        _RX_LON + range_km * math.sin(rad) / (KM_PER_DEG_LAT * math.cos(math.radians(_RX_LAT))),
    )


class TestClampFollowsTheEllipse:
    def test_a_far_anti_tx_point_is_refused(self):
        """The clamp is per bearing, so 'too far' depends on which way you look.

        50 km due south is inside 2x a 60 km circle — the old scalar clamp took
        it — but the footprint only reaches 30 km there, so at 2x margin the
        bound is 60 km and a 50 km point... is still admitted.  Push to 70 km,
        beyond even the doubled anti-TX reach, and it must go.
        """
        ec = _state()
        lat, lon = _at_bearing(180.0, 70.0)
        ec.add_point(lat, lon)
        assert ec.n_points == 0

    def test_the_same_range_toward_the_transmitter_is_kept(self):
        """73 km of real reach north against 30 km south — same node."""
        ec = _state()
        lat, lon = _at_bearing(0.0, 70.0)
        ec.add_point(lat, lon)
        assert ec.n_points == 1

    def test_reach_matches_the_shared_formula(self):
        ec = _state()
        baseline = 43.0
        for psi in (0.0, 90.0, 180.0):
            bearing = (0.0 + psi) % 360.0  # TX is due north
            assert ec._reach_at(bearing) == pytest.approx(
                bistatic_range_limit_km(psi, baseline, _DELTA),
                abs=0.2,
            )

    def test_a_node_without_a_bistatic_limit_keeps_the_circle(self):
        ec = _state(bistatic=None)
        for bearing in (0.0, 90.0, 180.0):
            assert ec._reach_at(bearing) == pytest.approx(_DELTA)


class TestObservedLimitAbstains:
    def test_no_evidence_means_no_constraint(self):
        """Silence is not a bound.  This is the property everything rests on."""
        assert _state().observed_limit_km(90.0) is None

    def test_a_thin_bin_still_abstains(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN - 1):
            ec.add_point(*_at_bearing(90.0, 20.0))
        assert ec.observed_limit_km(90.0) is None

    def test_enough_evidence_yields_a_limit(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 20.0))
        assert ec.observed_limit_km(90.0) == pytest.approx(20.0, abs=1.0)

    def test_a_constrained_bearing_does_not_constrain_its_neighbours(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 20.0))
        assert ec.observed_limit_km(180.0) is None

    def test_digest_tracks_what_would_constrain(self):
        ec = _state()
        before = ec.constraint_digest()
        assert set(before) == {None}
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 20.0))
        after = ec.constraint_digest()
        assert after != before
        assert sum(1 for v in after if v is not None) == 1


def _geo(coverage_limit=None):
    return NodeGeometry(
        node_id="n",
        rx_lat=_RX_LAT,
        rx_lon=_RX_LON,
        rx_alt_km=0.3,
        tx_lat=_TX_LAT,
        tx_lon=_TX_LON,
        tx_alt_km=0.6,
        fc_hz=183e6,
        beam_azimuth_deg=0.0,
        beam_width_deg=360.0,
        max_range_km=_DELTA,
        max_bistatic_range_km=_DELTA,
        coverage_limit=coverage_limit,
    )


class TestPriorOnlyTightens:
    def test_an_abstaining_bearing_is_untouched(self):
        """A node with an empty polygon gates exactly as one with no polygon."""
        ec = _state()
        lat, lon = _at_bearing(90.0, 25.0)
        assert _point_in_beam(lat, lon, _geo(ec.observed_limit_km)) == _point_in_beam(lat, lon, _geo(None))

    def test_observed_coverage_pulls_the_gate_in(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 12.0))
        far = _at_bearing(90.0, 40.0)  # inside the ellipse, far past what was seen
        assert _point_in_beam(*far, _geo(None)) is True
        assert _point_in_beam(*far, _geo(ec.observed_limit_km)) is False

    def test_the_margin_is_honoured(self):
        """Traffic does not fly to the edge, so the observed P85 is a floor."""
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 20.0))
        just_inside = _at_bearing(90.0, 20.0 * OBSERVED_LIMIT_MARGIN * 0.9)
        just_outside = _at_bearing(90.0, 20.0 * OBSERVED_LIMIT_MARGIN * 1.2)
        assert _point_in_beam(*just_inside, _geo(ec.observed_limit_km)) is True
        assert _point_in_beam(*just_outside, _geo(ec.observed_limit_km)) is False

    def test_it_can_never_extend_the_footprint(self):
        """Observations beyond the theoretical reach do not buy coverage.

        The ellipse stays the hard bound; the prior is one-directional.
        """
        ec = _state()
        # Force a generous observed limit by writing the bin directly — add_point
        # would clamp it, which is itself the belt to this braces.
        ec._bins[_bin_for_bearing_of(180.0)] = [200.0] * 40
        beyond = _at_bearing(180.0, 55.0)  # past the 30 km anti-TX reach
        geo = _geo(ec.observed_limit_km)
        geo.beam_width_deg = 360.0
        # Still admitted by the sector+radius test here, because the *range*
        # rule lives on the exact delay in compute_overlap_zone — the point is
        # that the prior did not widen anything.
        assert _point_in_beam(*beyond, geo) is True
        zone = compute_overlap_zone(
            geo,
            _geo(ec.observed_limit_km),
            grid_step_km=3.0,
            altitudes_km=(7.0,),
        )
        c_km_us = 0.299792458
        assert all(d * c_km_us <= _DELTA + 1e-9 for d, _ in zone.delay_pairs)


def _bin_for_bearing_of(bearing):
    from retina_analytics.empirical_coverage import _bin_for_bearing

    return _bin_for_bearing(bearing)


class TestRebuildOnTightening:
    def test_rebuilding_applies_a_polygon_that_arrived_later(self):
        """Grids are built at registration; a later tightening needs a rebuild.

        Without this the constraint would only ever take effect on a restart.
        """
        from retina_analytics.association import InterNodeAssociator

        states = {}

        def provider(node_id):
            ec = states.get(node_id)
            return ec.observed_limit_km if ec else None

        cfg_a = {
            "rx_lat": _RX_LAT,
            "rx_lon": _RX_LON,
            "rx_alt_ft": 1000,
            "tx_lat": _TX_LAT,
            "tx_lon": _TX_LON,
            "tx_alt_ft": 2000,
            "fc_hz": 183e6,
            "beam_width_deg": 360,
            "max_range_km": _DELTA,
            "max_bistatic_range_km": _DELTA,
            "beam_azimuth_deg": 0.0,
        }
        cfg_b = dict(cfg_a, rx_lon=-82.31, fc_hz=195e6)

        a = InterNodeAssociator(grid_step_km=3.0, coverage_provider=provider)
        a.register_node("n1", cfg_a)
        a.register_node("n2", cfg_b)
        before = len(a.overlap_zones[("n1", "n2")].grid_points)
        assert before > 0

        # n1 turns out to see only 8 km, in every direction.
        ec = _state()
        for bearing in range(0, 360, 5):
            for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
                ec.add_point(*_at_bearing(float(bearing), 8.0))
        states["n1"] = ec

        assert a.rebuild_zones_for("n1") >= 1
        after = len(a.overlap_zones[("n1", "n2")].grid_points)
        assert after < before

    def test_a_zone_tightened_to_nothing_stops_being_a_neighbour(self):
        """Otherwise every round keeps pairing frames against an empty grid."""
        from retina_analytics.association import InterNodeAssociator

        states = {}

        def provider(node_id):
            ec = states.get(node_id)
            return ec.observed_limit_km if ec else None

        cfg_a = {
            "rx_lat": _RX_LAT,
            "rx_lon": _RX_LON,
            "rx_alt_ft": 1000,
            "tx_lat": _TX_LAT,
            "tx_lon": _TX_LON,
            "tx_alt_ft": 2000,
            "fc_hz": 183e6,
            "beam_width_deg": 360,
            "max_range_km": _DELTA,
            "max_bistatic_range_km": _DELTA,
            "beam_azimuth_deg": 0.0,
        }
        cfg_b = dict(cfg_a, rx_lon=-82.31, fc_hz=195e6)

        a = InterNodeAssociator(grid_step_km=3.0, coverage_provider=provider)
        a.register_node("n1", cfg_a)
        a.register_node("n2", cfg_b)
        assert "n2" in a._neighbors.get("n1", set())

        ec = _state()
        for bearing in range(0, 360, 5):
            for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
                ec.add_point(*_at_bearing(float(bearing), 0.6))
        states["n1"] = ec
        a.rebuild_zones_for("n1")

        assert not a.overlap_zones[("n1", "n2")].delay_pairs
        assert "n2" not in a._neighbors.get("n1", set())
        assert "n1" not in a._neighbors.get("n2", set())


class TestBinBoundaries:
    """A constraint must not flicker across a 5° bin edge.

    add_point derives its bearing with a flat approximation and the association
    gate with a spherical one, so a query routinely lands one bin over from the
    evidence it belongs to.  Sampling a single bin made the constraint vanish
    at every boundary — found by a test that queried 89.87° against evidence
    filed at 90.0°, which is the difference between the two formulas.
    """

    def test_a_query_just_off_the_evidence_bearing_still_constrains(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 12.0))
        for probe in (89.87, 90.0, 90.2, 88.0, 92.0):
            assert ec.observed_limit_km(probe) is not None, probe

    def test_it_stays_local(self):
        """Neighbour widening is 3 bins, not a global smear."""
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 12.0))
        assert ec.observed_limit_km(120.0) is None
        assert ec.observed_limit_km(270.0) is None

    def test_the_most_permissive_neighbour_wins(self):
        """Erring permissive is the safe direction for a shrink-only bound."""
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 10.0))
            ec.add_point(*_at_bearing(95.0, 25.0))
        # 92.5° sits between them; the wider of the two must win.
        assert ec.observed_limit_km(92.5) == pytest.approx(25.0, abs=1.5)

    def test_wraparound_is_handled(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(0.5, 15.0))
        assert ec.observed_limit_km(359.0) is not None
