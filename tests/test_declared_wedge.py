"""The declared wedge: what a SYNTHETIC node publishes as its detection area.

A real node publishes evidence only (test_learned_fov.py's
TestFovOffPublishesEvidenceOnly) because its declared azimuth and width are
unsurveyed configuration.  A simulator node is the opposite case: the
simulator emits a detection only inside the declared cone, so the cone IS the
detection area and the accumulated bins — fed by ADS-B hexes bound to tracks,
about a third of them to the wrong aircraft — are the unreliable half.  See
EmpiricalCoverageState.declared_wedge_polygon and
NodeAnalyticsManager.register_node(declared_geometry_is_truth=True).
"""

import math

import pytest

from retina_analytics.constants import KM_PER_DEG_LAT, YAGI_BEAM_WIDTH_DEG, bearing_deg, haversine_km
from retina_analytics.empirical_coverage import EmpiricalCoverageState
from retina_analytics.manager import NodeAnalyticsManager

_RX_LAT, _RX_LON = 34.85, -82.40
_TX_LAT, _TX_LON = 35.236, -82.40  # ~43 km due north


def _at_bearing(bearing, range_km, rx_lat=_RX_LAT, rx_lon=_RX_LON):
    rad = math.radians(bearing)
    return (
        rx_lat + range_km * math.cos(rad) / KM_PER_DEG_LAT,
        rx_lon + range_km * math.sin(rad) / (KM_PER_DEG_LAT * math.cos(math.radians(rx_lat))),
    )


def _apex(ec):
    return [round(ec.rx_lat, 5), round(ec.rx_lon, 5)]


def _angle_from(az, bearing):
    """Signed angular distance from *az* to *bearing*, in (-180, 180]."""
    return (bearing - az + 180.0) % 360.0 - 180.0


# Vertices are placed with offset_latlon (the flat east/north step every
# polygon in empirical_coverage.py is built from), so reading a vertex back
# with the spherical bearing_deg returns the intended bearing plus a
# convergence term — about 0.2 deg at 20 deg off axis and 50 km out, growing
# with latitude and range.  The tests below measure spherically on purpose, as
# an external check, and carry this slack rather than re-deriving the same
# flat-earth arithmetic they are trying to verify.
_CONVERGENCE_SLACK_DEG = 0.25


def _bearings_of(poly, rx_lat=_RX_LAT, rx_lon=_RX_LON):
    return [bearing_deg(rx_lat, rx_lon, lat, lon) for lat, lon in poly]


# ── The polygon itself ───────────────────────────────────────────────────────


class TestDeclaredWedgePolygon:
    def _directional(self, width=40.0, az=90.0, max_range_km=50.0):
        return EmpiricalCoverageState(
            rx_lat=_RX_LAT,
            rx_lon=_RX_LON,
            max_range_km=max_range_km,
            prior_azimuth_deg=az,
            prior_width_deg=width,
        )

    def test_it_is_a_closed_ring_with_the_rx_as_apex(self):
        ec = self._directional()
        poly = ec.declared_wedge_polygon()
        assert poly is not None
        assert poly[0] == poly[-1] == _apex(ec)

    def test_no_vertex_falls_outside_the_declared_half_width(self):
        ec = self._directional(width=40.0, az=90.0)
        poly = ec.declared_wedge_polygon()
        for lat, lon in poly[1:-1]:
            bearing = bearing_deg(_RX_LAT, _RX_LON, lat, lon)
            assert abs(_angle_from(90.0, bearing)) <= 20.0 + _CONVERGENCE_SLACK_DEG

    def test_every_vertex_sits_at_the_reach_for_its_bearing(self):
        ec = self._directional(width=40.0, az=90.0)
        poly = ec.declared_wedge_polygon()
        for lat, lon in poly[1:-1]:
            bearing = bearing_deg(_RX_LAT, _RX_LON, lat, lon)
            r = haversine_km(_RX_LAT, _RX_LON, lat, lon)
            assert r == pytest.approx(ec._reach_at(bearing), rel=0.01)

    def test_both_wedge_edges_are_drawn_exactly(self):
        """The edges are the whole point of the shape, so they are vertices in
        their own right rather than whatever the last step_deg landed on."""
        ec = self._directional(width=41.0, az=90.0)  # not a multiple of 2.5
        poly = ec.declared_wedge_polygon()
        edges = sorted(_angle_from(90.0, b) for b in _bearings_of(poly[1:-1]))
        assert edges[0] == pytest.approx(-20.5, abs=_CONVERGENCE_SLACK_DEG)
        assert edges[-1] == pytest.approx(20.5, abs=_CONVERGENCE_SLACK_DEG)

    def test_a_wedge_is_sampled_every_step_deg(self):
        ec = self._directional(width=40.0, az=90.0)
        fine = ec.declared_wedge_polygon(step_deg=2.5)
        coarse = ec.declared_wedge_polygon(step_deg=10.0)
        assert len(fine) == 2 + (40 // 2.5 + 1)  # apex, edge..edge inclusive, apex
        assert len(coarse) == 2 + (40 // 10 + 1)

    def test_a_missing_width_falls_back_to_the_yagi_default(self):
        """Same fallback _in_theoretical_wedge uses, so the two cannot drift."""
        ec = self._directional(width=None, az=90.0)
        poly = ec.declared_wedge_polygon()
        spans = [abs(_angle_from(90.0, b)) for b in _bearings_of(poly[1:-1])]
        assert max(spans) == pytest.approx(YAGI_BEAM_WIDTH_DEG / 2.0, abs=_CONVERGENCE_SLACK_DEG)

    def test_evidence_does_not_change_it(self):
        """The whole point: the bins are not consulted at all."""
        ec = self._directional(width=40.0, az=90.0)
        before = ec.declared_wedge_polygon()
        for i in range(40):
            ec.add_point(*_at_bearing(270.0, 20.0 + i * 0.01))  # a lobe behind the node
        assert ec.n_points == 40
        assert ec.declared_wedge_polygon() == before

    def test_it_is_published_with_no_evidence_at_all(self):
        ec = self._directional()
        assert ec.n_points == 0
        assert ec.to_polygon(evidence_only=True) is None
        assert ec.declared_wedge_polygon() is not None

    def test_a_non_positive_reach_has_no_polygon(self):
        ec = EmpiricalCoverageState(
            rx_lat=_RX_LAT,
            rx_lon=_RX_LON,
            max_range_km=0.0,
            prior_azimuth_deg=90.0,
            prior_width_deg=40.0,
        )
        assert ec.declared_wedge_polygon() is None


class TestOmniPrior:
    def _omni(self):
        return EmpiricalCoverageState(
            rx_lat=_RX_LAT,
            rx_lon=_RX_LON,
            max_range_km=50.0,
            prior_azimuth_deg=None,
            prior_width_deg=None,
        )

    def test_it_is_a_full_ring_with_no_apex(self):
        """An omni node has no direction to exclude, so there is no apex to
        collapse to and the ring closes on its own first vertex."""
        ec = self._omni()
        poly = ec.declared_wedge_polygon()
        assert poly[0] == poly[-1]
        assert _apex(ec) not in poly
        bearings = sorted(_bearings_of(poly[:-1]))
        assert len(bearings) == 144  # 360 / 2.5
        assert min(bearings) < 2.5
        assert max(bearings) > 357.5
        gaps = [b - a for a, b in zip(bearings, bearings[1:])]
        assert max(gaps) == pytest.approx(2.5, abs=_CONVERGENCE_SLACK_DEG)

    def test_the_ring_sits_at_the_reach_all_round(self):
        ec = self._omni()
        for lat, lon in ec.declared_wedge_polygon()[:-1]:
            r = haversine_km(_RX_LAT, _RX_LON, lat, lon)
            assert r == pytest.approx(50.0, rel=0.01)


class TestReachRule:
    """_reach_at is the simulator's own range rule: the bistatic ellipse when a
    differential limit and the TX are declared, else the circle on the RX."""

    def _state(self, bistatic):
        ec = EmpiricalCoverageState(
            rx_lat=_RX_LAT,
            rx_lon=_RX_LON,
            max_range_km=50.0,
            tx_lat=_TX_LAT,
            tx_lon=_TX_LON,
            prior_azimuth_deg=None,
            prior_width_deg=None,
        )
        ec.max_bistatic_range_km = bistatic
        return ec

    def _reach_toward_and_away(self, ec):
        """Vertex range at the bearing nearest the TX and nearest away from it."""
        poly = ec.declared_wedge_polygon(step_deg=5.0)[:-1]
        to_tx = bearing_deg(_RX_LAT, _RX_LON, _TX_LAT, _TX_LON)

        def _nearest(target):
            lat, lon = min(poly, key=lambda v: abs(_angle_from(target, bearing_deg(_RX_LAT, _RX_LON, *v))))
            return haversine_km(_RX_LAT, _RX_LON, lat, lon)

        return _nearest(to_tx), _nearest((to_tx + 180.0) % 360.0)

    def test_a_declared_differential_limit_gives_an_ellipse(self):
        """Long axis along the baseline: a target toward the TX is nearly on
        the line between the two foci, where R_tx + R_rx - L barely grows,
        while directly away from the TX the differential is 2r."""
        toward, away = self._reach_toward_and_away(self._state(60.0))
        assert away == pytest.approx(30.0, rel=0.01)  # 2r <= 60
        assert toward > away * 1.5

    def test_without_one_the_reach_is_a_circle(self):
        toward, away = self._reach_toward_and_away(self._state(None))
        assert toward == pytest.approx(away, rel=0.01)
        assert toward == pytest.approx(50.0, rel=0.01)


# ── What the manager publishes ───────────────────────────────────────────────


def _manager_with_a_narrow_wedge(declared_truth):
    """A node aimed due north with a 20 deg wedge, evidence on both sides.

    Mirrors test_learned_fov.TestFovOffPublishesEvidenceOnly's fixture — the
    same node, registered the two different ways.
    """
    m = NodeAnalyticsManager()  # fov_mode defaults to "off"
    m.register_node(
        "N",
        dict(
            rx_lat=_RX_LAT,
            rx_lon=_RX_LON,
            tx_lat=_TX_LAT,
            tx_lon=_TX_LON,
            max_range_km=50,
            beam_azimuth_deg=0.0,
            beam_width_deg=20.0,
        ),
        declared_geometry_is_truth=declared_truth,
    )
    ec = m.empirical_coverages["N"]
    for bearing in (2.5, 182.5):  # bin 0 (in wedge) and bin 36 (opposite)
        for i in range(12):
            ec.add_point(*_at_bearing(bearing, 20.0 + i * 0.01))
    return m


class TestManagerPublishesTheDeclaredWedge:
    def test_a_flagged_node_publishes_only_in_wedge_vertices(self):
        m = _manager_with_a_narrow_wedge(True)
        cov = m.get_node_summary("N")["empirical_coverage"]
        assert cov["polygon_source"] == "declared"
        out_of_wedge = [
            (lat, lon)
            for lat, lon in cov["polygon"]
            if haversine_km(_RX_LAT, _RX_LON, lat, lon) > 1.0
            and abs(_angle_from(0.0, bearing_deg(_RX_LAT, _RX_LON, lat, lon))) > 10.0 + 1e-6
        ]
        assert not out_of_wedge, "the southern lobe is mis-attributed evidence, not coverage"

    def test_it_is_the_declared_wedge_shape(self):
        """Not merely "clipped" — the same list declared_wedge_polygon returns,
        so the two cannot drift apart unnoticed."""
        m = _manager_with_a_narrow_wedge(True)
        cov = m.get_node_summary("N")["empirical_coverage"]
        assert cov["polygon"] == m.empirical_coverages["N"].declared_wedge_polygon()

    def test_the_evidence_counts_are_still_reported(self):
        m = _manager_with_a_narrow_wedge(True)
        cov = m.get_node_summary("N")["empirical_coverage"]
        assert cov["n_points"] == 24
        assert cov["n_filled_bins"] == 2

    def test_the_fov_diagnostics_are_omitted_for_a_flagged_node(self):
        """They describe the learned wedge, which a declared-truth node does
        not publish."""
        m = NodeAnalyticsManager(fov_mode="shadow")
        m.register_node(
            "N",
            dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON, max_range_km=50),
            declared_geometry_is_truth=True,
        )
        cov = m.get_node_summary("N")["empirical_coverage"]
        assert "fov" not in cov
        assert cov["polygon_source"] == "declared"

    def test_an_unflagged_node_still_publishes_the_evidence_only_shape(self):
        m = _manager_with_a_narrow_wedge(False)
        cov = m.get_node_summary("N")["empirical_coverage"]
        assert cov["polygon_source"] == "evidence"
        assert cov["polygon"] == m.empirical_coverages["N"].to_polygon(evidence_only=True)

    def test_a_flagged_node_with_no_calibration_points_still_publishes(self):
        """to_polygon's MIN_POINTS gate is an evidence gate; the declared
        wedge is known from registration, before any traffic flies."""
        m = NodeAnalyticsManager()
        m.register_node(
            "N",
            dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON, max_range_km=50),
            declared_geometry_is_truth=True,
        )
        cov = m.get_node_summary("N")["empirical_coverage"]
        assert cov["n_points"] == 0
        assert cov["polygon"]
        assert cov["polygon_source"] == "declared"

    def test_the_learned_source_is_named_under_fov_active(self):
        m = NodeAnalyticsManager(fov_mode="active")
        m.register_node("N", dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON, max_range_km=50))
        assert m.get_node_summary("N")["empirical_coverage"]["polygon_source"] == "learned"


class TestFlagLifecycle:
    def _cfg(self):
        return dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON, max_range_km=50)

    def test_re_registering_without_the_flag_drops_it(self):
        m = NodeAnalyticsManager()
        m.register_node("N", self._cfg(), declared_geometry_is_truth=True)
        m.register_node("N", self._cfg())
        assert "N" not in m._declared_truth
        assert m.get_node_summary("N")["empirical_coverage"]["polygon_source"] == "evidence"

    def test_losing_geometry_drops_it(self):
        m = NodeAnalyticsManager()
        m.register_node("N", self._cfg(), declared_geometry_is_truth=True)
        m.register_node("N", {"rx_lat": _RX_LAT, "rx_lon": _RX_LON}, declared_geometry_is_truth=True)
        assert "N" not in m._declared_truth

    def test_retire_node_drops_it(self):
        m = NodeAnalyticsManager()
        m.register_node("N", self._cfg(), declared_geometry_is_truth=True)
        m.retire_node("N")
        assert "N" not in m._declared_truth

    def test_reset_for_tests_drops_it(self):
        m = NodeAnalyticsManager()
        m.register_node("N", self._cfg(), declared_geometry_is_truth=True)
        m._reset_for_tests()
        assert "N" not in m._declared_truth

    def test_a_flag_change_invalidates_the_summaries_cache(self):
        """get_all_summaries memoises for 60 s, so a flag flip that did not
        also rebuild the detection area would otherwise be invisible until the
        TTL expired."""
        m = NodeAnalyticsManager()
        m.register_node("N", self._cfg())
        assert m.get_all_summaries()["N"]["empirical_coverage"]["polygon_source"] == "evidence"
        m.register_node("N", self._cfg(), declared_geometry_is_truth=True)
        assert m.get_all_summaries()["N"]["empirical_coverage"]["polygon_source"] == "declared"

    def test_an_unchanged_flag_leaves_the_cache_alone(self):
        m = NodeAnalyticsManager()
        m.register_node("N", self._cfg(), declared_geometry_is_truth=True)
        first = m.get_all_summaries()
        m.register_node("N", self._cfg(), declared_geometry_is_truth=True)
        assert m.get_all_summaries() is first
