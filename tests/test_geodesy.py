"""The shared geodesy surface in constants.py.

These primitives replaced six haversines, six bearings, five bistatic-delay
computations and four different values of kilometres-per-degree scattered across
this library and the backend.  The tests that matter here are the invariants —
round-trips, degenerate cases, and the boundary conditions the old copies
disagreed on.
"""

import math

import pytest

from retina_analytics.constants import (
    C_KM_US,
    KM_PER_DEG_LAT,
    R_EARTH,
    bearing_deg,
    bistatic_delay_us,
    bistatic_differential_km,
    bistatic_max_radius_km,
    enu_km,
    haversine_km,
    km_per_deg_lon,
    offset_latlon,
    offset_latlon_m,
    point_in_beam,
)

RX = (34.85, -82.40)
TX = (34.90, -82.30)


class TestKmPerDegree:
    def test_it_is_derived_from_the_earth_radius(self):
        """Not typed. The four hand-typed values in circulation (111.0, 111.32,
        111.320, 111.1949) disagreed by 0.3%, which is 300 m over 100 km."""
        assert KM_PER_DEG_LAT == pytest.approx(R_EARTH * math.pi / 180.0)
        assert KM_PER_DEG_LAT == pytest.approx(111.1949, abs=1e-4)

    def test_longitude_shrinks_with_latitude(self):
        assert km_per_deg_lon(0.0) == pytest.approx(KM_PER_DEG_LAT)
        assert km_per_deg_lon(60.0) == pytest.approx(KM_PER_DEG_LAT / 2, rel=1e-6)

    def test_the_pole_does_not_divide_by_zero(self):
        assert km_per_deg_lon(90.0) > 0.0


class TestOffsetRoundTrip:
    def test_enu_and_offset_are_inverses(self):
        lat, lon = offset_latlon(*RX, east_km=12.0, north_km=-7.5)
        east, north = enu_km(*RX, lat, lon)
        assert east == pytest.approx(12.0, abs=1e-9)
        assert north == pytest.approx(-7.5, abs=1e-9)

    def test_a_zero_offset_is_the_identity(self):
        assert offset_latlon(*RX, 0.0, 0.0) == RX

    def test_metres_and_kilometres_agree(self):
        assert offset_latlon_m(*RX, 1500.0, -800.0) == \
               pytest.approx(offset_latlon(*RX, 1.5, -0.8))

    def test_a_north_offset_moves_only_latitude(self):
        lat, lon = offset_latlon(*RX, 0.0, 10.0)
        assert lon == RX[1]
        assert lat > RX[0]

    def test_offset_distance_agrees_with_haversine(self):
        """The two must not drift: one is used to dead-reckon and the other to
        judge how far the result moved."""
        lat, lon = offset_latlon(*RX, 0.0, 25.0)
        assert haversine_km(*RX, lat, lon) == pytest.approx(25.0, rel=2e-3)


class TestBistaticDelay:
    def test_a_target_on_the_baseline_has_zero_differential(self):
        """R_tx + R_rx = L when the target sits between the two.  The tolerance
        is 1 cm because the lat/lon midpoint is not exactly the great-circle
        midpoint — that gap, not the formula, is what is being measured here."""
        mid = ((RX[0] + TX[0]) / 2, (RX[1] + TX[1]) / 2)
        assert bistatic_differential_km(*TX, *RX, *mid) == pytest.approx(0.0, abs=1e-5)

    def test_differential_is_never_negative(self):
        for d_lat in (-0.5, 0.0, 0.3):
            for d_lon in (-0.5, 0.0, 0.4):
                got = bistatic_differential_km(*TX, *RX, RX[0] + d_lat, RX[1] + d_lon)
                assert got >= -1e-9

    def test_delay_is_the_differential_over_c(self):
        tgt = (34.95, -82.15)
        assert bistatic_delay_us(*TX, *RX, *tgt) == pytest.approx(
            bistatic_differential_km(*TX, *RX, *tgt) / C_KM_US)

    def test_a_colocated_tx_gives_twice_the_range(self):
        """Monostatic degenerate case: L = 0 so the differential is 2r."""
        tgt = (35.05, -82.40)
        got = bistatic_differential_km(*RX, *RX, *tgt)
        assert got == pytest.approx(2 * haversine_km(*RX, *tgt), rel=1e-9)


class TestPointInBeam:
    CFG = dict(rx_lat=RX[0], rx_lon=RX[1], tx_lat=TX[0], tx_lon=TX[1],
               beam_width_deg=41.0, max_range_km=50.0)

    def test_boresight_is_in(self):
        az = bearing_deg(*RX, 35.20, -82.40)
        assert point_in_beam(35.20, -82.40, beam_azimuth_deg=az, **self.CFG)

    def test_outside_the_half_width_is_out(self):
        az = (bearing_deg(*RX, 35.20, -82.40) + 30.0) % 360.0
        assert not point_in_beam(35.20, -82.40, beam_azimuth_deg=az, **self.CFG)

    def test_the_bearing_test_wraps(self):
        """359° and 1° are 2° apart, not 358°."""
        tgt = offset_latlon(*RX, east_km=0.35, north_km=20.0)   # bearing ~1°
        assert point_in_beam(*tgt, beam_azimuth_deg=359.0, **self.CFG)

    def test_no_azimuth_means_omnidirectional(self):
        for brg in (0.0, 90.0, 180.0, 270.0):
            tgt = offset_latlon(*RX, 20 * math.sin(math.radians(brg)),
                                20 * math.cos(math.radians(brg)))
            assert point_in_beam(*tgt, beam_azimuth_deg=None, **self.CFG)

    def test_the_bistatic_limit_beats_the_circle_away_from_the_tx(self):
        """The case the ellipse exists for: 45 km from the RX directly away from
        the transmitter is inside a 60 km circle but outside a 60 km bistatic
        footprint, whose anti-TX reach is Δ/2 = 30 km."""
        away = (bearing_deg(*RX, *TX) + 180.0) % 360.0
        tgt = offset_latlon(*RX, 45 * math.sin(math.radians(away)),
                            45 * math.cos(math.radians(away)))
        cfg = dict(self.CFG, max_range_km=60.0)
        assert point_in_beam(*tgt, beam_azimuth_deg=None, **cfg)
        assert not point_in_beam(*tgt, beam_azimuth_deg=None,
                                 max_bistatic_range_km=60.0, **cfg)

    def test_toward_the_tx_the_ellipse_reaches_further_than_the_circle(self):
        """r(0) = Δ/2 + L, so a distant tower extends reach rather than clipping
        it — the other half of the same correction."""
        baseline = haversine_km(*RX, *TX)
        toward = bearing_deg(*RX, *TX)
        reach = bistatic_max_radius_km(baseline, 60.0)
        tgt = offset_latlon(*RX, (reach - 1) * math.sin(math.radians(toward)),
                            (reach - 1) * math.cos(math.radians(toward)))
        assert point_in_beam(*tgt, beam_azimuth_deg=None,
                             max_bistatic_range_km=60.0,
                             **dict(self.CFG, max_range_km=30.0))

    def test_a_node_without_a_bistatic_limit_keeps_the_circle(self):
        tgt = offset_latlon(*RX, 0.0, 55.0)
        assert not point_in_beam(*tgt, beam_azimuth_deg=None, **self.CFG)
        assert point_in_beam(*tgt, beam_azimuth_deg=None,
                             **dict(self.CFG, max_range_km=60.0))

    def test_a_bistatic_limit_without_a_tx_falls_back_to_the_circle(self):
        """Half-declared geometry must not silently admit everything."""
        cfg = {k: v for k, v in self.CFG.items() if not k.startswith("tx_")}
        tgt = offset_latlon(*RX, 0.0, 55.0)
        assert not point_in_beam(*tgt, beam_azimuth_deg=None,
                                 max_bistatic_range_km=60.0, **cfg)
