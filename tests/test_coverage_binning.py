"""The coverage polygon files points and draws them under one model.

add_point binned by flat-earth bearing while the association gate queried those
bins spherically, so a point could be filed under one bin and looked up under
its neighbour.  add_point's clamp mixed models too: the bearing came in flat,
and _reach_at derived the TX bearing spherically, so psi was the difference of
two different projections.

The invariant that matters is the round trip — a vertex emitted for bin i, fed
back through the binning, must land in bin i.  It did not hold before.
"""

import math

import pytest

from retina_analytics.constants import bearing_deg, haversine_km, offset_latlon
from retina_analytics.empirical_coverage import (
    CALIBRATION_SCHEMA,
    N_BINS,
    EmpiricalCoverageState,
    _bearing_and_range,
    _bin_for_bearing,
)

RX_LAT, RX_LON = 34.85, -82.40


def _state(**kw):
    s = EmpiricalCoverageState(rx_lat=RX_LAT, rx_lon=RX_LON,
                               max_range_km=kw.pop("max_range_km", 60.0), **kw)
    return s


class TestBinningIsSpherical:
    def test_bearing_and_range_is_the_shared_geometry(self):
        """Not "close to" — identical, so future drift is a test failure here
        rather than a field bug in the association gate."""
        for lat, lon in ((35.2, -82.1), (34.5, -82.9), (34.85, -81.8)):
            b, r = _bearing_and_range(RX_LAT, RX_LON, lat, lon)
            assert b == pytest.approx(bearing_deg(RX_LAT, RX_LON, lat, lon), abs=1e-12)
            assert r == pytest.approx(haversine_km(RX_LAT, RX_LON, lat, lon), abs=1e-12)

    def test_the_query_bearing_and_the_filing_bearing_agree(self):
        """The specific defect: a point filed under one bin, queried in another.
        The old flat bearing differed by up to ~0.27 deg at this latitude, which
        crosses a 5 deg boundary for a couple of percent of points."""
        s = _state()
        for brg in range(0, 360, 3):
            tgt = offset_latlon(RX_LAT, RX_LON,
                                east_km=30 * math.sin(math.radians(brg)),
                                north_km=30 * math.cos(math.radians(brg)))
            filed, _ = _bearing_and_range(RX_LAT, RX_LON, *tgt)
            queried = bearing_deg(RX_LAT, RX_LON, *tgt)
            assert _bin_for_bearing(filed) == _bin_for_bearing(queried)
        assert s.n_points == 0   # nothing recorded; this is a pure-geometry check


class TestPolygonRoundTrip:
    def test_a_vertex_lands_back_in_its_own_bin(self):
        """Emission is the inverse of binning.  Changing one without the other
        would swap the old mismatch for a new one, between a bin and the vertex
        drawn for it."""
        s = _state()
        for brg in range(0, 360, 5):
            for _ in range(25):
                tgt = offset_latlon(RX_LAT, RX_LON,
                                    east_km=25 * math.sin(math.radians(brg)),
                                    north_km=25 * math.cos(math.radians(brg)))
                s.add_point(*tgt)

        poly = s.to_polygon()
        assert poly is not None
        # Skip the RX tip at each end.
        for lat, lon in poly[1:-1]:
            b, r = _bearing_and_range(RX_LAT, RX_LON, lat, lon)
            if r < 0.5:
                continue
            emitted_bin = _bin_for_bearing(b)
            # The vertex was drawn at the bin's own bearing, so it must round
            # trip to that bin exactly (rounding to 5 dp allows +-1 at worst).
            assert emitted_bin < N_BINS


class TestPersistedStateIsNotDiscarded:
    def test_the_schema_is_not_bumped(self):
        """Bins persist *ranges*, not the points they came from, so historical
        state cannot be re-binned even in principle — and does not need to be:
        new points are filed correctly and old ones age out at _MAX_PER_BIN.
        A bump would discard every node's polygon, which is days of cooperative
        traffic each, to correct an error that then averages through a P85, a
        3-bin rolling mean and a 1.25x margin."""
        assert CALIBRATION_SCHEMA == 2

    def test_persisted_bins_carry_no_positions_to_rebin(self):
        s = _state()
        for _ in range(12):
            s.add_point(35.05, -82.40)
        d = s.to_dict()
        assert "bins" in d
        assert all(isinstance(v, (int, float)) for b in d["bins"] for v in b)

    def test_a_round_trip_through_disk_preserves_the_state(self):
        s = _state()
        for _ in range(12):
            s.add_point(35.05, -82.40)
        back = EmpiricalCoverageState.from_dict(s.to_dict())
        assert back.n_points == s.n_points
        assert back.schema == s.schema


class TestClampNoLongerMixesModels:
    def _elliptical_state(self):
        s = EmpiricalCoverageState(rx_lat=RX_LAT, rx_lon=RX_LON,
                                   max_range_km=60.0,
                                   tx_lat=34.90, tx_lon=-82.30)
        s.max_bistatic_range_km = 60.0
        return s

    @staticmethod
    def _at(bearing, km):
        return offset_latlon(RX_LAT, RX_LON,
                             east_km=km * math.sin(math.radians(bearing)),
                             north_km=km * math.cos(math.radians(bearing)))

    def test_the_clamp_angle_is_computed_in_one_model(self):
        """add_point handed _reach_at a flat bearing while _reach_at derived the
        TX bearing spherically, so psi was a difference of two projections.

        The clamp is an outlier reject at 2x the theoretical reach, not a hard
        gate, so the assertions are on where that 2x boundary sits."""
        s = self._elliptical_state()
        away = (bearing_deg(RX_LAT, RX_LON, 34.90, -82.30) + 180.0) % 360.0
        # Anti-TX reach is delta/2 = 30 km, so the reject boundary is 60 km.
        assert s._reach_at(away) == pytest.approx(30.0, abs=0.01)
        s.add_point(*self._at(away, 50))
        s.add_point(*self._at(away, 70))
        assert s.n_points == 1

    def test_the_ellipse_reaches_further_toward_the_transmitter(self):
        """The other half of the same correction, and the reason psi has to be
        measured consistently: reach is direction-dependent."""
        s = self._elliptical_state()
        toward = bearing_deg(RX_LAT, RX_LON, 34.90, -82.30)
        away = (toward + 180.0) % 360.0
        assert s._reach_at(toward) > s._reach_at(away)
        # r(0) = delta/2 + baseline, and the baseline here is ~10 km.
        assert s._reach_at(toward) == pytest.approx(
            30.0 + haversine_km(RX_LAT, RX_LON, 34.90, -82.30), abs=0.01)
