"""InterNodeAssociator honours an explicit ring Yagi aim.

register_node keeps a supplied beam_azimuth_deg and computes the overlap grid
from it. Two receivers aimed at a shared core overlap there; left broadside they
point away from the core and never overlap.
"""

import math

from retina_analytics.association import InterNodeAssociator, _bearing_deg
from retina_analytics.constants import resolve_beam_azimuth_deg, bearing_deg as _c_bearing

_CORE_LAT, _CORE_LON = 32.8968, -97.0380
_TX_LAT, _TX_LON = 32.78060, -96.80060
_R = 6371.0


def _point_at(bearing_deg, dist_km, lat0, lon0):
    br = math.radians(bearing_deg)
    lat = lat0 + math.degrees((dist_km * math.cos(br)) / _R)
    lon = lon0 + math.degrees(
        (dist_km * math.sin(br)) / (_R * math.cos(math.radians(lat0)))
    )
    return lat, lon


_RX_A = _point_at(0.0, 18.0, _CORE_LAT, _CORE_LON)
_RX_B = _point_at(180.0, 18.0, _CORE_LAT, _CORE_LON)

_COMMON = dict(
    rx_alt_ft=500, tx_lat=_TX_LAT, tx_lon=_TX_LON, tx_alt_ft=1600,
    fc_hz=195e6, beam_width_deg=50, max_range_km=60,
)


def _bearing_to_core(rx):
    return _bearing_deg(rx[0], rx[1], _CORE_LAT, _CORE_LON)


def _broadside(rx):
    return (_bearing_deg(rx[0], rx[1], _TX_LAT, _TX_LON) + 90.0) % 360.0


class TestExplicitAimRegistration:
    def test_explicit_azimuth_kept(self):
        assoc = InterNodeAssociator(grid_step_km=10.0)
        assoc.register_node("X", {
            **_COMMON, "rx_lat": _RX_A[0], "rx_lon": _RX_A[1],
            "beam_azimuth_deg": 123.0,
        })
        assert assoc.node_geometries["X"].beam_azimuth_deg == 123.0

    def test_missing_azimuth_falls_back_to_broadside(self):
        assoc = InterNodeAssociator(grid_step_km=10.0)
        assoc.register_node("Y", {
            **_COMMON, "rx_lat": _RX_A[0], "rx_lon": _RX_A[1],
        })
        assert abs(
            assoc.node_geometries["Y"].beam_azimuth_deg - _broadside(_RX_A)
        ) < 1e-4

    def test_none_azimuth_falls_back_to_broadside(self):
        assoc = InterNodeAssociator(grid_step_km=10.0)
        assoc.register_node("Z", {
            **_COMMON, "rx_lat": _RX_A[0], "rx_lon": _RX_A[1],
            "beam_azimuth_deg": None,
        })
        assert abs(
            assoc.node_geometries["Z"].beam_azimuth_deg - _broadside(_RX_A)
        ) < 1e-4


class TestAimDrivesOverlap:
    def test_core_aimed_pair_overlaps_at_core(self):
        assoc = InterNodeAssociator(grid_step_km=3.0)
        assoc.register_node("RA", {
            **_COMMON, "rx_lat": _RX_A[0], "rx_lon": _RX_A[1],
            "beam_azimuth_deg": _bearing_to_core(_RX_A),
        })
        assoc.register_node("RB", {
            **_COMMON, "rx_lat": _RX_B[0], "rx_lon": _RX_B[1],
            "beam_azimuth_deg": _bearing_to_core(_RX_B),
        })
        zone = assoc.overlap_zones[tuple(sorted(["RA", "RB"]))]
        assert zone.delay_pairs

    def test_broadside_pair_does_not_overlap_at_core(self):
        assoc = InterNodeAssociator(grid_step_km=3.0)
        assoc.register_node("RA", {
            **_COMMON, "rx_lat": _RX_A[0], "rx_lon": _RX_A[1],
        })
        assoc.register_node("RB", {
            **_COMMON, "rx_lat": _RX_B[0], "rx_lon": _RX_B[1],
        })
        zone = assoc.overlap_zones[tuple(sorted(["RA", "RB"]))]
        assert not zone.delay_pairs


class TestResolveBeamAzimuth:
    def test_explicit_azimuth_kept(self):
        az = resolve_beam_azimuth_deg(
            {"beam_azimuth_deg": 123.0}, _RX_A[0], _RX_A[1], _TX_LAT, _TX_LON
        )
        assert az == 123.0

    def test_missing_falls_back_to_broadside(self):
        az = resolve_beam_azimuth_deg({}, _RX_A[0], _RX_A[1], _TX_LAT, _TX_LON)
        assert abs(az - _broadside(_RX_A)) < 1e-9

    def test_unparseable_falls_back_to_broadside(self):
        az = resolve_beam_azimuth_deg(
            {"beam_azimuth_deg": "north"}, _RX_A[0], _RX_A[1], _TX_LAT, _TX_LON
        )
        assert abs(az - _broadside(_RX_A)) < 1e-4
