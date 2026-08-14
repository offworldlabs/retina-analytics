"""A null beam_width_deg resolves to the nominal Yagi width (86cb5dh4d).

beam_width_deg is nullable on the node wire contract, and the backend builds a
node's config field by field, so an unset width arrives present and None. The
registries previously read it with dict.get plus a default, which only fires on
an *absent* key, so the None reached the half-width divisions and raised
TypeError.

The associator regression needs TWO nodes: the crash sits in the pairwise
overlap-zone precompute, which never runs for a lone node, so a single-node
test passes against the unfixed code.
"""

import math

from retina_analytics.association import InterNodeAssociator, _bearing_deg
from retina_analytics.constants import YAGI_BEAM_WIDTH_DEG, resolve_beam_width_deg
from retina_analytics.manager import NodeAnalyticsManager

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

# No beam_width_deg here: each test supplies its own, null or otherwise.
_COMMON = dict(
    rx_alt_ft=500, tx_lat=_TX_LAT, tx_lon=_TX_LON, tx_alt_ft=1600,
    fc_hz=195e6, max_range_km=60,
)

_MGR_RX_LAT, _MGR_RX_LON = 32.90, -97.00
_MGR_CFG = dict(
    rx_lat=_MGR_RX_LAT, rx_lon=_MGR_RX_LON,
    tx_lat=_TX_LAT, tx_lon=_TX_LON, max_range_km=50,
)


class TestResolveBeamWidth:
    def test_missing_falls_back_to_nominal(self):
        assert resolve_beam_width_deg({}) == YAGI_BEAM_WIDTH_DEG

    def test_none_falls_back_to_nominal(self):
        assert resolve_beam_width_deg({"beam_width_deg": None}) == YAGI_BEAM_WIDTH_DEG

    def test_explicit_width_kept(self):
        assert resolve_beam_width_deg({"beam_width_deg": 50.0}) == 50.0

    def test_integer_width_is_coerced_to_float(self):
        width = resolve_beam_width_deg({"beam_width_deg": 50})
        assert isinstance(width, float)
        assert width == 50.0

    def test_unparseable_falls_back_to_nominal(self):
        assert resolve_beam_width_deg({"beam_width_deg": "wide"}) == YAGI_BEAM_WIDTH_DEG

    def test_nan_falls_back_to_nominal(self):
        width = resolve_beam_width_deg({"beam_width_deg": float("nan")})
        assert math.isfinite(width)
        assert width == YAGI_BEAM_WIDTH_DEG

    def test_inf_falls_back_to_nominal(self):
        width = resolve_beam_width_deg({"beam_width_deg": float("inf")})
        assert math.isfinite(width)
        assert width == YAGI_BEAM_WIDTH_DEG


class TestAssociatorNullWidthRegistration:
    def test_two_null_width_nodes_register_without_raising(self):
        """The regression proper: the second registration runs the pairwise
        overlap grid, which divides by the beam width for every grid point."""
        assoc = InterNodeAssociator(grid_step_km=3.0)
        assoc.register_node("RA", {
            **_COMMON, "rx_lat": _RX_A[0], "rx_lon": _RX_A[1],
            "beam_width_deg": None,
            "beam_azimuth_deg": _bearing_deg(_RX_A[0], _RX_A[1], _CORE_LAT, _CORE_LON),
        })
        assoc.register_node("RB", {
            **_COMMON, "rx_lat": _RX_B[0], "rx_lon": _RX_B[1],
            "beam_width_deg": None,
            "beam_azimuth_deg": _bearing_deg(_RX_B[0], _RX_B[1], _CORE_LAT, _CORE_LON),
        })
        assert assoc.node_geometries["RA"].beam_width_deg == YAGI_BEAM_WIDTH_DEG
        assert assoc.node_geometries["RB"].beam_width_deg == YAGI_BEAM_WIDTH_DEG
        # Both are aimed at the shared core, so the zone must be a real overlap
        # rather than an empty one the crash could hide behind.
        assert assoc.overlap_zones[tuple(sorted(["RA", "RB"]))].delay_pairs

    def test_explicit_width_still_kept(self):
        assoc = InterNodeAssociator(grid_step_km=10.0)
        assoc.register_node("X", {
            **_COMMON, "rx_lat": _RX_A[0], "rx_lon": _RX_A[1],
            "beam_width_deg": 50,
        })
        assert assoc.node_geometries["X"].beam_width_deg == 50.0


class TestManagerNullWidthRegistration:
    def test_null_width_stores_nominal(self):
        m = NodeAnalyticsManager()
        m.register_node("N", {**_MGR_CFG, "beam_width_deg": None})
        assert m.detection_areas["N"].beam_width_deg == YAGI_BEAM_WIDTH_DEG

    def test_missing_width_stores_nominal(self):
        m = NodeAnalyticsManager()
        m.register_node("N", dict(_MGR_CFG))
        assert m.detection_areas["N"].beam_width_deg == YAGI_BEAM_WIDTH_DEG

    def test_explicit_width_still_kept(self):
        m = NodeAnalyticsManager()
        m.register_node("N", {**_MGR_CFG, "beam_width_deg": 50})
        assert m.detection_areas["N"].beam_width_deg == 50.0
