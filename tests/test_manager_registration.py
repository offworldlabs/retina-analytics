"""Behaviour of NodeAnalyticsManager.register_node for coverage review follow-ups."""

import os

from retina_analytics.constants import bearing_deg
from retina_analytics.manager import NodeAnalyticsManager

_RX_LAT, _RX_LON = 32.90, -97.00
_TX_LAT, _TX_LON = 32.78, -96.80
_CFG = dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON, max_range_km=50)


def _broadside():
    return (bearing_deg(_RX_LAT, _RX_LON, _TX_LAT, _TX_LON) + 90.0) % 360.0


def test_manager_honors_explicit_aim():
    m = NodeAnalyticsManager()
    m.register_node("N", {**_CFG, "beam_azimuth_deg": 123.0})
    assert m.detection_areas["N"].beam_azimuth_deg == 123.0


def test_manager_defaults_to_broadside_when_unaimed():
    m = NodeAnalyticsManager()
    m.register_node("N", dict(_CFG))
    assert abs(m.detection_areas["N"].beam_azimuth_deg - _broadside()) < 1e-4


def test_placeholder_zero_azimuth_is_honored_as_explicit_aim():
    # Contract guard (86caqqdah): resolve_beam_azimuth_deg treats ANY numeric
    # beam_azimuth_deg as an explicit aim — including 0.0. Cross-repo investigation
    # confirmed real configs OMIT the key for un-aimed nodes (they never send a
    # placeholder 0.0), so honoring 0.0 is safe. This pins that boundary: if a
    # future config starts emitting a schema-default 0.0 for un-aimed nodes, they
    # would be aimed due north — this asserts the honored-explicit behavior so the
    # regression surfaces here rather than silently.
    from retina_analytics.constants import resolve_beam_azimuth_deg

    az = resolve_beam_azimuth_deg({"beam_azimuth_deg": 0.0}, _RX_LAT, _RX_LON, _TX_LAT, _TX_LON)
    assert az == 0.0  # honored as an explicit aim, NOT broadside


def test_same_rx_reregistration_updates_max_range():
    m = NodeAnalyticsManager()
    m.register_node("N", {**_CFG, "max_range_km": 50})
    m.register_node("N", {**_CFG, "max_range_km": 300})  # same RX, retuned range
    ec = m.empirical_coverages["N"]
    assert ec.max_range_km == 300
    # A ~250 km detection is now within 300*2 and must be accepted (was rejected at 50).
    before = ec.n_points
    m.record_calibration_point("N", _RX_LAT + 250.0 / 111.320, _RX_LON)
    assert ec.n_points == before + 1


def _accumulate(m, node_id="N", n=25):
    for i in range(n):
        m.record_calibration_point(node_id, _RX_LAT + 0.2, _RX_LON + i * 1e-4)  # ~22 km N


def test_sub_metre_jitter_keeps_calibration():
    m = NodeAnalyticsManager()
    m.register_node("N", dict(_CFG))
    _accumulate(m)
    pts = m.empirical_coverages["N"].n_points
    assert pts >= 25
    # Reconnect with ~5 m RX jitter (4.5e-5 deg lat ≈ 5 m).
    m.register_node("N", {**_CFG, "rx_lat": _RX_LAT + 4.5e-5})
    assert m.empirical_coverages["N"].n_points == pts  # retained


def test_genuine_relocation_rebuilds_state():
    m = NodeAnalyticsManager()
    m.register_node("N", dict(_CFG))
    _accumulate(m)
    m.register_node("N", {**_CFG, "rx_lat": _RX_LAT + 0.01})  # ~1.1 km move
    assert m.empirical_coverages["N"].n_points == 0  # rebuilt


def test_recreate_on_move_removes_stale_empirical_file(tmp_path):
    storage = str(tmp_path)
    m = NodeAnalyticsManager(storage_dir=storage)
    m.register_node("N", dict(_CFG))
    _accumulate(m)
    m.save_coverage_maps()
    path = os.path.join(storage, "empirical_N.json")
    assert os.path.exists(path)
    # Node relocates > 50 m → in-memory state recreated empty.
    m.register_node("N", {**_CFG, "rx_lat": _RX_LAT + 0.01})
    assert not os.path.exists(path)  # stale old-location file removed
    # Restart: a fresh manager on the same dir must not resurrect the old polygon.
    m2 = NodeAnalyticsManager(storage_dir=storage)
    assert m2.empirical_coverages.get("N", None) is None or m2.empirical_coverages["N"].n_points == 0
