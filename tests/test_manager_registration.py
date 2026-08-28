"""Behaviour of NodeAnalyticsManager.register_node for coverage review follow-ups."""

import os

import pytest

from retina_analytics.constants import KM_PER_DEG_LAT, bearing_deg
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
    m.record_calibration_point("N", _RX_LAT + 250.0 / KM_PER_DEG_LAT, _RX_LON)
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


# ── Coverage invalidation when the range rule changes ────────────────────────


def test_switching_to_bistatic_rebuilds_coverage():
    """A polygon accumulated under a monostatic limit describes a circle on the
    RX; under a bistatic limit the footprint is an ellipse with foci at RX and
    TX.  Keeping the old points would serve a shape the node no longer has."""
    m = NodeAnalyticsManager()
    m.register_node("N", dict(_CFG))
    _accumulate(m)
    assert m.empirical_coverages["N"].n_points >= 25
    m.register_node("N", {**_CFG, "max_bistatic_range_km": 60})
    assert m.empirical_coverages["N"].n_points == 0


def test_stable_bistatic_config_keeps_coverage():
    """Re-registering with the same rule must not throw away calibration —
    nodes reconnect routinely."""
    cfg = {**_CFG, "max_bistatic_range_km": 60}
    m = NodeAnalyticsManager()
    m.register_node("N", cfg)
    _accumulate(m)
    pts = m.empirical_coverages["N"].n_points
    assert pts >= 25
    m.register_node("N", dict(cfg))
    assert m.empirical_coverages["N"].n_points == pts


def test_monostatic_node_is_untouched_by_the_new_key():
    """Real hardware carries only max_range_km. Its coverage must survive
    reconnects exactly as before."""
    m = NodeAnalyticsManager()
    m.register_node("N", dict(_CFG))
    _accumulate(m)
    pts = m.empirical_coverages["N"].n_points
    m.register_node("N", dict(_CFG))
    assert m.empirical_coverages["N"].n_points == pts


def test_max_range_retune_still_keeps_calibration():
    """Guard the pre-existing decision: a retuned clamp is not a shape change,
    so accumulated points are deliberately retained."""
    m = NodeAnalyticsManager()
    m.register_node("N", {**_CFG, "max_range_km": 50})
    _accumulate(m)
    pts = m.empirical_coverages["N"].n_points
    m.register_node("N", {**_CFG, "max_range_km": 300})
    assert m.empirical_coverages["N"].n_points == pts
    assert m.empirical_coverages["N"].max_range_km == 300


def test_rule_change_removes_the_stale_on_disk_polygon(tmp_path):
    storage = str(tmp_path)
    m = NodeAnalyticsManager(storage_dir=storage)
    m.register_node("N", dict(_CFG))
    _accumulate(m)
    m.save_coverage_maps()
    path = os.path.join(storage, "empirical_N.json")
    assert os.path.exists(path)
    # Production mounts coverage_data as a named volume that survives rebuilds,
    # so leaving the file would resurrect the stale polygon on restart.
    m.register_node("N", {**_CFG, "max_bistatic_range_km": 60})
    assert not os.path.exists(path)


def test_older_calibration_schema_rebuilds_coverage():
    """A polygon built from solver positions is discarded, not served.

    Nothing configured changes when the calibration feed switches from solver
    output to ADS-B fixes, so the bistatic-rule key cannot catch it — and both
    staging and production mount coverage_data as a named volume that survives
    rebuilds.  Without this the ghost-shaped polygon would outlive every deploy.
    """
    from retina_analytics.empirical_coverage import CALIBRATION_SCHEMA

    m = NodeAnalyticsManager()
    m.register_node("N", dict(_CFG))
    ec = m.empirical_coverages["N"]
    for i in range(30):
        ec.add_point(_RX_LAT + 0.05 + i * 1e-4, _RX_LON + 0.05)
    assert ec.n_points == 30

    ec.schema = CALIBRATION_SCHEMA - 1  # as if loaded from an older file
    m.register_node("N", dict(_CFG))  # same RX, same range rule

    assert m.empirical_coverages["N"].n_points == 0
    assert m.empirical_coverages["N"].schema == CALIBRATION_SCHEMA


def test_current_schema_keeps_calibration():
    """The version check must not throw away every polygon on every restart."""
    m = NodeAnalyticsManager()
    m.register_node("N", dict(_CFG))
    ec = m.empirical_coverages["N"]
    for i in range(30):
        ec.add_point(_RX_LAT + 0.05 + i * 1e-4, _RX_LON + 0.05)

    m.register_node("N", dict(_CFG))
    assert m.empirical_coverages["N"].n_points == 30


def test_schema_survives_a_save_load_round_trip(tmp_path):
    """A file written before the field existed reads back as v1."""
    from retina_analytics.empirical_coverage import (
        CALIBRATION_SCHEMA,
        EmpiricalCoverageState,
    )

    ec = EmpiricalCoverageState(rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=50)
    path = str(tmp_path / "e.json")
    ec.save_to_file(path)
    assert EmpiricalCoverageState.load_from_file(path).schema == CALIBRATION_SCHEMA

    import json

    with open(path) as f:
        d = json.load(f)
    d.pop("schema")
    with open(path, "w") as f:
        json.dump(d, f)
    assert EmpiricalCoverageState.load_from_file(path).schema == 1


# ── Registration without geometry ────────────────────────────────────────────

_POSITIONED = {
    "rx_lat": 34.85,
    "rx_lon": -82.39,
    "rx_alt_ft": 900.0,
    "tx_lat": 34.90,
    "tx_lon": -82.45,
    "tx_alt_ft": 1200.0,
    "fc_hz": 195e6,
    "beam_width_deg": None,
    "beam_azimuth_deg": None,
}


def _cfg(**overrides):
    return {**_POSITIONED, **overrides}


@pytest.mark.parametrize(
    "overrides",
    [
        {"rx_lat": None, "rx_lon": None},
        {"tx_lat": None, "tx_lon": None},
        {"rx_lat": None, "rx_lon": None, "tx_lat": None, "tx_lon": None},
    ],
    ids=["no-rx", "no-tx", "neither"],
)
def test_registration_without_geometry_does_not_raise(overrides):
    m = NodeAnalyticsManager()
    m.register_node("n1", _cfg(**overrides))
    assert "n1" in m.metrics
    assert "n1" in m.trust_scores
    assert "n1" in m.reputations
    assert "n1" not in m.detection_areas
    assert "n1" not in m.empirical_coverages


def test_positionless_summary_omits_detection_area():
    m = NodeAnalyticsManager()
    m.register_node("n1", _cfg(rx_lat=None, rx_lon=None))
    summary = m.get_node_summary("n1")
    assert "detection_area" not in summary
    assert "metrics" in summary


def test_null_altitude_still_positions_the_node():
    m = NodeAnalyticsManager()
    m.register_node("n1", _cfg(rx_alt_ft=None, tx_alt_ft=None))
    assert "n1" in m.detection_areas


def test_frames_are_counted_for_a_positionless_node():
    m = NodeAnalyticsManager()
    m.register_node("n1", _cfg(rx_lat=None, rx_lon=None))
    assert m.record_detection_frame("n1", {"timestamp": 1.0, "detections": []}) is True
    assert m.metrics["n1"].total_frames == 1


def test_losing_geometry_drops_a_stale_detection_area():
    m = NodeAnalyticsManager()
    m.register_node("n1", _cfg())
    assert "n1" in m.detection_areas
    m.register_node("n1", _cfg(rx_lat=None, rx_lon=None))
    assert "n1" not in m.detection_areas
    assert "n1" in m.empirical_coverages  # retained, not popped: see the guard's comment


def test_losing_geometry_stops_publishing_empirical_coverage():
    """The empirical_coverages entry survives the loss of geometry (see
    above), but with no detection area there is no beam or range left to
    constrain its polygon, so the summary must not publish one."""
    m = NodeAnalyticsManager()
    m.register_node("n1", _cfg())
    m.register_node("n1", _cfg(rx_lat=None, rx_lon=None))
    assert "n1" in m.empirical_coverages
    summary = m.get_node_summary("n1")
    assert "empirical_coverage" not in summary


def test_reregistration_that_loses_geometry_invalidates_the_summary_cache():
    m = NodeAnalyticsManager()
    m.register_node("n1", _cfg())
    assert "detection_area" in m.get_all_summaries()["n1"]

    m.register_node("n1", _cfg(rx_lat=None, rx_lon=None))

    assert "detection_area" not in m.get_all_summaries()["n1"]


def test_reregistration_that_gains_geometry_invalidates_the_summary_cache():
    m = NodeAnalyticsManager()
    m.register_node("n1", _cfg(rx_lat=None, rx_lon=None))
    assert "detection_area" not in m.get_all_summaries()["n1"]

    m.register_node("n1", _cfg())

    assert "detection_area" in m.get_all_summaries()["n1"]
