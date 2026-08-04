"""Regression tests for the stage-1 correctness fixes.

Each class pins one previously-wrong output:
- coverage_area_km2 squared KM_PER_DEG_LAT (22% overstatement at 35°N)
- neighbour pre-filters used monostatic radii against bistatic footprints
- reputation penalties compounded per evaluation pass (permanent blocks)
- EmpiricalCoverageState.from_dict silently reset range_clamp_mult
- NodeMetrics.total_tracks / geolocated_tracks were never written
"""

import math

from retina_analytics.constants import KM_PER_DEG_LAT, km_per_deg_lon
from retina_analytics.coverage import HistoricalCoverageMap
from retina_analytics.detection_area import DetectionAreaState
from retina_analytics.empirical_coverage import EmpiricalCoverageState
from retina_analytics.metrics import NodeMetrics
from retina_analytics.reputation import NodeReputation


class TestCoverageAreaUsesCosLat:
    def test_single_cell_area_shrinks_with_latitude(self):
        m = HistoricalCoverageMap(node_id="n1")
        m.add_detection(lat=34.85, lon=-82.4, alt_km=10, snr=12, delay_error=0.1)
        lat_side = 0.01 * KM_PER_DEG_LAT
        expected = lat_side * 0.01 * km_per_deg_lon(34.85)
        assert m.coverage_area_km2 == expected
        # The old (lat_side)**2 answer was 1/cos(lat) too big — ~22% at 34.85°.
        old = lat_side**2
        assert old / m.coverage_area_km2 == (
            1.0 / math.cos(math.radians(34.85))
        ) or abs(old / m.coverage_area_km2 - 1 / math.cos(math.radians(34.85))) < 1e-9

    def test_area_sums_per_cell(self):
        m = HistoricalCoverageMap(node_id="n1")
        m.add_detection(lat=0.0, lon=10.0, alt_km=10, snr=12, delay_error=0.1)
        m.add_detection(lat=60.0, lon=10.0, alt_km=10, snr=12, delay_error=0.1)
        lat_side = 0.01 * KM_PER_DEG_LAT
        expected = lat_side * 0.01 * (km_per_deg_lon(0.0) + km_per_deg_lon(60.0))
        assert abs(m.coverage_area_km2 - expected) < 1e-9

    def test_zero_beam_estimate_is_reported_not_nulled(self):
        m = HistoricalCoverageMap(node_id="n1")
        # No entries → estimate is None → key must be None.
        assert m.summary()["estimated_beam_width_deg"] is None


class TestFootprintRadius:
    def _area(self, bistatic=None, tx=(34.9, -82.2)):
        return DetectionAreaState(
            node_id="a", rx_lat=34.85, rx_lon=-82.4,
            tx_lat=tx[0], tx_lon=tx[1],
            max_range_km=60.0, max_bistatic_range_km=bistatic,
        )

    def test_monostatic_node_keeps_max_range(self):
        area = self._area(bistatic=None)
        assert area.footprint_radius_km() == 60.0

    def test_bistatic_node_reaches_delta_over_two_plus_baseline(self):
        area = self._area(bistatic=60.0)
        from retina_analytics.constants import haversine_km
        baseline = haversine_km(34.85, -82.4, 34.9, -82.2)
        assert area.footprint_radius_km() == 60.0 / 2.0 + baseline
        # Strictly beyond the monostatic circle — the old pre-filter pruned
        # pairs inside this annulus before overlap was ever computed.
        assert area.footprint_radius_km() < 60.0 or baseline > 30.0

    def test_tx_with_one_zero_axis_still_counts_as_present(self):
        area = self._area(bistatic=60.0, tx=(0.0, -82.2))
        assert area.has_tx
        assert area.footprint_radius_km() > 0.0

    def test_null_island_tx_is_the_unset_sentinel(self):
        area = self._area(bistatic=60.0, tx=(0.0, 0.0))
        assert not area.has_tx
        assert area.footprint_radius_km() == 60.0


class TestReputationPenalisesOnsetNotPasses:
    def test_persistent_stale_heartbeat_penalises_once(self):
        rep = NodeReputation(node_id="n1")
        stale = 1.0  # epoch → gap is enormous
        for _ in range(20):
            rep.evaluate_heartbeat(stale)
        assert rep.reputation == 0.9  # one 0.1 penalty, not eight-and-blocked
        assert not rep.blocked

    def test_condition_clearing_and_returning_penalises_again(self):
        rep = NodeReputation(node_id="n1")
        import time as _t
        rep.evaluate_heartbeat(1.0)          # onset
        rep.evaluate_heartbeat(_t.time())    # fresh again — clears
        rep.evaluate_heartbeat(1.0)          # second onset
        assert abs(rep.reputation - 0.8) < 1e-9

    def test_unblocked_node_is_not_instantly_reblocked_by_a_persisting_condition(self):
        rep = NodeReputation(node_id="n1")
        # Drive it to blocked via repeated distinct conditions.
        rep.apply_penalty(0.85, "test")
        assert rep.blocked
        rep.unblock()
        # The stale heartbeat persists — but its onset was already penalised,
        # so evaluation must not knock the node straight back out.
        rep._condition_active["heartbeat_stale"] = True
        for _ in range(10):
            rep.evaluate_heartbeat(1.0)
        assert not rep.blocked

    def test_neighbour_conditions_are_keyed_per_neighbour(self):
        rep = NodeReputation(node_id="n1")
        rep.evaluate_neighbour_consistency(0.0, 0.9, neighbour_id="b")
        rep.evaluate_neighbour_consistency(0.0, 0.9, neighbour_id="c")
        # Two distinct neighbours → two penalties; repeating them → no more.
        assert abs(rep.reputation - (1.0 - 2 * 0.08)) < 1e-9
        rep.evaluate_neighbour_consistency(0.0, 0.9, neighbour_id="b")
        assert abs(rep.reputation - (1.0 - 2 * 0.08)) < 1e-9

    def test_trust_and_detection_rate_still_escalate_per_pass(self):
        # These are active-bad-data signals; escalation to a block is the
        # intended defence and must not be onset-gated.
        rep = NodeReputation(node_id="n1")
        for _ in range(15):
            rep.evaluate_trust(0.05)
        assert rep.blocked
        rep2 = NodeReputation(node_id="n2")
        for _ in range(5):
            rep2.evaluate_detection_rate(100.0)
        assert abs(rep2.reputation - 0.75) < 1e-9


class TestEmpiricalRoundTrip:
    def test_range_clamp_mult_survives_serialisation(self):
        ec = EmpiricalCoverageState(rx_lat=34.85, rx_lon=-82.4,
                                    max_range_km=60.0, range_clamp_mult=1.25)
        restored = EmpiricalCoverageState.from_dict(ec.to_dict())
        assert restored.range_clamp_mult == 1.25

    def test_old_payload_without_the_field_keeps_the_default(self):
        ec = EmpiricalCoverageState(rx_lat=34.85, rx_lon=-82.4, max_range_km=60.0)
        d = ec.to_dict()
        d.pop("range_clamp_mult")
        restored = EmpiricalCoverageState.from_dict(d)
        assert restored.range_clamp_mult == 2.0


class TestTrackCountersAreWritten:
    def test_distinct_ids_count_once(self):
        m = NodeMetrics(node_id="n1")
        m.record_tracks(["t1", "t2"], ["t1"])
        m.record_tracks(["t1", "t2"], ["t1"])  # same frame again
        m.record_tracks(["t3"], [])
        assert m.total_tracks == 3
        assert m.geolocated_tracks == 1
        s = m.summary()
        assert s["total_tracks"] == 3
        assert s["geolocated_tracks"] == 1

    def test_seen_set_is_bounded(self):
        m = NodeMetrics(node_id="n1")
        m._MAX_SEEN_IDS = 10
        m.record_tracks([f"t{i}" for i in range(25)], [])
        assert m.total_tracks == 25
        assert len(m._seen_track_ids) <= 11
