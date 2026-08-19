"""Retiring a node that has left the fleet.

Registration only ever adds.  Without a retirement path a decommissioned or
superseded receiver keeps its trust score, detection area, coverage polygon and
empirical calibration for the life of the process, its files keep being
rewritten on every save, and the associator keeps a grid for every pair it was
in.  Staging showed all three after a fleet layout change.
"""

from retina_analytics.association import InterNodeAssociator
from retina_analytics.manager import NodeAnalyticsManager

_CFG_A = dict(rx_lat=34.85, rx_lon=-82.40, tx_lat=34.90, tx_lon=-82.30, max_range_km=50, max_bistatic_range_km=50)
_CFG_B = dict(rx_lat=34.86, rx_lon=-82.36, tx_lat=34.90, tx_lon=-82.30, max_range_km=50, max_bistatic_range_km=50)


class TestManagerRetirement:
    def test_every_in_memory_store_drops_the_node(self):
        m = NodeAnalyticsManager()
        m.register_node("gone", dict(_CFG_A))
        m.register_node("stays", dict(_CFG_B))

        m.retire_node("gone")

        for store in (
            m.trust_scores,
            m.detection_areas,
            m.metrics,
            m.reputations,
            m.coverage_maps,
            m.empirical_coverages,
        ):
            assert "gone" not in store
        assert "stays" in m.trust_scores
        assert "stays" in m.detection_areas

    def test_it_reports_what_it_dropped(self):
        m = NodeAnalyticsManager()
        m.register_node("gone", dict(_CFG_A))

        report = m.retire_node("gone")

        assert report["node_id"] == "gone"
        assert report["dropped"]["trust_score"] is True
        assert report["dropped"]["detection_area"] is True

    def test_retiring_an_unknown_node_is_a_no_op(self):
        m = NodeAnalyticsManager()
        m.register_node("stays", dict(_CFG_A))

        report = m.retire_node("never-existed")

        assert all(v is False for v in report["dropped"].values())
        assert "stays" in m.trust_scores

    def test_the_files_go_too(self, tmp_path):
        """Otherwise the next boot loads the node straight back in —
        _load_coverage_maps rebuilds from whatever is on disk."""
        m = NodeAnalyticsManager(storage_dir=str(tmp_path))
        m.register_node("gone", dict(_CFG_A))
        for lat, lon in ((34.9, -82.35), (34.95, -82.32), (34.88, -82.38)):
            m.record_calibration_point("gone", lat, lon)
        m.save_coverage_maps()
        empirical = tmp_path / "empirical_gone.json"
        assert empirical.exists()

        report = m.retire_node("gone")

        assert not empirical.exists()
        assert "empirical_gone.json" in report["files_removed"]

        # And a fresh manager over the same directory does not resurrect it.
        assert "gone" not in NodeAnalyticsManager(storage_dir=str(tmp_path)).empirical_coverages

    def test_a_cached_summary_does_not_outlive_the_node(self):
        m = NodeAnalyticsManager()
        m.register_node("gone", dict(_CFG_A))
        m.register_node("stays", dict(_CFG_B))
        assert "gone" in m.get_all_summaries()

        m.retire_node("gone")

        assert "gone" not in m.get_all_summaries()


class TestAssociatorRetirement:
    def _pair(self):
        a = InterNodeAssociator(grid_step_km=5.0)
        a.register_node("gone", dict(_CFG_A))
        a.register_node("stays", dict(_CFG_B))
        return a

    def test_the_node_and_its_configs_are_dropped(self):
        a = self._pair()

        a.unregister_node("gone")

        assert "gone" not in a.node_geometries
        assert "gone" not in a.node_configs
        assert "stays" in a.node_geometries

    def test_every_zone_it_was_in_goes_with_it(self):
        a = self._pair()
        assert any("gone" in key for key in a.overlap_zones)

        removed = a.unregister_node("gone")

        assert removed >= 1
        assert not any("gone" in key for key in a.overlap_zones)

    def test_the_surviving_neighbour_forgets_it(self):
        """A dangling adjacency entry makes every round walk a node that is
        no longer there."""
        a = self._pair()

        a.unregister_node("gone")

        assert "gone" not in a._neighbors.get("stays", set())
        assert "gone" not in a._neighbors

    def test_pending_submissions_are_discarded(self):
        a = self._pair()
        a._pending_frames["gone"] = {"detections": []}
        a._pending_tracks["gone"] = []

        a.unregister_node("gone")

        assert "gone" not in a._pending_frames
        assert "gone" not in a._pending_tracks

    def test_rate_limit_and_cursor_state_are_discarded(self):
        """Fleet regenerations reuse node ids: a stale _last_assoc suppressed
        the replacement node's first round, and a stale cursor started its
        neighbour rotation at a leftover offset."""
        a = self._pair()
        a._last_assoc["gone"] = 1e12
        a._neighbor_cursor["gone"] = 7

        a.unregister_node("gone")

        assert "gone" not in a._last_assoc
        assert "gone" not in a._neighbor_cursor

    def test_unregistering_an_unknown_node_is_a_no_op(self):
        a = self._pair()
        before = dict(a.overlap_zones)

        assert a.unregister_node("never-existed") == 0
        assert a.overlap_zones == before
