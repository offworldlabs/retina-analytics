"""A node registered without a receiver position takes no part in overlap.

`NodeGeometry` defaults rx and tx to (0, 0) with a nominal range and beam, so
every node registered from a config that carries no position lands on the same
footprint off the west coast of Africa.  Each such node then overlaps every
other one completely, which is wrong twice over: the pairing is fictitious, and
a fully-overlapping pair produces the densest grid the geometry can yield, so
it is also the most expensive one to compute.

Registering the fleet that way turns the neighbour graph complete, which is
what the multinode solver then sees as a 14-node candidate.
"""

import pytest

from retina_analytics.association import InterNodeAssociator

POSITIONED_A = {
    "rx_lat": 33.939, "rx_lon": -84.651, "rx_alt_ft": 950,
    "tx_lat": 33.756, "tx_lon": -84.331, "tx_alt_ft": 1600,
    "fc_hz": 195e6, "beam_width_deg": 41, "max_range_km": 50,
}
# Close enough to A to genuinely share sky, so the positioned-pair assertions
# are testing the guard rather than a geographic miss.
POSITIONED_B = {
    "rx_lat": 34.05, "rx_lon": -84.4, "rx_alt_ft": 980,
    "tx_lat": 33.85, "tx_lon": -84.15, "tx_alt_ft": 1600,
    "fc_hz": 195e6, "beam_width_deg": 41, "max_range_km": 50,
}
# What routes/radar.py sends for a node first seen over POST /api/radar/detections.
BARE = {"node_id": "whatever"}


@pytest.fixture
def assoc():
    return InterNodeAssociator(grid_step_km=5.0)


class TestUnpositionedRegistration:
    def test_positioned_pair_still_overlaps(self, assoc):
        """Regression guard: the fix must not silence real nodes."""
        assoc.register_node("A", POSITIONED_A)
        assoc.register_node("B", POSITIONED_B)

        assert assoc.overlap_zones[("A", "B")].delay_pairs
        assert "B" in assoc._neighbors.get("A", set())

    def test_unpositioned_node_builds_no_zone_against_a_positioned_one(self, assoc):
        assoc.register_node("A", POSITIONED_A)
        assoc.register_node("bare", BARE)

        assert ("A", "bare") not in assoc.overlap_zones

    def test_a_positioned_node_registered_later_skips_unpositioned_peers(self, assoc):
        """Order must not matter — the guard has to hold on both sides of the loop."""
        assoc.register_node("bare", BARE)
        assoc.register_node("A", POSITIONED_A)

        assert ("A", "bare") not in assoc.overlap_zones

    def test_two_unpositioned_nodes_are_not_neighbours(self, assoc):
        """The degenerate case: both sit on (0, 0), so they overlap perfectly."""
        assoc.register_node("bare-1", BARE)
        assoc.register_node("bare-2", BARE)

        assert assoc._neighbors.get("bare-1", set()) == set()
        assert assoc._neighbors.get("bare-2", set()) == set()

    def test_a_fleet_of_unpositioned_nodes_stays_a_null_graph(self, assoc):
        """What production actually hit: N config-less nodes, N*(N-1)/2 dense grids."""
        for i in range(20):
            assoc.register_node(f"bare-{i}", BARE)

        assert assoc.overlap_zones == {}
        assert all(not peers for peers in assoc._neighbors.values())

    def test_the_node_is_still_registered(self, assoc):
        """Skipping overlap is not the same as refusing the node: it still streams
        frames, and single-node tracking does not need a peer."""
        assoc.register_node("bare", BARE)

        assert "bare" in assoc.node_geometries
        assert "bare" in assoc.node_configs

    def test_supplying_a_position_later_builds_the_zones(self, assoc):
        """The upgrade path: a node that registers bare and re-registers with real
        geometry must become a full participant."""
        assoc.register_node("A", POSITIONED_A)
        assoc.register_node("late", BARE)
        assert ("A", "late") not in assoc.overlap_zones

        assoc.register_node("late", POSITIONED_B)

        assert assoc.overlap_zones[("A", "late")].delay_pairs
        assert "late" in assoc._neighbors.get("A", set())

    def test_rebuild_skips_unpositioned_nodes(self, assoc):
        """rebuild_zones_for walks every geometry too, so it needs the same guard —
        otherwise the background cadence puts the dense grids straight back."""
        assoc.register_node("A", POSITIONED_A)
        assoc.register_node("B", POSITIONED_B)
        assoc.register_node("bare", BARE)

        rebuilt = assoc.rebuild_zones_for("A")

        assert ("A", "bare") not in assoc.overlap_zones
        assert rebuilt == 1  # B only

    def test_rebuilding_an_unpositioned_node_is_a_no_op(self, assoc):
        assoc.register_node("A", POSITIONED_A)
        assoc.register_node("bare", BARE)

        assert assoc.rebuild_zones_for("bare") == 0
        assert assoc.overlap_zones == {}


class TestPositionPredicate:
    """(0, 0) is the sentinel the defaults produce; a real node never sits there.

    The equator and the prime meridian are each perfectly valid on their own,
    so only the exact pair is treated as absent.
    """

    def test_a_node_on_the_equator_is_positioned(self, assoc):
        cfg = dict(POSITIONED_A, rx_lat=0.0)
        assoc.register_node("A", POSITIONED_A)
        assoc.register_node("equator", cfg)

        assert ("A", "equator") in assoc.overlap_zones

    def test_a_node_on_the_prime_meridian_is_positioned(self, assoc):
        cfg = dict(POSITIONED_A, rx_lon=0.0)
        assoc.register_node("A", POSITIONED_A)
        assoc.register_node("meridian", cfg)

        assert ("A", "meridian") in assoc.overlap_zones

    def test_an_explicit_null_position_counts_as_absent(self, assoc):
        """A v1 registration may carry the keys with no value."""
        cfg = dict(POSITIONED_A, rx_lat=None, rx_lon=None)
        assoc.register_node("A", POSITIONED_A)
        assoc.register_node("null", cfg)

        assert ("A", "null") not in assoc.overlap_zones
