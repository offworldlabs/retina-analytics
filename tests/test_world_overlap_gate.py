"""Bottom-up pairing does not cross worlds.

The ADS-B gate added a world check to seeding and claiming, but the overlap
zones pairing draws its candidates from were still built between every pair of
positioned nodes.  A synthetic fleet sharing a footprint with real hardware —
the test droplet's 50 simulated nodes over the same city as 8 receivers —
therefore kept a grid for every sim/real pair, and real node ids turned up
inside synthetic dark solves.  Two nodes in different worlds can never see the
same echo, so those grids have no true pairing to find.

Same fail-open rule as the seed gate: only two known, different worlds are
refused.
"""

from retina_analytics.association import InterNodeAssociator

# Two receivers a few km apart on the same illuminator: a real, non-empty
# overlap, so "no zone" below is the world gate and not the geometry.
_CFG_A = dict(rx_lat=34.85, rx_lon=-82.40, tx_lat=34.90, tx_lon=-82.30, max_range_km=50, max_bistatic_range_km=50)
_CFG_B = dict(rx_lat=34.86, rx_lon=-82.36, tx_lat=34.90, tx_lon=-82.30, max_range_km=50, max_bistatic_range_km=50)

_PAIR = ("a", "b")


def _register_pair(**kwargs) -> InterNodeAssociator:
    assoc = InterNodeAssociator(grid_step_km=5.0, **kwargs)
    assoc.register_node("a", dict(_CFG_A))
    assoc.register_node("b", dict(_CFG_B))
    return assoc


class TestOverlapZoneWorldGate:
    def test_cross_world_pair_gets_no_zone(self):
        worlds = {"a": "sim", "b": "real"}
        assoc = _register_pair(node_world_provider=worlds.get)

        assert _PAIR not in assoc.overlap_zones
        assert assoc.overlap_zones == {}
        assert assoc._neighbors.get("a", set()) == set()
        assert assoc._neighbors.get("b", set()) == set()
        assert assoc.assoc_world_skipped_pairs == 1

    def test_same_world_pair_is_unchanged(self):
        assoc = _register_pair(node_world_provider=lambda nid: "sim")

        assert assoc.overlap_zones[_PAIR].delay_pairs
        assert assoc._neighbors["a"] == {"b"}
        assert assoc.assoc_world_skipped_pairs == 0

    def test_one_untagged_node_still_pairs(self):
        """Fail-open: an unknown world is compatible with every world, which is
        what a node the provider has not seen yet has."""
        worlds = {"a": "real"}
        assoc = _register_pair(node_world_provider=worlds.get)

        assert assoc.overlap_zones[_PAIR].delay_pairs
        assert assoc._neighbors["a"] == {"b"}
        assert assoc.assoc_world_skipped_pairs == 0

    def test_an_empty_world_tag_is_not_a_world(self):
        worlds = {"a": "", "b": "real"}
        assoc = _register_pair(node_world_provider=worlds.get)

        assert assoc.overlap_zones[_PAIR].delay_pairs
        assert assoc.assoc_world_skipped_pairs == 0

    def test_no_provider_pairs_everything(self):
        assoc = _register_pair()

        assert assoc.overlap_zones[_PAIR].delay_pairs
        assert assoc._neighbors["a"] == {"b"}
        assert assoc.assoc_world_skipped_pairs == 0

    def test_a_node_that_changes_world_loses_its_zone_on_rebuild(self):
        """The gate has to be able to take a grid away, not only decline to
        build one: registration order and a late handshake both mean a pair can
        be compatible when it is first built and cross-world afterwards."""
        worlds = {"a": "sim", "b": "sim"}
        assoc = _register_pair(node_world_provider=worlds.get)
        assert assoc.overlap_zones[_PAIR].delay_pairs

        worlds["b"] = "real"
        rebuilt = assoc.rebuild_zones_for("b")

        assert rebuilt == 0
        assert _PAIR not in assoc.overlap_zones
        assert assoc._neighbors.get("a", set()) == set()
        assert assoc.assoc_world_skipped_pairs == 1

    def test_rebuild_keeps_same_world_zones(self):
        assoc = _register_pair(node_world_provider=lambda nid: "sim")

        rebuilt = assoc.rebuild_zones_for("b")

        assert rebuilt == 1
        assert assoc.overlap_zones[_PAIR].delay_pairs
        assert assoc._neighbors["a"] == {"b"}
        assert assoc.assoc_world_skipped_pairs == 0

    def test_a_re_registration_that_changes_world_drops_the_zone(self):
        """register_node's own drop path, which rebuild_zones_for cannot cover:
        a reconnecting node re-registers with a moved receiver."""
        worlds = {"a": "sim", "b": "sim"}
        assoc = _register_pair(node_world_provider=worlds.get)
        assert _PAIR in assoc.overlap_zones

        worlds["b"] = "real"
        moved = dict(_CFG_B, rx_lat=34.87)
        assoc.register_node("b", moved)

        assert _PAIR not in assoc.overlap_zones
        assert assoc._neighbors.get("a", set()) == set()
        assert assoc.assoc_world_skipped_pairs == 1

    def test_the_counter_resets_with_the_others(self):
        worlds = {"a": "sim", "b": "real"}
        assoc = _register_pair(node_world_provider=worlds.get)
        assert assoc.assoc_world_skipped_pairs == 1

        assoc._reset_for_tests()

        assert assoc.assoc_world_skipped_pairs == 0
