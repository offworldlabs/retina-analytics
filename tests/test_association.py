"""Tests for inter-node association logic."""

from retina_analytics.association import (
    NodeGeometry, compute_overlap_zone, InterNodeAssociator, _bistatic_delay_at, _lla_to_enu,
)


# ── Overlap zone & bistatic delay ────────────────────────────────────────────


class TestOverlapZone:
    def test_overlap_zone_ids(self):
        geo_a = NodeGeometry(
            node_id="assoc-A", rx_lat=33.939, rx_lon=-84.651, rx_alt_km=0.29,
            tx_lat=33.756, tx_lon=-84.331, tx_alt_km=0.49,
            beam_azimuth_deg=135, beam_width_deg=41, max_range_km=50,
        )
        geo_b = NodeGeometry(
            node_id="assoc-B", rx_lat=34.05, rx_lon=-84.4, rx_alt_km=0.3,
            tx_lat=33.85, tx_lon=-84.15, tx_alt_km=0.5,
            beam_azimuth_deg=210, beam_width_deg=41, max_range_km=50,
        )
        zone = compute_overlap_zone(geo_a, geo_b, grid_step_km=5.0)
        assert zone.node_a_id == "assoc-A"
        assert len(zone.delay_pairs) == len(zone.grid_points)

    def test_bistatic_delay_positive(self):
        ref_lat, ref_lon = 33.9, -84.5
        tx_enu = _lla_to_enu(33.756, -84.331, 0.49, ref_lat, ref_lon, 0.0)
        target_enu = (10.0, 10.0, 8.0)
        delay = _bistatic_delay_at(target_enu, tx_enu)
        assert delay > 0
        assert delay < 300


# ── InterNodeAssociator ──────────────────────────────────────────────────────


class TestInterNodeAssociator:
    def _make_assoc(self):
        assoc = InterNodeAssociator(grid_step_km=5.0)
        assoc.register_node("assoc-A", {
            "rx_lat": 33.939, "rx_lon": -84.651, "rx_alt_ft": 950,
            "tx_lat": 33.756, "tx_lon": -84.331, "tx_alt_ft": 1600,
            "fc_hz": 195e6, "beam_width_deg": 41, "max_range_km": 50,
        })
        assoc.register_node("assoc-B", {
            "rx_lat": 34.05, "rx_lon": -84.4, "rx_alt_ft": 980,
            "tx_lat": 33.85, "tx_lon": -84.15, "tx_alt_ft": 1600,
            "fc_hz": 195e6, "beam_width_deg": 41, "max_range_km": 50,
        })
        return assoc

    def test_register_two_nodes(self):
        assoc = self._make_assoc()
        assert len(assoc.node_geometries) == 2

    def test_overlap_summary(self):
        assoc = self._make_assoc()
        summary = assoc.get_overlap_summary()
        assert isinstance(summary, list)

    def test_beam_width_in_geometry(self):
        assoc = self._make_assoc()
        assert assoc.node_geometries["assoc-A"].beam_width_deg == 41

    def test_submit_frame_returns_list(self):
        assoc = self._make_assoc()
        candidates = assoc.submit_frame("assoc-A", {
            "delay": [30.0, 45.0], "doppler": [60.0, -20.0], "snr": [15.0, 10.0],
        }, timestamp_ms=1000)
        assert isinstance(candidates, list)

    def test_submit_both_frames(self):
        assoc = self._make_assoc()
        assoc.submit_frame("assoc-A", {
            "delay": [30.0, 45.0], "doppler": [60.0, -20.0], "snr": [15.0, 10.0],
        }, timestamp_ms=1000)
        candidates = assoc.submit_frame("assoc-B", {
            "delay": [31.0, 46.0], "doppler": [58.0, -22.0], "snr": [14.0, 9.0],
        }, timestamp_ms=1000)
        assert isinstance(candidates, list)

    def test_solver_format(self):
        assoc = self._make_assoc()
        assoc.submit_frame("assoc-A", {
            "delay": [30.0, 45.0], "doppler": [60.0, -20.0], "snr": [15.0, 10.0],
        }, timestamp_ms=1000)
        candidates = assoc.submit_frame("assoc-B", {
            "delay": [31.0, 46.0], "doppler": [58.0, -22.0], "snr": [14.0, 9.0],
        }, timestamp_ms=1000)
        solver_input = assoc.format_candidates_for_solver(candidates)
        assert isinstance(solver_input, list)

    def test_format_candidates_merges_cross_zone_pairs(self):
        """Candidates from pairs (A-B) and (A-C) near the same aircraft must be
        merged into one solver input with n_nodes == 3, not two separate 2-node
        inputs.  Proximity-based clustering (6 km threshold) ensures grid points
        from different overlap zones that are < 6 km apart collapse into one group.
        """
        from retina_analytics.association import AssociationCandidate

        # Simulate two candidates from pairs (A,B) and (A,C) both pointing at
        # roughly the same aircraft.  Grid positions differ by ~2 km (within one
        # grid step) — as expected when different overlap zones triangulate the
        # same target.
        base_lat, base_lon = 33.90, -84.60
        offset = 0.018  # ~2 km in latitude

        c_ab = AssociationCandidate(
            node_a_id="A", node_b_id="B",
            det_a_idx=0, det_b_idx=0,
            delay_a=30.0, delay_b=31.0,
            doppler_a=10.0, doppler_b=11.0,
            snr_a=15.0, snr_b=14.0,
            grid_delay_a=30.1, grid_delay_b=31.1,
            grid_lat=base_lat, grid_lon=base_lon, grid_alt_km=10.0,
            timestamp_ms=1000,
        )
        c_ac = AssociationCandidate(
            node_a_id="A", node_b_id="C",
            det_a_idx=0, det_b_idx=0,
            delay_a=30.1, delay_b=32.0,
            doppler_a=10.5, doppler_b=12.0,
            snr_a=13.0, snr_b=12.0,
            grid_delay_a=30.2, grid_delay_b=32.1,
            grid_lat=base_lat + offset, grid_lon=base_lon + offset, grid_alt_km=10.0,
            timestamp_ms=1000,
        )

        assoc = InterNodeAssociator(grid_step_km=3.0)
        solver_inputs = assoc.format_candidates_for_solver([c_ab, c_ac])

        # Both candidates fall in the same 0.05° bin → one merged group
        assert len(solver_inputs) == 1, (
            f"Expected 1 merged solver input (n_nodes=3), got {len(solver_inputs)}: "
            f"{[s['n_nodes'] for s in solver_inputs]}"
        )
        assert solver_inputs[0]["n_nodes"] == 3, (
            f"Expected n_nodes=3 (A, B, C merged), got {solver_inputs[0]['n_nodes']}"
        )

    def test_ghost_filter_rejects_clutter_pairing(self):
        """Clutter × real pairing (adsb[i_a]=None) must be rejected."""
        from retina_analytics.association import OverlapZone, find_associations

        zone = OverlapZone(
            node_a_id="A", node_b_id="B",
            grid_points=[(33.9, -84.5, 9.0)],
            delay_pairs=[(10.0, 20.0)],
            delay_gate_us=5.0,
            doppler_gate_hz=30.0,
        )
        frame_a = {"delay": [9.5], "doppler": [0.0], "snr": [10.0],
                   "adsb": [None]}  # i_a=0 is clutter
        frame_b = {"delay": [20.5], "doppler": [0.0], "snr": [10.0],
                   "adsb": [{"hex": "abc123", "alt_baro": 35000, "lat": 33.9, "lon": -84.5}]}
        candidates = find_associations(zone, frame_a, frame_b, timestamp_ms=1000)
        assert len(candidates) == 0, "Clutter×real ghost should be rejected"

    def test_ghost_filter_accepts_real_real_pairing(self):
        """Both detections have valid ADS-B → pairing accepted with ADS-B altitude."""
        from retina_analytics.association import OverlapZone, find_associations

        zone = OverlapZone(
            node_a_id="A", node_b_id="B",
            grid_points=[(33.9, -84.5, 9.0)],
            delay_pairs=[(10.0, 20.0)],
            delay_gate_us=5.0,
            doppler_gate_hz=30.0,
        )
        ac = {"hex": "abc123", "alt_baro": 35000, "lat": 33.9, "lon": -84.5}
        frame_a = {"delay": [9.5], "doppler": [0.0], "snr": [10.0], "adsb": [ac]}
        frame_b = {"delay": [20.5], "doppler": [0.0], "snr": [10.0], "adsb": [ac]}
        candidates = find_associations(zone, frame_a, frame_b, timestamp_ms=1000)
        assert len(candidates) == 1, "Real×real pairing should be accepted"
        # ADS-B altitude (35000 ft → 10.668 km) should override the 9.0 km grid layer
        expected_alt_km = 35000 * 0.3048 / 1000
        assert abs(candidates[0].grid_alt_km - expected_alt_km) < 0.01

    def test_ghost_filter_no_adsb_passes_through(self):
        """No adsb in either frame → ghost filter is a no-op (non-ADS-B aircraft)."""
        from retina_analytics.association import OverlapZone, find_associations

        zone = OverlapZone(
            node_a_id="A", node_b_id="B",
            grid_points=[(33.9, -84.5, 9.0)],
            delay_pairs=[(10.0, 20.0)],
            delay_gate_us=5.0,
            doppler_gate_hz=30.0,
        )
        frame_a = {"delay": [9.5], "doppler": [0.0], "snr": [10.0]}  # no adsb field
        frame_b = {"delay": [20.5], "doppler": [0.0], "snr": [10.0]}
        candidates = find_associations(zone, frame_a, frame_b, timestamp_ms=1000)
        assert len(candidates) == 1, "No-ADS-B aircraft should pass through"
        assert candidates[0].grid_alt_km == 9.0, "Grid altitude used when no ADS-B"

    def test_ghost_filter_real_clutter_pairing_rejected(self):
        """Real × clutter pairing (adsb[i_b]=None) must also be rejected."""
        from retina_analytics.association import OverlapZone, find_associations

        zone = OverlapZone(
            node_a_id="A", node_b_id="B",
            grid_points=[(33.9, -84.5, 9.0)],
            delay_pairs=[(10.0, 20.0)],
            delay_gate_us=5.0,
            doppler_gate_hz=30.0,
        )
        ac = {"hex": "abc123", "alt_baro": 35000, "lat": 33.9, "lon": -84.5}
        frame_a = {"delay": [9.5], "doppler": [0.0], "snr": [10.0], "adsb": [ac]}
        frame_b = {"delay": [20.5], "doppler": [0.0], "snr": [10.0], "adsb": [None]}
        candidates = find_associations(zone, frame_a, frame_b, timestamp_ms=1000)
        assert len(candidates) == 0, "Real×clutter ghost should be rejected"

    def test_same_aircraft_filter_rejects_cross_pairing(self):
        """Cross-pairing of two different real aircraft must be rejected."""
        from retina_analytics.association import OverlapZone, find_associations

        zone = OverlapZone(
            node_a_id="A", node_b_id="B",
            grid_points=[(33.9, -84.5, 9.0)],
            delay_pairs=[(10.0, 20.0)],
            delay_gate_us=5.0,
            doppler_gate_hz=30.0,
        )
        ac_a = {"hex": "aaa111", "alt_baro": 35000}
        ac_b = {"hex": "bbb222", "alt_baro": 35000}
        frame_a = {"delay": [9.5], "doppler": [0.0], "snr": [10.0], "adsb": [ac_a]}
        frame_b = {"delay": [20.5], "doppler": [0.0], "snr": [10.0], "adsb": [ac_b]}
        candidates = find_associations(zone, frame_a, frame_b, timestamp_ms=1000)
        assert len(candidates) == 0, "Different-aircraft cross-pairing should be rejected"

    def test_same_aircraft_filter_accepts_same_hex(self):
        """Pairing where both frames carry the same hex must pass through."""
        from retina_analytics.association import OverlapZone, find_associations

        zone = OverlapZone(
            node_a_id="A", node_b_id="B",
            grid_points=[(33.9, -84.5, 9.0)],
            delay_pairs=[(10.0, 20.0)],
            delay_gate_us=5.0,
            doppler_gate_hz=30.0,
        )
        ac = {"hex": "abc123", "alt_baro": 35000}
        frame_a = {"delay": [9.5], "doppler": [0.0], "snr": [10.0], "adsb": [ac]}
        frame_b = {"delay": [20.5], "doppler": [0.0], "snr": [10.0], "adsb": [ac]}
        candidates = find_associations(zone, frame_a, frame_b, timestamp_ms=1000)
        assert len(candidates) == 1, "Same-aircraft pairing should be accepted"
        assert abs(candidates[0].grid_alt_km - 35000 * 0.3048 / 1000) < 0.01

    def test_format_candidates_keeps_distant_aircraft_separate(self):
        """Two aircraft > 5.6 km apart must produce separate solver inputs."""
        from retina_analytics.association import AssociationCandidate

        c1 = AssociationCandidate(
            node_a_id="A", node_b_id="B",
            det_a_idx=0, det_b_idx=0,
            delay_a=30.0, delay_b=31.0,
            doppler_a=10.0, doppler_b=11.0,
            snr_a=15.0, snr_b=14.0,
            grid_delay_a=30.1, grid_delay_b=31.1,
            grid_lat=33.90, grid_lon=-84.60, grid_alt_km=10.0,
            timestamp_ms=1000,
        )
        c2 = AssociationCandidate(
            node_a_id="C", node_b_id="D",
            det_a_idx=0, det_b_idx=0,
            delay_a=35.0, delay_b=36.0,
            doppler_a=5.0, doppler_b=6.0,
            snr_a=12.0, snr_b=11.0,
            grid_delay_a=35.1, grid_delay_b=36.1,
            grid_lat=34.20, grid_lon=-84.90, grid_alt_km=10.0,  # ~34 km away
            timestamp_ms=1000,
        )

        assoc = InterNodeAssociator(grid_step_km=3.0)
        solver_inputs = assoc.format_candidates_for_solver([c1, c2])

        assert len(solver_inputs) == 2, (
            f"Expected 2 separate solver inputs for distant aircraft, got {len(solver_inputs)}"
        )
