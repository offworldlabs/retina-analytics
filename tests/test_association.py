"""Tests for inter-node association logic."""

import math

from retina_analytics.association import (
    InterNodeAssociator,
    NodeGeometry,
    _bistatic_delay_at,
    _lla_to_enu,
    compute_overlap_zone,
)

# ── Overlap zone & bistatic delay ────────────────────────────────────────────


class TestOverlapZone:
    def test_overlap_zone_ids(self):
        geo_a = NodeGeometry(
            node_id="assoc-A",
            rx_lat=33.939,
            rx_lon=-84.651,
            rx_alt_km=0.29,
            tx_lat=33.756,
            tx_lon=-84.331,
            tx_alt_km=0.49,
            beam_azimuth_deg=135,
            beam_width_deg=41,
            max_range_km=50,
        )
        geo_b = NodeGeometry(
            node_id="assoc-B",
            rx_lat=34.05,
            rx_lon=-84.4,
            rx_alt_km=0.3,
            tx_lat=33.85,
            tx_lon=-84.15,
            tx_alt_km=0.5,
            beam_azimuth_deg=210,
            beam_width_deg=41,
            max_range_km=50,
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
        assoc.register_node(
            "assoc-A",
            {
                "rx_lat": 33.939,
                "rx_lon": -84.651,
                "rx_alt_ft": 950,
                "tx_lat": 33.756,
                "tx_lon": -84.331,
                "tx_alt_ft": 1600,
                "fc_hz": 195e6,
                "beam_width_deg": 41,
                "max_range_km": 50,
            },
        )
        assoc.register_node(
            "assoc-B",
            {
                "rx_lat": 34.05,
                "rx_lon": -84.4,
                "rx_alt_ft": 980,
                "tx_lat": 33.85,
                "tx_lon": -84.15,
                "tx_alt_ft": 1600,
                "fc_hz": 195e6,
                "beam_width_deg": 41,
                "max_range_km": 50,
            },
        )
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


_SCALING_CFG = {
    "rx_lat": 33.939,
    "rx_lon": -84.651,
    "rx_alt_ft": 950,
    "tx_lat": 33.756,
    "tx_lon": -84.331,
    "tx_alt_ft": 1600,
    "fc_hz": 195e6,
    "beam_width_deg": 41,
    "max_range_km": 50,
}


class TestAssocInterval:
    """One configured interval, not scaled by fleet size.

    It was briefly scaled down for small fleets, reasoning that their CPU
    budget allowed associating far more often -- at 15 nodes ~2 ms/s against
    the 57% of a core the limit exists to avoid.  The arithmetic held; the
    premise did not.  Measured offline over 6 seeds
    (backend/scripts/association_bench.py), ghost tracks as a share of all
    tracks ran 40% (sd 13) at a 2 s interval against 6% (sd 9) at 30 s, t=4.9,
    while the real-track count (8-13) and matched position error (~0.28 km
    median) stayed flat across the whole sweep.  Associating more often bought
    no extra targets and no extra accuracy -- it re-sampled the same geometry
    and minted more false tracks.

    Ghost rate is driven by how many aircraft share an overlap zone, which node
    count does not predict, so the scaling optimised a resource that was never
    the constraint.
    """

    def _register(self, assoc, n):
        for i in range(n):
            cfg = dict(_SCALING_CFG)
            cfg["rx_lat"] += i * 0.01
            assoc.register_node(f"scale-{i}", cfg)

    def test_default_is_the_budgeted_interval(self):
        assert InterNodeAssociator(grid_step_km=30.0)._ASSOC_MIN_INTERVAL_S == 30.0

    def test_interval_is_configurable(self):
        assoc = InterNodeAssociator(grid_step_km=30.0, assoc_interval_s=10.0)
        assert assoc._ASSOC_MIN_INTERVAL_S == 10.0

    def test_small_fleet_does_not_relax_the_interval(self):
        """The regression this guards: a 15-node fleet used to drop to 2 s."""
        assoc = InterNodeAssociator(grid_step_km=30.0)
        self._register(assoc, 15)
        assert assoc._ASSOC_MIN_INTERVAL_S == 30.0

    def test_three_node_fleet_is_unchanged_too(self):
        # (Renamed: it registered 3 nodes while claiming a large fleet, and was
        # a strictly weaker duplicate of the 15-node test above.)
        assoc = InterNodeAssociator(grid_step_km=30.0)
        self._register(assoc, 3)
        assert assoc._ASSOC_MIN_INTERVAL_S == 30.0


# ── Doppler as a velocity projection ─────────────────────────────────────────


_FC_VHF, _FC_UHF = 183e6, 599e6


def _lam(fc):
    from retina_analytics.association import C_KM_S

    return C_KM_S * 1000.0 / fc


def _doppler_hz(v_ms, tgt, tx, rx, fc):
    """Forward model: f_d = (1/lambda) * v . (u_tx + u_rx)."""
    from retina_analytics.association import _bisector

    b = _bisector(tgt, tx, rx)
    return sum(v_ms[i] * b[i] for i in range(3)) / _lam(fc)


class TestDopplerIsAVelocityProjection:
    """Each node measures one projection of v onto its own bistatic bisector.

    The axis depends on TX position, RX position and *target* position, so two
    nodes watching one aircraft report genuinely different numbers.  Comparing
    them — in Hz or normalised to m/s — subtracts components of the same vector
    resolved along different directions, which is why the old
    |f_a - f_b| <= 90 Hz gate was not merely mis-tuned but meaningless.
    """

    RX = (0.0, 0.0, 0.0)
    TX1 = (-20.0, 25.0, 0.0)
    TX2 = (30.0, -18.0, 0.0)
    TGT = (12.0, 18.0, 9.0)

    def test_projection_identity_holds(self):
        from retina_analytics.association import _bisector

        v = (180.0, 120.0, 0.0)
        for tx, fc in ((self.TX1, _FC_VHF), (self.TX2, _FC_UHF)):
            f = _doppler_hz(v, self.TGT, tx, self.RX, fc)
            b = _bisector(self.TGT, tx, self.RX)
            assert abs(f * _lam(fc) - sum(v[i] * b[i] for i in range(3))) < 1e-6

    def test_one_aircraft_gives_wildly_different_hz_across_bands(self):
        """The case the old gate rejected: a genuine cross-band pair."""
        v = (180.0, 120.0, 0.0)
        f1 = _doppler_hz(v, self.TGT, self.TX1, self.RX, _FC_VHF)
        f2 = _doppler_hz(v, self.TGT, self.TX2, self.RX, _FC_UHF)
        assert abs(f1 - f2) > 90.0, "must exceed the old gate to be a regression guard"

    def test_velocity_is_recovered_from_the_two_projections(self):
        from retina_analytics.association import _bisector, implied_horizontal_speed

        v = (180.0, 120.0, 0.0)
        f1 = _doppler_hz(v, self.TGT, self.TX1, self.RX, _FC_VHF)
        f2 = _doppler_hz(v, self.TGT, self.TX2, self.RX, _FC_UHF)
        got = implied_horizontal_speed(
            f1 * _lam(_FC_VHF),
            _bisector(self.TGT, self.TX1, self.RX),
            f2 * _lam(_FC_UHF),
            _bisector(self.TGT, self.TX2, self.RX),
        )
        assert abs(got - math.hypot(v[0], v[1])) < 0.5

    def test_impossible_pairing_is_rejected(self):
        """Two projections implying a speed no aircraft can fly."""
        from retina_analytics.association import _bisector, implied_horizontal_speed

        b1 = _bisector(self.TGT, self.TX1, self.RX)
        b2 = _bisector(self.TGT, self.TX2, self.RX)
        # Opposed projections along near-orthogonal axes force a huge |v|.
        got = implied_horizontal_speed(600.0, b1, -600.0, b2)
        assert got is not None and got > 340.0

    def test_abstains_on_parallel_axes(self):
        """Same axis twice cannot determine a horizontal velocity."""
        from retina_analytics.association import _bisector, implied_horizontal_speed

        b1 = _bisector(self.TGT, self.TX1, self.RX)
        assert implied_horizontal_speed(50.0, b1, 50.0, b1) is None

    def test_abstains_when_the_bisector_collapses(self):
        """On the baseline |b| -> 0 and Doppler carries no velocity information."""
        from retina_analytics.association import _bisector, implied_horizontal_speed

        b2 = _bisector(self.TGT, self.TX2, self.RX)
        assert implied_horizontal_speed(50.0, (0.01, 0.0, 0.0), 50.0, b2) is None

    def test_true_pairings_are_essentially_never_rejected(self):
        """The test proves impossibility only; it must not discard real traffic."""
        import random

        from retina_analytics.association import _bisector, implied_horizontal_speed

        rng = random.Random(11)
        judged = rejected = 0
        for _ in range(2000):
            tgt = (rng.uniform(-40, 40), rng.uniform(-40, 40), rng.uniform(3, 11))
            sp, th = rng.uniform(80, 270), rng.uniform(0, 2 * math.pi)
            v = (sp * math.cos(th), sp * math.sin(th), rng.gauss(0, 5))
            f1 = _doppler_hz(v, tgt, self.TX1, self.RX, _FC_VHF)
            f2 = _doppler_hz(v, tgt, self.TX2, self.RX, _FC_UHF)
            got = implied_horizontal_speed(
                f1 * _lam(_FC_VHF),
                _bisector(tgt, self.TX1, self.RX),
                f2 * _lam(_FC_UHF),
                _bisector(tgt, self.TX2, self.RX),
            )
            if got is None:
                continue
            judged += 1
            if got > 340.0:
                rejected += 1
        assert judged > 1500
        assert rejected / judged < 0.02, f"false-rejection rate {rejected / judged:.3f}"
