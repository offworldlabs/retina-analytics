"""Tests for inter-node association logic."""

import math
import random

import pytest

from retina_analytics.association import (
    InterNodeAssociator,
    NodeGeometry,
    _bistatic_delay_at,
    _lla_to_enu,
    compute_overlap_zone,
    predict_observation,
)
from retina_analytics.constants import C_KM_S, C_KM_US, R_EARTH

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


class TestOverlapGridCost:
    """The both-beams test is 2-D, so it must not be repeated per altitude.

    compute_overlap_zone evaluates six altitude layers over one bounding box,
    and _point_in_beam takes (lat, lon) only.  Re-deriving it per layer was 87%
    of a node rebuild on the 52-node test deployment (855k calls where 143k
    distinct columns exist), which is what put analytics_refresh past its 120 s
    health budget.  These pin the property the restructure rests on, not the
    restructure itself.
    """

    ALTS = (1.5, 3.0, 5.0, 7.0, 9.0, 11.0)

    def _pair(self):
        geo_a = NodeGeometry(
            node_id="cost-A",
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
            node_id="cost-B",
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
        return geo_a, geo_b

    def test_beam_membership_is_tested_once_per_column(self, monkeypatch):
        import retina_analytics.association as A

        geo_a, geo_b = self._pair()
        seen = []
        real = A._point_in_beam
        monkeypatch.setattr(
            A, "_point_in_beam", lambda lat, lon, g: (seen.append((lat, lon, g.node_id)), real(lat, lon, g))[1]
        )
        zone = compute_overlap_zone(geo_a, geo_b, grid_step_km=5.0, altitudes_km=self.ALTS)
        assert zone.grid_points  # the pair does overlap, so this is a real grid
        assert len(seen) == len(set(seen))

    def test_every_altitude_layer_is_still_emitted(self):
        geo_a, geo_b = self._pair()
        zone = compute_overlap_zone(geo_a, geo_b, grid_step_km=5.0, altitudes_km=self.ALTS)
        # Column set is altitude-independent, so each layer contributes the same
        # (lat, lon) columns modulo the per-altitude delay gates.
        by_alt = {}
        for lat, lon, alt in zone.grid_points:
            by_alt.setdefault(alt, set()).add((lat, lon))
        assert set(by_alt) == set(self.ALTS)

    def test_baseline_km_is_memoised_but_follows_a_moved_node(self):
        geo_a, _ = self._pair()
        first = geo_a.baseline_km
        assert geo_a.baseline_km == first  # cached, same answer
        geo_a.tx_lat += 0.5
        assert geo_a.baseline_km != first  # keyed on the coordinates, not sticky

    @pytest.mark.parametrize("field", ["rx_lat", "rx_lon", "tx_lat", "tx_lon"])
    def test_baseline_km_follows_every_coordinate_it_is_keyed_on(self, field):
        """Both ends, in place — a node re-registers by rewriting this object.

        The RX half was untested: a memo keyed on the TX coordinates alone
        passes the test above and still answers from the old receiver site
        forever.
        """
        geo_a, _ = self._pair()
        first = geo_a.baseline_km
        setattr(geo_a, field, getattr(geo_a, field) + 0.5)
        assert geo_a.baseline_km != first


# ── Hot-path memoisation ─────────────────────────────────────────────────────


def _hot_path_geo(**over) -> NodeGeometry:
    base = dict(
        node_id="memo-A",
        rx_lat=33.939,
        rx_lon=-84.651,
        rx_alt_km=0.29,
        tx_lat=33.756,
        tx_lon=-84.331,
        tx_alt_km=0.49,
        fc_hz=195e6,
        beam_azimuth_deg=135,
        beam_width_deg=41,
        max_range_km=50.0,
    )
    base.update(over)
    return NodeGeometry(**base)


def _predict_reference(geo, lat, lon, alt_km, ve_ms, vn_ms, vu_ms=0.0):
    """predict_observation as it read before the ENU frame was memoised.

    Written out straight rather than imported, so it keeps saying what the
    answer used to be even as the module is optimised further: a shared helper
    would follow the production code and stop being a reference.
    """

    def lla_to_enu(la, lo, al, ref_la, ref_lo, ref_al):
        dlat = math.radians(la - ref_la)
        dlon = math.radians(lo - ref_lo)
        north = dlat * R_EARTH
        east = dlon * R_EARTH * math.cos(math.radians(ref_la))
        return (east, north, al - ref_al)

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    ref = (geo.rx_lat, geo.rx_lon, geo.rx_alt_km)
    rx = lla_to_enu(geo.rx_lat, geo.rx_lon, geo.rx_alt_km, *ref)
    tx = lla_to_enu(geo.tx_lat, geo.tx_lon, geo.tx_alt_km, *ref)
    tgt = lla_to_enu(lat, lon, alt_km, *ref)

    d_tx = norm([tgt[i] - tx[i] for i in range(3)])
    d_rx = norm([rx[i] - tgt[i] for i in range(3)])
    d_bl = norm([rx[i] - tx[i] for i in range(3)])
    delay_us = (d_tx + d_rx - d_bl) / C_KM_US

    b_tx = norm([tgt[i] - tx[i] for i in range(3)]) or 1e-9
    b_rx = norm([tgt[i] - rx[i] for i in range(3)]) or 1e-9
    b = tuple((tx[i] - tgt[i]) / b_tx + (rx[i] - tgt[i]) / b_rx for i in range(3))
    v_dot_b = ve_ms * b[0] + vn_ms * b[1] + vu_ms * b[2]
    return delay_us, v_dot_b * geo.fc_hz / (C_KM_S * 1000.0)


class TestPredictObservationMemoisation:
    """The node's own ENU frame is a constant; caching it must not move a number.

    predict_observation anchors ENU at the node's own RX, which makes rx_enu the
    origin and leaves TX, the RX→TX baseline length and the longitude scale
    factor fixed by the node's own coordinates — all three were rebuilt on every
    call, on a path the claiming and ADS-B seeding rounds run per tracklet per
    round.  A memo keyed on too few fields is invisible until a node
    re-registers at a new site and every prediction quietly answers from the old
    one, so both halves are pinned here: the values against the pre-memo maths,
    and invalidation against an in-place rewrite of each coordinate in turn.
    """

    TARGET = (33.99, -84.50, 8.0, 200.0, 50.0)

    def test_matches_the_pre_memo_maths_over_a_geometry_grid(self):
        rng = random.Random(20260826)
        for _ in range(400):
            rx_lat = rng.uniform(-60.0, 60.0)
            rx_lon = rng.uniform(-179.0, 179.0)
            rx_alt_km = rng.uniform(0.0, 2.5)
            if rng.random() < 0.35:  # monostatic: TX co-sited with RX
                tx_lat, tx_lon, tx_alt_km = rx_lat, rx_lon, rx_alt_km
            else:
                tx_lat = rx_lat + rng.uniform(-0.5, 0.5)
                tx_lon = rx_lon + rng.uniform(-0.5, 0.5)
                tx_alt_km = rng.uniform(0.0, 2.5)
            geo = _hot_path_geo(
                rx_lat=rx_lat,
                rx_lon=rx_lon,
                rx_alt_km=rx_alt_km,
                tx_lat=tx_lat,
                tx_lon=tx_lon,
                tx_alt_km=tx_alt_km,
                fc_hz=rng.choice([98e6, 195e6, 430e6, 750e6]),
            )
            target = (
                rx_lat + rng.uniform(-0.6, 0.6),
                rx_lon + rng.uniform(-0.6, 0.6),
                rng.uniform(0.0, 13.0),
                rng.uniform(-300.0, 300.0),
                rng.uniform(-300.0, 300.0),
                rng.uniform(-20.0, 20.0),
            )
            got = predict_observation(geo, *target)
            want = _predict_reference(geo, *target)
            assert got[0] == pytest.approx(want[0], rel=1e-9, abs=1e-9)
            assert got[1] == pytest.approx(want[1], rel=1e-9, abs=1e-9)

    def test_a_target_sitting_on_the_receiver_still_agrees(self):
        """d_rx == 0 exactly, which is the one branch the random grid cannot hit."""
        geo = _hot_path_geo()
        target = (geo.rx_lat, geo.rx_lon, geo.rx_alt_km, 200.0, 50.0, 0.0)
        got = predict_observation(geo, *target)
        want = _predict_reference(geo, *target)
        assert got[0] == pytest.approx(want[0], rel=1e-9, abs=1e-9)
        assert got[1] == pytest.approx(want[1], rel=1e-9, abs=1e-9)

    def test_the_memo_is_warm_and_repeatable(self):
        geo = _hot_path_geo()
        first = predict_observation(geo, *self.TARGET)
        assert predict_observation(geo, *self.TARGET) == first

    @pytest.mark.parametrize(
        "field, delta",
        [
            ("rx_lat", 0.4),
            ("rx_lon", 0.4),
            ("rx_alt_km", 0.35),
            ("tx_lat", 0.5),
            ("tx_lon", 0.5),
            ("tx_alt_km", 0.4),
        ],
    )
    def test_an_in_place_coordinate_rewrite_invalidates_the_frame(self, field, delta):
        geo = _hot_path_geo()
        before = predict_observation(geo, *self.TARGET)
        assert predict_observation(geo, *self.TARGET) == before  # memo is warm

        setattr(geo, field, getattr(geo, field) + delta)
        after = predict_observation(geo, *self.TARGET)
        assert after[0] != before[0], f"a rewritten {field} never reached the delay"
        want = _predict_reference(geo, *self.TARGET)
        assert after[0] == pytest.approx(want[0], rel=1e-9, abs=1e-9)
        assert after[1] == pytest.approx(want[1], rel=1e-9, abs=1e-9)


class TestFootprintRadiusMemoisation:
    """footprint_radius_km is read per grid point per node, and is a constant.

    Keyed on every field it reads: the bistatic limit, the monostatic range it
    falls back to, and the four coordinates baseline_km is itself keyed on.
    Each is rewritten in place below, because that is how a node re-registering
    with a new configuration reaches this object.
    """

    def test_monostatic_radius_follows_a_rewritten_max_range(self):
        geo = _hot_path_geo(max_bistatic_range_km=None)
        assert geo.footprint_radius_km == pytest.approx(50.0)
        geo.max_range_km = 72.0
        assert geo.footprint_radius_km == pytest.approx(72.0)

    def test_declaring_a_bistatic_limit_switches_the_formula(self):
        geo = _hot_path_geo(max_bistatic_range_km=None)
        assert geo.footprint_radius_km == pytest.approx(50.0)
        geo.max_bistatic_range_km = 60.0
        assert geo.footprint_radius_km == pytest.approx(30.0 + geo.baseline_km)

    def test_a_rewritten_bistatic_limit_follows(self):
        geo = _hot_path_geo(max_bistatic_range_km=60.0)
        first = geo.footprint_radius_km
        geo.max_bistatic_range_km = 90.0
        assert geo.footprint_radius_km == pytest.approx(first + 15.0)

    @pytest.mark.parametrize("field", ["rx_lat", "rx_lon", "tx_lat", "tx_lon"])
    def test_a_moved_station_moves_the_bistatic_radius(self, field):
        geo = _hot_path_geo(max_bistatic_range_km=60.0)
        first = geo.footprint_radius_km
        setattr(geo, field, getattr(geo, field) + 0.5)
        assert geo.footprint_radius_km == pytest.approx(30.0 + geo.baseline_km)
        assert geo.footprint_radius_km != first

    def test_effective_radius_still_tracks_a_learning_fov(self):
        """The FOV keeps widening with no field of this dataclass changing, which
        is why effective_radius_km is left unmemoised over a memoised footprint."""

        class _Fov:
            reach = 40.0

            def max_limit_km(self):
                return self.reach

        geo = _hot_path_geo(max_bistatic_range_km=None)
        geo.fov = _Fov()
        assert geo.effective_radius_km == pytest.approx(50.0)  # footprint still wider
        geo.fov.reach = 85.0
        assert geo.effective_radius_km == pytest.approx(85.0)
