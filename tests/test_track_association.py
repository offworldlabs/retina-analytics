"""Track-to-track association: pairing N=1 tracks instead of detections.

At n=2 a detection pairing is untestable — two nodes give 4 measurements
against 6 unknowns, so a cross pairing between two real aircraft leaves zero
residual exactly as a real target does.  Pairing confirmed single-node tracks
supplies 4K measurements from the same two nodes, which is what makes the
constant-velocity fit able to tell them apart.
"""

import math

import pytest

from retina_analytics.association import (
    InterNodeAssociator,
    TrackPairCandidate,
    _merge_epochs,
    velocity_conflict,
)
from retina_analytics.constants import KM_PER_DEG_LAT

_C_KM_S = 299792.458
_R_EARTH_KM = 6371.0

# A dual-illuminator site: two nodes, one receiver, two transmitters.  This is
# the arrangement the n=2 case actually arises in.
_NODE_A = {
    "rx_lat": 34.85,
    "rx_lon": -82.40,
    "rx_alt_ft": 1000,
    "tx_lat": 34.9412,
    "tx_lon": -82.4103,
    "tx_alt_ft": 2000,
    "fc_hz": 183e6,
    "beam_width_deg": 90,
    "max_range_km": 60,
    "beam_azimuth_deg": 45.0,
}
_NODE_B = {
    "rx_lat": 34.85,
    "rx_lon": -82.40,
    "rx_alt_ft": 1000,
    "tx_lat": 34.9701,
    "tx_lon": -81.9484,
    "tx_alt_ft": 800,
    "fc_hz": 195e6,
    "beam_width_deg": 90,
    "max_range_km": 60,
    "beam_azimuth_deg": 45.0,
}


def _enu_km(lat, lon, alt_km, ref_lat, ref_lon, ref_alt_km):
    east = math.radians(lon - ref_lon) * _R_EARTH_KM * math.cos(math.radians(ref_lat))
    north = math.radians(lat - ref_lat) * _R_EARTH_KM
    return (east, north, alt_km - ref_alt_km)


def _measure(cfg, lat, lon, alt_km, ve, vn):
    ref_alt = cfg["rx_alt_ft"] * 0.0003048
    tgt = _enu_km(lat, lon, alt_km, cfg["rx_lat"], cfg["rx_lon"], ref_alt)
    tx = _enu_km(cfg["tx_lat"], cfg["tx_lon"], cfg["tx_alt_ft"] * 0.0003048, cfg["rx_lat"], cfg["rx_lon"], ref_alt)
    d_tx = math.dist(tgt, tx)
    d_rx = math.dist(tgt, (0.0, 0.0, 0.0))
    delay_us = (d_tx + d_rx - math.dist(tx, (0.0, 0.0, 0.0))) / 0.299792458
    b = [(tx[i] - tgt[i]) / d_tx + (0.0 - tgt[i]) / d_rx for i in range(3)]
    lam_m = (_C_KM_S * 1000.0) / cfg["fc_hz"]
    return delay_us, (ve * b[0] + vn * b[1]) / lam_m


# Sample count and spacing giving a ~20 s observation span.  Span, not epoch
# count, is what separates a real pairing from a crossed one: measured on this
# geometry a crossed pairing fits at χ²/dof 0.40 over 10 s and 0.57 over 14 s,
# and stops being fittable at all by 20 s.  Beyond ~30 s the flat-earth
# generator below starts diverging from the fit's ellipsoidal geometry and the
# *true* pairing's χ² climbs too, which is a limit of these fixtures rather
# than of the fit.
_N = 11
_DT = 2.0


def _history(cfg, lat, lon, alt_km, ve, vn, n=_N, dt=_DT, anchor="start"):
    """One node's view of a straight, level target, oldest sample first.

    anchor="end" puts the target at (lat, lon) on the *last* sample instead of
    the first.  That is how a crossed pairing is actually born: two aircraft
    whose bistatic ellipses intersect right now, which is exactly what the
    coarse delay gate tests.  Anchoring both trajectories at the start instead
    would let them drift apart, and the coarse gate would throw the pairing out
    before the χ² test ever saw it — which is what a first version of these
    tests accidentally measured.
    """
    t_anchor = (n - 1) * dt if anchor == "end" else 0.0
    out = []
    for k in range(n):
        t = k * dt
        la = lat + vn * (t - t_anchor) / (KM_PER_DEG_LAT * 1000.0)
        lo = lon + ve * (t - t_anchor) / ((KM_PER_DEG_LAT * 1000.0) * math.cos(math.radians(lat)))
        d, f = _measure(cfg, la, lo, alt_km, ve, vn)
        out.append({"t_s": t, "delay_us": d, "doppler_hz": f, "snr": 15.0})
    return out


def _crossing_pair(n=_N, dt=_DT):
    """Node A's view of one aircraft, node B's of another, coincident right now.

    Both are real detections of real aircraft on straight, level courses; they
    simply happen to be in the same place at the latest epoch with different
    velocities.  Every single-epoch residual gate passes this, because with 4
    measurements against 6 unknowns the residuals are structurally zero.
    """
    hist_a = _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0, n, dt, anchor="end")
    hist_b = _history(_NODE_B, 34.88, -82.35, 7.0, -150.0, 170.0, n, dt, anchor="end")
    return hist_a, hist_b


def _assoc(cv_fit=None, **kw):
    a = InterNodeAssociator(grid_step_km=3.0, cv_fit=cv_fit, **kw)
    a.register_node("site-a", _NODE_A)
    a.register_node("site-b", _NODE_B)
    return a


def _candidate(track_a_id, track_b_id, **kw):
    """A TrackPairCandidate at a default position, for the clustering tests.

    Clustering is judged on position, per-node track ids and implied velocity
    alone, so these need no zone geometry — building them by hand keeps a
    two-aircraft scene readable and exactly reproducible.
    """
    fields = dict(
        timestamp_ms=1000,
        node_a_id="site-a",
        node_b_id="site-b",
        delay_a=30.0,
        delay_b=40.0,
        doppler_a=5.0,
        doppler_b=-5.0,
        snr_a=15.0,
        snr_b=15.0,
        lat=34.88,
        lon=-82.35,
        alt_km=7.0,
        vel_east_ms=180.0,
        vel_north_ms=-90.0,
        implied_vel=(180.0, -90.0),
        dof=14,
        n_epochs=6,
    )
    fields.update(kw)
    return TrackPairCandidate(track_a_id=track_a_id, track_b_id=track_b_id, **fields)


def _cv_fit():
    """The real fit, imported lazily so the rest of the module runs without it."""
    pytest.importorskip("retina_geolocator")
    from retina_geolocator.multinode_solver import fit_constant_velocity

    return fit_constant_velocity


class TestMergeEpochs:
    def test_samples_keep_their_own_timestamps(self):
        """No resampling — each sample is its own epoch.

        Aligning the two tracks onto shared epochs would reintroduce exactly the
        error the frame stagger causes: nodes send on their own cadence, and at
        250 m/s a 2 s misalignment invents 500 m of position error.
        """
        ha = [
            {"t_s": 0.0, "delay_us": 10.0, "doppler_hz": 5.0, "snr": 12.0},
            {"t_s": 2.0, "delay_us": 11.0, "doppler_hz": 6.0, "snr": 12.0},
        ]
        hb = [{"t_s": 1.0, "delay_us": 20.0, "doppler_hz": -5.0, "snr": 9.0}]
        epochs = _merge_epochs(ha, "a", hb, "b")

        assert [e["t_s"] for e in epochs] == [0.0, 1.0, 2.0]
        assert [e["measurements"][0]["node_id"] for e in epochs] == ["a", "b", "a"]

    def test_simultaneous_samples_share_an_epoch(self):
        ha = [{"t_s": 4.0, "delay_us": 10.0, "doppler_hz": 5.0, "snr": 12.0}]
        hb = [{"t_s": 4.0, "delay_us": 20.0, "doppler_hz": -5.0, "snr": 9.0}]
        epochs = _merge_epochs(ha, "a", hb, "b")
        assert len(epochs) == 1
        assert len(epochs[0]["measurements"]) == 2


class TestSubmitTracks:
    def test_needs_both_sides(self):
        """One node's tracks alone cannot pair with anything."""
        a = _assoc()
        assert (
            a.submit_tracks(
                "site-a",
                [
                    {"track_id": "t1", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0)},
                ],
                1000,
            )
            == []
        )

    def test_pairs_two_tracks(self):
        a = _assoc()
        a.submit_tracks(
            "site-a",
            [
                {"track_id": "a1", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0)},
            ],
            1000,
        )
        pairs = a.submit_tracks(
            "site-b",
            [
                {"track_id": "b1", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0)},
            ],
            2000,
        )

        assert len(pairs) == 1
        assert isinstance(pairs[0], TrackPairCandidate)
        assert {pairs[0].track_a_id, pairs[0].track_b_id} == {"a1", "b1"}

    def test_empty_history_is_skipped(self):
        a = _assoc()
        a.submit_tracks("site-a", [{"track_id": "a1", "history": []}], 1000)
        pairs = a.submit_tracks(
            "site-b",
            [
                {"track_id": "b1", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0)},
            ],
            2000,
        )
        assert pairs == []

    def test_short_history_passes_on_the_coarse_gate(self):
        """A pairing too young to fit is held, not dropped.

        It is re-tested every round and a real target accumulates the history it
        needs within a few frames; dropping it would lose the target outright.
        Downstream decides whether an unfitted pairing may be published.
        """
        a = _assoc(cv_fit=_cv_fit())
        a.submit_tracks(
            "site-a",
            [
                {"track_id": "a1", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0, n=2)},
            ],
            1000,
        )
        pairs = a.submit_tracks(
            "site-b",
            [
                {"track_id": "b1", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0, n=2)},
            ],
            2000,
        )

        assert len(pairs) == 1
        assert pairs[0].chi2_per_dof is None
        assert a.track_pairs_unfitted == 1


class TestConstantVelocityGate:
    """The part that a detection-level pairing structurally cannot do."""

    def _run(self, hist_a, hist_b, **kw):
        a = _assoc(cv_fit=_cv_fit(), **kw)
        a.submit_tracks("site-a", [{"track_id": "a1", "history": hist_a}], 1000)
        return a, a.submit_tracks("site-b", [{"track_id": "b1", "history": hist_b}], 2000)

    def test_true_pairing_is_accepted_and_fitted(self):
        a, pairs = self._run(
            _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0),
            _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0),
        )
        assert len(pairs) == 1
        p = pairs[0]
        assert p.chi2_per_dof is not None and p.chi2_per_dof < 1.0
        assert a.track_pairs_accepted == 1
        # The fit replaces the 3 km grid point with a real position estimate,
        # reported at the *last* epoch — where the target is now — so the
        # expectation is the start dead-reckoned across the observation window.
        span = (_N - 1) * _DT
        exp_lat = 34.88 + -90.0 * span / (KM_PER_DEG_LAT * 1000.0)
        exp_lon = -82.35 + 180.0 * span / ((KM_PER_DEG_LAT * 1000.0) * math.cos(math.radians(34.88)))
        assert math.hypot((p.lat - exp_lat) * KM_PER_DEG_LAT, (p.lon - exp_lon) * 91.3) < 1.0
        assert p.vel_east_ms == pytest.approx(180.0, abs=15.0)
        assert p.vel_north_ms == pytest.approx(-90.0, abs=15.0)

    def test_cross_pairing_is_rejected(self):
        """Two real aircraft, coincident right now — the ghost mechanism."""
        a, pairs = self._run(*_crossing_pair())
        assert pairs == []
        assert a.track_pairs_rejected >= 1

    def test_the_chi2_test_is_what_rejects_it(self):
        """Loosen the threshold and the same pairing survives, with a bad fit.

        Pins that the rejection comes from the fit and not from the coarse
        delay grid happening to miss — the two are easy to confuse, and a first
        version of this test measured the grid by accident.
        """
        a, pairs = self._run(*_crossing_pair(), cv_chi2_max=float("inf"))
        assert len(pairs) == 1
        assert pairs[0].chi2_per_dof > InterNodeAssociator().cv_chi2_max

    def test_true_and_crossed_are_orders_of_magnitude_apart(self):
        """The separation, not the threshold, is what the design rests on."""
        _, true_pairs = self._run(
            _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0),
            _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0),
            cv_chi2_max=float("inf"),
        )
        _, cross_pairs = self._run(*_crossing_pair(), cv_chi2_max=float("inf"))
        assert cross_pairs[0].chi2_per_dof > 100 * true_pairs[0].chi2_per_dof

    def test_a_short_span_cannot_separate(self):
        """Over a 4 s window the crossed pairing passes as a real target.

        Two aircraft are close enough to constant-velocity over a short window;
        it is accumulated curvature that gives them away.  This is the finding
        that makes cv_min_span_s the gate rather than cv_min_epochs — otherwise
        a 22 fps node would qualify on 5 epochs spanning 0.2 s, which carries no
        information at all.
        """
        a, pairs = self._run(*_crossing_pair(n=3, dt=2.0), cv_min_span_s=2.0, cv_min_epochs=2)
        assert len(pairs) == 1
        assert pairs[0].chi2_per_dof < a.cv_chi2_max

    def test_separation_grows_with_span(self):
        """Measured: crossed χ²/dof 1.41 at 4 s, 2.46 at 10 s, 3.72 at 20 s.

        Monotone in span, which is why the default cv_min_span_s sits above the
        point where the crossed value clears the threshold rather than at the
        first span that yields enough samples.
        """

        def crossed_chi2(n, dt):
            _, pairs = self._run(
                *_crossing_pair(n=n, dt=dt), cv_chi2_max=float("inf"), cv_min_span_s=0.0, cv_min_epochs=2
            )
            return pairs[0].chi2_per_dof

        short, mid, long = crossed_chi2(3, 2.0), crossed_chi2(6, 2.0), crossed_chi2(11, 2.0)
        assert short < mid < long
        assert short < InterNodeAssociator().cv_chi2_max < long

    def test_span_gate_defers_rather_than_fits(self):
        """Below cv_min_span_s nothing is fitted — the pairing waits."""
        a, pairs = self._run(
            _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0, n=4, dt=1.0),
            _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0, n=4, dt=1.0),
        )
        assert len(pairs) == 1 and pairs[0].chi2_per_dof is None
        assert a.track_pairs_unfitted == 1

    def test_no_fit_injected_means_coarse_gate_only(self):
        """Without a solver the library still works, just without the fine test.

        The pairing the fit would have rejected comes straight through, which is
        the point: the coarse gate is a coverage question — do the two ellipses
        cross inside both beams — and it was never able to answer this one.
        """
        hist_a, hist_b = _crossing_pair()
        a = _assoc()
        a.submit_tracks("site-a", [{"track_id": "a1", "history": hist_a}], 1000)
        pairs = a.submit_tracks("site-b", [{"track_id": "b1", "history": hist_b}], 2000)
        assert len(pairs) == 1 and pairs[0].chi2_per_dof is None


class TestFormatTrackPairsForSolver:
    def test_emits_the_shape_the_solver_takes(self):
        a = _assoc(cv_fit=_cv_fit())
        a.submit_tracks(
            "site-a",
            [
                {"track_id": "a1", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0)},
            ],
            1000,
        )
        pairs = a.submit_tracks(
            "site-b",
            [
                {"track_id": "b1", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0)},
            ],
            2000,
        )

        inputs = a.format_track_pairs_for_solver(pairs)
        assert len(inputs) == 1
        s_in = inputs[0]
        assert s_in["n_nodes"] == 2
        assert {m["node_id"] for m in s_in["measurements"]} == {"site-a", "site-b"}
        assert set(s_in["initial_guess"]) == {"lat", "lon", "alt_km"}
        assert s_in["chi2_per_dof"] is not None
        assert s_in["track_ids"] == ["a1", "b1"]
        assert s_in["track_ids_by_node"] == {"site-a": ["a1"], "site-b": ["b1"]}
        assert set().union(*s_in["track_ids_by_node"].values()) == set(s_in["track_ids"])

    def test_cluster_reports_its_worst_fit(self):
        """A poor pairing must not be laundered by a good one beside it.

        The cluster is published as one target, so its quality is the quality of
        the weakest pairing holding it together.

        The two pairings here share node-a track a1, which is what makes them
        one cluster rather than two: same node, same track, two neighbours —
        the 3-node case, not a conflict.  It used to be a2/b2 against a1/b1,
        which _split_node_conflicts now (correctly) separates into two targets,
        so the worst-fit rule would never have been reached.
        """
        pairs = [
            _candidate("a1", "b1", chi2_per_dof=0.4),
            _candidate("a1", "b2", node_b_id="site-c", chi2_per_dof=9.9),
        ]
        inputs = InterNodeAssociator().format_track_pairs_for_solver(pairs)
        assert len(inputs) == 1
        assert inputs[0]["chi2_per_dof"] == 9.9

    def test_unfitted_cluster_reports_no_chi2(self):
        """Production defers the fit, so the field must say so rather than 0.

        With cv_fit=None nothing here is scored; a numeric chi2 would be the
        solver worker's gate reading a quality nobody measured.
        """
        inputs = InterNodeAssociator().format_track_pairs_for_solver([_candidate("a1", "b1")])
        assert inputs[0]["chi2_per_dof"] is None

    def test_empty_input(self):
        assert InterNodeAssociator().format_track_pairs_for_solver([]) == []


class TestSameNodeConflictSplits:
    """A node contributing two tracks to one cluster means two aircraft.

    A node's tracker gives one track per aircraft, so the cluster is not one
    target and cannot be made into one by keeping the louder track — that
    hands the solver a candidate no position explains, and 58-65% of dark
    candidates measured on the live fleet carried a node that could not see
    the aircraft they were finally published as.
    """

    def test_two_aircraft_split_instead_of_being_merged(self):
        """Both nodes see both aircraft, 4 km apart — one input each."""
        a = InterNodeAssociator()
        inputs = a.format_track_pairs_for_solver(
            [
                _candidate("aP", "bP", lat=34.88, snr_a=20.0, snr_b=20.0),
                _candidate("aQ", "bQ", lat=34.88 + 4.0 / KM_PER_DEG_LAT, snr_a=6.0, snr_b=6.0),
            ]
        )
        assert len(inputs) == 2
        assert a.cluster_splits == 1
        by_node = [s_in["track_ids_by_node"] for s_in in inputs]
        assert {"site-a": ["aP"], "site-b": ["bP"]} in by_node
        assert {"site-a": ["aQ"], "site-b": ["bQ"]} in by_node
        # No SNR pick: the quiet aircraft keeps its own measurements rather
        # than borrowing the loud one's.
        assert sorted(m["snr"] for s_in in inputs for m in s_in["measurements"]) == [6.0, 6.0, 20.0, 20.0]

    def test_a_false_pairing_between_them_stands_alone(self):
        """The cross pairing is emitted too, but never welded onto a true one.

        This stage cannot tell which of three hypotheses is real — the solver's
        gates and the resolve slot decide that — so the requirement is only
        that no emitted input mixes two aircraft.
        """
        a = InterNodeAssociator()
        inputs = a.format_track_pairs_for_solver(
            [
                _candidate("aP", "bP", lat=34.88, grid_resid_us=0.1),
                _candidate("aQ", "bQ", lat=34.88 + 4.0 / KM_PER_DEG_LAT, grid_resid_us=0.1),
                # The cross pairing: node a's aircraft P against node b's Q,
                # landing between the two true clusters and inside the merge
                # radius of both.
                _candidate("aP", "bQ", lat=34.88 + 2.0 / KM_PER_DEG_LAT, grid_resid_us=0.9),
            ]
        )
        assert len(inputs) == 3
        for s_in in inputs:
            assert all(len(ids) == 1 for ids in s_in["track_ids_by_node"].values())
        assert {"site-a": ["aP"], "site-b": ["bP"]} in [s["track_ids_by_node"] for s in inputs]
        assert {"site-a": ["aQ"], "site-b": ["bQ"]} in [s["track_ids_by_node"] for s in inputs]

    def test_one_aircraft_on_three_nodes_is_not_split(self):
        """The legitimate multi-node cluster is untouched.

        Two pairings sharing node-a track a1 are the same aircraft seen by
        three nodes — one track per node, no conflict — and must still merge
        into a single 3-node solver input.
        """
        a = InterNodeAssociator()
        inputs = a.format_track_pairs_for_solver(
            [
                _candidate("a1", "b1"),
                _candidate("a1", "c1", node_b_id="site-c"),
            ]
        )
        assert len(inputs) == 1
        assert inputs[0]["n_nodes"] == 3
        assert a.cluster_splits == 0

    def test_velocity_disagreement_blocks_the_merge(self):
        """Two coincident pairings heading opposite ways are two targets.

        Proximity alone made this one cluster, which is the crossing case the
        old 6 km radius could not distinguish from one aircraft's own
        neighbouring pairings.
        """
        a = InterNodeAssociator()
        inputs = a.format_track_pairs_for_solver(
            [
                _candidate("a1", "b1", implied_vel=(200.0, 0.0)),
                _candidate("a2", "b2", implied_vel=(-200.0, 0.0), lat=34.881),
            ]
        )
        assert len(inputs) == 2
        # Split by the merge edge, not by the conflict split, so nothing was
        # ever a single cluster to begin with.
        assert a.cluster_splits == 0


class TestVelocityExclusivity:
    """Deferred-path exclusivity: production has no chi2 to rank on.

    cv_fit is None live, so stage 2's chi2 selection never runs and every
    hypothesis the coarse grid passed is emitted — including the several that
    claim the same track.  The Doppler-implied velocity cannot reject a
    pairing on its own (0% power, measured), but two claims about one track's
    velocity that cannot both be true mean at most one pairing is.
    """

    def _run(self, **kw):
        a = _assoc(**kw)
        a.submit_tracks(
            "site-a",
            [{"track_id": "aP", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0, anchor="end")}],
            1000,
        )
        return a, a.submit_tracks(
            "site-b",
            [
                {"track_id": "bP", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0, anchor="end")},
                {"track_id": "bQ", "history": _history(_NODE_B, 34.88, -82.35, 7.0, -150.0, 170.0, anchor="end")},
            ],
            2000,
        )

    def test_the_contradicting_hypothesis_is_dropped(self):
        """Both pairings pass the coarse gate at the same grid point and the
        same delay residual — the residual is ~0 for a crossed pairing at n=2,
        which is why an assignment on it alone cost recall.  The implied
        headings differ by 75°, and that is decisive."""
        a, pairs = self._run()
        assert [(p.track_a_id, p.track_b_id) for p in pairs] == [("aP", "bP")]
        assert a.track_pairs_superseded == 1

    def test_off_by_flag_restores_both(self):
        a, pairs = self._run(pair_vel_exclusive=False)
        assert len(pairs) == 2
        assert a.track_pairs_superseded == 0

    def test_agreement_keeps_both(self):
        """Where the two hypotheses agree, nothing has been learned.

        A three-node target legitimately pairs one track against two
        neighbours; this must never become an excuse to drop one of them, and
        it is the reason the test is on disagreement rather than on rank.
        """
        a = _assoc()
        a.submit_tracks(
            "site-a",
            [{"track_id": "aP", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0, anchor="end")}],
            1000,
        )
        pairs = a.submit_tracks(
            "site-b",
            [
                {"track_id": "bP", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0, anchor="end")},
                {"track_id": "bP2", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 182.0, -88.0, anchor="end")},
            ],
            2000,
        )
        assert len(pairs) == 2
        assert a.track_pairs_superseded == 0

    def test_a_supplied_fit_keeps_the_chi2_path(self):
        """Gated on cv_fit is None, so an inline-fitting caller is untouched."""
        a, pairs = self._run(cv_fit=_cv_fit(), cv_min_span_s=2.0, cv_min_epochs=2)
        assert len(pairs) == 1
        assert pairs[0].chi2_per_dof is not None


class TestVelocityConflict:
    def test_speed_alone_can_decide(self):
        assert velocity_conflict((200.0, 0.0), (60.0, 0.0), 80.0, 40.0)
        assert not velocity_conflict((200.0, 0.0), (150.0, 0.0), 80.0, 40.0)

    def test_heading_alone_can_decide(self):
        assert velocity_conflict((200.0, 0.0), (0.0, 200.0), 80.0, 40.0)
        assert not velocity_conflict((200.0, 0.0), (190.0, 40.0), 80.0, 40.0)

    def test_missing_inference_never_conflicts(self):
        """No information is not evidence — the abstention this shares with
        implied_horizontal_velocity, which returns None on geometry that
        cannot support the inference at all."""
        assert not velocity_conflict(None, (200.0, 0.0), 80.0, 40.0)
        assert not velocity_conflict((200.0, 0.0), None, 80.0, 40.0)

    def test_near_zero_speeds_compare_magnitude_only(self):
        """Below 30 m/s the heading is noise, so opposite directions at a
        crawl are not called a conflict."""
        assert not velocity_conflict((5.0, 0.0), (-5.0, 0.0), 80.0, 40.0)


# Third node at the same receiver — a triple-illuminator site — for exercising
# cross-node-pair sharing, which hypothesis selection must never forbid.
_NODE_C = {
    "rx_lat": 34.85,
    "rx_lon": -82.40,
    "rx_alt_ft": 1000,
    "tx_lat": 35.1702,
    "tx_lon": -82.2905,
    "tx_alt_ft": 3000,
    "fc_hz": 201e6,
    "beam_width_deg": 90,
    "max_range_km": 60,
    "beam_azimuth_deg": 45.0,
}


class TestHypothesisSelection:
    """Pairings are competing hypotheses, not independent candidates.

    One track is one aircraft, so two pairings sharing a track are mutually
    exclusive, and the χ² from the CV fit is the first non-degenerate score
    this competition has had — assignment on the delay residual failed (the
    cost is ~0 for every pairing at n=2) and ranking by cluster size failed
    (reverted; ghosts went 52% → 85%).
    """

    def test_selection_rejects_what_the_threshold_cannot(self):
        """The decisive case: a crossed pairing *under* the χ² bar.

        Over a 4 s span a crossed pairing fits at χ²/dof ≈ 1.4 — below any
        threshold that keeps real targets, which is why the absolute gate needs
        a 12 s span.  Selection does not need the crossed hypothesis to fail
        the bar, only the true one to beat it: ~1e-4 beats 1.4 at any span.
        """
        hist_p_a = _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0, n=3, dt=2.0, anchor="end")
        hist_p_b = _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0, n=3, dt=2.0, anchor="end")
        hist_q_b = _history(_NODE_B, 34.88, -82.35, 7.0, -150.0, 170.0, n=3, dt=2.0, anchor="end")

        def run(exclusive):
            a = _assoc(cv_fit=_cv_fit(), cv_min_span_s=2.0, cv_min_epochs=2, cv_exclusive=exclusive)
            a.submit_tracks("site-a", [{"track_id": "aP", "history": hist_p_a}], 1000)
            return a, a.submit_tracks(
                "site-b",
                [
                    {"track_id": "bP", "history": hist_p_b},
                    {"track_id": "bQ", "history": hist_q_b},
                ],
                2000,
            )

        # Threshold alone passes both hypotheses — the ghost publishes.
        _, both = run(exclusive=False)
        assert len(both) == 2

        # Selection keeps only the better explanation of track aP.
        a, pairs = run(exclusive=True)
        assert len(pairs) == 1
        assert {pairs[0].track_a_id, pairs[0].track_b_id} == {"aP", "bP"}
        assert a.track_pairs_superseded == 1

    def test_claim_then_vet(self):
        """A failing best hypothesis still claims its tracks.

        If the best explanation of two tracks is implausible, a worse or
        unscored one must not inherit them — otherwise rejection would promote
        exactly the pairing the fit ranked lower.
        """
        crossed_a, crossed_b = _crossing_pair()  # long span → χ² fails
        short_b = _history(_NODE_B, 34.88, -82.35, 7.0, -150.0, 170.0, n=2, dt=2.0, anchor="end")  # unscoreable

        a = _assoc(cv_fit=_cv_fit())
        a.submit_tracks("site-a", [{"track_id": "a1", "history": crossed_a}], 1000)
        pairs = a.submit_tracks(
            "site-b",
            [
                {"track_id": "b1", "history": crossed_b},
                {"track_id": "b2", "history": short_b},
            ],
            2000,
        )

        assert pairs == []
        assert a.track_pairs_rejected == 1  # (a1, b1): fitted, failed, claimed
        assert a.track_pairs_superseded >= 1  # (a1, b2): held, blocked by claim

    def test_sharing_across_node_pairs_is_allowed(self):
        """A 3-node target pairs its A-track with a B-track AND a C-track.

        Exclusivity is per node pair — the Ta×Tb and Ta×Tc matrices are
        separate — because cross-pair sharing is not a conflict, it is the
        n≥3 structure the solver wants.
        """

        def h(cfg):
            return _history(cfg, 34.88, -82.35, 7.0, 180.0, -90.0)

        a = InterNodeAssociator(grid_step_km=3.0, cv_fit=_cv_fit())
        a.register_node("site-a", _NODE_A)
        a.register_node("site-b", _NODE_B)
        a.register_node("site-c", _NODE_C)
        a.submit_tracks("site-a", [{"track_id": "a1", "history": h(_NODE_A)}], 1000)
        a.submit_tracks("site-b", [{"track_id": "b1", "history": h(_NODE_B)}], 1500)
        pairs = a.submit_tracks("site-c", [{"track_id": "c1", "history": h(_NODE_C)}], 2000)

        assert len(pairs) == 2
        assert all("c1" in (p.track_a_id, p.track_b_id) for p in pairs)
        assert a.track_pairs_superseded == 0


class TestPairingCostIsBounded:
    """Association runs in the frame worker, so its cost is frame latency.

    The first version scanned the whole overlap grid once per (track_a,
    track_b) pairing.  On staging that put the frame queue at 92% depth with
    the processor 21 s behind a 6 frame/s feed, and the map went empty — the
    offline bench never showed it, running unthrottled on one core with no
    queue to fall behind.
    """

    def _tracks(self, n, node_cfg, prefix):
        # n aircraft strung along one bearing so they all share the zone.
        return [
            {"track_id": f"{prefix}{i}", "history": _history(node_cfg, 34.88 + i * 0.01, -82.35, 7.0, 180.0, -90.0)}
            for i in range(n)
        ]

    def test_gate_is_one_contraction_not_one_scan_per_pairing(self):
        """_batch_grid_match is called once per node pair, not Ta x Tb times."""
        from unittest import mock

        import retina_analytics.association as assoc_mod

        a = _assoc()
        a.submit_tracks("site-a", self._tracks(6, _NODE_A, "a"), 1000)
        with mock.patch.object(assoc_mod, "_batch_grid_match", wraps=assoc_mod._batch_grid_match) as spy:
            a.submit_tracks("site-b", self._tracks(6, _NODE_B, "b"), 2000)
        assert spy.call_count == 1, "one contraction per node pair"

    def test_batch_match_agrees_with_a_per_pair_scan(self):
        """Vectorising must not change which pairings pass, or which point wins."""
        import numpy as np

        from retina_analytics.association import _batch_grid_match

        a = _assoc()
        zone = next(z for z in a.overlap_zones.values() if z.delay_pairs)
        zone._ensure_np()
        da = [float(zone._np_pred_a[10]), float(zone._np_pred_a[400]) + 0.4]
        db = [float(zone._np_pred_b[10]), 9999.0]
        batched = _batch_grid_match(zone, da, db)

        gate = zone.delay_gate_us
        for i, d_a in enumerate(da):
            for j, d_b in enumerate(db):
                valid = np.nonzero((np.abs(zone._np_pred_a - d_a) < gate) & (np.abs(zone._np_pred_b - d_b) < gate))[0]
                if valid.size == 0:
                    assert (i, j) not in batched
                    continue
                res = np.abs(zone._np_pred_a[valid] - d_a) + np.abs(zone._np_pred_b[valid] - d_b)
                assert batched[(i, j)] == int(valid[np.argmin(res)])

    def test_fits_per_round_are_capped(self):
        """A crowded zone cannot make one round unboundedly slow."""
        calls = []

        def counting_fit(fit_input, cfgs):
            calls.append(1)
            return _cv_fit()(fit_input, cfgs)

        a = _assoc(cv_fit=counting_fit)
        a._MAX_FITS_PER_ROUND = 3
        a.submit_tracks("site-a", self._tracks(6, _NODE_A, "a"), 1000)
        a.submit_tracks("site-b", self._tracks(6, _NODE_B, "b"), 2000)
        assert len(calls) <= 3

    def test_deferred_pairings_are_not_lost(self):
        """Raising the cap recovers pairings a tight one skipped."""

        def run(cap):
            a = _assoc(cv_fit=_cv_fit())
            a._MAX_FITS_PER_ROUND = cap
            a.submit_tracks("site-a", self._tracks(5, _NODE_A, "a"), 1000)
            return len(a.submit_tracks("site-b", self._tracks(5, _NODE_B, "b"), 2000))

        assert run(1) <= run(25)
