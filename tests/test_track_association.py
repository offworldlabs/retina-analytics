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
)

_C_KM_S = 299792.458
_R_EARTH_KM = 6371.0

# A dual-illuminator site: two nodes, one receiver, two transmitters.  This is
# the arrangement the n=2 case actually arises in.
_NODE_A = {
    "rx_lat": 34.85, "rx_lon": -82.40, "rx_alt_ft": 1000,
    "tx_lat": 34.9412, "tx_lon": -82.4103, "tx_alt_ft": 2000,
    "fc_hz": 183e6, "beam_width_deg": 90, "max_range_km": 60,
    "beam_azimuth_deg": 45.0,
}
_NODE_B = {
    "rx_lat": 34.85, "rx_lon": -82.40, "rx_alt_ft": 1000,
    "tx_lat": 34.9701, "tx_lon": -81.9484, "tx_alt_ft": 800,
    "fc_hz": 195e6, "beam_width_deg": 90, "max_range_km": 60,
    "beam_azimuth_deg": 45.0,
}


def _enu_km(lat, lon, alt_km, ref_lat, ref_lon, ref_alt_km):
    east = math.radians(lon - ref_lon) * _R_EARTH_KM * math.cos(math.radians(ref_lat))
    north = math.radians(lat - ref_lat) * _R_EARTH_KM
    return (east, north, alt_km - ref_alt_km)


def _measure(cfg, lat, lon, alt_km, ve, vn):
    ref_alt = cfg["rx_alt_ft"] * 0.0003048
    tgt = _enu_km(lat, lon, alt_km, cfg["rx_lat"], cfg["rx_lon"], ref_alt)
    tx = _enu_km(cfg["tx_lat"], cfg["tx_lon"], cfg["tx_alt_ft"] * 0.0003048,
                 cfg["rx_lat"], cfg["rx_lon"], ref_alt)
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
        la = lat + vn * (t - t_anchor) / 111_320.0
        lo = lon + ve * (t - t_anchor) / (111_320.0 * math.cos(math.radians(lat)))
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
        ha = [{"t_s": 0.0, "delay_us": 10.0, "doppler_hz": 5.0, "snr": 12.0},
              {"t_s": 2.0, "delay_us": 11.0, "doppler_hz": 6.0, "snr": 12.0}]
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
        assert a.submit_tracks("site-a", [
            {"track_id": "t1", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0)},
        ], 1000) == []

    def test_pairs_two_tracks(self):
        a = _assoc()
        a.submit_tracks("site-a", [
            {"track_id": "a1", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0)},
        ], 1000)
        pairs = a.submit_tracks("site-b", [
            {"track_id": "b1", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0)},
        ], 2000)

        assert len(pairs) == 1
        assert isinstance(pairs[0], TrackPairCandidate)
        assert {pairs[0].track_a_id, pairs[0].track_b_id} == {"a1", "b1"}

    def test_empty_history_is_skipped(self):
        a = _assoc()
        a.submit_tracks("site-a", [{"track_id": "a1", "history": []}], 1000)
        pairs = a.submit_tracks("site-b", [
            {"track_id": "b1", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0)},
        ], 2000)
        assert pairs == []

    def test_short_history_passes_on_the_coarse_gate(self):
        """A pairing too young to fit is held, not dropped.

        It is re-tested every round and a real target accumulates the history it
        needs within a few frames; dropping it would lose the target outright.
        Downstream decides whether an unfitted pairing may be published.
        """
        a = _assoc(cv_fit=_cv_fit())
        a.submit_tracks("site-a", [
            {"track_id": "a1", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0, n=2)},
        ], 1000)
        pairs = a.submit_tracks("site-b", [
            {"track_id": "b1", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0, n=2)},
        ], 2000)

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
        # The fit replaces the 3 km grid point with a real position estimate.
        assert math.hypot((p.lat - 34.88) * 111.32, (p.lon + 82.35) * 91.3) < 1.0
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
        a, pairs = self._run(*_crossing_pair(n=3, dt=2.0),
                             cv_min_span_s=2.0, cv_min_epochs=2)
        assert len(pairs) == 1
        assert pairs[0].chi2_per_dof < a.cv_chi2_max

    def test_separation_grows_with_span(self):
        """Measured: crossed χ²/dof 1.41 at 4 s, 2.46 at 10 s, 3.72 at 20 s.

        Monotone in span, which is why the default cv_min_span_s sits above the
        point where the crossed value clears the threshold rather than at the
        first span that yields enough samples.
        """
        def crossed_chi2(n, dt):
            _, pairs = self._run(*_crossing_pair(n=n, dt=dt),
                                 cv_chi2_max=float("inf"),
                                 cv_min_span_s=0.0, cv_min_epochs=2)
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
        a.submit_tracks("site-a", [
            {"track_id": "a1", "history": _history(_NODE_A, 34.88, -82.35, 7.0, 180.0, -90.0)},
        ], 1000)
        pairs = a.submit_tracks("site-b", [
            {"track_id": "b1", "history": _history(_NODE_B, 34.88, -82.35, 7.0, 180.0, -90.0)},
        ], 2000)

        inputs = a.format_track_pairs_for_solver(pairs)
        assert len(inputs) == 1
        s_in = inputs[0]
        assert s_in["n_nodes"] == 2
        assert {m["node_id"] for m in s_in["measurements"]} == {"site-a", "site-b"}
        assert set(s_in["initial_guess"]) == {"lat", "lon", "alt_km"}
        assert s_in["chi2_per_dof"] is not None
        assert s_in["track_ids"] == ["a1", "b1"]

    def test_cluster_reports_its_worst_fit(self):
        """A poor pairing must not be laundered by a good one beside it.

        The cluster is published as one target, so its quality is the quality of
        the weakest pairing holding it together.
        """
        base = dict(timestamp_ms=1000, node_a_id="site-a", node_b_id="site-b",
                    delay_a=30.0, delay_b=40.0, doppler_a=5.0, doppler_b=-5.0,
                    snr_a=15.0, snr_b=15.0, lat=34.88, lon=-82.35, alt_km=7.0,
                    vel_east_ms=180.0, vel_north_ms=-90.0, dof=14, n_epochs=6)
        pairs = [
            TrackPairCandidate(track_a_id="a1", track_b_id="b1",
                               chi2_per_dof=0.4, **base),
            TrackPairCandidate(track_a_id="a2", track_b_id="b2",
                               chi2_per_dof=9.9, **base),
        ]
        inputs = InterNodeAssociator().format_track_pairs_for_solver(pairs)
        assert len(inputs) == 1
        assert inputs[0]["chi2_per_dof"] == 9.9

    def test_empty_input(self):
        assert InterNodeAssociator().format_track_pairs_for_solver([]) == []
