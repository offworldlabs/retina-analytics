"""Top-down tracklet claiming (ASSOC_CLAIM_MODE) — the claiming stage added
to InterNodeAssociator alongside the bottom-up track-pairing round.

Fresh, eligible global tracks are projected into each node's delay-Doppler
space; matching local tracklet views are claimed one-to-one per node; a
global claimed at >=2 nodes including the triggering one earns an anchored
solver input carrying its key.  See association.py's _claim_round /
submit_tracks_round for the mechanism this pins.

Mirrors test_track_association.py's fixture shapes (_NODE_A/_NODE_B,
_measure, _history) rather than importing them, so this file stands alone.
"""

import math

import pytest

from retina_analytics.association import (
    CLAIM_MAX_DR_AGE_S,
    InterNodeAssociator,
    predict_observation,
)
from retina_analytics.constants import KM_PER_DEG_LAT

_C_KM_S = 299792.458
_R_EARTH_KM = 6371.0

# Same dual-illuminator site as test_track_association.py: one receiver, two
# transmitters — the arrangement the n=2 case actually arises in.
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
# Third node at the same receiver — needed only for the trigger-must-match
# test, which is structurally a 3-node scenario.
_NODE_C = {
    "rx_lat": 34.85, "rx_lon": -82.40, "rx_alt_ft": 1000,
    "tx_lat": 35.1702, "tx_lon": -82.2905, "tx_alt_ft": 3000,
    "fc_hz": 201e6, "beam_width_deg": 90, "max_range_km": 60,
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


# Same sample count/spacing as test_track_association.py — a ~20 s span.
_N = 11
_DT = 2.0
_LAST_T_S = (_N - 1) * _DT


def _history(cfg, lat, lon, alt_km, ve, vn, n=_N, dt=_DT, anchor="start"):
    """One node's view of a straight, level target, oldest sample first.

    anchor="end" puts the target at (lat, lon) on the *last* sample — the
    position/instant every claiming test anchors its global track to, so the
    dead-reckoned projection and the tracklet's latest measurement describe
    the same instant with dt=0.
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


def _assoc(**kw) -> InterNodeAssociator:
    a = InterNodeAssociator(grid_step_km=3.0, **kw)
    a.register_node("site-a", _NODE_A)
    a.register_node("site-b", _NODE_B)
    return a


def _make_assoc(claim_mode, provider) -> InterNodeAssociator:
    return _assoc(claim_mode=claim_mode, global_track_provider=provider)


# A "true aircraft state": both tracklets and the global track it should
# claim describe the same target, at the same instant, by construction.
_TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM = 34.88, -82.35, 7.0
_TRUE_VE, _TRUE_VN = 180.0, -90.0


def _global(key, *, lat=_TRUE_LAT, lon=_TRUE_LON, alt_km=_TRUE_ALT_KM,
           ve=_TRUE_VE, vn=_TRUE_VN, ts_s=_LAST_T_S, n_nodes=0,
           solve_count=2, vel_up=0.0) -> dict:
    """A provider-contract global track dict, defaulting to the true state."""
    return {
        "key": key, "lat": lat, "lon": lon, "alt_m": alt_km * 1000.0,
        "vel_east": ve, "vel_north": vn, "vel_up": vel_up,
        "timestamp_ms": ts_s * 1000.0, "n_nodes": n_nodes,
        "solve_count": solve_count,
    }


def _true_hist(cfg):
    return _history(cfg, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM, _TRUE_VE, _TRUE_VN,
                    anchor="end")


class TestPredictObservation:
    """Round-trip against _measure — the same math this claiming stage
    compares a claim's prediction to a tracklet's actual measurement with."""

    def test_round_trips_measure_both_doppler_signs(self):
        a = _assoc()
        geo = a.node_geometries["site-a"]
        for ve, vn in [(180.0, -90.0), (-150.0, 170.0), (0.0, 0.0)]:
            exp_delay, exp_doppler = _measure(_NODE_A, _TRUE_LAT, _TRUE_LON,
                                              _TRUE_ALT_KM, ve, vn)
            delay, doppler = predict_observation(
                geo, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM, ve, vn)
            assert delay == pytest.approx(exp_delay, abs=1e-6)
            assert doppler == pytest.approx(exp_doppler, abs=1e-6)

    def test_positive_is_closing_negative_is_receding(self):
        a = _assoc()
        geo = a.node_geometries["site-a"]
        _, closing = predict_observation(geo, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM,
                                         180.0, -90.0)
        _, receding = predict_observation(geo, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM,
                                          -180.0, 90.0)
        assert closing == pytest.approx(-receding, abs=1e-6)


class TestClaimMatching:
    def test_true_state_global_claims_its_tracklet_at_both_nodes(self):
        g = _global("mn-dark-1")
        a = _make_assoc("active", lambda: [g])
        a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        round_ = a.submit_tracks_round(
            "site-b", [{"track_id": "b1", "history": _true_hist(_NODE_B)}], 2000)

        won_nodes = {c["node_id"] for c in round_.claims
                     if c["anchor_key"] == "mn-dark-1" and c["won"]}
        assert won_nodes == {"site-a", "site-b"}
        assert len(round_.anchored_inputs) == 1
        assert round_.anchored_inputs[0]["anchor_key"] == "mn-dark-1"

    def test_a_distant_global_claims_nothing(self):
        # ~15 km north — delay sensitivity is ~3-6.7 us/km, so this is many
        # multiples of CLAIM_DELAY_GATE_US away.
        far_lat = _TRUE_LAT + 15.0 / KM_PER_DEG_LAT
        g = _global("mn-dark-far", lat=far_lat)
        a = _make_assoc("active", lambda: [g])
        a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        round_ = a.submit_tracks_round(
            "site-b", [{"track_id": "b1", "history": _true_hist(_NODE_B)}], 2000)

        assert round_.claims == []
        assert round_.anchored_inputs == []

    def test_doppler_gate_rejects_a_co_delay_tracklet_with_wrong_velocity(self):
        """Two aircraft coincident right now (anchor='end'): the delay
        ellipses intersect, but the global's velocity is not what this
        tracklet actually measured — the mechanism claiming exists to
        reject that the bottom-up path cannot at n=2."""
        g = _global("mn-dark-1")  # true velocity: 180, -90
        a = _make_assoc("active", lambda: [g])
        hist_wrong_vel = _history(_NODE_A, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM,
                                  -150.0, 170.0, anchor="end")
        round_a = a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": hist_wrong_vel}], 1000)
        assert round_a.claims == []  # gated out before becoming a candidate

        round_b = a.submit_tracks_round(
            "site-b", [{"track_id": "b1", "history": _true_hist(_NODE_B)}], 2000)
        a_claims = [c for c in round_b.claims if c["node_id"] == "site-a"]
        assert a_claims == []
        # Only one node matched (site-b) — never anchors.
        assert round_b.anchored_inputs == []

    def test_two_globals_competing_for_one_tracklet_produce_one_winner(self):
        """Both globals describe the identical true state, so their scores
        tie exactly — the first (stable-sorted) one wins and the second is
        a conflict, deterministically."""
        g_exact = _global("mn-dark-exact")
        g_dup = _global("mn-dark-dup")
        a = _make_assoc("active", lambda: [g_exact, g_dup])
        round_ = a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)

        a_claims = [c for c in round_.claims if c["node_id"] == "site-a"]
        assert {c["anchor_key"] for c in a_claims} == {"mn-dark-exact", "mn-dark-dup"}
        winners = [c for c in a_claims if c["won"]]
        assert len(winners) == 1
        assert winners[0]["anchor_key"] == "mn-dark-exact"
        assert a.claim_conflicts >= 1


class TestEligibility:
    def test_low_n_nodes_and_solve_count_one_never_claims(self):
        g = _global("mn-dark-1", n_nodes=2, solve_count=1)
        a = _make_assoc("active", lambda: [g])
        round_ = a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        assert round_.claims == []

    def test_solve_count_two_is_enough_even_at_n_nodes_zero(self):
        g = _global("mn-dark-1", n_nodes=0, solve_count=2)
        a = _make_assoc("active", lambda: [g])
        round_ = a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        assert any(c["won"] for c in round_.claims)

    def test_n_nodes_three_is_enough_even_at_solve_count_zero(self):
        g = _global("mn-dark-1", n_nodes=3, solve_count=0)
        a = _make_assoc("active", lambda: [g])
        round_ = a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        assert any(c["won"] for c in round_.claims)

    def test_dr_age_beyond_the_cap_is_not_claimed(self):
        """The global's own last solve is old enough that dead-reckoning it
        forward to the tracklet's latest sample exceeds claim_max_dr_age_s —
        withheld regardless of how good the position match would be."""
        g = _global("mn-dark-1", ts_s=_LAST_T_S - (CLAIM_MAX_DR_AGE_S + 1.0))
        a = _make_assoc("active", lambda: [g])
        round_ = a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        assert round_.claims == []


class TestShadowMode:
    def test_shadow_counts_but_pairs_and_anchored_inputs_match_off(self):
        g = _global("mn-dark-1")

        off = _make_assoc("off", lambda: [g])
        off.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        off_round = off.submit_tracks_round(
            "site-b", [{"track_id": "b1", "history": _true_hist(_NODE_B)}], 2000)

        shadow = _make_assoc("shadow", lambda: [g])
        shadow.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        shadow_round = shadow.submit_tracks_round(
            "site-b", [{"track_id": "b1", "history": _true_hist(_NODE_B)}], 2000)

        # Computed and counted ...
        assert shadow.claim_rounds > 0
        assert shadow.claims_matched > 0
        assert shadow_round.claims != []
        assert any(c["won"] for c in shadow_round.claims)
        # ... but provably inert downstream.
        assert shadow_round.anchored_inputs == []
        assert len(shadow_round.pairs) == len(off_round.pairs) == 1
        assert ({p.track_a_id for p in shadow_round.pairs}
                == {p.track_a_id for p in off_round.pairs})
        assert ({p.track_b_id for p in shadow_round.pairs}
                == {p.track_b_id for p in off_round.pairs})


class TestActiveMode:
    def test_claimed_tracklets_excluded_round_locally_pending_tracks_unmutated(self):
        g = _global("mn-dark-1")
        a = _make_assoc("active", lambda: [g])
        tracks_a = [{"track_id": "a1", "history": _true_hist(_NODE_A)}]
        tracks_b = [{"track_id": "b1", "history": _true_hist(_NODE_B)}]
        a.submit_tracks_round("site-a", tracks_a, 1000)
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        # Both tracklets were claimed by the same global, so nothing is left
        # for the bottom-up pairing to pair.
        assert round_.pairs == []
        assert a.tracklets_excluded >= 1

        # The exclusion is round-local: _pending_tracks holds the original,
        # un-filtered lists exactly as submitted.
        assert a._pending_tracks["site-a"] == tracks_a
        assert a._pending_tracks["site-b"] == tracks_b

        assert len(round_.anchored_inputs) == 1
        anc = round_.anchored_inputs[0]
        assert anc["anchor_key"] == "mn-dark-1"
        assert anc["cv_epochs"]
        assert anc["cv_epochs"] == sorted(anc["cv_epochs"], key=lambda e: e["t_s"])
        assert len(anc["track_pair_ids"]) <= 1
        assert anc["n_nodes"] == len(anc["measurements"]) == 2

    def test_trigger_must_be_among_the_matched_nodes(self):
        """site-a and site-b both match the global, but the round is
        triggered by site-c, which does not — no anchored input, even
        though 2 (non-triggering) nodes matched."""
        g = _global("mn-dark-1")
        a = InterNodeAssociator(grid_step_km=3.0, claim_mode="active",
                                global_track_provider=lambda: [g])
        a.register_node("site-a", _NODE_A)
        a.register_node("site-b", _NODE_B)
        a.register_node("site-c", _NODE_C)
        a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        a.submit_tracks_round(
            "site-b", [{"track_id": "b1", "history": _true_hist(_NODE_B)}], 1500)

        # site-c's own tracklet is a different target entirely, ~111 km
        # away — it will never gate against this global.
        hist_c_other = _history(_NODE_C, _TRUE_LAT + 1.0, _TRUE_LON,
                                _TRUE_ALT_KM, _TRUE_VE, _TRUE_VN, anchor="end")
        round_ = a.submit_tracks_round(
            "site-c", [{"track_id": "c1", "history": hist_c_other}], 2000)

        assert round_.anchored_inputs == []
        matched = {c["node_id"] for c in round_.claims
                  if c["anchor_key"] == "mn-dark-1" and c["won"]}
        assert matched == {"site-a", "site-b"}


class TestResetAndWrapper:
    def test_reset_for_tests_clears_the_new_counters(self):
        g = _global("mn-dark-1")
        a = _make_assoc("active", lambda: [g])
        a.submit_tracks_round(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        a.submit_tracks_round(
            "site-b", [{"track_id": "b1", "history": _true_hist(_NODE_B)}], 2000)
        assert a.claim_rounds > 0
        assert a.claims_matched > 0

        a._reset_for_tests()
        for name in ("claim_rounds", "claims_matched", "claim_conflicts",
                     "anchored_inputs_emitted", "tracklets_excluded"):
            assert getattr(a, name) == 0

    def test_submit_tracks_wrapper_returns_a_plain_list(self):
        a = _assoc()
        a.submit_tracks(
            "site-a", [{"track_id": "a1", "history": _true_hist(_NODE_A)}], 1000)
        pairs = a.submit_tracks(
            "site-b", [{"track_id": "b1", "history": _true_hist(_NODE_B)}], 2000)
        assert type(pairs) is list
        assert len(pairs) == 1
