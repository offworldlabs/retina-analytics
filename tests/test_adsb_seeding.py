"""ADS-B-seeded detection assignment (ADSB_SEED_MODE) — the seeding stage
added to InterNodeAssociator alongside top-down claiming and the bottom-up
track-pairing round.

A tracklet the node (or backend) has already tagged with an ADS-B hex is
verified against the predicted bistatic observation, excluded from
bottom-up dark pairing when verified, and — if >=2 nodes verify the same
hex including the triggering node — re-emitted as a same-hex seeded solver
input.  See association.py's _adsb_seed_round / submit_tracks_round for the
mechanism this pins.

Mirrors test_track_claiming.py's fixture shapes (_NODE_A/_NODE_B, _measure,
_history) rather than importing them, so this file stands alone.
"""

import math

import pytest

from retina_analytics.association import (
    ADSB_SEED_MAX_DR_AGE_S,
    InterNodeAssociator,
    associate_detections_to_adsb,
    predict_observation,
)
from retina_analytics.constants import KM_PER_DEG_LAT

# Same dual-illuminator site as test_track_claiming.py: one receiver, two
# transmitters — the arrangement the n=2 case actually arises in.
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

_C_KM_S = 299792.458
_R_EARTH_KM = 6371.0


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


# Same sample count/spacing as test_track_claiming.py — a ~20 s span.
_N = 11
_DT = 2.0
_LAST_T_S = (_N - 1) * _DT


def _history(cfg, lat, lon, alt_km, ve, vn, n=_N, dt=_DT, anchor="start"):
    """One node's view of a straight, level target, oldest sample first.

    anchor="end" puts the target at (lat, lon) on the *last* sample — the
    position/instant every seeding test anchors its ADS-B state to, so the
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


def _make_seed_assoc(mode, provider) -> InterNodeAssociator:
    return _assoc(adsb_seed_mode=mode, adsb_provider=provider)


# A "true aircraft state": both tracklets and the ADS-B fix it should verify
# against describe the same target, at the same instant, by construction.
_TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM = 34.88, -82.35, 7.0
_TRUE_VE, _TRUE_VN = 180.0, -90.0


def _adsb_state(
    hexn, *, lat=_TRUE_LAT, lon=_TRUE_LON, alt_km=_TRUE_ALT_KM, ve=_TRUE_VE, vn=_TRUE_VN, ts_s=_LAST_T_S
) -> dict:
    """A provider-contract ADS-B state dict, defaulting to the true state."""
    return {
        "hex": hexn,
        "lat": lat,
        "lon": lon,
        "alt_m": alt_km * 1000.0,
        "vel_east": ve,
        "vel_north": vn,
        "timestamp_ms": ts_s * 1000.0,
        "alt_baro": alt_km * 3280.84,
        "gs": math.hypot(ve, vn) / 0.514444,
        "track": math.degrees(math.atan2(ve, vn)) % 360.0,
        "flight": "TEST123",
    }


def _true_hist(cfg):
    return _history(cfg, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM, _TRUE_VE, _TRUE_VN, anchor="end")


def _view(track_id, cfg, hexn=None):
    v = {"track_id": track_id, "history": _true_hist(cfg)}
    if hexn is not None:
        v["adsb_hex"] = hexn
    return v


class TestAdsbSeedActive:
    def test_two_nodes_same_hex_fresh_state_emits_one_input_and_excludes_both(self):
        st = _adsb_state("abc123")
        a = _make_seed_assoc("active", lambda: {"abc123": st})
        tracks_a = [_view("a1", _NODE_A, "ABC123")]  # mixed case: verifies
        tracks_b = [_view("b1", _NODE_B, "abc123")]  # normalization too
        # Seed site-a's pending view directly rather than through its own
        # round: _adsb_seed_round rescans the whole round-node set every
        # round (identical assembly to _claim_round), so a real site-a
        # round here would process a1 a second time when site-b triggers.
        a._pending_tracks["site-a"] = tracks_a
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        assert len(round_.adsb_inputs) == 1
        inp = round_.adsb_inputs[0]
        assert inp["adsb_hex"] == "abc123"
        assert "anchor_key" not in inp
        assert inp["cv_epochs"]
        assert inp["n_epochs"] == len(inp["cv_epochs"])
        assert {m["node_id"] for m in inp["measurements"]} == {"site-a", "site-b"}
        assert inp["initial_guess"]["lat"] == pytest.approx(_TRUE_LAT, abs=1e-6)
        assert inp["initial_guess"]["lon"] == pytest.approx(_TRUE_LON, abs=1e-6)

        assert round_.pairs == []
        assert a.adsb_tracklets_tagged == 2
        assert a.adsb_tracklets_excluded == 2
        assert a.adsb_inputs_emitted == 1

    def test_gate_reject_leaves_tracklet_in_dark_pairing(self):
        """A displaced ADS-B fix: delay residual far past the gate — the
        tag is not trusted, and the tracklet stays available for bottom-up
        pairing (fail open)."""
        far_lat = _TRUE_LAT + 15.0 / KM_PER_DEG_LAT  # ~15 km north
        st = _adsb_state("abc123", lat=far_lat)
        a = _make_seed_assoc("active", lambda: {"abc123": st})
        tracks_a = [_view("a1", _NODE_A, "abc123")]
        tracks_b = [_view("b1", _NODE_B, "abc123")]
        a._pending_tracks["site-a"] = tracks_a
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        assert a.adsb_seed_gate_rejects >= 1
        assert round_.adsb_inputs == []
        assert len(round_.pairs) == 1

    def test_missing_state_entry_fails_open(self):
        """The tagged hex has no matching provider entry at all."""
        other = _adsb_state("zzzzzz")
        a = _make_seed_assoc("active", lambda: {"zzzzzz": other})
        tracks_a = [_view("a1", _NODE_A, "abc123")]
        tracks_b = [_view("b1", _NODE_B, "abc123")]
        a._pending_tracks["site-a"] = tracks_a
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        assert a.adsb_seed_no_state >= 1
        assert round_.adsb_inputs == []
        assert len(round_.pairs) == 1

    def test_stale_state_fails_open(self):
        st = _adsb_state("abc123", ts_s=_LAST_T_S - (ADSB_SEED_MAX_DR_AGE_S + 1.0))
        a = _make_seed_assoc("active", lambda: {"abc123": st})
        tracks_a = [_view("a1", _NODE_A, "abc123")]
        tracks_b = [_view("b1", _NODE_B, "abc123")]
        a._pending_tracks["site-a"] = tracks_a
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        assert a.adsb_seed_no_state >= 1
        assert round_.adsb_inputs == []
        assert len(round_.pairs) == 1

    def test_single_node_tagged_hex_excludes_but_emits_no_input(self):
        st = _adsb_state("abc123")
        a = _make_seed_assoc("active", lambda: {"abc123": st})
        tracks_a = [_view("a1", _NODE_A)]  # untagged
        tracks_b = [_view("b1", _NODE_B, "abc123")]  # only this one tagged
        a._pending_tracks["site-a"] = tracks_a
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        assert round_.adsb_inputs == []
        assert a.adsb_inputs_emitted == 0
        assert a.adsb_tracklets_excluded == 1
        # b1 excluded leaves a1 with no partner to pair against.
        assert round_.pairs == []

    def test_two_views_same_node_same_hex_keeps_only_better_scoring(self):
        """Two tracklets at site-a both tagged with the same hex: the
        loser must not be tagged/excluded, only the better-scoring one."""
        st = _adsb_state("abc123")
        a = _make_seed_assoc("active", lambda: {"abc123": st})
        good_hist = _true_hist(_NODE_A)
        # Same final position (anchor="end" fixes it regardless of
        # velocity), but a 1% velocity mismatch versus the ADS-B state —
        # verifies (well inside the doppler gate) with a nonzero score.
        bad_hist = _history(_NODE_A, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM, _TRUE_VE * 1.01, _TRUE_VN * 1.01, anchor="end")
        tracks_a = [
            {"track_id": "a1-good", "history": good_hist, "adsb_hex": "abc123"},
            {"track_id": "a1-bad", "history": bad_hist, "adsb_hex": "abc123"},
        ]
        tracks_b = [_view("b1", _NODE_B, "abc123")]
        a._pending_tracks["site-a"] = tracks_a
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        assert len(round_.adsb_inputs) == 1
        assert round_.adsb_inputs[0]["track_ids_by_node"]["site-a"] == ["a1-good"]
        # Only one of the two site-a views was excluded.
        assert a.adsb_tracklets_excluded == 2  # a1-good + b1


class TestAdsbSeedShadowAndOff:
    def test_shadow_counts_move_but_pairs_and_inputs_match_off(self):
        st = _adsb_state("abc123")
        tracks_a = [_view("a1", _NODE_A, "abc123")]
        tracks_b = [_view("b1", _NODE_B, "abc123")]

        off = _make_seed_assoc("off", lambda: {"abc123": st})
        off._pending_tracks["site-a"] = tracks_a
        off_round = off.submit_tracks_round("site-b", tracks_b, 2000)

        shadow = _make_seed_assoc("shadow", lambda: {"abc123": st})
        shadow._pending_tracks["site-a"] = tracks_a
        shadow_round = shadow.submit_tracks_round("site-b", tracks_b, 2000)

        # Computed and counted ...
        assert shadow.adsb_tracklets_tagged == 2
        assert shadow.adsb_inputs_emitted == 1
        # ... but provably inert downstream.
        assert shadow.adsb_tracklets_excluded == 0
        assert shadow_round.adsb_inputs == []
        assert len(shadow_round.pairs) == len(off_round.pairs) == 1
        assert {p.track_a_id for p in shadow_round.pairs} == {p.track_a_id for p in off_round.pairs}
        assert {p.track_b_id for p in shadow_round.pairs} == {p.track_b_id for p in off_round.pairs}

    def test_off_mode_all_counters_stay_zero(self):
        st = _adsb_state("abc123")
        a = _make_seed_assoc("off", lambda: {"abc123": st})
        tracks_a = [_view("a1", _NODE_A, "abc123")]
        tracks_b = [_view("b1", _NODE_B, "abc123")]
        a._pending_tracks["site-a"] = tracks_a
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        for name in (
            "adsb_seed_rounds",
            "adsb_tracklets_tagged",
            "adsb_seed_no_state",
            "adsb_seed_gate_rejects",
            "adsb_tracklets_excluded",
            "adsb_inputs_emitted",
        ):
            assert getattr(a, name) == 0
        assert round_.adsb_inputs == []
        assert len(round_.pairs) == 1


class TestAdsbSeedClaimInteraction:
    def test_active_seed_blocks_a_dark_global_from_claiming_the_tagged_tracklet(self):
        st = _adsb_state("abc123")
        g = {
            "key": "mn-dark-1",
            "lat": _TRUE_LAT,
            "lon": _TRUE_LON,
            "alt_m": _TRUE_ALT_KM * 1000.0,
            "vel_east": _TRUE_VE,
            "vel_north": _TRUE_VN,
            "timestamp_ms": _LAST_T_S * 1000.0,
            "n_nodes": 0,
            "solve_count": 2,
        }
        a = _assoc(
            adsb_seed_mode="active",
            adsb_provider=lambda: {"abc123": st},
            claim_mode="active",
            global_track_provider=lambda: [g],
        )
        tracks_a = [_view("a1", _NODE_A, "abc123")]
        tracks_b = [_view("b1", _NODE_B, "abc123")]
        a._pending_tracks["site-a"] = tracks_a
        round_ = a.submit_tracks_round("site-b", tracks_b, 2000)

        # Both tracklets verified against the ADS-B tag, so the claim
        # candidate loop must skip both — no claim record at all, won or
        # lost, for either track_id.
        claim_track_ids = {c["track_id"] for c in round_.claims}
        assert "a1" not in claim_track_ids
        assert "b1" not in claim_track_ids
        # The seeded input still formed — claiming's exclusion did not
        # interfere with seeding itself.
        assert len(round_.adsb_inputs) == 1


class TestAssociateDetectionsToAdsb:
    def test_index_alignment_plane_then_clutter(self):
        a = _assoc()
        geo = a.node_geometries["site-a"]
        d_true, f_true = predict_observation(geo, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM, _TRUE_VE, _TRUE_VN)
        st = _adsb_state("abc123")
        frame_ts_ms = int(_LAST_T_S * 1000)

        out = associate_detections_to_adsb(
            geo,
            [d_true, d_true + 500.0],
            [f_true, f_true + 500.0],
            {"abc123": st},
            frame_ts_ms,
        )
        assert out is not None
        assert out[0]["hex"] == "abc123"
        assert out[1] is None

    def test_one_to_one_two_planes_two_detections(self):
        a = _assoc()
        geo = a.node_geometries["site-a"]
        lat2 = _TRUE_LAT + 0.05  # ~5.5 km north — a distinct target
        d1, f1 = predict_observation(geo, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM, _TRUE_VE, _TRUE_VN)
        d2, f2 = predict_observation(geo, lat2, _TRUE_LON, _TRUE_ALT_KM, _TRUE_VE, _TRUE_VN)
        st1 = _adsb_state("plane1")
        st2 = _adsb_state("plane2", lat=lat2)
        frame_ts_ms = int(_LAST_T_S * 1000)

        out = associate_detections_to_adsb(
            geo,
            [d1, d2],
            [f1, f2],
            {"plane1": st1, "plane2": st2},
            frame_ts_ms,
        )
        assert out is not None
        assert out[0]["hex"] == "plane1"
        assert out[1]["hex"] == "plane2"

    def test_all_states_stale_returns_none(self):
        a = _assoc()
        geo = a.node_geometries["site-a"]
        st = _adsb_state("abc123", ts_s=_LAST_T_S - (ADSB_SEED_MAX_DR_AGE_S + 5.0))
        frame_ts_ms = int(_LAST_T_S * 1000)

        out = associate_detections_to_adsb(
            geo,
            [50.0],
            [10.0],
            {"abc123": st},
            frame_ts_ms,
        )
        assert out is None

    def test_attached_entry_carries_reported_fields(self):
        a = _assoc()
        geo = a.node_geometries["site-a"]
        d_true, f_true = predict_observation(geo, _TRUE_LAT, _TRUE_LON, _TRUE_ALT_KM, _TRUE_VE, _TRUE_VN)
        st = _adsb_state("abc123")
        frame_ts_ms = int(_LAST_T_S * 1000)

        out = associate_detections_to_adsb(
            geo,
            [d_true],
            [f_true],
            {"abc123": st},
            frame_ts_ms,
        )
        assert out[0]["lat"] == st["lat"]
        assert out[0]["lon"] == st["lon"]
        assert out[0]["alt_baro"] == st["alt_baro"]
        assert out[0]["gs"] == st["gs"]
        assert out[0]["track"] == st["track"]
        assert out[0]["flight"] == st["flight"]


class TestAdsbSeedReset:
    def test_reset_for_tests_clears_the_new_counters(self):
        st = _adsb_state("abc123")
        a = _make_seed_assoc("active", lambda: {"abc123": st})
        tracks_a = [_view("a1", _NODE_A, "abc123")]
        tracks_b = [_view("b1", _NODE_B, "abc123")]
        a._pending_tracks["site-a"] = tracks_a
        a.submit_tracks_round("site-b", tracks_b, 2000)
        assert a.adsb_seed_rounds > 0
        assert a.adsb_tracklets_tagged > 0

        a._reset_for_tests()
        for name in (
            "adsb_seed_rounds",
            "adsb_tracklets_tagged",
            "adsb_seed_no_state",
            "adsb_seed_gate_rejects",
            "adsb_seed_world_rejects",
            "adsb_tracklets_excluded",
            "adsb_inputs_emitted",
        ):
            assert getattr(a, name) == 0


class TestSeedWorldGate:
    """A tagged state from the other world is no verification target.

    The provider's cache is keyed by bare hex and mixes simulated
    transponders with real traffic, so a hex collision hands the wrong
    world's fix to the tag check — and the residual gates are two numbers a
    wrong aircraft can pass by coincidence.  Fail-open in every unknown
    case: only a positive world mismatch rejects."""

    def _seed_round(self, st, node_world_provider=None):
        kw = {"adsb_seed_mode": "active", "adsb_provider": lambda: {"abc123": st}}
        if node_world_provider is not None:
            kw["node_world_provider"] = node_world_provider
        a = _assoc(**kw)
        a._pending_tracks["site-a"] = [_view("a1", _NODE_A, "abc123")]
        round_ = a.submit_tracks_round("site-b", [_view("b1", _NODE_B, "abc123")], 2000)
        return a, round_

    def test_cross_world_state_rejects_and_fails_open(self):
        st = _adsb_state("abc123")
        st["world"] = "real"
        a, round_ = self._seed_round(st, node_world_provider=lambda nid: "sim")

        assert a.adsb_seed_world_rejects == 2  # both round nodes' tags
        assert round_.adsb_inputs == []
        assert len(round_.pairs) == 1  # tracklets stay in dark pairing

    def test_same_world_state_still_seeds(self):
        st = _adsb_state("abc123")
        st["world"] = "sim"
        a, round_ = self._seed_round(st, node_world_provider=lambda nid: "sim")

        assert a.adsb_seed_world_rejects == 0
        assert len(round_.adsb_inputs) == 1

    def test_untagged_state_is_not_gated(self):
        a, round_ = self._seed_round(_adsb_state("abc123"), node_world_provider=lambda nid: "sim")

        assert a.adsb_seed_world_rejects == 0
        assert len(round_.adsb_inputs) == 1

    def test_unknown_node_world_is_not_gated(self):
        st = _adsb_state("abc123")
        st["world"] = "real"
        a, round_ = self._seed_round(st, node_world_provider=lambda nid: None)

        assert a.adsb_seed_world_rejects == 0
        assert len(round_.adsb_inputs) == 1

    def test_no_provider_ignores_world_tags(self):
        st = _adsb_state("abc123")
        st["world"] = "real"
        a, round_ = self._seed_round(st)

        assert a.adsb_seed_world_rejects == 0
        assert len(round_.adsb_inputs) == 1
