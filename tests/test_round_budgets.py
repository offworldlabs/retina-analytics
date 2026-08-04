"""Neighbour and pairing budgets on the track path.

Both fixes here address the same failure mode: a bound that was written for one
configuration and silently did something else in the one that ships.

submit_frame capped neighbours at _ASSOC_MAX_NEIGHBORS from the start;
submit_tracks never did, so the live path was the uncapped one.  And the
per-round pairing cap was keyed to the *fit* budget, which is only decremented
when a fit actually runs — with cv_fit=None (production) it never depleted, so
its truncation applied as a fixed cap per node pair, discarding candidates to
bound a cost that was not being paid.
"""

from retina_analytics.association import InterNodeAssociator


def _cfg(i):
    """Nodes on a tight grid so every pair overlaps and no zone is empty."""
    return {
        "rx_lat": 34.85 + (i % 5) * 0.05,
        "rx_lon": -82.40 + (i // 5) * 0.05,
        "rx_alt_ft": 900,
        "tx_lat": 34.90, "tx_lon": -82.30, "tx_alt_ft": 1600,
        "fc_hz": 195e6, "beam_width_deg": 360, "max_range_km": 80,
    }


def _tracks(prefix, n, delay=30.0):
    return [
        {
            "track_id": f"{prefix}-{k}",
            "history": [
                {"t_s": t, "delay_us": delay + k, "doppler_hz": 10.0, "snr": 12.0}
                for t in range(0, 20, 2)
            ],
        }
        for k in range(n)
    ]


def _tracks_on_grid(assoc, a_id, b_id, n):
    """n tracks per node whose delays sit on real grid points of their zone.

    Invented delays are rejected by the coarse gate, so a budget test built on
    them measures nothing: the cap is never the binding constraint.
    """
    zone = assoc.overlap_zones[tuple(sorted([a_id, b_id]))]
    assert zone.delay_pairs, "test fleet produced an empty overlap zone"
    step = max(len(zone.delay_pairs) // n, 1)
    picks = [zone.delay_pairs[(i * step) % len(zone.delay_pairs)] for i in range(n)]

    def build(prefix, idx):
        return [
            {
                "track_id": f"{prefix}-{k}",
                "history": [
                    {"t_s": t, "delay_us": float(picks[k][idx]),
                     "doppler_hz": 10.0, "snr": 12.0}
                    for t in range(0, 20, 2)
                ],
            }
            for k in range(n)
        ]

    lo, hi = tuple(sorted([a_id, b_id]))
    return build(lo, 0), build(hi, 1)


def _fleet(n_nodes, max_neighbors=50, max_pairs_per_round=64):
    a = InterNodeAssociator(
        grid_step_km=25.0, assoc_interval_s=0.0,
        max_neighbors=max_neighbors, max_pairs_per_round=max_pairs_per_round,
    )
    for i in range(n_nodes):
        a.register_node(f"n{i}", _cfg(i))
    return a


class TestNeighbourCap:
    def test_a_round_visits_at_most_the_cap(self):
        a = _fleet(12, max_neighbors=3)
        for i in range(12):
            a._pending_tracks[f"n{i}"] = _tracks(f"n{i}", 1)

        visited = set()
        original = a._pair_tracks

        def spy(zone, *args, **kw):
            visited.add(zone.node_a_id)
            visited.add(zone.node_b_id)
            return original(zone, *args, **kw)

        a._pair_tracks = spy
        a.submit_tracks("n0", a._pending_tracks["n0"], 1000)

        # n0 plus at most 3 neighbours.
        assert len(visited - {"n0"}) <= 3

    def test_the_cursor_rotates_so_the_tail_is_not_starved(self):
        """Set iteration order is fixed for the process, so a plain slice would
        hand the same neighbours every round and never reach the rest.  The
        'nothing is lost, only deferred' claim depends on this."""
        a = _fleet(12, max_neighbors=2)
        for i in range(12):
            a._pending_tracks[f"n{i}"] = _tracks(f"n{i}", 1)

        seen = set()
        original = a._pair_tracks

        def spy(zone, *args, **kw):
            seen.add(zone.node_a_id)
            seen.add(zone.node_b_id)
            return original(zone, *args, **kw)

        a._pair_tracks = spy
        neighbours = len(a._neighbors.get("n0", ()))
        for _ in range(neighbours * 2):
            a.submit_tracks("n0", a._pending_tracks["n0"], 1000)

        assert len(seen - {"n0"}) == neighbours, (
            f"only {len(seen) - 1} of {neighbours} neighbours ever visited")

    def test_a_small_fleet_is_unaffected(self):
        a = _fleet(4, max_neighbors=50)
        for i in range(4):
            a._pending_tracks[f"n{i}"] = _tracks(f"n{i}", 1)
        a.submit_tracks("n0", a._pending_tracks["n0"], 1000)
        # No rotation needed, so no deferral is recorded.
        assert a.track_pairs_deferred == 0


class TestPairBudget:
    def test_the_deferred_path_is_not_capped_at_the_fit_budget(self):
        """The regression: cv_fit=None means the fit budget never decrements,
        so tying the truncation to it capped candidates at 8 per node pair."""
        a = _fleet(2, max_pairs_per_round=64)
        assert a.cv_fit is None
        t0, t1 = _tracks_on_grid(a, "n0", "n1", 12)
        a._pending_tracks["n0"], a._pending_tracks["n1"] = t0, t1

        out = a.submit_tracks("n0", a._pending_tracks["n0"], 1000)

        assert len(out) > a._MAX_FITS_PER_ROUND, (
            f"only {len(out)} pairings emitted; the fit budget is still capping "
            "a path that runs no fit")

    def test_the_pair_budget_still_bounds_the_round(self):
        """Unbounded would convert a frame-path bound into a solver-queue
        overflow, which fails more quietly."""
        a = _fleet(2, max_pairs_per_round=5)
        t0, t1 = _tracks_on_grid(a, "n0", "n1", 12)
        a._pending_tracks["n0"], a._pending_tracks["n1"] = t0, t1

        out = a.submit_tracks("n0", a._pending_tracks["n0"], 1000)

        assert len(out) <= 5

    def test_an_inline_fit_still_uses_the_fit_budget(self):
        """The inline path pays ~86 ms per pairing and must stay bounded by the
        smaller number, whatever the pair budget says."""
        calls = {"n": 0}

        def fake_fit(fit_input, node_cfgs):
            calls["n"] += 1
            return {"success": True, "chi2_per_dof": 0.5, "dof": 10,
                    "lat": 34.9, "lon": -82.35, "alt_m": 8000.0,
                    "vel_east": 100.0, "vel_north": 0.0, "n_epochs": 10}

        a = InterNodeAssociator(
            grid_step_km=25.0, assoc_interval_s=0.0, cv_fit=fake_fit,
            cv_min_epochs=2, cv_min_span_s=1.0,
            max_pairs_per_round=64,
        )
        for i in range(2):
            a.register_node(f"n{i}", _cfg(i))
        t0, t1 = _tracks_on_grid(a, "n0", "n1", 12)
        a._pending_tracks["n0"], a._pending_tracks["n1"] = t0, t1

        a.submit_tracks("n0", a._pending_tracks["n0"], 1000)

        assert calls["n"] <= a._MAX_FITS_PER_ROUND


class TestDeferredCounterMeansDeferred:
    """track_pairs_deferred increments once per round that actually cut work
    short — not on every rotated round of a large fleet, and not twice when
    rotation and budget exhaustion coincide."""

    def test_rotation_alone_does_not_count_as_deferral(self):
        a = InterNodeAssociator(grid_step_km=25.0, assoc_interval_s=0.0,
                                max_neighbors=2)
        for i in range(6):
            a.register_node(f"n{i}", _cfg(i))
        # No pending tracks anywhere: the round rotates the cursor but defers
        # no actual work.
        a.submit_tracks("n0", [], 1000)
        assert a.track_pairs_deferred == 0

    def test_budget_exhaustion_counts_once(self):
        a = InterNodeAssociator(grid_step_km=25.0, assoc_interval_s=0.0,
                                max_pairs_per_round=1)
        for i in range(2):
            a.register_node(f"n{i}", _cfg(i))
        t0, t1 = _tracks_on_grid(a, "n0", "n1", 12)
        a._pending_tracks["n0"], a._pending_tracks["n1"] = t0, t1
        before = a.track_pairs_deferred
        a.submit_tracks("n0", a._pending_tracks["n0"], 1000)
        assert a.track_pairs_deferred - before <= 1
