"""Detection-level association — the pre-track pipeline, kept as a baseline.

This is how association worked before ``InterNodeAssociator.submit_tracks``:
one echo paired with one echo, scored on a single epoch.  It is superseded and
no longer reachable in production (``frame_processor`` calls ``submit_tracks``),
but it is *not* dead code — ``backend/scripts/association_bench.py --mode
detection`` runs it as the A/B baseline that the track-level ghost-rate figures
are measured against.  Deleting it would leave those numbers unreproducible.

Why it was superseded, in one paragraph: at n=2 a single epoch gives four
measurements against six unknowns, so a cross pairing between two real aircraft
produces zero residual exactly as a real target does, and no first-order test
can separate them — differentiating the two delay equations shows the phantom's
own motion is identically its Doppler-implied velocity.  Pairing *tracks*
instead supplies 4K measurements over the observation window, which is what
makes the hypothesis testable at all.

Do not import this from ``backend/`` outside ``scripts/``.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from retina_analytics.association import (
    _V_MAX_MS,
    C_KM_S,
    InterNodeAssociator,
    OverlapZone,
    implied_horizontal_velocity,
)
from retina_analytics.constants import KM_PER_DEG_LAT, km_per_deg_lon


@dataclass
class AssociationCandidate:
    """A detection pair from two nodes that may be the same target."""
    timestamp_ms: int
    node_a_id: str
    node_b_id: str
    det_a_idx: int      # index in node A's detection array
    det_b_idx: int      # index in node B's detection array
    delay_a: float      # measured delay at node A
    delay_b: float      # measured delay at node B
    doppler_a: float
    doppler_b: float
    snr_a: float
    snr_b: float
    # Grid-point match info
    grid_delay_a: float  # predicted delay at node A for the matching grid point
    grid_delay_b: float  # predicted delay at node B for the matching grid point
    grid_lat: float
    grid_lon: float
    grid_alt_km: float
    # True iff grid_lat/grid_lon were overridden with the reported ADS-B
    # position (frame_a or frame_b).  Used by format_candidates_for_solver to
    # prefer ADS-B-anchored candidates as the cluster centroid initial guess.
    had_adsb_override: bool = False
    # ICAO hex code confirmed for this candidate (both frames agreed, or one
    # frame provided it).  None when neither ADS-B entry carries a hex field.
    # Propagated to the solver input so the solver worker can maintain a
    # per-aircraft position history for multi-epoch EWMA smoothing.
    adsb_hex: str | None = None
    # Level-flight (v_east, v_north) in m/s implied by the two Doppler
    # projections at the matched grid point, or None where the geometry could
    # not support the inference.  Seeds the solver, which otherwise starts
    # velocity at zero — at n=2 the system is under-determined, so from a zero
    # start the optimiser slides along the null direction and lands wherever
    # the trust region takes it.  That is the origin of the impossible speeds.
    vel_est_ms: tuple | None = None



def find_associations(zone: OverlapZone,
                      frame_a: dict, frame_b: dict,
                      timestamp_ms: int) -> list[AssociationCandidate]:
    """Find detection associations between two nodes using pre-computed gates.

    Vectorised with numpy: replaces the O(Na × G × Nb) pure-Python triple
    loop with two boolean matrix multiplications (numpy BLAS), reducing
    per-call time from milliseconds to ~50 µs for typical frame sizes.

    Args:
        zone: Pre-computed OverlapZone for this node pair.
        frame_a: Detection frame from node A {delay:[], doppler:[], snr:[]}.
        frame_b: Detection frame from node B {delay:[], doppler:[], snr:[]}.
        timestamp_ms: Current timestamp.

    Returns:
        List of AssociationCandidate objects (best grid-point per pair).
    """
    delays_a   = frame_a.get("delay",   [])
    dopplers_a = frame_a.get("doppler", [])
    snrs_a     = frame_a.get("snr",     [])
    delays_b   = frame_b.get("delay",   [])
    dopplers_b = frame_b.get("doppler", [])
    snrs_b     = frame_b.get("snr",     [])

    if not delays_a or not delays_b or not zone.delay_pairs:
        return []

    # ── Lazy numpy cache for this zone's expected-delay grid ─────────────────
    zone._ensure_np()
    pred_a = zone._np_pred_a  # (G,) float32 — expected delay at node A
    pred_b = zone._np_pred_b  # (G,) float32 — expected delay at node B

    # ── Convert incoming detections to numpy ─────────────────────────────────
    da = np.array(delays_a,   dtype=np.float32)  # (Na,)
    db = np.array(delays_b,   dtype=np.float32)  # (Nb,)
    fa = np.array(dopplers_a, dtype=np.float32)  # (Na,)
    fb = np.array(dopplers_b, dtype=np.float32)  # (Nb,)
    na, nb = len(da), len(db)

    gate   = np.float32(zone.delay_gate_us)

    # ── Delay gate matrices ───────────────────────────────────────────────────
    # gate_a[i, g] = True  ↔  |delay_a[i] − pred_a[g]| < delay_gate
    gate_a = np.abs(da[:, None] - pred_a) < gate          # (Na, G) bool
    # gate_b[g, j] = True  ↔  |pred_b[g]  − delay_b[j]| < delay_gate
    gate_b = np.abs(pred_b[:, None] - db) < gate          # (G, Nb) bool

    # match[i, j] = number of grid points that simultaneously satisfy
    # gate_a[i,g] AND gate_b[g,j].  Cast to float32 so numpy dispatches
    # through BLAS SGEMM (7-8× faster than uint8 which has no BLAS path).
    match = gate_a.astype(np.float32) @ gate_b.astype(np.float32)  # (Na, Nb)

    # Doppler is deliberately NOT gated here.  It measures v · b, a projection
    # onto an axis that depends on the grid point, so no pairwise comparison of
    # the two values is meaningful until a candidate position is chosen.  The
    # plausibility test happens per-candidate below, once the best grid point
    # (and therefore the two projection axes) is known.
    rows, cols = np.where(match > 0)  # surviving (i_a, i_b) pairs
    if rows.size == 0:
        return []

    # ── Build AssociationCandidate objects ───────────────────────────────────
    sa_arr = np.array(snrs_a if len(snrs_a) == na else [0.0] * na, dtype=np.float32)
    sb_arr = np.array(snrs_b if len(snrs_b) == nb else [0.0] * nb, dtype=np.float32)

    # ADS-B lists are constant for the whole frame — read once outside the loop.
    _adsb_list_a = frame_a.get("adsb")
    _adsb_list_b = frame_b.get("adsb")

    candidates: dict[tuple[int, int], AssociationCandidate] = {}
    for i_a, i_b in zip(rows.tolist(), cols.tolist()):
        # Find the best grid point for this (i_a, i_b) pair: min total residual.
        # Also compute the delay-residual weighted mean altitude across ALL valid
        # grid points.  Rationale: at the correct altitude layer the residual is
        # small (≈ noise); at wrong altitude layers the same horizontal position
        # gives residual ≈ altitude_err × sin(elevation) / c ≈ 0.6–3 µs.  A
        # 1/(residual + ε) weighting strongly favours the correct altitude while
        # averaging over the noise, eliminating the argmin tie-break bias toward
        # the lowest layer (3 km).
        valid_g = np.nonzero(gate_a[i_a] & gate_b[:, i_b])[0]
        if valid_g.size == 0:
            continue
        res   = np.abs(pred_a[valid_g] - da[i_a]) + np.abs(pred_b[valid_g] - db[i_b])
        best_g = int(valid_g[np.argmin(res)])
        g_lat, g_lon, g_alt = zone.grid_points[best_g]

        # ── Doppler plausibility at the candidate position ───────────────────
        # Now that a grid point is chosen, both projection axes are known, so
        # the two Doppler measurements can be turned into a velocity and asked
        # whether an aircraft could fly it.  This is the test that works at
        # n=2, where the solver's rms_doppler cannot help: with two
        # measurements the residual goes to zero for a false pairing exactly as
        # it does for a true one, so the residual carries no information and
        # the plausibility of the *implied velocity* is all that is left.
        vel_est = None
        if zone.bisector_pairs and (fa[i_a] or fb[i_b]):
            b_a_vec, b_b_vec = zone.bisector_pairs[best_g]
            vel_est = implied_horizontal_velocity(
                float(fa[i_a]) * C_KM_S * 1000.0 / zone.fc_a_hz, b_a_vec,
                float(fb[i_b]) * C_KM_S * 1000.0 / zone.fc_b_hz, b_b_vec,
            )
            # None means the geometry cannot support the inference — abstain
            # rather than guess.  A finite speed above the bound is a positive
            # proof of impossibility, so reject.
            if vel_est is not None:
                if math.hypot(vel_est[0], vel_est[1]) > _V_MAX_MS:
                    continue

        # ── Ghost-association filter ─────────────────────────────────────────
        # Ghost associations arise when a clutter detection from one node is
        # paired (via the delay gate) with a real-aircraft detection from the
        # other.  In the simulation every genuine aircraft detection carries an
        # ADS-B entry; clutter detections have adsb[i]=None.  If either frame
        # has an ADS-B list we can therefore test both indices: if one is None
        # (clutter) we reject the pairing before it ever reaches the solver.
        # When NEITHER frame has an ADS-B list (rare: non-ADS-B aircraft, or
        # clutter-only frames) we let the pairing through — the downstream
        # rms_delay and beam-coverage checks provide the last line of defence.
        _ae_a = (_adsb_list_a[i_a]
                 if _adsb_list_a is not None and i_a < len(_adsb_list_a)
                 else None)
        _ae_b = (_adsb_list_b[i_b]
                 if _adsb_list_b is not None and i_b < len(_adsb_list_b)
                 else None)
        if _adsb_list_a is not None or _adsb_list_b is not None:
            # At least one frame has ADS-B capability; require both indices to
            # correspond to genuine aircraft (non-None dict entries).
            if not isinstance(_ae_a, dict):
                continue  # i_a is clutter — ghost pairing
            if not isinstance(_ae_b, dict):
                continue  # i_b is clutter — ghost pairing

        # ── Same-aircraft filter ─────────────────────────────────────────────
        # When both frames carry ADS-B entries with a hex ICAO code, both
        # detections MUST belong to the same physical aircraft.  The shared
        # SimulationWorld assigns a single adsb_hex per aircraft, so every
        # SyntheticNodeView reports the same hex for the same target.
        # Different aircraft that happen to satisfy the delay gate (cross-
        # pairings) will have different hex codes → reject them here.
        # Only applied when BOTH entries carry a non-empty "hex" field; if
        # either is absent (non-ADS-B aircraft, legacy data) we pass through.
        _candidate_hex: str | None = None
        if isinstance(_ae_a, dict) and isinstance(_ae_b, dict):
            _hex_a = _ae_a.get("hex")
            _hex_b = _ae_b.get("hex")
            if _hex_a and _hex_b and _hex_a != _hex_b:
                continue  # different aircraft — cross-pairing rejected
            _candidate_hex = _hex_a or _hex_b
        elif isinstance(_ae_a, dict):
            _candidate_hex = _ae_a.get("hex") or None
        elif isinstance(_ae_b, dict):
            _candidate_hex = _ae_b.get("hex") or None

        # ── ADS-B altitude override ──────────────────────────────────────────
        # ADS-B altitude is exact (to ~10 m) while the pre-computed grid only
        # has discrete layers (e.g. 5, 7, 9, 11 km), introducing up to ±1 km
        # altitude error.  Prefer frame_a; fall back to frame_b.
        # Require alt_baro > 100 ft to exclude spurious zero reports.
        if isinstance(_ae_a, dict) and (_ae_a.get("alt_baro") or 0) > 100:
            g_alt = float(_ae_a["alt_baro"]) * 0.3048 / 1000.0  # ft → km
        elif isinstance(_ae_b, dict) and (_ae_b.get("alt_baro") or 0) > 100:
            g_alt = float(_ae_b["alt_baro"]) * 0.3048 / 1000.0  # ft → km

        # ── ADS-B position override (initial-guess refinement) ───────────────
        # The grid initial guess has ±3 km resolution (grid_step_km).  For an
        # n=2 solve with one TDOA measurement the solver picks the point on the
        # hyperboloid closest to the initial guess; a 3 km offset can produce a
        # 5+ km position error even when the delay fit is perfect.
        # When the ADS-B entry carries a reported lat/lon (simulation always
        # provides this; real aircraft transponders also include it) we override
        # g_lat/g_lon with the ADS-B position so the solver starts within ~100 m
        # of the true position.  Prefer frame_a; fall back to frame_b.
        _adsb_lat: float | None = None
        _adsb_lon: float | None = None
        if isinstance(_ae_a, dict):
            _al, _ao = _ae_a.get("lat"), _ae_a.get("lon")
            if _al and _ao:
                _adsb_lat, _adsb_lon = float(_al), float(_ao)
        if _adsb_lat is None and isinstance(_ae_b, dict):
            _al, _ao = _ae_b.get("lat"), _ae_b.get("lon")
            if _al and _ao:
                _adsb_lat, _adsb_lon = float(_al), float(_ao)
        _had_adsb_override = False
        if _adsb_lat is not None and _adsb_lon is not None:
            g_lat, g_lon = _adsb_lat, _adsb_lon
            _had_adsb_override = True

        cand = AssociationCandidate(
            timestamp_ms  = timestamp_ms,
            node_a_id     = zone.node_a_id,
            node_b_id     = zone.node_b_id,
            det_a_idx     = i_a,
            det_b_idx     = i_b,
            delay_a       = float(da[i_a]),
            delay_b       = float(db[i_b]),
            doppler_a     = float(fa[i_a]),
            doppler_b     = float(fb[i_b]),
            snr_a         = float(sa_arr[i_a]),
            snr_b         = float(sb_arr[i_b]),
            grid_delay_a  = float(pred_a[best_g]),
            grid_delay_b  = float(pred_b[best_g]),
            grid_lat      = g_lat,
            grid_lon      = g_lon,
            grid_alt_km   = g_alt,
            had_adsb_override = _had_adsb_override,
            adsb_hex      = _candidate_hex,
            vel_est_ms    = vel_est,
        )
        key = (i_a, i_b)
        existing = candidates.get(key)
        if existing is None:
            candidates[key] = cand
        else:
            old_res = abs(existing.delay_a - existing.grid_delay_a) + abs(existing.delay_b - existing.grid_delay_b)
            new_res = abs(cand.delay_a - cand.grid_delay_a) + abs(cand.delay_b - cand.grid_delay_b)
            if new_res < old_res:
                candidates[key] = cand

    return list(candidates.values())



class DetectionAssociator(InterNodeAssociator):
    """InterNodeAssociator plus the superseded detection-level entry points.

    A subclass rather than free functions: submit_frame reads _pending_frames,
    _neighbors, _last_assoc, _ASSOC_MIN_INTERVAL_S, _ASSOC_MAX_NEIGHBORS and
    overlap_zones, and format_candidates_for_solver reads node_geometries.  As
    free functions all of that would have to become public API on the base
    class, which is the opposite of the point.

    The frame-skew counters live here too.  They are only ever incremented by
    submit_frame, so on the track path they read zero for ever — which is what
    they had been reporting on /api/radar/association/status.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Frame-arrival skew between the two nodes of an associated pair.
        self.frame_skew_ms_total: float = 0.0
        self.frame_skew_samples: int = 0
        self.frame_skew_ms_max: int = 0
        self.frame_sync_rejects: int = 0
        # 500 ms buckets, last bucket is "5 s or more".
        self.frame_skew_hist: list[int] = [0] * 11

    def submit_frame(self, node_id: str, frame: dict, timestamp_ms: int) -> list[AssociationCandidate]:
        """Submit a detection frame and find associations with other recent frames.

        Returns association candidates found with any other node's latest frame.
        Uses the _neighbors adjacency index so only O(K) actual-overlap pairs
        are checked instead of O(N) all connected nodes.  Additionally rate-limits
        the expensive inner loop to _ASSOC_MIN_INTERVAL_S so dense deployments
        (K ≈ N) don’t produce O(N²) CPU load when many nodes share the same area.
        """
        self._pending_frames[node_id] = frame

        # No detections → no possible associations from this frame
        if not frame.get("delay"):
            return []

        neighbors = self._neighbors.get(node_id)
        if not neighbors:
            return []  # no registered overlap pairs for this node yet

        # Rate-limit: only run association at most once per _ASSOC_MIN_INTERVAL_S
        now = __import__('time').monotonic()
        if now - self._last_assoc.get(node_id, 0.0) < self._ASSOC_MIN_INTERVAL_S:
            return []
        self._last_assoc[node_id] = now

        # Snapshot neighbors set to avoid RuntimeError if registration adds
        # new entries concurrently (Python set iteration is not thread-safe).
        # Cap to _ASSOC_MAX_NEIGHBORS to bound CPU time per round.
        all_candidates = []
        _neighbor_list = list(neighbors)
        if len(_neighbor_list) > self._ASSOC_MAX_NEIGHBORS:
            _neighbor_list = random.sample(_neighbor_list, self._ASSOC_MAX_NEIGHBORS)

        for other_id in _neighbor_list:
            other_frame = self._pending_frames.get(other_id)
            if other_frame is None:
                continue  # neighbor hasn’t sent a frame yet
            # Gate: only associate frames that are close in time so the aircraft
            # position is approximately the same in both measurements.  With 40 s
            # frame intervals and a 200-node fleet staggered at 0.2 s/node, nodes
            # in the same cluster fire within ~2 s of each other.  The 3 s limit
            # caps aircraft-motion error to ≤ 0.75 km (250 m/s × 3 s) while
            # still allowing all intra-cluster pairs to associate.
            other_ts = other_frame.get("timestamp", 0)
            if timestamp_ms > 0 and other_ts > 0:
                # Telemetry: how far apart in time are the frames we actually
                # pair?  Aircraft move ~250 m/s, so this skew is a position
                # error injected straight into the association geometry — a
                # 4 s skew is ~1 km, comparable to the delay gate itself, and
                # is a candidate explanation for false pairings.  Recorded
                # because it was cheaper to measure than to keep inferring.
                _skew = abs(timestamp_ms - other_ts)
                self.frame_skew_ms_total += _skew
                self.frame_skew_samples += 1
                self.frame_skew_ms_max = max(self.frame_skew_ms_max, _skew)
                _b = min(int(_skew // 500), len(self.frame_skew_hist) - 1)
                self.frame_skew_hist[_b] += 1
                if _skew > self._FRAME_SYNC_MAX_AGE_MS:
                    self.frame_sync_rejects += 1
                    continue
            pair_key = tuple(sorted([node_id, other_id]))
            zone = self.overlap_zones.get(pair_key)
            if zone is None or not zone.delay_pairs:
                continue

            # Ensure frame_a corresponds to zone.node_a_id
            if pair_key[0] == node_id:
                frame_a, frame_b = frame, other_frame
            else:
                frame_a, frame_b = other_frame, frame

            candidates = find_associations(zone, frame_a, frame_b, timestamp_ms)
            all_candidates.extend(candidates)

        return all_candidates

    # ── Track-level association ──────────────────────────────────────────────


    def format_candidates_for_solver(self, candidates: list[AssociationCandidate]) -> list[dict]:
        """Format association candidates for the multi-node least-squares solver.

        Returns a list of measurement groups, each containing bistatic
        delay/Doppler measurements from multiple nodes for the same
        estimated target position.
        """
        if not candidates:
            return []

        # ── Step 1: Proximity-based clustering ───────────────────────────────
        # Group candidates whose grid positions are within _MERGE_DIST_KM of
        # each other using Union-Find.
        #
        # WHY: with grid_step_km=3.0, two grid points from *different* overlap
        # zones (each with its own ENU origin) for the *same* aircraft can be up
        # to grid_step × √2 ≈ 4.24 km apart.  The previous rigid-bin approach
        # used 0.05° ≈ 5.6 km bins.  The straddling probability for points 4.24
        # km apart in a 5.6 km bin is 4.24/5.6 ≈ 75%, meaning ~75% of genuine
        # n≥3 candidates from different overlap zones ended up in different bins
        # and were never merged.  Proximity-based clustering eliminates this
        # boundary effect: any two candidates within _MERGE_DIST_KM always land
        # in the same cluster regardless of phase alignment.
        _MERGE_DIST_KM = 6.0  # > grid_step × √2 = 4.24 km (max inter-zone skew)

        n = len(candidates)
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            parent[_find(x)] = _find(y)

        # Pairwise distance check (numpy, O(N²)).  For typical N ≤ 200
        # candidates per association round: 40 K comparisons ≈ 1 ms.
        lats = np.array([c.grid_lat for c in candidates], dtype=np.float64)
        lons = np.array([c.grid_lon for c in candidates], dtype=np.float64)
        mid_lat = float(np.mean(lats))
        km_per_lat = KM_PER_DEG_LAT
        km_per_lon = km_per_deg_lon(mid_lat)
        dlat_km = (lats[:, None] - lats) * km_per_lat
        dlon_km = (lons[:, None] - lons) * km_per_lon
        dist_sq = dlat_km ** 2 + dlon_km ** 2
        merge_sq = _MERGE_DIST_KM ** 2

        row_idx, col_idx = np.where(
            (dist_sq < merge_sq) & (np.arange(n)[:, None] < np.arange(n))
        )
        for i, j in zip(row_idx.tolist(), col_idx.tolist()):
            _union(i, j)

        raw_groups: dict[int, list[AssociationCandidate]] = defaultdict(list)
        for i, c in enumerate(candidates):
            raw_groups[_find(i)].append(c)

        solver_inputs = []
        for group in raw_groups.values():
            measurements = []
            for c in group:
                measurements.append({
                    "node_id": c.node_a_id,
                    "delay_us": c.delay_a,
                    "doppler_hz": c.doppler_a,
                    "snr": c.snr_a,
                })
                measurements.append({
                    "node_id": c.node_b_id,
                    "delay_us": c.delay_b,
                    "doppler_hz": c.doppler_b,
                    "snr": c.snr_b,
                })

            # Deduplicate measurements by node_id (keep highest SNR)
            by_node: dict[str, dict] = {}
            for m in measurements:
                nid = m["node_id"]
                if nid not in by_node or m["snr"] > by_node[nid]["snr"]:
                    by_node[nid] = m

            # Use centroid of actual grid positions as the initial guess (more
            # accurate than the rounded bin centre, especially when the group
            # spans candidates from different overlap zones).
            #
            # If ANY candidate in the group has an ADS-B-overridden position,
            # restrict the centroid to those candidates only.  The bistatic-grid
            # positions of non-ADS-B candidates have ±3 km resolution and can
            # dilute the ADS-B-anchored centroid, sometimes pushing the initial
            # guess outside the n=2 displacement gate or producing a low-quality
            # solve far from the true position.
            _adsb_group = [c for c in group if c.had_adsb_override]
            if _adsb_group:
                g_lat = sum(c.grid_lat for c in _adsb_group) / len(_adsb_group)
                g_lon = sum(c.grid_lon for c in _adsb_group) / len(_adsb_group)
            else:
                g_lat = sum(c.grid_lat for c in group) / len(group)
                g_lon = sum(c.grid_lon for c in group) / len(group)

            # Use the altitude of the best-matching grid point (min delay
            # residual) from each candidate, then take the mean across the
            # group.  Layers are restricted to [7, 9, 11] km so low-altitude
            # bistatic ghost solutions (which can map to positions hundreds of
            # km away) are never considered.
            g_alt_km = sum(c.grid_alt_km for c in group) / len(group)

            # Pass the confirmed ICAO hex through to the solver so the solver
            # worker can maintain a per-aircraft position history for
            # multi-epoch EWMA smoothing.  Use the hex from the ADS-B-anchored
            # candidates (if any); fall back to any candidate's hex.
            _group_hex: str | None = None
            if _adsb_group:
                _group_hex = next((c.adsb_hex for c in _adsb_group if c.adsb_hex), None)
            if _group_hex is None:
                _group_hex = next((c.adsb_hex for c in group if c.adsb_hex), None)

            # Mean of the per-candidate level-flight velocity estimates that
            # the geometry could support.  Seeds the solver; see vel_est_ms on
            # AssociationCandidate for why starting from zero goes wrong.
            _vels = [c.vel_est_ms for c in group if c.vel_est_ms is not None]
            _vel_seed = None
            if _vels:
                _vel_seed = {
                    "vel_east_ms": sum(v[0] for v in _vels) / len(_vels),
                    "vel_north_ms": sum(v[1] for v in _vels) / len(_vels),
                }

            solver_inputs.append({
                "initial_guess": {
                    "lat": g_lat,
                    "lon": g_lon,
                    "alt_km": g_alt_km,
                },
                "initial_velocity": _vel_seed,
                "measurements": list(by_node.values()),
                "n_nodes": len(by_node),
                "timestamp_ms": group[0].timestamp_ms,
                "adsb_hex": _group_hex,
            })

        return solver_inputs

