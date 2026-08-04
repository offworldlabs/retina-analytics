"""Central analytics aggregator for all connected nodes."""

import os
import time
import threading

from retina_analytics.trust import AdsReportEntry, TrustScoreState
from retina_analytics.detection_area import DetectionAreaState
from retina_analytics.metrics import NodeMetrics
from retina_analytics.reputation import NodeReputation
from retina_analytics.coverage import HistoricalCoverageMap
from retina_analytics.empirical_coverage import (
    CALIBRATION_SCHEMA,
    EmpiricalCoverageState,
)
from retina_analytics.cross_node import compute_delay_bin_overlap, coverage_suggestion
from retina_analytics.constants import YAGI_BEAM_WIDTH_DEG, YAGI_MAX_RANGE_KM, haversine_km, resolve_beam_azimuth_deg

_RX_RELOCATE_THRESHOLD_KM = 0.05   # 50 m — above real GPS/reporting jitter


class NodeAnalyticsManager:
    """Central analytics aggregator for all connected nodes."""

    _ANALYSIS_CACHE_TTL = 60  # seconds

    def __init__(self, storage_dir: str = ""):
        self.trust_scores: dict[str, TrustScoreState] = {}
        self.detection_areas: dict[str, DetectionAreaState] = {}
        self.metrics: dict[str, NodeMetrics] = {}
        self.reputations: dict[str, NodeReputation] = {}
        self.coverage_maps: dict[str, HistoricalCoverageMap] = {}
        self.empirical_coverages: dict[str, EmpiricalCoverageState] = {}
        self._storage_dir = storage_dir
        self._last_save_time = 0.0
        self._save_interval_s = 300.0
        self._cross_node_cache: dict | None = None
        self._cross_node_cache_ts: float = 0.0
        self._summaries_cache: dict | None = None
        self._summaries_cache_ts: float = 0.0
        self._analytics_lock = threading.Lock()
        # RLock: guards the per-node dicts against structural mutation
        # (register/retire) racing the save/load iterations.  Re-entrant so
        # maybe_auto_save → save_coverage_maps can hold it across both.
        self._save_lock = threading.RLock()
        if storage_dir:
            self._load_coverage_maps()

    def register_node(self, node_id: str, config: dict):
        # Locked: save_coverage_maps / _load_coverage_maps iterate these dicts
        # from other threads, and an unlocked insert mid-iteration raises
        # "dictionary changed size during iteration".
        with self._save_lock:
            self._register_node_locked(node_id, config)

    def _register_node_locked(self, node_id: str, config: dict):
        if node_id not in self.trust_scores:
            self.trust_scores[node_id] = TrustScoreState(node_id=node_id)

        rx_lat = config.get("rx_lat", 0)
        rx_lon = config.get("rx_lon", 0)
        tx_lat = config.get("tx_lat", 0)
        tx_lon = config.get("tx_lon", 0)
        # Honour an explicit aim if the config supplies one; else broadside.
        beam_az = resolve_beam_azimuth_deg(config, rx_lat, rx_lon, tx_lat, tx_lon)
        self.detection_areas[node_id] = DetectionAreaState(
            node_id=node_id,
            rx_lat=rx_lat,
            rx_lon=rx_lon,
            tx_lat=tx_lat,
            tx_lon=tx_lon,
            fc_hz=config.get("fc_hz", config.get("FC", 195e6)),
            beam_azimuth_deg=beam_az,
            beam_width_deg=config.get("beam_width_deg", YAGI_BEAM_WIDTH_DEG),
            max_range_km=config.get("max_range_km", YAGI_MAX_RANGE_KM),
            max_bistatic_range_km=config.get("max_bistatic_range_km"),
        )

        # Preserve accumulated metrics across reconnects.  Every other
        # per-node store here is conditional, but this one was replaced
        # unconditionally — a reconnect wiped total_frames / SNR / gap
        # history and then fed reputation a fresh 0.0 detection rate.
        existing_metrics = self.metrics.get(node_id)
        if existing_metrics is None:
            self.metrics[node_id] = NodeMetrics(
                node_id=node_id,
                connected_at=time.time(),
            )
        else:
            existing_metrics.connected_at = time.time()

        if node_id not in self.reputations:
            self.reputations[node_id] = NodeReputation(node_id=node_id)

        if node_id not in self.coverage_maps:
            self.coverage_maps[node_id] = HistoricalCoverageMap(node_id=node_id)

        # Recreate empirical coverage when the node is new OR its RX moved — node
        # IDs are reused across fleet regenerations at different positions, so a
        # persisted polygon from the old location would otherwise be served for
        # the new one (stale, beam-mismatched, collapsed).
        ec = self.empirical_coverages.get(node_id)
        cfg_max_range = config.get("max_range_km", YAGI_MAX_RANGE_KM)
        moved = ec is not None and haversine_km(ec.rx_lat, ec.rx_lon, rx_lat, rx_lon) > _RX_RELOCATE_THRESHOLD_KM
        # A change in the *range rule* invalidates the accumulated polygon just
        # as surely as the RX physically moving: switching a node from a
        # monostatic limit to a bistatic one reshapes its footprint from a
        # circle on the RX to an ellipse with foci at RX and TX.  Detecting it
        # here means persisted coverage self-heals on the next registration —
        # important because production mounts coverage_data as a named volume
        # that survives rebuilds, so a stale polygon would otherwise be served
        # indefinitely with no operator action to prompt it.
        # Scoped to the bistatic key alone.  A max_range_km retune deliberately
        # *keeps* accumulated calibration (see the else branch) because it only
        # moves the clamp; switching range rules changes the footprint's shape,
        # which is a different thing.
        cfg_bistatic = config.get("max_bistatic_range_km")
        rule_changed = (
            ec is not None
            and getattr(ec, "max_bistatic_range_km", None) != cfg_bistatic
        )
        # A polygon accumulated under an older calibration input is discarded on
        # the same footing.  The bistatic key cannot catch this one: switching
        # the feed from solver output to ADS-B positions changes what the bins
        # *mean* without changing any configured value, so a v1 polygon — shaped
        # partly by ghosts — would otherwise survive every restart on the named
        # coverage volume, with nothing to prompt an operator.
        schema_changed = (
            ec is not None and getattr(ec, "schema", 1) != CALIBRATION_SCHEMA
        )
        if ec is None or moved or rule_changed or schema_changed:
            self.empirical_coverages[node_id] = EmpiricalCoverageState(
                rx_lat=rx_lat, rx_lon=rx_lon,
                max_range_km=cfg_max_range,
                tx_lat=tx_lat, tx_lon=tx_lon,
            )
            # Record the rule this polygon was built under so the next
            # registration can tell whether it is still valid.
            self.empirical_coverages[node_id].max_bistatic_range_km = cfg_bistatic
            # Remove stale on-disk file from previous location; save_coverage_maps
            # skips n_points==0 states, so the old file would persist and be loaded
            # on restart, resurrecting the stale polygon.
            if self._storage_dir:
                try:
                    os.remove(self._empirical_path(node_id))
                except FileNotFoundError:
                    pass
        else:
            # Same RX (within jitter) and same range rule — keep accumulated
            # calibration but track bound.
            ec.max_range_km = cfg_max_range
            ec.max_bistatic_range_km = cfg_bistatic
            # A reconnect can carry a corrected TX; the clamp follows it.
            ec.tx_lat, ec.tx_lon = tx_lat, tx_lon

    def coverage_limit_for(self, node_id: str):
        """A bearing → observed-limit-km callable for one node, or None.

        Handed to InterNodeAssociator so the overlap grid can be tightened to
        what a node has actually been seen to detect.  Returns None when the
        node has no polygon at all, so a fresh node is unconstrained rather than
        blind.
        """
        ec = self.empirical_coverages.get(node_id)
        if ec is None:
            return None
        return ec.observed_limit_km

    def coverage_digest(self, node_id: str):
        """Change token for a node's observed limits; see constraint_digest."""
        ec = self.empirical_coverages.get(node_id)
        return ec.constraint_digest() if ec is not None else None

    def retire_node(self, node_id: str) -> dict:
        """Forget a node entirely — in-memory state and its files on disk.

        Nothing else removes a node.  register_node only ever adds, and the
        refresh/save paths iterate whatever is in these dicts, so a receiver
        that leaves the fleet keeps its trust score, detection area, coverage
        polygon and empirical calibration for the life of the deployment, and
        every subsequent pass pays for them.  On staging, 10 receivers from a
        superseded fleet layout were still being iterated and re-saved 40
        minutes after they stopped existing.

        This is deliberately *not* driven by a staleness timer.  A real
        receiver offline for a week is still a real receiver whose accumulated
        coverage we want when it comes back; only an explicit decision that the
        node is gone should discard it.  Callers own that decision.

        Returns what was actually dropped, so a caller can report it rather
        than guess.
        """
        with self._save_lock:
            dropped = {
                name: (node_id in store)
                for name, store in (
                    ("trust_score", self.trust_scores),
                    ("detection_area", self.detection_areas),
                    ("metrics", self.metrics),
                    ("reputation", self.reputations),
                    ("coverage_map", self.coverage_maps),
                    ("empirical_coverage", self.empirical_coverages),
                )
            }
            for store in (
                self.trust_scores, self.detection_areas, self.metrics,
                self.reputations, self.coverage_maps, self.empirical_coverages,
            ):
                store.pop(node_id, None)

        files = []
        if self._storage_dir:
            for path in (self._coverage_map_path(node_id), self._empirical_path(node_id)):
                try:
                    os.remove(path)
                    files.append(os.path.basename(path))
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

        # Any summary cached before this call still names the node.
        self._summaries_cache = None
        self._cross_node_cache = None

        return {"node_id": node_id, "dropped": dropped, "files_removed": files}

    def is_node_blocked(self, node_id: str) -> bool:
        rep = self.reputations.get(node_id)
        return rep.blocked if rep else False

    def record_calibration_point(self, node_id: str, lat: float, lon: float) -> None:
        """Record a detection at an independently-known target position.

        ADS-B only.  Callers used to pass solver output here, which made the
        polygon a picture of what the solver believed rather than of what the
        node can see; see CALIBRATION_SCHEMA.
        """
        ec = self.empirical_coverages.get(node_id)
        if ec is not None:
            ec.add_point(lat, lon)

    def record_detection_frame(self, node_id: str, frame: dict):
        if self.is_node_blocked(node_id):
            return False
        if node_id in self.detection_areas:
            self.detection_areas[node_id].update_from_frame(frame)
        if node_id in self.metrics:
            self.metrics[node_id].record_frame(frame)
        return True

    def record_adsb_correlation(self, node_id: str, entry: AdsReportEntry):
        if node_id not in self.trust_scores:
            self.trust_scores[node_id] = TrustScoreState(node_id=node_id)
        self.trust_scores[node_id].add_sample(entry)

        delay_err = abs(entry.predicted_delay - entry.measured_delay)
        if node_id in self.coverage_maps:
            self.coverage_maps[node_id].add_detection(
                lat=entry.adsb_lat, lon=entry.adsb_lon, alt_km=0.0,
                snr=0.0, delay_error=delay_err,
            )

    def record_heartbeat(self, node_id: str):
        if node_id in self.metrics:
            self.metrics[node_id].record_heartbeat()

    def record_node_tracks(self, node_id: str, confirmed_ids, geolocated_ids=()):
        """Feed the per-node distinct-track counters (see NodeMetrics.record_tracks)."""
        m = self.metrics.get(node_id)
        if m is not None:
            m.record_tracks(confirmed_ids, geolocated_ids)

    def evaluate_reputations(self):
        for node_id, rep in self.reputations.items():
            ts = self.trust_scores.get(node_id)
            if ts and ts.samples:
                rep.evaluate_trust(ts.score)

            metrics = self.metrics.get(node_id)
            if metrics:
                rep.evaluate_heartbeat(metrics.last_heartbeat)
                rep.evaluate_detection_rate(metrics.avg_detections_per_frame)

        node_ids = sorted(self.reputations.keys())
        for i, a_id in enumerate(node_ids):
            for b_id in node_ids[i + 1:]:
                area_a = self.detection_areas.get(a_id)
                area_b = self.detection_areas.get(b_id)
                if area_a and area_b and area_a.n_detections > 0 and area_b.n_detections > 0:
                    dist = haversine_km(area_a.rx_lat, area_a.rx_lon,
                                        area_b.rx_lat, area_b.rx_lon)
                    # Bistatic footprints reach delta/2 + L from the RX; the
                    # monostatic sum pruned genuinely overlapping pairs.
                    if dist > area_a.footprint_radius_km() + area_b.footprint_radius_km():
                        continue
                    overlap = compute_delay_bin_overlap(area_a, area_b)
                    ts_a = self.trust_scores.get(a_id)
                    ts_b = self.trust_scores.get(b_id)
                    if ts_a and ts_b:
                        self.reputations[a_id].evaluate_neighbour_consistency(
                            overlap["overlap_ratio"], ts_b.score, neighbour_id=b_id
                        )
                        self.reputations[b_id].evaluate_neighbour_consistency(
                            overlap["overlap_ratio"], ts_a.score, neighbour_id=a_id
                        )

    def unblock_node(self, node_id: str):
        rep = self.reputations.get(node_id)
        if rep:
            rep.unblock()

    def get_node_summary(self, node_id: str) -> dict:
        result = {"node_id": node_id}
        if node_id in self.trust_scores:
            result["trust"] = self.trust_scores[node_id].summary()
        if node_id in self.detection_areas:
            result["detection_area"] = self.detection_areas[node_id].summary()
        if node_id in self.metrics:
            result["metrics"] = self.metrics[node_id].summary()
        if node_id in self.reputations:
            result["reputation"] = self.reputations[node_id].summary()
        if node_id in self.coverage_maps:
            result["coverage_map"] = self.coverage_maps[node_id].summary()
        ec = self.empirical_coverages.get(node_id)
        if ec is not None:
            da = self.detection_areas.get(node_id)
            poly_kwargs = {}
            if da is not None:
                poly_kwargs["beam_azimuth_deg"] = da.beam_azimuth_deg
                poly_kwargs["beam_width_deg"] = da.beam_width_deg
                poly_kwargs["max_range_km"] = da.max_range_km
            result["empirical_coverage"] = {
                "n_points": ec.n_points,
                "n_filled_bins": ec.n_filled_bins,
                "polygon": ec.to_polygon(**poly_kwargs),
            }
        return result

    def get_all_summaries(self) -> dict:
        now = time.monotonic()
        if self._summaries_cache is not None and now - self._summaries_cache_ts < self._ANALYSIS_CACHE_TTL:
            return self._summaries_cache
        with self._analytics_lock:
            # Double-check after acquiring lock
            now = time.monotonic()
            if self._summaries_cache is not None and now - self._summaries_cache_ts < self._ANALYSIS_CACHE_TTL:
                return self._summaries_cache
            all_nodes = (set(self.trust_scores) | set(self.detection_areas)
                         | set(self.metrics) | set(self.reputations))
            result = {nid: self.get_node_summary(nid) for nid in sorted(all_nodes)}
            self._summaries_cache = result
            self._summaries_cache_ts = now
            return result

    def get_cross_node_analysis(self) -> dict:
        now = time.monotonic()
        if self._cross_node_cache is not None and now - self._cross_node_cache_ts < self._ANALYSIS_CACHE_TTL:
            return self._cross_node_cache
        with self._analytics_lock:
            # Double-check after acquiring lock
            now = time.monotonic()
            if self._cross_node_cache is not None and now - self._cross_node_cache_ts < self._ANALYSIS_CACHE_TTL:
                return self._cross_node_cache

            node_ids = sorted(self.detection_areas.keys())
            pair_overlaps = []
            # Only compute pair overlaps for nodes within range of each other
            for i, a_id in enumerate(node_ids):
                area_a = self.detection_areas[a_id]
                for b_id in node_ids[i + 1:]:
                    area_b = self.detection_areas[b_id]
                    dist = haversine_km(area_a.rx_lat, area_a.rx_lon,
                                        area_b.rx_lat, area_b.rx_lon)
                    # Bistatic footprints reach delta/2 + L from the RX; the
                    # monostatic sum pruned genuinely overlapping pairs.
                    if dist > area_a.footprint_radius_km() + area_b.footprint_radius_km():
                        continue
                    overlap = compute_delay_bin_overlap(area_a, area_b)
                    if overlap["overlap_ratio"] > 0:
                        pair_overlaps.append({
                            "node_a": a_id,
                            "node_b": b_id,
                            **overlap,
                        })

            if self.detection_areas:
                areas = list(self.detection_areas.values())
                avg_lat = sum(a.rx_lat for a in areas) / len(areas)
                avg_lon = sum(a.rx_lon for a in areas) / len(areas)
                suggestions = coverage_suggestion(
                    areas, avg_lat, avg_lon,
                    trust_scores=self.trust_scores,
                )
            else:
                suggestions = []

            blocked = [
                nid for nid, rep in self.reputations.items() if rep.blocked
            ]

            result = {
                "pair_overlaps": pair_overlaps,
                "coverage_suggestions": suggestions,
                "blocked_nodes": blocked,
            }
            self._cross_node_cache = result
            self._cross_node_cache_ts = now
            return result

    # ── Persistent storage ────────────────────────────────────────────────

    def _coverage_map_path(self, node_id: str) -> str:
        safe_id = node_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self._storage_dir, f"coverage_{safe_id}.json")

    def _empirical_path(self, node_id: str) -> str:
        safe_id = node_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self._storage_dir, f"empirical_{safe_id}.json")

    def _load_coverage_maps(self):
        if not self._storage_dir or not os.path.isdir(self._storage_dir):
            return
        for fname in os.listdir(self._storage_dir):
            if fname.startswith("coverage_") and fname.endswith(".json"):
                try:
                    path = os.path.join(self._storage_dir, fname)
                    cmap = HistoricalCoverageMap.load_from_file(path)
                    self.coverage_maps[cmap.node_id] = cmap
                except Exception:
                    pass
            elif fname.startswith("empirical_") and fname.endswith(".json"):
                try:
                    path = os.path.join(self._storage_dir, fname)
                    ec = EmpiricalCoverageState.load_from_file(path)
                    # Derive node_id from filename: empirical_<safe_id>.json
                    node_id = fname[len("empirical_"):-len(".json")]
                    self.empirical_coverages[node_id] = ec
                except Exception:
                    pass

    def save_coverage_maps(self):
        if not self._storage_dir:
            return
        os.makedirs(self._storage_dir, exist_ok=True)
        # Under the (re-entrant) save lock: register/retire mutate these
        # dicts from other threads, and an unguarded iteration raised
        # RuntimeError mid-save.
        with self._save_lock:
            for node_id, cmap in list(self.coverage_maps.items()):
                if cmap.entries:
                    cmap.save_to_file(self._coverage_map_path(node_id))
            for node_id, ec in list(self.empirical_coverages.items()):
                if ec.n_points > 0:
                    ec.save_to_file(self._empirical_path(node_id))
            self._last_save_time = time.time()

    def maybe_auto_save(self):
        if not self._storage_dir:
            return
        if (time.time() - self._last_save_time) <= self._save_interval_s:
            return
        # Non-blocking: only one thread runs the save; others skip immediately.
        if not self._save_lock.acquire(blocking=False):
            return
        try:
            # Double-check interval after acquiring lock (another thread may have
            # already updated _last_save_time inside save_coverage_maps).
            if (time.time() - self._last_save_time) <= self._save_interval_s:
                return
            self.save_coverage_maps()
        finally:
            self._save_lock.release()
