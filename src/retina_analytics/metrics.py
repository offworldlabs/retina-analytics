"""Per-node availability, SNR, and track quality metrics."""

import math
import time
from dataclasses import dataclass, field

from retina_analytics.availability import DAY_MINUTES, MinuteRing, minute_of, share

# What a restart carries over (see to_state).  The last heartbeat is not among
# them: restored, it would read as stale until the node next beat.
_PERSISTED_COUNTS = (
    "total_frames",
    "total_detections",
    "total_tracks",
    "geolocated_tracks",
    "_snr_sum",
    "_snr_count",
    "_snr_max",
)


@dataclass
class NodeMetrics:
    """Availability / SNR / track quality metrics for one node."""

    node_id: str
    # Where its availability is measured from.  Kept across re-registration.
    first_seen: float = field(default_factory=lambda: time.time())
    last_heartbeat: float = 0.0
    total_frames: int = 0
    total_detections: int = 0
    total_tracks: int = 0
    geolocated_tracks: int = 0
    # SNR stats
    _snr_sum: float = 0.0
    _snr_count: int = 0
    _snr_max: float = 0.0
    # Track quality / gap detection
    _frame_timestamps: list = field(default_factory=list)
    _max_frame_ts: int = 500
    gap_threshold_s: float = 60.0
    # Dedup state for the track counters.  Track ids are monotonically new per
    # node, so a bounded seen-set gives exact distinct counts while capping
    # memory: an id evicted by the clear can never reappear.
    _seen_track_ids: set = field(default_factory=set)
    _seen_geo_ids: set = field(default_factory=set)
    _MAX_SEEN_IDS: int = 4096
    # The minutes of the trailing week in which it delivered a frame.
    _minutes: MinuteRing = field(default_factory=MinuteRing)

    def record_tracks(self, confirmed_ids, geolocated_ids=()):
        """Count distinct confirmed / geolocated track ids.

        total_tracks and geolocated_tracks were declared, exported through
        summary(), and never incremented anywhere — the admin per-node track
        count read a permanent 0.
        """
        for tid in confirmed_ids:
            if tid not in self._seen_track_ids:
                self._seen_track_ids.add(tid)
                self.total_tracks += 1
        for tid in geolocated_ids:
            if tid not in self._seen_geo_ids:
                self._seen_geo_ids.add(tid)
                self.geolocated_tracks += 1
        if len(self._seen_track_ids) > self._MAX_SEEN_IDS:
            self._seen_track_ids.clear()
        if len(self._seen_geo_ids) > self._MAX_SEEN_IDS:
            self._seen_geo_ids.clear()

    def record_delivery(self, now: float | None = None):
        """Count the minute as one in which the node delivered."""
        self._minutes.mark(minute_of(time.time() if now is None else now))

    def record_frame(self, frame: dict, now: float | None = None):
        self.record_delivery(now)
        self.total_frames += 1
        delays = frame.get("delay", [])
        self.total_detections += len(delays)
        for s in frame.get("snr", []):
            self._snr_sum += s
            self._snr_count += 1
            if s > self._snr_max:
                self._snr_max = s
        ts = frame.get("timestamp")
        if ts is not None:
            self._frame_timestamps.append(ts / 1000.0 if ts > 1e12 else ts)
            if len(self._frame_timestamps) > self._max_frame_ts:
                self._frame_timestamps = self._frame_timestamps[-self._max_frame_ts :]

    def record_heartbeat(self):
        self.last_heartbeat = time.time()

    @property
    def avg_snr(self) -> float:
        return self._snr_sum / self._snr_count if self._snr_count else 0.0

    @property
    def avg_detections_per_frame(self) -> float:
        return self.total_detections / self.total_frames if self.total_frames else 0.0

    @property
    def gap_stats(self) -> dict:
        if len(self._frame_timestamps) < 2:
            return {"gap_count": 0, "avg_gap_s": 0.0, "max_gap_s": 0.0, "continuity_ratio": 1.0}
        ts_sorted = sorted(self._frame_timestamps)
        gaps = []
        total_intervals = 0
        good_intervals = 0
        for i in range(1, len(ts_sorted)):
            dt = ts_sorted[i] - ts_sorted[i - 1]
            total_intervals += 1
            if dt > self.gap_threshold_s:
                gaps.append(dt)
            else:
                good_intervals += 1
        return {
            "gap_count": len(gaps),
            "avg_gap_s": round(sum(gaps) / len(gaps), 2) if gaps else 0.0,
            "max_gap_s": round(max(gaps), 2) if gaps else 0.0,
            "continuity_ratio": round(good_intervals / total_intervals, 4) if total_intervals else 1.0,
        }

    def availability(self, up: MinuteRing, now: float | None = None) -> dict:
        """The share of the minutes the server was `up` in which this node
        delivered a frame, over the trailing week and the trailing day.

        Counted from the first whole minute after first_seen to the last whole
        minute before now, so neither a partial first minute nor the minute
        in progress counts against it.  None until one such minute has passed.
        """
        start = math.ceil(self.first_seen / 60)
        last = minute_of(time.time() if now is None else now) - 1
        seen_week, up_week = share(self._minutes, up, start, last)
        seen_day, up_day = share(self._minutes, up, max(start, last - DAY_MINUTES + 1), last)
        return {
            "availability_7d": round(seen_week / up_week, 4) if up_week else None,
            "availability_24h": round(seen_day / up_day, 4) if up_day else None,
            "availability_measured_s": up_week * 60,
        }

    def to_state(self) -> dict:
        return {
            "node_id": self.node_id,
            "first_seen": self.first_seen,
            **{name: getattr(self, name) for name in _PERSISTED_COUNTS},
            "minutes": self._minutes.to_state(),
        }

    @classmethod
    def from_state(cls, state: dict) -> "NodeMetrics":
        metrics = cls(node_id=state["node_id"], first_seen=state["first_seen"])
        # A count saved before it was persisted starts from zero.
        for name in _PERSISTED_COUNTS:
            setattr(metrics, name, state.get(name, getattr(metrics, name)))
        metrics._minutes = MinuteRing.from_state(state["minutes"])
        return metrics

    def summary(self, up: MinuteRing, now: float | None = None) -> dict:
        return {
            "node_id": self.node_id,
            **self.availability(up, now),
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "avg_detections_per_frame": round(self.avg_detections_per_frame, 2),
            "avg_snr": round(self.avg_snr, 2),
            "max_snr": round(self._snr_max, 2),
            "total_tracks": self.total_tracks,
            "geolocated_tracks": self.geolocated_tracks,
            "track_quality": self.gap_stats,
        }
