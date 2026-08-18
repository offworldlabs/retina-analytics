"""Cross-node comparison and coverage suggestion algorithms."""

import math

from retina_analytics.constants import offset_latlon, point_in_beam
from retina_analytics.detection_area import DetectionAreaState


def compute_delay_bin_overlap(
    area_a: DetectionAreaState, area_b: DetectionAreaState, bin_width_us: float = 2.0
) -> dict:
    """Compare two nodes' detection areas via delay-bin overlap (Jaccard)."""
    if area_a.n_detections == 0 or area_b.n_detections == 0:
        return {"overlap_ratio": 0.0, "shared_bins": 0, "total_bins_union": 0, "a_only": 0, "b_only": 0}

    def _bins(lo, hi):
        if lo > hi:
            return set()
        start = int(lo // bin_width_us)
        end = int(hi // bin_width_us) + 1
        return set(range(start, end))

    bins_a = _bins(area_a.min_delay, area_a.max_delay)
    bins_b = _bins(area_b.min_delay, area_b.max_delay)
    shared = bins_a & bins_b
    union = bins_a | bins_b

    return {
        "overlap_ratio": len(shared) / len(union) if union else 0.0,
        "shared_bins": len(shared),
        "total_bins_union": len(union),
        "a_only": len(bins_a - bins_b),
        "b_only": len(bins_b - bins_a),
    }


def _point_in_beam(area: DetectionAreaState, lat: float, lon: float) -> bool:
    """Check whether a lat/lon point falls inside a node's detection cone.

    Delegates to the shared gate.  This used to test a monostatic circle with a
    flat-earth bearing, ignoring the bistatic limit the node declares — so it
    scored coverage over a region up to 2x too far in radius away from the
    transmitter, and under-reached toward it.
    """
    # has_tx tests the pair: (0, 0) is the unset sentinel, but `tx_lat or
    # None` alone also nulled genuine transmitters with one zero axis.
    return point_in_beam(
        lat,
        lon,
        rx_lat=area.rx_lat,
        rx_lon=area.rx_lon,
        tx_lat=area.tx_lat if area.has_tx else None,
        tx_lon=area.tx_lon if area.has_tx else None,
        beam_azimuth_deg=area.beam_azimuth_deg,
        beam_width_deg=area.beam_width_deg,
        max_range_km=area.max_range_km,
        max_bistatic_range_km=area.max_bistatic_range_km,
    )


def _count_covering_nodes(areas: list[DetectionAreaState], lat: float, lon: float) -> int:
    return sum(1 for a in areas if _point_in_beam(a, lat, lon))


def coverage_suggestion(
    areas: list[DetectionAreaState],
    center_lat: float,
    center_lon: float,
    desired_range_km: float = 80.0,
    trust_scores: dict | None = None,
) -> list[dict]:
    """Suggest where to place additional nodes for better coverage.

    A `solver_rms_history` "saturation" strategy used to live here: when a
    caller supplied ≥10 RMS samples with <5% improvement AND overlap density
    exceeded 0.3, covered directions were suppressed in favour of
    expansion-only output.  No production caller ever passed the history —
    only tests did — so the branch was unreachable, and it dragged a full
    O(N²) pairwise-overlap pass along with it on every cache miss.  Deleted;
    the covered/uncovered logic below is the whole strategy.
    """
    suggestions = []
    directions = [
        ("N", 0),
        ("NE", 45),
        ("E", 90),
        ("SE", 135),
        ("S", 180),
        ("SW", 225),
        ("W", 270),
        ("NW", 315),
    ]

    for label, bearing_deg in directions:
        # The old expression divided by both R_EARTH *and* 111.32, so the "80 km"
        # probe points landed 0.72 km from the centre — the gap analysis was
        # measuring the middle of the metro in eight directions.  Advisory
        # output only, but it was wrong by a factor of 111.
        bearing_rad = math.radians(bearing_deg)
        test_lat, test_lon = offset_latlon(
            center_lat,
            center_lon,
            east_km=desired_range_km * math.sin(bearing_rad),
            north_km=desired_range_km * math.cos(bearing_rad),
        )

        covered = any(_point_in_beam(a, test_lat, test_lon) for a in areas)

        if covered:
            n_covering = _count_covering_nodes(areas, test_lat, test_lon)
            if n_covering < 3:
                best_trust = 0.0
                if trust_scores:
                    for a in areas:
                        ts = trust_scores.get(a.node_id)
                        if ts and ts.score > best_trust and _point_in_beam(a, test_lat, test_lon):
                            best_trust = ts.score
                suggestions.append(
                    {
                        "direction": label,
                        "bearing_deg": bearing_deg,
                        "test_point": {"lat": round(test_lat, 5), "lon": round(test_lon, 5)},
                        "gap_km": round(desired_range_km, 1),
                        "strategy": "densification",
                        "overlap_count": n_covering,
                        "nearest_trust": round(best_trust, 3),
                    }
                )
        else:
            suggestions.append(
                {
                    "direction": label,
                    "bearing_deg": bearing_deg,
                    "test_point": {"lat": round(test_lat, 5), "lon": round(test_lon, 5)},
                    "gap_km": round(desired_range_km, 1),
                    "strategy": "expansion",
                    "overlap_count": 0,
                }
            )

    suggestions.sort(key=lambda s: (-1 if s["strategy"] == "densification" else 0, -s.get("overlap_count", 0)))
    return suggestions
