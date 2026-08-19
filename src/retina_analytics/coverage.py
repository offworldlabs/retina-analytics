"""Historical ADS-B-validated coverage map accumulation with persistent storage."""

import json
import os
import time
from dataclasses import dataclass, field

from retina_analytics.constants import KM_PER_DEG_LAT, bearing_deg, km_per_deg_lon


@dataclass
class CoverageMapEntry:
    """A single ADS-B-validated detection position."""

    lat: float
    lon: float
    alt_km: float
    timestamp: float
    snr: float
    delay_error: float


@dataclass
class HistoricalCoverageMap:
    """Accumulates ADS-B-validated detection positions over time to build
    a factual coverage map for each node."""

    node_id: str
    entries: list[CoverageMapEntry] = field(default_factory=list)
    max_entries: int = 10000
    _grid: dict[tuple[int, int], dict] = field(default_factory=dict)
    _grid_resolution_deg: float = 0.01  # ~1.1 km
    # Cell cap: `entries` was capped but the grid was not — a node observed
    # across a 60 km footprint fills 10^4–10^5 cells, all serialised to disk
    # on every autosave.  20k cells ≈ a 165×165 km footprint at 0.01°;
    # beyond that the least-recently-seen cells are evicted.
    max_grid_cells: int = 20000

    def add_detection(self, lat: float, lon: float, alt_km: float, snr: float, delay_error: float):
        entry = CoverageMapEntry(
            lat=lat,
            lon=lon,
            alt_km=alt_km,
            timestamp=time.time(),
            snr=snr,
            delay_error=delay_error,
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

        grid_key = (
            round(lat / self._grid_resolution_deg),
            round(lon / self._grid_resolution_deg),
        )
        cell = self._grid.get(grid_key)
        if cell is None:
            if len(self._grid) >= self.max_grid_cells:
                self._evict_oldest_cells()
            self._grid[grid_key] = {
                "lat": lat,
                "lon": lon,
                "count": 1,
                "avg_snr": snr,
                "first_seen": time.time(),
                "last_seen": time.time(),
            }
        else:
            cell["count"] += 1
            cell["avg_snr"] = (cell["avg_snr"] * (cell["count"] - 1) + snr) / cell["count"]
            cell["last_seen"] = time.time()

    def _evict_oldest_cells(self) -> None:
        """Drop the least-recently-seen 10% of cells to make room."""
        n_drop = max(1, len(self._grid) // 10)
        for key, _ in sorted(self._grid.items(), key=lambda kv: kv[1].get("last_seen", 0.0))[:n_drop]:
            del self._grid[key]

    @property
    def coverage_area_km2(self) -> float:
        # A 0.01° × 0.01° cell is not square: the lat side is fixed but the
        # lon side shrinks by cos(lat).  Squaring KM_PER_DEG_LAT overstated
        # every area by 1/cos(lat) — 22% at the Greenville latitude.
        lat_side_km = self._grid_resolution_deg * KM_PER_DEG_LAT
        return sum(
            lat_side_km * self._grid_resolution_deg * km_per_deg_lon(cell["lat"]) for cell in self._grid.values()
        )

    @property
    def n_grid_cells(self) -> int:
        return len(self._grid)

    def get_coverage_grid(self) -> list[dict]:
        return [
            {
                "lat": cell["lat"],
                "lon": cell["lon"],
                "count": cell["count"],
                "avg_snr": round(cell["avg_snr"], 2),
                "first_seen": cell["first_seen"],
                "last_seen": cell["last_seen"],
            }
            for cell in self._grid.values()
        ]

    def estimate_beam_width(self) -> float | None:
        if len(self.entries) < 20:
            return None
        lats = [e.lat for e in self.entries]
        lons = [e.lon for e in self.entries]
        lats_sorted = sorted(lats)
        lons_sorted = sorted(lons)
        mid = len(lats_sorted) // 2
        center_lat = lats_sorted[mid]
        center_lon = lons_sorted[mid]
        bearings = [bearing_deg(center_lat, center_lon, e.lat, e.lon) for e in self.entries]
        if not bearings:
            return None
        bearings.sort()
        gaps = [(bearings[i + 1] - bearings[i]) for i in range(len(bearings) - 1)]
        gaps.append(360 - bearings[-1] + bearings[0])
        max_gap_idx = gaps.index(max(gaps))
        rotated = bearings[max_gap_idx + 1 :] + bearings[: max_gap_idx + 1]
        if not rotated:
            return None
        spread = (rotated[-1] - rotated[0]) % 360
        return min(spread, 180.0)

    def summary(self) -> dict:
        beam_est = self.estimate_beam_width()
        return {
            "node_id": self.node_id,
            "total_entries": len(self.entries),
            "grid_cells": self.n_grid_cells,
            "coverage_area_km2": round(self.coverage_area_km2, 1),
            # `is not None`: a genuine 0.0° estimate is a value, not an absence.
            "estimated_beam_width_deg": round(beam_est, 1) if beam_est is not None else None,
        }

    def save_to_file(self, path: str):
        data = {
            "node_id": self.node_id,
            "entries": [
                {
                    "lat": e.lat,
                    "lon": e.lon,
                    "alt_km": e.alt_km,
                    "timestamp": e.timestamp,
                    "snr": e.snr,
                    "delay_error": e.delay_error,
                }
                for e in self.entries
            ],
            "grid": {f"{k[0]},{k[1]}": v for k, v in self._grid.items()},
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)

    @classmethod
    def load_from_file(cls, path: str) -> "HistoricalCoverageMap":
        with open(path) as f:
            data = json.load(f)
        cmap = cls(node_id=data["node_id"])
        for e in data.get("entries", []):
            cmap.entries.append(
                CoverageMapEntry(
                    lat=e["lat"],
                    lon=e["lon"],
                    alt_km=e["alt_km"],
                    timestamp=e["timestamp"],
                    snr=e["snr"],
                    delay_error=e["delay_error"],
                )
            )
        # Re-apply the caps: a file written before they existed (or under
        # larger ones) must not resurrect unbounded state.
        if len(cmap.entries) > cmap.max_entries:
            cmap.entries = cmap.entries[-cmap.max_entries :]
        for k_str, v in data.get("grid", {}).items():
            parts = k_str.split(",")
            cmap._grid[(int(parts[0]), int(parts[1]))] = v
        while len(cmap._grid) > cmap.max_grid_cells:
            cmap._evict_oldest_cells()
        return cmap
