"""Regression tests for two empirical-coverage fixes.

BUG A — bow-tie sector: to_polygon emitted sector vertices in bin-index order,
which self-intersects when the beam straddles north (in-beam bins wrap 71→0).
The fix sorts in-beam bins by signed angular offset from beam_azimuth_deg.

BUG B — giant polygon: a single mis-attributed far detection flung a vertex
thousands of km out. add_point now rejects detections beyond
max_range_km × range_clamp_mult, and to_polygon clamps each bin's P85 to the
same bound (covering states loaded from disk without max_range_km).
"""

import math

import pytest
from shapely.geometry import Polygon

from retina_analytics.empirical_coverage import (
    _DEG_PER_BIN,
    MIN_POINTS,
    N_BINS,
    EmpiricalCoverageState,
    _bearing_and_range,
    _p85,
)

RX_LAT, RX_LON = 33.4484, -112.0740  # Phoenix, AZ
BEAM_WIDTH_DEG = 43.9


def _offset_point(bearing_deg: float, range_km: float) -> tuple[float, float]:
    bearing_rad = math.radians(bearing_deg)
    cos_lat = math.cos(math.radians(RX_LAT))
    lat = RX_LAT + (range_km * math.cos(bearing_rad)) / 111.320
    lon = RX_LON + (range_km * math.sin(bearing_rad)) / (111.320 * cos_lat)
    return lat, lon


def _in_beam_bin_count(beam_azimuth_deg: float, beam_width_deg: float) -> int:
    half = beam_width_deg / 2.0
    count = 0
    for bin_idx in range(N_BINS):
        centre = bin_idx * _DEG_PER_BIN
        diff = (centre - beam_azimuth_deg + 180.0) % 360.0 - 180.0
        if abs(diff) <= half:
            count += 1
    return count


def _state_with_beam_calibration(
    beam_azimuth_deg: float,
    beam_width_deg: float = BEAM_WIDTH_DEG,
    range_km: float = 30.0,
    n_points: int = MIN_POINTS + 10,
    max_range_km: float | None = None,
) -> EmpiricalCoverageState:
    cov = EmpiricalCoverageState(RX_LAT, RX_LON, max_range_km=max_range_km)
    half = beam_width_deg / 2.0
    for i in range(n_points):
        frac = i / (n_points - 1)
        bearing = (beam_azimuth_deg - half + frac * beam_width_deg) % 360.0
        cov.add_point(*_offset_point(bearing, range_km))
    return cov


def _ring_to_polygon(poly: list[list[float]]) -> Polygon:
    return Polygon([(lon, lat) for lat, lon in poly])


class TestSectorPolygonValidity:
    @pytest.mark.parametrize("az", [0.0, 10.0, 12.9, 90.0, 180.0, 270.0, 350.0, 355.0])
    def test_sector_polygon_is_simple(self, az):
        cov = _state_with_beam_calibration(az)
        poly = cov.to_polygon(beam_azimuth_deg=az, beam_width_deg=BEAM_WIDTH_DEG)
        assert poly is not None
        shape = _ring_to_polygon(poly)
        assert shape.is_valid

    @pytest.mark.parametrize("az", [0.0, 10.0, 12.9, 90.0, 180.0, 270.0, 350.0, 355.0])
    def test_sector_has_apex_at_rx_and_expected_vertex_count(self, az):
        cov = _state_with_beam_calibration(az)
        poly = cov.to_polygon(beam_azimuth_deg=az, beam_width_deg=BEAM_WIDTH_DEG)
        rx = [round(RX_LAT, 5), round(RX_LON, 5)]
        assert poly[0] == rx
        assert poly[-1] == rx
        expected = _in_beam_bin_count(az, BEAM_WIDTH_DEG) + 2
        assert len(poly) == expected

    def test_full_circle_polygon_is_valid(self):
        cov = EmpiricalCoverageState(RX_LAT, RX_LON)
        n_points = 100
        for i in range(n_points):
            bearing = (360.0 / n_points) * i
            cov.add_point(*_offset_point(bearing, 30.0))
        poly = cov.to_polygon(beam_azimuth_deg=None)
        assert poly is not None
        assert _ring_to_polygon(poly).is_valid


class TestRangeClampAndOutlierReject:
    def test_add_point_rejects_far_detection(self):
        cov = EmpiricalCoverageState(RX_LAT, RX_LON, max_range_km=50.0)
        for _ in range(25):
            cov.add_point(*_offset_point(30.0, 30.0))
        cov.add_point(*_offset_point(45.0, 3000.0))
        assert cov.n_points == 25
        for b in cov._bins:
            for r in b:
                assert r < 100.0

    def test_to_polygon_clamps_when_state_has_no_max_range(self):
        cov = EmpiricalCoverageState(RX_LAT, RX_LON, max_range_km=None)
        for _ in range(25):
            cov.add_point(*_offset_point(30.0, 30.0))
        cov.add_point(*_offset_point(35.0, 3000.0))
        poly = cov.to_polygon(beam_azimuth_deg=30.0, beam_width_deg=BEAM_WIDTH_DEG, max_range_km=50.0)
        assert poly is not None
        clamp = 50.0 * cov.range_clamp_mult
        for lat, lon in poly[1:-1]:
            _, range_km = _bearing_and_range(RX_LAT, RX_LON, lat, lon)
            assert range_km <= clamp + 1.0

    def test_clamp_does_not_distort_normal_ranges(self):
        cov = _state_with_beam_calibration(30.0, range_km=30.0, max_range_km=50.0)
        poly = cov.to_polygon(beam_azimuth_deg=30.0, beam_width_deg=BEAM_WIDTH_DEG, max_range_km=50.0)
        assert poly is not None
        for lat, lon in poly[1:-1]:
            _, range_km = _bearing_and_range(RX_LAT, RX_LON, lat, lon)
            assert 25.0 < range_km < 35.0

    def test_add_point_rejects_far_when_state_max_range_is_none(self):
        cov = EmpiricalCoverageState(RX_LAT, RX_LON, max_range_km=None)
        for _ in range(25):
            cov.add_point(*_offset_point(30.0, 30.0))
        cov.add_point(*_offset_point(45.0, 3000.0))
        assert cov.n_points == 25  # far point rejected via YAGI fallback bound
        for b in cov._bins:
            for r in b:
                assert r < 200.0

    def test_to_polygon_clamps_when_no_max_range_anywhere(self):
        cov = EmpiricalCoverageState(RX_LAT, RX_LON, max_range_km=None)
        for _ in range(25):
            cov.add_point(*_offset_point(30.0, 30.0))
        poly = cov.to_polygon(beam_azimuth_deg=30.0, beam_width_deg=BEAM_WIDTH_DEG)
        assert poly is not None
        clamp = 50.0 * cov.range_clamp_mult  # YAGI_MAX_RANGE_KM * mult
        for lat, lon in poly[1:-1]:
            _, range_km = _bearing_and_range(RX_LAT, RX_LON, lat, lon)
            assert range_km <= clamp + 1.0

    def test_from_dict_without_max_range_is_bounded(self):
        cov = EmpiricalCoverageState(RX_LAT, RX_LON, max_range_km=None)
        for _ in range(25):
            cov.add_point(*_offset_point(30.0, 30.0))
        d = cov.to_dict()
        d.pop("max_range_km", None)  # simulate a pre-commit persisted state
        restored = EmpiricalCoverageState.from_dict(d)
        restored.add_point(*_offset_point(45.0, 3000.0))
        assert restored.n_points == 25  # still rejects the far point


class TestP85OutlierRobustness:
    def test_p85_degenerates_to_max_in_sparse_bin(self):
        # Characterisation: with few samples, P85 == max, so a lone mid-range
        # mis-attribution spikes that bearing. Documents current behaviour.
        assert _p85([30.0, 30.0, 250.0]) == 250.0
