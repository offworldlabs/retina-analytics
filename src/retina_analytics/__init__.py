"""Node analytics package — trust, reputation, coverage, cross-node analysis."""

from retina_analytics.manager import NodeAnalyticsManager
from retina_analytics.trust import AdsReportEntry, TrustScoreState
from retina_analytics.detection_area import DetectionAreaState
from retina_analytics.metrics import NodeMetrics
from retina_analytics.reputation import NodeReputation
from retina_analytics.coverage import HistoricalCoverageMap, CoverageMapEntry
from retina_analytics.cross_node import compute_delay_bin_overlap, coverage_suggestion
from retina_analytics.association import InterNodeAssociator
from retina_analytics.constants import (
    C_KM_US, R_EARTH, YAGI_BEAM_WIDTH_DEG, YAGI_MAX_RANGE_KM, haversine_km,
)

__all__ = [
    "NodeAnalyticsManager",
    "AdsReportEntry",
    "TrustScoreState",
    "DetectionAreaState",
    "NodeMetrics",
    "NodeReputation",
    "HistoricalCoverageMap",
    "CoverageMapEntry",
    "InterNodeAssociator",
    "compute_delay_bin_overlap",
    "coverage_suggestion",
    "C_KM_US",
    "R_EARTH",
    "YAGI_BEAM_WIDTH_DEG",
    "YAGI_MAX_RANGE_KM",
    "haversine_km",
]