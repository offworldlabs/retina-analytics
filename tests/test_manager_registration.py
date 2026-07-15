"""Behaviour of NodeAnalyticsManager.register_node for coverage review follow-ups."""

import os

from retina_analytics.manager import NodeAnalyticsManager
from retina_analytics.constants import bearing_deg

_RX_LAT, _RX_LON = 32.90, -97.00
_TX_LAT, _TX_LON = 32.78, -96.80
_CFG = dict(rx_lat=_RX_LAT, rx_lon=_RX_LON, tx_lat=_TX_LAT, tx_lon=_TX_LON, max_range_km=50)


def _broadside():
    return (bearing_deg(_RX_LAT, _RX_LON, _TX_LAT, _TX_LON) + 90.0) % 360.0


def test_manager_honors_explicit_aim():
    m = NodeAnalyticsManager()
    m.register_node("N", {**_CFG, "beam_azimuth_deg": 123.0})
    assert m.detection_areas["N"].beam_azimuth_deg == 123.0


def test_manager_defaults_to_broadside_when_unaimed():
    m = NodeAnalyticsManager()
    m.register_node("N", dict(_CFG))
    assert abs(m.detection_areas["N"].beam_azimuth_deg - _broadside()) < 1e-4
