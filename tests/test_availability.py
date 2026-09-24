"""Node availability: the share of the minutes this server was up in which a node delivered a frame."""

import base64
import json
import time

from retina_analytics.availability import DAY_MINUTES, WINDOW_MINUTES, MinuteRing
from retina_analytics.manager import NodeAnalyticsManager
from retina_analytics.metrics import NodeMetrics

M0 = 29_833_334  # an epoch minute
FRAME = {"delay": [1.0, 2.0], "doppler": [0.0, 0.0], "snr": [12.0, 14.0]}
CFG = {"rx_lat": 34.8, "rx_lon": -82.4, "tx_lat": 34.9, "tx_lon": -82.2}


def at(minute, second=30):
    return (M0 + minute) * 60.0 + second


def server_up(minutes):
    ring = MinuteRing()
    for minute in minutes:
        ring.mark(M0 + minute)
    return ring


def delivering(minutes, first_seen=0):
    node = NodeMetrics(node_id="n1", first_seen=at(first_seen, 0))
    for minute in minutes:
        node.record_frame(FRAME, now=at(minute))
    return node


def test_the_share_of_up_minutes_with_a_frame():
    s = delivering(range(80)).summary(server_up(range(100)), now=at(100))
    assert s["availability_7d"] == 0.8
    assert s["availability_measured_s"] == 100 * 60


def test_minutes_the_server_was_down_count_for_no_one():
    s = delivering(range(100)).summary(server_up([*range(50), *range(60, 100)]), now=at(100))
    assert s["availability_7d"] == 1.0
    assert s["availability_measured_s"] == 90 * 60


def test_a_node_is_measured_from_when_it_was_first_seen():
    s = delivering(range(50, 100), first_seen=50).summary(server_up(range(100)), now=at(100))
    assert s["availability_7d"] == 1.0
    assert s["availability_measured_s"] == 50 * 60


def test_the_minute_in_progress_is_not_counted():
    s = delivering(range(10)).summary(server_up(range(11)), now=at(10))
    assert s["availability_7d"] == 1.0


def test_nothing_to_report_before_a_whole_minute_is_measured():
    s = delivering([5], first_seen=5).summary(server_up(range(6)), now=at(5))
    assert s["availability_7d"] is None
    assert s["availability_24h"] is None
    assert s["availability_measured_s"] == 0


def test_the_day_figure_reads_only_the_last_day():
    s = delivering(range(DAY_MINUTES)).summary(server_up(range(2 * DAY_MINUTES)), now=at(2 * DAY_MINUTES))
    assert s["availability_7d"] == 0.5
    assert s["availability_24h"] == 0.0


def test_the_week_forgets_what_came_before_it():
    end = WINDOW_MINUTES + 100
    s = delivering(range(100)).summary(server_up(range(end)), now=at(end))
    assert s["availability_7d"] == 0.0
    assert s["availability_measured_s"] == WINDOW_MINUTES * 60


# A minute a week old shares its slot with the minute now, so a slot must be
# cleared as the week moves past it rather than read as this week's.
def test_a_minute_a_week_old_does_not_read_as_this_weeks():
    end = WINDOW_MINUTES + 100
    node = delivering([*range(100), *range(WINDOW_MINUTES + 50, end)])
    s = node.summary(server_up(range(end)), now=at(end))
    assert s["availability_7d"] == round(50 / WINDOW_MINUTES, 4)


def test_a_frame_recorded_late_still_counts():
    s = delivering([7, 5]).summary(server_up(range(10)), now=at(10))
    assert s["availability_7d"] == 0.2


def test_a_restored_node_keeps_its_counts_and_its_week():
    node = delivering(range(80))
    up = server_up(range(100))
    restored = NodeMetrics.from_state(json.loads(json.dumps(node.to_state())))
    restored_up = MinuteRing.from_state(json.loads(json.dumps(up.to_state())))
    assert restored.summary(restored_up, now=at(100)) == node.summary(up, now=at(100))
    assert restored.total_detections == 160


def test_the_manager_measures_nodes_against_its_own_clock(monkeypatch):
    clock = [at(0, 0)]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    mgr = NodeAnalyticsManager()
    mgr.register_node("n1", CFG)
    for minute in range(10):
        clock[0] = at(minute)
        mgr.mark_server_up()
        if minute < 8:
            mgr.record_detection_frame("n1", FRAME)
    clock[0] = at(10)
    assert mgr.get_node_summary("n1")["metrics"]["availability_7d"] == 0.8


def test_reregistration_does_not_restart_the_measurement(monkeypatch):
    clock = [at(0)]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    mgr = NodeAnalyticsManager()
    mgr.register_node("n1", CFG)
    clock[0] = at(30)
    mgr.register_node("n1", CFG)
    assert mgr.metrics["n1"].first_seen == at(0)


# Availability is whether a node delivers; whether it is believed is
# reputation's to say.
def test_a_blocked_node_still_counts_as_delivering(monkeypatch):
    clock = [at(0, 0)]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    mgr = NodeAnalyticsManager()
    mgr.register_node("n1", CFG)
    mgr.reputations["n1"].blocked = True
    for minute in range(10):
        clock[0] = at(minute)
        mgr.mark_server_up()
        assert mgr.record_detection_frame("n1", FRAME) is False
    clock[0] = at(10)
    assert mgr.metrics["n1"].total_frames == 0
    assert mgr.get_node_summary("n1")["metrics"]["availability_7d"] == 1.0


def test_a_count_the_saved_state_lacks_starts_at_zero():
    state = delivering(range(3)).to_state()
    del state["geolocated_tracks"]
    restored = NodeMetrics.from_state(state)
    assert (restored.total_frames, restored.geolocated_tracks) == (3, 0)


def test_a_saved_week_of_another_length_is_not_restored():
    state = server_up(range(10)).to_state()
    state["bits"] = base64.b64encode(b"\xff" * (WINDOW_MINUTES // 8 + 1)).decode()
    assert MinuteRing.from_state(state).bits(M0, M0 + 9) == 0
