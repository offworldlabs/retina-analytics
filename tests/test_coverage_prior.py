"""The empirical coverage prior (off/shadow) and the learned FOV (FOV_MODE
shadow/active) that extends the same accumulated bins.

The off-mode polygon is fed from ADS-B fixes alone, so it grows where
cooperative traffic flies and stays empty elsewhere.  That asymmetry is the
whole safety argument for TestObservedLimitAbstains/TestBinBoundaries below:
an empty bearing means nobody flew there, not that the node is deaf, so the
observed limit may only ever *tighten* the theoretical footprint.  Read as an
upper bound it would carve blind spots into perfectly good coverage.

TestAsymmetricLearning covers the learned FOV's different, deliberate
asymmetry: it broadens fast off positives (K=3 to open a bin, no percentile
floor) and shrinks only on sustained negative evidence — never on mere
absence — because a tracked target disappearing where it was predicted
detectable is a real signal that traffic-percentile absence is not.
"""

import pytest

from retina_analytics.association import (
    InterNodeAssociator,
    NodeGeometry,
    _point_in_beam,
    compute_overlap_zone,
)
from retina_analytics.constants import bistatic_range_limit_km
from retina_analytics.empirical_coverage import (
    FOV_OPEN_MIN_POINTS,
    FOV_OPEN_MIN_SPAN_S,
    OBSERVED_LIMIT_MARGIN,
    EmpiricalCoverageState,
    _MIN_BIN_POINTS_TO_CONSTRAIN,
)
from retina_analytics.constants import KM_PER_DEG_LAT

_RX_LAT, _RX_LON = 34.85, -82.40
_TX_LAT, _TX_LON = 35.236, -82.40      # 43 km due north
_DELTA = 60.0


def _state(bistatic=_DELTA, tx=(_TX_LAT, _TX_LON)):
    ec = EmpiricalCoverageState(
        rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=_DELTA,
        tx_lat=tx[0] if tx else None, tx_lon=tx[1] if tx else None,
    )
    ec.max_bistatic_range_km = bistatic
    return ec


def _at_bearing(bearing_deg, range_km):
    """A lat/lon that many km from the RX on that bearing (flat approximation)."""
    import math
    rad = math.radians(bearing_deg)
    return (_RX_LAT + range_km * math.cos(rad) / KM_PER_DEG_LAT,
            _RX_LON + range_km * math.sin(rad) / (KM_PER_DEG_LAT * math.cos(math.radians(_RX_LAT))))


class TestClampFollowsTheEllipse:
    def test_a_far_anti_tx_point_is_refused(self):
        """The clamp is per bearing, so 'too far' depends on which way you look.

        50 km due south is inside 2x a 60 km circle — the old scalar clamp took
        it — but the footprint only reaches 30 km there, so at 2x margin the
        bound is 60 km and a 50 km point... is still admitted.  Push to 70 km,
        beyond even the doubled anti-TX reach, and it must go.
        """
        ec = _state()
        lat, lon = _at_bearing(180.0, 70.0)
        ec.add_point(lat, lon)
        assert ec.n_points == 0

    def test_the_same_range_toward_the_transmitter_is_kept(self):
        """73 km of real reach north against 30 km south — same node."""
        ec = _state()
        lat, lon = _at_bearing(0.0, 70.0)
        ec.add_point(lat, lon)
        assert ec.n_points == 1

    def test_reach_matches_the_shared_formula(self):
        ec = _state()
        baseline = 43.0
        for psi in (0.0, 90.0, 180.0):
            bearing = (0.0 + psi) % 360.0   # TX is due north
            assert ec._reach_at(bearing) == pytest.approx(
                bistatic_range_limit_km(psi, baseline, _DELTA), abs=0.2,
            )

    def test_a_node_without_a_bistatic_limit_keeps_the_circle(self):
        ec = _state(bistatic=None)
        for bearing in (0.0, 90.0, 180.0):
            assert ec._reach_at(bearing) == pytest.approx(_DELTA)

    def test_a_monostatic_node_admits_up_to_four_times_reach(self):
        """FOV_CLAMP_MULT_MONOSTATIC: a node with no declared bistatic limit
        is scored against a config guess (max_range_km — radar3's is a
        default 140), so it gets 4x admit room instead of the 2x a real
        physical ellipse earns."""
        ec = _state(bistatic=None)
        lat, lon = _at_bearing(90.0, _DELTA * 3.5)   # beyond 2x, inside 4x
        ec.add_point(lat, lon)
        assert ec.n_points == 1
        lat, lon = _at_bearing(45.0, _DELTA * 4.5)   # beyond even 4x
        ec.add_point(lat, lon)
        assert ec.n_points == 1   # the second point was rejected

    def test_a_bistatic_declared_node_keeps_the_2x_ceiling(self):
        """The ellipse is physical once max_bistatic_range_km is declared —
        2x it stays implausible, unlike the monostatic config guess."""
        ec = _state()   # bistatic declared (default)
        lat, lon = _at_bearing(90.0, ec._reach_at(90.0) * 3.5)   # beyond 2x
        ec.add_point(lat, lon)
        assert ec.n_points == 0


class TestObservedLimitAbstains:
    def test_no_evidence_means_no_constraint(self):
        """Silence is not a bound.  This is the property everything rests on."""
        assert _state().observed_limit_km(90.0) is None

    def test_a_thin_bin_still_abstains(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN - 1):
            ec.add_point(*_at_bearing(90.0, 20.0))
        assert ec.observed_limit_km(90.0) is None

    def test_enough_evidence_yields_a_limit(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 20.0))
        assert ec.observed_limit_km(90.0) == pytest.approx(20.0, abs=1.0)

    def test_a_constrained_bearing_does_not_constrain_its_neighbours(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 20.0))
        assert ec.observed_limit_km(180.0) is None

    def test_digest_tracks_what_would_constrain(self):
        ec = _state()
        before = ec.constraint_digest()
        assert set(before) == {None}
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 20.0))
        after = ec.constraint_digest()
        assert after != before
        assert sum(1 for v in after if v is not None) == 1


def _geo(coverage_limit=None):
    return NodeGeometry(
        node_id="n", rx_lat=_RX_LAT, rx_lon=_RX_LON, rx_alt_km=0.3,
        tx_lat=_TX_LAT, tx_lon=_TX_LON, tx_alt_km=0.6, fc_hz=183e6,
        beam_azimuth_deg=0.0, beam_width_deg=360.0,
        max_range_km=_DELTA, max_bistatic_range_km=_DELTA,
        coverage_limit=coverage_limit,
    )


class TestAsymmetricLearning:
    """FOV_MODE's read surface: _bin_open / limit_km / contains / wedge_state.

    Deliberately different asymmetry from TestObservedLimitAbstains above —
    this is the shadow/active-mode API over the SAME accumulated bins, not a
    second copy of the shrink-only prior.  See the module docstring.
    """

    def test_absence_never_shrinks_an_in_wedge_bin(self):
        """10 SHORT points inside the theoretical wedge must not pull the
        limit in below the theoretical reach — the old shrink-only P85 prior
        did exactly this (TestObservedLimitAbstains/TestPriorOnlyTightens's
        former subject); limit_km does not."""
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 90.0, 40.0
        theoretical = ec._reach_at(92.5)   # bin centre — limit_km's reference
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(92.5, 5.0))   # short detections only
        assert ec.limit_km(92.5) == pytest.approx(theoretical)

    def test_an_out_of_wedge_bin_opens_at_fov_open_min_points_spanning_min_span_s(self):
        """Both conditions are required: the count floor (K=3, a single
        mis-matched hex should not open a bin permanently) AND the span floor
        (>= FOV_OPEN_MIN_SPAN_S, so the count floor cannot be satisfied by
        one correlated burst — see test_a_single_pass_does_not_open_a_bin)."""
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 0.0, 20.0   # narrow wedge, north
        t0 = 1_000_000.0
        assert ec.wedge_state(180.0) == "closed"
        for dt in (0.0, FOV_OPEN_MIN_SPAN_S / 2):
            ec.add_point(*_at_bearing(180.0, 20.0), ts=t0 + dt)
        assert ec.wedge_state(180.0) == "closed"   # one short of K=3
        ec.add_point(*_at_bearing(180.0, 20.0), ts=t0 + FOV_OPEN_MIN_SPAN_S)
        assert ec.wedge_state(180.0) != "closed"

    def test_a_single_pass_does_not_open_a_bin(self):
        """FOV_OPEN_MIN_POINTS positives clustered in a few seconds — one
        aircraft's pass, or an exit smear's few emit cycles — must not open
        an out-of-wedge bin no matter how many of them land: the count alone
        cannot certify independence, only the span can."""
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 0.0, 20.0
        t0 = 1_000_000.0
        for i in range(FOV_OPEN_MIN_POINTS + 5):
            ec.add_point(*_at_bearing(180.0, 20.0), ts=t0 + i)   # a few seconds apart
        assert ec.wedge_state(180.0) == "closed"

    def test_the_same_points_spread_over_the_span_floor_do_open_it(self):
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 0.0, 20.0
        t0 = 1_000_000.0
        for i in range(FOV_OPEN_MIN_POINTS + 5):
            ec.add_point(*_at_bearing(180.0, 20.0),
                        ts=t0 + i * (FOV_OPEN_MIN_SPAN_S / FOV_OPEN_MIN_POINTS))
        assert ec.wedge_state(180.0) != "closed"

    def test_range_extends_past_theoretical_via_p95_times_margin(self):
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 90.0, 40.0
        theoretical = ec._reach_at(92.5)
        far = theoretical + 20.0   # past theoretical, inside the 2x bistatic admit ceiling
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(92.5, far))
        assert ec.limit_km(92.5) == pytest.approx(far * OBSERVED_LIMIT_MARGIN, rel=0.02)

    def test_extension_ceiling_is_2x_bistatic_but_4x_monostatic(self):
        """Same raw far detection: a bistatic-declared node cannot see past
        its 2x admit ceiling to extend from, a monostatic-guess node can."""
        far_bistatic = _state()._reach_at(92.5) * 2.5   # beyond 2x
        bi = _state()
        bi.prior_azimuth_deg, bi.prior_width_deg = 90.0, 40.0
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            bi.add_point(*_at_bearing(92.5, far_bistatic))
        assert bi.n_points == 0   # nothing admitted, nothing to extend from
        assert bi.limit_km(92.5) == pytest.approx(bi._reach_at(92.5))

        mono = _state(bistatic=None)
        mono.prior_azimuth_deg, mono.prior_width_deg = 90.0, 40.0
        far_mono = mono._reach_at(92.5) * 2.5   # beyond 2x, still inside 4x
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            mono.add_point(*_at_bearing(92.5, far_mono))
        assert mono.n_points == _MIN_BIN_POINTS_TO_CONSTRAIN
        assert mono.limit_km(92.5) == pytest.approx(far_mono * OBSERVED_LIMIT_MARGIN, rel=0.02)

    def test_shrink_needs_three_events_spanning_ten_minutes(self):
        t0 = 1_000_000.0

        # Two events, however far apart in time, are not a pattern.
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 90.0, 40.0
        theoretical = ec._reach_at(92.5)
        ec.record_disappearance(*_at_bearing(92.5, 20.0), ts=t0)
        ec.record_disappearance(*_at_bearing(92.5, 20.0), ts=t0 + 5000.0)
        assert ec.limit_km(92.5) == pytest.approx(theoretical)

        # Three events, but clustered inside one bad minute -- not sustained.
        ec2 = _state()
        ec2.prior_azimuth_deg, ec2.prior_width_deg = 90.0, 40.0
        for dt in (0.0, 20.0, 45.0):
            ec2.record_disappearance(*_at_bearing(92.5, 20.0), ts=t0 + dt)
        assert ec2.limit_km(92.5) == pytest.approx(ec2._reach_at(92.5))

        # Three events spanning >= 600 s: a sustained pattern, shrinks.
        ec3 = _state()
        ec3.prior_azimuth_deg, ec3.prior_width_deg = 90.0, 40.0
        for dt in (0.0, 300.0, 650.0):
            ec3.record_disappearance(*_at_bearing(92.5, 20.0), ts=t0 + dt)
        assert ec3.limit_km(92.5) < ec3._reach_at(92.5)
        assert ec3.limit_km(92.5) == pytest.approx(20.0 * 0.95, abs=0.5)

    def test_a_fresh_positive_reopens_a_capped_bin(self):
        """Positives supersede every negative event older than themselves —
        broaden fast, shrink slow: a bin that reopens is not haunted by
        evidence that predates the reopening."""
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 90.0, 40.0
        t0 = 1_000_000.0
        for dt in (0.0, 300.0, 650.0):
            ec.record_disappearance(*_at_bearing(92.5, 20.0), ts=t0 + dt)
        assert ec.limit_km(92.5) < ec._reach_at(92.5)
        ec.add_point(*_at_bearing(92.5, 25.0), ts=t0 + 1000.0)
        assert ec.limit_km(92.5) == pytest.approx(ec._reach_at(92.5))


def _mono_state(max_range_km=50.0):
    """A monostatic state (no TX) — reach_at is a flat max_range_km at every
    bearing, which keeps the evidence-proportional-limit tests below about
    the open/evidence machinery, not the bistatic ellipse TestClampFollows
    TheEllipse already covers.  Admit clamp is 4x (FOV_CLAMP_MULT_MONOSTATIC),
    so up to 200 km is accepted by add_point/record_disappearance."""
    return EmpiricalCoverageState(rx_lat=_RX_LAT, rx_lon=_RX_LON, max_range_km=max_range_km)


class TestOutOfWedgeEvidenceProportionalLimit:
    """limit_km's asymmetry, opposite in direction to _bin_open's: inside the
    wedge the theoretical prior is trusted and evidence may only extend it;
    outside, the prior already said "no", so an open bin reaches only as far
    as its own evidence says — never the theoretical reach it sits outside
    of.  Before this, an open out-of-wedge bin was granted the FULL
    theoretical _reach_at, which is how 3 near-RX exit-smear points at ~1 km
    used to draw and gate a 20-160 km lobe (staging 2026-08-10)."""

    def test_regression_in_wedge_bin_keeps_full_theoretical_reach_below_the_constrain_floor(self):
        """Guards the asymmetry's other half: this is the behaviour the
        out-of-wedge change must NOT touch."""
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 90.0, 40.0
        theoretical = ec._reach_at(92.5)
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN - 1):
            ec.add_point(*_at_bearing(92.5, 5.0))   # short, and below the floor
        assert ec.limit_km(92.5) == pytest.approx(theoretical)

    def test_below_own_bin_floor_the_limit_is_p95_of_the_pooled_bin_pm_1_evidence(self):
        ec = _mono_state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 0.0, 20.0   # wedge north; 180 is out
        t0 = 1_000_000.0
        i = _bin_for_bearing_of(180.0)
        # Own bin (180 deg): 2 points, short of _MIN_BIN_POINTS_TO_CONSTRAIN.
        ec.add_point(*_at_bearing(180.0, 15.0), ts=t0)
        ec.add_point(*_at_bearing(180.0, 15.0), ts=t0 + FOV_OPEN_MIN_SPAN_S)
        # Neighbour bin (176 deg, bin i-1): makes up the rest of the count
        # and the span the pool needs to OPEN the bin in the first place.
        ec.add_point(*_at_bearing(176.0, 40.0), ts=t0)
        ec.add_point(*_at_bearing(176.0, 40.0), ts=t0 + FOV_OPEN_MIN_SPAN_S / 2)
        ec.add_point(*_at_bearing(176.0, 40.0), ts=t0 + FOV_OPEN_MIN_SPAN_S)
        assert ec.wedge_state(180.0) != "closed"
        assert len(ec._bins[i]) < _MIN_BIN_POINTS_TO_CONSTRAIN

        from retina_analytics.empirical_coverage import N_BINS, _percentile
        pooled = [r for off in (-1, 0, 1) for r in ec._bins[(i + off) % N_BINS]]
        expected = _percentile(pooled, 95) * OBSERVED_LIMIT_MARGIN
        assert ec.limit_km(180.0) == pytest.approx(expected)
        # And NOT the theoretical reach this bin sits outside of.
        assert ec.limit_km(180.0) != pytest.approx(ec._reach_at(180.0))

    def test_at_own_bin_floor_the_limit_is_p95_of_only_its_own_evidence(self):
        """Once the bin's own count reaches _MIN_BIN_POINTS_TO_CONSTRAIN, a
        neighbour with wildly different ranges must NOT move the limit —
        the extension (in-wedge) and this evidence claim (out-of-wedge) are
        both a statement about what THIS bin itself has shown."""
        ec = _mono_state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 0.0, 20.0
        t0 = 1_000_000.0
        i = _bin_for_bearing_of(180.0)
        own_ranges = [10.0] * (_MIN_BIN_POINTS_TO_CONSTRAIN - 1) + [50.0]
        for k, r in enumerate(own_ranges):
            ec.add_point(*_at_bearing(180.0, r),
                        ts=t0 + k * (FOV_OPEN_MIN_SPAN_S / (len(own_ranges) - 1)))
        # A neighbour with a much larger (but still admissible) range — must
        # be excluded once the bin's own evidence clears the floor.
        ec.add_point(*_at_bearing(176.0, 150.0), ts=t0)

        from retina_analytics.empirical_coverage import _percentile
        expected = _percentile(own_ranges, 95) * OBSERVED_LIMIT_MARGIN
        assert ec.limit_km(180.0) == pytest.approx(expected)
        assert ec.limit_km(180.0) < 150.0 * OBSERVED_LIMIT_MARGIN

    def test_negative_cap_still_applies_to_the_evidence_proportional_limit(self):
        """The negative cap is the ONLY thing that shrinks a limit_km result
        below either the theoretical reach (in-wedge) or the evidence claim
        (out-of-wedge) — it must keep applying in the out-of-wedge branch,
        as a final min(), same as it always has in-wedge."""
        from retina_analytics.empirical_coverage import FOV_NEG_MIN_SPAN_S

        ec = _mono_state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 0.0, 20.0
        t0 = 1_000_000.0
        for k in range(5):
            ec.add_point(*_at_bearing(180.0, 15.0),
                        ts=t0 + k * (FOV_OPEN_MIN_SPAN_S / 4))
        before = ec.limit_km(180.0)
        assert before == pytest.approx(15.0 * OBSERVED_LIMIT_MARGIN, rel=0.02)

        neg_t0 = t0 + FOV_OPEN_MIN_SPAN_S + 1000.0   # after the last positive
        for dt in (0.0, FOV_NEG_MIN_SPAN_S / 2, FOV_NEG_MIN_SPAN_S):
            ec.record_disappearance(*_at_bearing(180.0, 8.0), ts=neg_t0 + dt)
        after = ec.limit_km(180.0)
        assert after < before
        assert after == pytest.approx(8.0 * 0.95, abs=0.5)


class TestDigestMovesOnOpenAndCap:
    """fov_digest() is the change-detection token _refresh_coverage_constraints
    (backend/services/tasks/analytics_refresh.py) compares to decide whether a
    node's overlap grids need rebuilding.  A bin's (state_code, limit) pair
    must move when the bin's *admission* changes (closed -> prior/observed,
    or a negative cap lands) — those are exactly the changes that alter which
    grid points _point_in_beam would admit.  A limit nudging within the same
    state, well under the backend's 2 km rebuild grain, is not load-bearing
    and is documented rather than asserted here (see
    TestCoverageConstraintRefresh in test_analytics_refresh_helpers.py for
    the quantization itself).
    """

    def test_a_bin_opening_changes_its_digest_entry(self):
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 0.0, 20.0   # narrow wedge, north
        i = _bin_for_bearing_of(180.0)
        before = ec.fov_digest()[i]
        assert before[0] == "closed"
        t0 = 1_000_000.0
        for k in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            # Spread across >= FOV_OPEN_MIN_SPAN_S so the bin actually opens
            # under the new span rule, not just the count floor.
            ec.add_point(*_at_bearing(180.0, 20.0),
                        ts=t0 + k * (FOV_OPEN_MIN_SPAN_S / (_MIN_BIN_POINTS_TO_CONSTRAIN - 1)))
        after = ec.fov_digest()[i]
        assert after != before
        assert after[0] in ("prior", "observed")

    def test_a_negative_cap_changes_its_digest_entry(self):
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 90.0, 40.0
        i = _bin_for_bearing_of(92.5)
        before = ec.fov_digest()[i]
        t0 = 1_000_000.0
        for dt in (0.0, 300.0, 650.0):
            ec.record_disappearance(*_at_bearing(92.5, 20.0), ts=t0 + dt)
        after = ec.fov_digest()[i]
        assert after != before
        assert after[1] < before[1]   # limit component dropped

    def test_a_full_bin_reopening_survives_the_round_trip_to_disk(self):
        """The digest is read from the same object callers persist — a
        rebuild decision made against a freshly-loaded state must see the
        same digest as the one that triggered the save."""
        ec = _state()
        ec.prior_azimuth_deg, ec.prior_width_deg = 0.0, 20.0
        t0 = 1_000_000.0
        for k in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(180.0, 20.0),
                        ts=t0 + k * (FOV_OPEN_MIN_SPAN_S / (_MIN_BIN_POINTS_TO_CONSTRAIN - 1)))
        assert ec.wedge_state(180.0) != "closed"   # sanity: the bin actually opened
        restored = EmpiricalCoverageState.from_dict(ec.to_dict())
        assert restored.fov_digest() == ec.fov_digest()


def _bin_for_bearing_of(bearing):
    from retina_analytics.empirical_coverage import _bin_for_bearing
    return _bin_for_bearing(bearing)


class TestFovGateInAssociation:
    """association.py's fov wiring: NodeGeometry.fov, _point_in_beam's fov
    branch, and rebuild_zones_for refreshing it.  fov=None (the off/shadow
    construction — InterNodeAssociator's fov_provider stays unset in those
    modes) must reproduce the legacy shrink-only behaviour exactly; a stub
    fov must REPLACE it, not add to it.
    """

    def test_legacy_path_is_pinned_with_fov_none(self):
        """With geo.fov unset this is byte-identical to the shrink-only
        prior TestPriorOnlyTightens used to pin directly."""
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 12.0))
        far = _at_bearing(90.0, 40.0)   # inside the ellipse, far past what was seen
        assert _point_in_beam(*far, _geo(None)) is True
        assert _point_in_beam(*far, _geo(ec.observed_limit_km)) is False

    def test_active_path_replaces_the_legacy_checks_entirely(self):
        """A stub fov with no bearing constraint at all, on a node whose
        THEORETICAL wedge is narrow and points somewhere else, proves the
        fov branch is a replacement — not an additional filter stacked on
        top.  Falling through to the legacy bearing test afterward would
        reject a point the fov explicitly admits."""
        class _StubFov:
            def max_limit_km(self):
                return 30.0

            def contains(self, bearing, range_km):
                return range_km <= 30.0   # no bearing constraint at all

        geo = _geo(None)
        geo.beam_azimuth_deg = 0.0
        geo.beam_width_deg = 10.0   # narrow theoretical wedge, pointed north
        geo.fov = _StubFov()

        # Due east, 20 km out: the legacy 10 deg-wide northward wedge would
        # reject this on bearing alone (90 deg off boresight).
        east_20km = _at_bearing(90.0, 20.0)
        assert _point_in_beam(*east_20km, geo) is True

        far = _at_bearing(90.0, 40.0)   # past the fov's own 30 km cap
        assert _point_in_beam(*far, geo) is False

    def test_bounding_box_grows_with_the_learned_fov(self):
        """compute_overlap_zone's grid must reach as far as the fov claims,
        even past the theoretical footprint — see NodeGeometry.effective_radius_km.

        Both nodes monostatic (no bistatic limit), so footprint_radius_km is a
        fixed 60 km regardless of position — isolating the widening to geo_a's
        fov rather than letting a baseline-sensitive bistatic ellipse
        confound the comparison.
        """
        class _WideFov:
            def max_limit_km(self):
                return 200.0   # well past the 60 km theoretical footprint

            def contains(self, bearing, range_km):
                return range_km <= self.max_limit_km()

        far_lat, far_lon = _at_bearing(90.0, 150.0)   # > 60+60, <= 200+60

        def _mono_geo(node_id, rx_lat, rx_lon, fov=None):
            return NodeGeometry(
                node_id=node_id, rx_lat=rx_lat, rx_lon=rx_lon, rx_alt_km=0.3,
                tx_lat=_TX_LAT, tx_lon=_TX_LON, tx_alt_km=0.6, fc_hz=183e6,
                beam_azimuth_deg=0.0, beam_width_deg=360.0,
                max_range_km=60.0, max_bistatic_range_km=None, fov=fov,
            )

        geo_a = _mono_geo("a", _RX_LAT, _RX_LON, fov=_WideFov())
        geo_b = _mono_geo("b", far_lat, far_lon)
        assert geo_a.footprint_radius_km + geo_b.footprint_radius_km < 150.0

        zone = compute_overlap_zone(geo_a, geo_b, grid_step_km=5.0, altitudes_km=(7.0,))
        assert zone.grid_points   # the widened bounding box actually reached node b

    def test_rebuild_zones_for_refreshes_geo_fov(self):
        class _Fov:
            def __init__(self, limit):
                self.limit = limit

            def max_limit_km(self):
                return self.limit

            def contains(self, bearing, range_km):
                return range_km <= self.limit

        fovs = {"n1": _Fov(5.0)}

        def fov_provider(node_id):
            return fovs.get(node_id)

        cfg_a = {"rx_lat": _RX_LAT, "rx_lon": _RX_LON, "rx_alt_ft": 1000,
                 "tx_lat": _TX_LAT, "tx_lon": _TX_LON, "tx_alt_ft": 2000,
                 "fc_hz": 183e6, "beam_width_deg": 360, "max_range_km": _DELTA,
                 "max_bistatic_range_km": _DELTA, "beam_azimuth_deg": 0.0}
        cfg_b = dict(cfg_a, rx_lon=-82.31, fc_hz=195e6)

        a = InterNodeAssociator(grid_step_km=3.0, fov_provider=fov_provider)
        a.register_node("n1", cfg_a)
        a.register_node("n2", cfg_b)
        assert a.node_geometries["n1"].fov is fovs["n1"]

        fovs["n1"] = _Fov(80.0)   # the node's learned fov widened
        a.rebuild_zones_for("n1")
        assert a.node_geometries["n1"].fov is fovs["n1"]
        assert a.node_geometries["n1"].fov.limit == 80.0


class TestBinBoundaries:
    """A constraint must not flicker across a 5° bin edge.

    add_point derives its bearing with a flat approximation and the association
    gate with a spherical one, so a query routinely lands one bin over from the
    evidence it belongs to.  Sampling a single bin made the constraint vanish
    at every boundary — found by a test that queried 89.87° against evidence
    filed at 90.0°, which is the difference between the two formulas.
    """

    def test_a_query_just_off_the_evidence_bearing_still_constrains(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 12.0))
        for probe in (89.87, 90.0, 90.2, 88.0, 92.0):
            assert ec.observed_limit_km(probe) is not None, probe

    def test_it_stays_local(self):
        """Neighbour widening is 3 bins, not a global smear."""
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 12.0))
        assert ec.observed_limit_km(120.0) is None
        assert ec.observed_limit_km(270.0) is None

    def test_the_most_permissive_neighbour_wins(self):
        """Erring permissive is the safe direction for a shrink-only bound."""
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(90.0, 10.0))
            ec.add_point(*_at_bearing(95.0, 25.0))
        # 92.5° sits between them; the wider of the two must win.
        assert ec.observed_limit_km(92.5) == pytest.approx(25.0, abs=1.5)

    def test_wraparound_is_handled(self):
        ec = _state()
        for _ in range(_MIN_BIN_POINTS_TO_CONSTRAIN):
            ec.add_point(*_at_bearing(0.5, 15.0))
        assert ec.observed_limit_km(359.0) is not None
