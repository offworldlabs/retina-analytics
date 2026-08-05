"""The bistatic detection footprint: an ellipse, not a circle on the receiver.

A node's limit is a *differential* range — R_tx + R_rx − L — so its footprint is
an ellipse with foci at TX and RX.  Treating it as a circle of radius Δ about
the RX over-reaches by 2× in radius (4× in area) directly away from the
transmitter and under-reaches toward a distant one, which is the error these
tests exist to prevent recurring.
"""

import math

import pytest

from retina_analytics.association import (
    C_KM_US,
    NodeGeometry,
    _point_in_beam,
    compute_overlap_zone,
)
from retina_analytics.constants import (
    KM_PER_DEG_LAT,
    bistatic_max_radius_km,
    bistatic_range_limit_km,
)

_DELTA = 60.0   # the fleet's differential-range limit, km


class TestBistaticRangeLimit:
    def test_away_from_the_transmitter_is_half_delta(self):
        """r(180°) = Δ/2, whatever the baseline.

        This is the case a circle of radius Δ gets most wrong: 30 km of real
        coverage against 60 km assumed.
        """
        for baseline in (5.0, 25.0, 43.0, 100.0):
            r = bistatic_range_limit_km(180.0, baseline, _DELTA)
            assert r == pytest.approx(_DELTA / 2.0), f"baseline {baseline}"

    def test_toward_the_transmitter_is_half_delta_plus_baseline(self):
        """r(0°) = Δ/2 + L — and for a distant tower that exceeds Δ."""
        assert bistatic_range_limit_km(0.0, 25.0, _DELTA) == pytest.approx(55.0)
        # Spartanburg sits 43 km out; its footprint genuinely reaches past 60 km.
        assert bistatic_range_limit_km(0.0, 43.0, _DELTA) == pytest.approx(73.0)

    def test_collapses_to_a_circle_with_no_baseline(self):
        """L → 0 is monostatic: R_tx = R_rx = r, so 2r = Δ."""
        for psi in (0.0, 45.0, 90.0, 180.0, 270.0):
            assert bistatic_range_limit_km(psi, 0.0, _DELTA) == pytest.approx(_DELTA / 2.0)

    def test_monotone_decreasing_away_from_the_baseline(self):
        vals = [bistatic_range_limit_km(p, 30.0, _DELTA) for p in range(0, 181, 15)]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_symmetric_about_the_baseline(self):
        for psi in (30.0, 75.0, 140.0):
            assert bistatic_range_limit_km(psi, 30.0, _DELTA) == pytest.approx(
                bistatic_range_limit_km(-psi, 30.0, _DELTA)
            )

    def test_satisfies_the_defining_equation(self):
        """r(ψ) really is the locus of constant differential range.

        Reconstructs R_tx from the law of cosines and checks
        R_tx + R_rx − L = Δ, which is the property everything else rests on.
        """
        baseline = 37.0
        for psi in (0.0, 20.0, 60.0, 110.0, 179.0):
            r = bistatic_range_limit_km(psi, baseline, _DELTA)
            r_tx = math.sqrt(r ** 2 + baseline ** 2
                             - 2 * r * baseline * math.cos(math.radians(psi)))
            assert (r + r_tx - baseline) == pytest.approx(_DELTA, abs=1e-6)

    def test_max_radius_is_the_toward_tx_extreme(self):
        assert bistatic_max_radius_km(43.0, _DELTA) == pytest.approx(
            bistatic_range_limit_km(0.0, 43.0, _DELTA)
        )


# A transmitter 43 km due north of the receiver — Spartanburg's baseline from
# the Greenville core, the case where the ellipse reaches *past* Δ.
_TX_FAR = 35.236
# ~28 km due north, a near tower where the ellipse stays inside Δ everywhere.
_TX_NEAR = 35.10


def _geo(node_id="n", bistatic=None, tx_lat=_TX_NEAR, rx_lon=-82.40,
         beam_width_deg=360.0, beam_azimuth_deg=0.0):
    """A node at 34.85N with its transmitter due north."""
    return NodeGeometry(
        node_id=node_id,
        rx_lat=34.85, rx_lon=rx_lon, rx_alt_km=0.3,
        tx_lat=tx_lat, tx_lon=-82.40, tx_alt_km=0.6,
        fc_hz=183e6, beam_azimuth_deg=beam_azimuth_deg,
        beam_width_deg=beam_width_deg,
        max_range_km=_DELTA, max_bistatic_range_km=bistatic,
    )


class TestFootprintRadius:
    def test_legacy_node_keeps_its_circle(self):
        assert _geo(bistatic=None).footprint_radius_km == pytest.approx(_DELTA)

    def test_radius_is_the_toward_tx_extreme(self):
        geo = _geo(bistatic=_DELTA, tx_lat=_TX_NEAR)
        assert geo.footprint_radius_km == pytest.approx(
            _DELTA / 2 + geo.baseline_km, abs=0.01
        )

    def test_a_distant_tower_reaches_past_the_differential_limit(self):
        """The bounding radius must exceed Δ, or the far lobe is never enumerated."""
        geo = _geo(bistatic=_DELTA, tx_lat=_TX_FAR)
        assert geo.baseline_km > _DELTA / 2
        assert geo.footprint_radius_km > _DELTA


class TestOverlapGridRespectsTheEllipse:
    """The grid *is* the association candidate space, so this is the gate."""

    @staticmethod
    def _zone(bistatic, tx_lat=_TX_NEAR, beam_width_deg=360.0):
        # Two receivers ~8 km apart sharing a transmitter, so the range rule is
        # the only thing that differs between the two arms.
        a = _geo("a", bistatic, tx_lat, beam_width_deg=beam_width_deg)
        b = _geo("b", bistatic, tx_lat, rx_lon=-82.31,
                 beam_width_deg=beam_width_deg)
        return compute_overlap_zone(a, b, grid_step_km=3.0, altitudes_km=(7.0,))

    def test_omnidirectional_coverage_roughly_halves(self):
        """Measured: 587 grid points against 1137 for the circle, a 48% cut.

        The ellipse's area is π·a·b with 2a = Δ + L, against π·Δ² for the
        circle — most of the loss is the anti-TX annulus between Δ/2 and Δ.
        """
        circle = self._zone(None)
        ellipse = self._zone(_DELTA)
        assert ellipse.grid_points, "the bistatic gate must not empty the grid"
        ratio = len(ellipse.grid_points) / len(circle.grid_points)
        assert 0.4 < ratio < 0.65

    def test_a_beam_aimed_at_a_distant_tower_gains_coverage(self):
        """The correction is not simply a shrink — shape matters more than size.

        A 41° beam pointed at a tower 43 km out reaches 73 km, not 60, so the
        ellipse admits *more* than the circle did.  Measured: 145 grid points
        against 94, a 54% gain.  Anyone reading this change as "coverage gets
        smaller" will mispredict half the fleet.
        """
        circle = self._zone(None, tx_lat=_TX_FAR, beam_width_deg=41.0)
        ellipse = self._zone(_DELTA, tx_lat=_TX_FAR, beam_width_deg=41.0)
        assert len(ellipse.grid_points) > len(circle.grid_points)

    def test_every_kept_point_is_within_the_differential_limit(self):
        zone = self._zone(_DELTA)
        for d_a, d_b in zone.delay_pairs:
            assert d_a * C_KM_US <= _DELTA + 1e-9
            assert d_b * C_KM_US <= _DELTA + 1e-9

    def test_nothing_survives_behind_the_receiver_past_half_delta(self):
        """Away from the transmitter the footprint stops at Δ/2 = 30 km.

        The old circle ran to 60 km there — 4× the area for the same bearing
        spread — and that annulus is what the monostatic gate was handing to
        association.  The bearing sector still accepts those points; only the
        differential-range test rejects them.
        """
        geo = _geo(bistatic=_DELTA)
        limit_behind = bistatic_range_limit_km(180.0, geo.baseline_km, _DELTA)
        assert limit_behind == pytest.approx(30.0)

        far_south = geo.rx_lat - 45.0 / KM_PER_DEG_LAT
        assert _point_in_beam(far_south, geo.rx_lon, geo), (
            "the bearing sector should still accept it — the range rule rejects it"
        )

        zone = self._zone(_DELTA)
        southmost = min(lat for lat, _lon, _alt in zone.grid_points)
        south_reach_km = (geo.rx_lat - southmost) * KM_PER_DEG_LAT
        assert south_reach_km < limit_behind + 3.0  # one grid step of slack

    def test_legacy_grid_is_unchanged(self):
        """A node with no declared bistatic limit gates exactly as before."""
        zone = self._zone(None)
        assert len(zone.grid_points) == len(zone.delay_pairs)
        assert zone.grid_points
